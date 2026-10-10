#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Sizes = std::array<size_t, 3>;
using Coords = std::array<int, 3>;

size_t checkedProduct(size_t a, size_t b) {
    if (b != 0 && a > std::numeric_limits<size_t>::max() / b)
        throw std::runtime_error("Grid size overflows addressable memory");
    return a * b;
}

// Minimize communication while accounting for uneven block sizes. Prefer keeping
// X contiguous and splitting Z when otherwise equivalent process grids exist.
Coords chooseProcessGrid(const Sizes& global, int& processes) {
    processes = static_cast<int>(std::min<size_t>(processes,
        checkedProduct(checkedProduct(global[0], global[1]), global[2])));
    for (; processes > 0; --processes) {
        Coords best{0, 0, 0};
        double bestScore = std::numeric_limits<double>::infinity();
        for (int px = 1; px <= processes && static_cast<size_t>(px) <= global[0]; ++px) {
            if (processes % px != 0) continue;
            const int rest = processes / px;
            for (int py = 1; py <= rest && static_cast<size_t>(py) <= global[1]; ++py) {
                if (rest % py != 0) continue;
                const int pz = rest / py;
                if (static_cast<size_t>(pz) > global[2]) continue;
                const double largestBlock =
                    static_cast<double>(global[0] / px + (global[0] % px != 0)) *
                    static_cast<double>(global[1] / py + (global[1] % py != 0)) *
                    static_cast<double>(global[2] / pz + (global[2] % pz != 0));
                const double score = largestBlock * (1.0 + 2.0 *
                    ((px - 1.0) / global[0] + (py - 1.0) / global[1] +
                     (pz - 1.0) / global[2]));
                if (score < bestScore) {
                    bestScore = score;
                    best = {px, py, pz};
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

void blockLayout(const Sizes& global, const Coords& dims, const Coords& coords,
                 Sizes& local, Sizes& offset) {
    for (int a = 0; a < 3; ++a) {
        const size_t base = global[a] / dims[a];
        const size_t remainder = global[a] % dims[a];
        local[a] = base + (static_cast<size_t>(coords[a]) < remainder);
        offset[a] = static_cast<size_t>(coords[a]) * base +
                    std::min(static_cast<size_t>(coords[a]), remainder);
    }
}

struct Domain {
    MPI_Comm comm;
    Sizes global, local, offset;
    size_t pitch, plane, storage;
    std::array<int, 6> neighbor;
    int rank, processes;

    Domain(MPI_Comm active, const Sizes& size, const Coords& dims) : global(size) {
        Coords periods{0, 0, 0}, coords;
        // Keep rank zero as the output rank.
        MPI_Cart_create(active, 3, dims.data(), periods.data(), 0, &comm);
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &processes);
        MPI_Cart_coords(comm, rank, 3, coords.data());
        blockLayout(global, dims, coords, local, offset);
        for (int a = 0; a < 3; ++a)
            MPI_Cart_shift(comm, a, 1, &neighbor[2 * a], &neighbor[2 * a + 1]);
        for (size_t n : local)
            if (n > std::numeric_limits<size_t>::max() - 2)
                throw std::runtime_error("Local grid size overflows addressable memory");
        pitch = local[0] + 2;
        plane = checkedProduct(pitch, local[1] + 2);
        storage = checkedProduct(plane, local[2] + 2);
        checkedProduct(storage, sizeof(Real));
    }

    size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * pitch + x;
    }
};

void initializeGrid(std::vector<Real>& grid, const Domain& d) {
    const size_t globalPlane = d.global[0] * d.global[1];
    for (size_t z = 1; z <= d.local[2]; ++z) {
        for (size_t y = 1; y <= d.local[1]; ++y) {
            const size_t first = (d.offset[2] + z - 1) * globalPlane +
                                 (d.offset[1] + y - 1) * d.global[0] + d.offset[0];
            Real* row = grid.data() + d.index(1, y, z);
            for (size_t x = 0; x < d.local[0]; ++x)
                row[x] = static_cast<Real>((first + x) % 19);
        }
    }
}

// Persistent requests amortize MPI setup. X and Y faces are packed explicitly;
// Z planes are sent and received directly. Their unused halo edges are included
// to keep the transfers contiguous; the seven-point stencil never reads corners.
class HaloExchange {
    const Domain& d;
    std::array<std::vector<Real>, 4> send, receive;
    std::array<std::array<MPI_Request, 12>, 2> requests;
    std::array<MPI_Datatype, 3> faceType{MPI_DOUBLE, MPI_DOUBLE, MPI_DOUBLE};
    int requestCount = 0;

public:
    HaloExchange(const Domain& domain, std::vector<Real>& first,
                 std::vector<Real>& second) : d(domain) {
        const std::array<size_t, 3> faceSize{
            checkedProduct(d.local[1], d.local[2]),
            checkedProduct(d.local[0], d.local[2]), d.plane};
        std::array<int, 3> faceCount{};
        for (int axis = 0; axis < 3; ++axis) {
            if (d.neighbor[2 * axis] == MPI_PROC_NULL &&
                d.neighbor[2 * axis + 1] == MPI_PROC_NULL) continue;
            if (faceSize[axis] <= INT_MAX) {
                faceCount[axis] = static_cast<int>(faceSize[axis]);
            } else {
                // Represent a large contiguous face with a count of one,
                // retaining compatibility with MPI implementations before MPI 4.
                MPI_Datatype chunk;
                MPI_Type_contiguous(INT_MAX, MPI_DOUBLE, &chunk);
                const int blocks[2]{static_cast<int>(faceSize[axis] / INT_MAX),
                                    static_cast<int>(faceSize[axis] % INT_MAX)};
                const MPI_Aint displacements[2]{0, static_cast<MPI_Aint>(
                    (faceSize[axis] / INT_MAX) * INT_MAX * sizeof(Real))};
                const MPI_Datatype types[2]{chunk, MPI_DOUBLE};
                MPI_Type_create_struct(2, blocks, displacements, types, &faceType[axis]);
                MPI_Type_commit(&faceType[axis]);
                MPI_Type_free(&chunk);
                faceCount[axis] = 1;
            }
        }
        for (int face = 0; face < 6; ++face) {
            if (d.neighbor[face] == MPI_PROC_NULL) continue;
            if (face < 4) {
                send[face].resize(faceSize[face / 2]);
                receive[face].resize(faceSize[face / 2]);
            }
        }
        for (int buffer = 0; buffer < 2; ++buffer) {
            Real* grid = (buffer == 0 ? first : second).data();
            int count = 0;
            // Receive requests precede send requests in the persistent set.
            for (int face = 0; face < 6; ++face) {
                if (d.neighbor[face] == MPI_PROC_NULL) continue;
                Real* target = face < 4 ? receive[face].data() :
                    grid + d.index(0, 0, face == 4 ? 0 : d.local[2] + 1);
                MPI_Recv_init(target, faceCount[face / 2], faceType[face / 2],
                              d.neighbor[face], face ^ 1, d.comm, &requests[buffer][count++]);
            }
            for (int face = 0; face < 6; ++face) {
                if (d.neighbor[face] == MPI_PROC_NULL) continue;
                Real* source = face < 4 ? send[face].data() :
                    grid + d.index(0, 0, face == 4 ? 1 : d.local[2]);
                MPI_Send_init(source, faceCount[face / 2], faceType[face / 2],
                              d.neighbor[face], face, d.comm, &requests[buffer][count++]);
            }
            requestCount = count;
        }
    }

    ~HaloExchange() {
        for (auto& set : requests)
            for (int i = 0; i < requestCount; ++i) MPI_Request_free(&set[i]);
        for (auto& type : faceType)
            if (type != MPI_DOUBLE) MPI_Type_free(&type);
    }

    void start(const std::vector<Real>& grid, int buffer) {
        for (int face = 0; face < 4; ++face) {
            if (d.neighbor[face] == MPI_PROC_NULL) continue;
            Real* packed = send[face].data();
            if (face < 2) {
                const size_t x = face == 0 ? 1 : d.local[0];
                for (size_t z = 1; z <= d.local[2]; ++z)
                    for (size_t y = 1; y <= d.local[1]; ++y)
                        *packed++ = grid[d.index(x, y, z)];
            } else {
                const size_t y = face == 2 ? 1 : d.local[1];
                for (size_t z = 1; z <= d.local[2]; ++z) {
                    std::memcpy(packed, grid.data() + d.index(1, y, z),
                                d.local[0] * sizeof(Real));
                    packed += d.local[0];
                }
            }
        }
        if (requestCount != 0) MPI_Startall(requestCount, requests[buffer].data());
    }

    bool progress(int buffer) {
        if (requestCount == 0) return true;
        int complete;
        MPI_Testall(requestCount, requests[buffer].data(), &complete, MPI_STATUSES_IGNORE);
        return complete != 0;
    }

    void finish(std::vector<Real>& grid, int buffer) {
        if (requestCount != 0)
            MPI_Waitall(requestCount, requests[buffer].data(), MPI_STATUSES_IGNORE);
        for (int face = 0; face < 4; ++face) {
            if (d.neighbor[face] == MPI_PROC_NULL) continue;
            const Real* packed = receive[face].data();
            if (face < 2) {
                const size_t x = face == 0 ? 0 : d.local[0] + 1;
                for (size_t z = 1; z <= d.local[2]; ++z)
                    for (size_t y = 1; y <= d.local[1]; ++y)
                        grid[d.index(x, y, z)] = *packed++;
            } else {
                const size_t y = face == 2 ? 0 : d.local[1] + 1;
                for (size_t z = 1; z <= d.local[2]; ++z) {
                    std::memcpy(grid.data() + d.index(1, y, z), packed,
                                d.local[0] * sizeof(Real));
                    packed += d.local[0];
                }
            }
        }
    }
};

struct Box {
    Sizes begin, end; // Half-open local coordinates.
    bool empty() const {
        return begin[0] >= end[0] || begin[1] >= end[1] || begin[2] >= end[2];
    }
};

// Restrict-qualified pointers let the compiler vectorize contiguous X rows.
// Keep the original operation order and division; do not enable fast-math.
void updateBox(const Real* __restrict__ input, Real* __restrict__ output,
               size_t pitch, size_t plane, const Box& box) {
    for (size_t z = box.begin[2]; z < box.end[2]; ++z) {
        for (size_t y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t row = z * plane + y * pitch;
            for (size_t x = box.begin[0]; x < box.end[0]; ++x) {
                const size_t i = row + x;
                output[i] = (input[i] + input[i - 1] + input[i + 1] +
                             input[i - pitch] + input[i + pitch] +
                             input[i - plane] + input[i + plane]) / 7.0;
            }
        }
    }
}

// Complement of the overlap region, partitioned into at most six disjoint boxes.
// This also handles blocks only one or two cells wide without duplicate updates.
void updateShell(const Real* input, Real* output, const Domain& d,
                 const Box& whole, const Box& core) {
    if (whole.empty()) return;
    if (core.empty()) {
        updateBox(input, output, d.pitch, d.plane, whole);
        return;
    }
    Box remaining = whole;
    for (int axis = 2; axis >= 0; --axis) {
        Box face = remaining;
        face.end[axis] = core.begin[axis];
        updateBox(input, output, d.pitch, d.plane, face);
        face = remaining;
        face.begin[axis] = core.end[axis];
        updateBox(input, output, d.pitch, d.plane, face);
        remaining.begin[axis] = core.begin[axis];
        remaining.end[axis] = core.end[axis];
    }
}

// Reassemble only for -r, outside the timed computation. Chunked transfers avoid
// MPI's int-count limit and preserve the serial result order, sum and byte hash.
void printDistributedResults(const std::vector<Real>& grid, const Domain& d,
                             const Coords& dims) {
    const size_t owned = d.local[0] * d.local[1] * d.local[2];
    std::vector<Real> packed(owned);
    size_t position = 0;
    for (size_t z = 1; z <= d.local[2]; ++z)
        for (size_t y = 1; y <= d.local[1]; ++y) {
            std::memcpy(packed.data() + position, grid.data() + d.index(1, y, z),
                        d.local[0] * sizeof(Real));
            position += d.local[0];
        }
    if (d.rank != 0) {
        for (size_t start = 0; start < owned;) {
            const int count = static_cast<int>(std::min<size_t>(owned - start, INT_MAX));
            MPI_Send(packed.data() + start, count, MPI_DOUBLE, 0, 6, d.comm);
            start += count;
        }
        return;
    }
    std::vector<Real> global(d.global[0] * d.global[1] * d.global[2]);
    for (int rank = 0; rank < d.processes; ++rank) {
        Coords coords;
        Sizes local, offset;
        MPI_Cart_coords(d.comm, rank, 3, coords.data());
        blockLayout(d.global, dims, coords, local, offset);
        const size_t count = local[0] * local[1] * local[2];
        if (rank != 0) {
            packed.resize(count);
            for (size_t start = 0; start < count;) {
                const int chunk = static_cast<int>(std::min<size_t>(count - start, INT_MAX));
                MPI_Recv(packed.data() + start, chunk, MPI_DOUBLE, rank, 6, d.comm,
                         MPI_STATUS_IGNORE);
                start += chunk;
            }
        }
        position = 0;
        for (size_t z = 0; z < local[2]; ++z)
            for (size_t y = 0; y < local[1]; ++y) {
                const size_t first = ((offset[2] + z) * d.global[1] + offset[1] + y) *
                                     d.global[0] + offset[0];
                std::memcpy(global.data() + first, packed.data() + position,
                            local[0] * sizeof(Real));
                position += local[0];
            }
    }
    print_results(global, "Grid");
}

bool validateResult(const std::vector<Real>& grid, const Domain& d) {
    Real minimum = std::numeric_limits<Real>::infinity();
    Real maximum = -minimum;
    int nonfinite = 0;
    for (size_t z = 1; z <= d.local[2]; ++z)
        for (size_t y = 1; y <= d.local[1]; ++y)
            for (size_t x = 1; x <= d.local[0]; ++x) {
                const Real value = grid[d.index(x, y, z)];
                nonfinite |= !std::isfinite(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
    Real globalMin, globalMax;
    int globalNonfinite;
    MPI_Allreduce(&nonfinite, &globalNonfinite, 1, MPI_INT, MPI_MAX, d.comm);
    MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (globalNonfinite) {
        if (d.rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    if (d.rank == 0) printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 1e6 || globalMin < -1e6) {
        if (d.rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool parseNumber(const char* text, size_t& value) {
    if (*text < '0' || *text > '9') return false;
    char* end;
    errno = 0;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > std::numeric_limits<size_t>::max())
        return false;
    value = static_cast<size_t>(parsed);
    return true;
}

int runBenchmark(int argc, char** argv, int worldRank, int worldSize) {
    Sizes global{128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "-y") == 0 ||
                   strcmp(argv[i], "-z") == 0 || strcmp(argv[i], "-i") == 0) {
            const char option = argv[i][1];
            size_t value;
            if (i + 1 >= argc || !parseNumber(argv[++i], value) ||
                (option == 'i' && value > INT_MAX) || (option == 'x' && value == 0)) {
                if (worldRank == 0) fprintf(stderr, "Invalid value for -%c\n", option);
                return 1;
            }
            if (option == 'i') iterations = static_cast<int>(value);
            else global[option == 'x' ? 0 : option == 'y' ? 1 : 2] = value;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (global[1] == 0) global[1] = global[0];
    if (global[2] == 0) global[2] = global[0];
    checkedProduct(checkedProduct(checkedProduct(global[0], global[1]), global[2]),
                   sizeof(Real));
    int activeProcesses = worldSize;
    const Coords dims = chooseProcessGrid(global, activeProcesses);
    MPI_Comm active;
    // Extra ranks remain in MPI, but cannot own empty grid blocks.
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeProcesses ? 0 : MPI_UNDEFINED,
                   worldRank, &active);
    int result = 0;
    if (active != MPI_COMM_NULL) {
        Domain d(active, global, dims);
        if (d.rank == 0) {
            printf("3D Stencil Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", global[0], global[1], global[2]);
            printf("Iterations: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d (process grid: %d x %d x %d)\n",
                   activeProcesses, dims[0], dims[1], dims[2]);
            printf("Initializing grid...\n");
        }
        std::vector<Real> grid1(d.storage);
        initializeGrid(grid1, d);
        // Initialize both buffers once: fixed physical boundaries never need
        // another copy or a traversal during the timed stencil iterations.
        std::vector<Real> grid2 = grid1;
        Box whole, core;
        for (int a = 0; a < 3; ++a) {
            whole.begin[a] = d.offset[a] == 0 ? 2 : 1;
            whole.end[a] = d.local[a] + 1 -
                (d.offset[a] + d.local[a] == d.global[a]);
            core.begin[a] = std::max(whole.begin[a],
                d.neighbor[2 * a] == MPI_PROC_NULL ? size_t{1} : size_t{2});
            core.end[a] = std::min(whole.end[a], d.local[a] +
                (d.neighbor[2 * a + 1] == MPI_PROC_NULL));
        }
        const bool hasInterior = global[0] > 2 && global[1] > 2 && global[2] > 2;
        const size_t progressPlanes = core.empty() ? 1 : std::max<size_t>(1,
            65536 / ((core.end[0] - core.begin[0]) * (core.end[1] - core.begin[1])));
        double elapsed;
        {
            HaloExchange halo(d, grid1, grid2);
            if (d.rank == 0) printf("Running stencil computation...\n");
            MPI_Barrier(d.comm);
            const double start = MPI_Wtime();
            if (hasInterior) {
                for (int iter = 0; iter < iterations; ++iter) {
                    auto& input = (iter % 2 == 0) ? grid1 : grid2;
                    auto& output = (iter % 2 == 0) ? grid2 : grid1;
                    halo.start(input, iter % 2);
                    if (!core.empty()) {
                        Box slice = core;
                        if (d.processes == 1) {
                            updateBox(input.data(), output.data(), d.pitch, d.plane, core);
                        } else {
                            // Drive MPI progress on implementations without a
                            // background progress thread, while doing useful work.
                            while (slice.begin[2] < core.end[2]) {
                                slice.end[2] = slice.begin[2] +
                                    std::min(progressPlanes, core.end[2] - slice.begin[2]);
                                updateBox(input.data(), output.data(), d.pitch, d.plane, slice);
                                slice.begin[2] = slice.end[2];
                                if (halo.progress(iter % 2)) {
                                    slice.end[2] = core.end[2];
                                    updateBox(input.data(), output.data(), d.pitch, d.plane, slice);
                                    break;
                                }
                            }
                        }
                    }
                    halo.finish(input, iter % 2);
                    updateShell(input.data(), output.data(), d, whole, core);
                }
            }
            const double localElapsed = MPI_Wtime() - start;
            MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
        }
        if (d.rank == 0) {
            printf("Computation time: %.0f ms\n", std::floor(elapsed * 1000.0));
            const double updates = hasInterior ?
                static_cast<double>(global[0] - 2) * (global[1] - 2) *
                (global[2] - 2) * iterations : 0.0;
            printf("Performance: %.3f MCellUpdates/s\n",
                   elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
        }
        const auto& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
        if (printResults) printDistributedResults(finalGrid, d, dims);
        if (validate) {
            if (d.rank == 0) printf("Validating result...\n");
            const bool valid = validateResult(finalGrid, d);
            if (d.rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
        MPI_Comm_free(&d.comm);
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int result = 1;
    try {
        result = runBenchmark(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "MPI rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
