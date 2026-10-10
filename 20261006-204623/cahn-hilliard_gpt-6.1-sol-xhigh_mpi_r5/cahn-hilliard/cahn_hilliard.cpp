#include <mpi.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Extent = std::array<size_t, 3>;

// Favor compact blocks, balanced work, and contiguous Z/Y faces over strided X
// faces. Restrict the process grid to nonempty blocks, even for very thin grids.
int chooseProcessGrid(const Extent& global, int processes, int dims[3]) {
    const size_t volume = global[0] * global[1] * global[2];
    for (int active = static_cast<int>(std::min(volume, static_cast<size_t>(processes)));
         active > 0; --active) {
        long double best = std::numeric_limits<long double>::infinity();
        auto consider = [&](int px, int py, int pz) {
            const int candidate[3] = {px, py, pz};
            long double largestBlock = 1.0L;
            for (int a = 0; a < 3; ++a) {
                if (static_cast<size_t>(candidate[a]) > global[a]) return;
                largestBlock *= global[a] / candidate[a] + (global[a] % candidate[a] != 0);
            }
            const long double x = global[0], y = global[1], z = global[2];
            const long double surface = 1.15L * (px - 1) * y * z
                                      + 1.03L * (py - 1) * x * z
                                      + (pz - 1) * x * y;
            const long double score = surface * largestBlock / (static_cast<long double>(volume) / active);
            if (score < best) {
                best = score;
                std::copy_n(candidate, 3, dims);
            }
        };
        auto considerX = [&](int px) {
            if (static_cast<size_t>(px) > global[0]) return;
            const int remaining = active / px;
            for (int py = 1; py <= remaining / py; ++py) {
                if (remaining % py == 0) {
                    consider(px, py, remaining / py);
                    consider(px, remaining / py, py);
                }
            }
        };
        for (int px = 1; px <= active / px; ++px) {
            if (active % px == 0) {
                considerX(px);
                if (px != active / px) considerX(active / px);
            }
        }
        if (std::isfinite(best)) return active;
    }
    return 0;
}

void blockGeometry(const Extent& global, const int dims[3], const int coords[3],
                   Extent& local, Extent& origin) {
    for (int a = 0; a < 3; ++a) {
        const size_t base = global[a] / dims[a];
        const size_t remainder = global[a] % dims[a];
        local[a] = base + (static_cast<size_t>(coords[a]) < remainder);
        origin[a] = static_cast<size_t>(coords[a]) * base
                  + std::min(static_cast<size_t>(coords[a]), remainder);
    }
}

// Nested types avoid the int count/displacement limit on the total block size.
// Only individual local dimensions must fit in an MPI count.
MPI_Datatype blockType(const Extent& size, size_t rowStride, size_t planeStride) {
    MPI_Datatype row, plane, block;
    MPI_Type_contiguous(static_cast<int>(size[0]), MPI_DOUBLE, &row);
    MPI_Type_create_hvector(static_cast<int>(size[1]), 1,
                           static_cast<MPI_Aint>(rowStride * sizeof(double)), row, &plane);
    MPI_Type_create_hvector(static_cast<int>(size[2]), 1,
                           static_cast<MPI_Aint>(planeStride * sizeof(double)), plane, &block);
    MPI_Type_commit(&block);
    MPI_Type_free(&plane);
    MPI_Type_free(&row);
    return block;
}

struct Domain {
    MPI_Comm comm;
    Extent local, origin;
    size_t rowStride, planeStride, storage;
    int neighbor[6];
    Extent innerBegin, innerEnd;
    MPI_Datatype face[3], interior;

    Domain(MPI_Comm communicator, const Extent& global, const int dims[3]) : comm(communicator) {
        int rank, coords[3];
        MPI_Comm_rank(comm, &rank);
        MPI_Cart_coords(comm, rank, 3, coords);
        blockGeometry(global, dims, coords, local, origin);
        for (int a = 0; a < 3; ++a) {
            if (local[a] > static_cast<size_t>(std::numeric_limits<int>::max() - 2)) {
                fprintf(stderr, "Local grid dimension exceeds MPI datatype limits\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            MPI_Cart_shift(comm, a, 1, &neighbor[2 * a], &neighbor[2 * a + 1]);
            innerBegin[a] = 1 + (neighbor[2 * a] != MPI_PROC_NULL);
            innerEnd[a] = std::max(innerBegin[a], local[a] + 1
                                  - (neighbor[2 * a + 1] != MPI_PROC_NULL));
        }
        const size_t limit = static_cast<size_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(double);
        rowStride = local[0] + 2;
        if (rowStride > limit / (local[1] + 2)
            || rowStride * (local[1] + 2) > limit / (local[2] + 2)) {
            fprintf(stderr, "Local grid storage is too large\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        planeStride = rowStride * (local[1] + 2);
        storage = planeStride * (local[2] + 2);
        interior = blockType(local, rowStride, planeStride);
        for (int a = 0; a < 3; ++a) {
            Extent size = local;
            size[a] = 1;
            face[a] = blockType(size, rowStride, planeStride);
        }
    }

    ~Domain() {
        for (auto& type : face) MPI_Type_free(&type);
        MPI_Type_free(&interior);
        MPI_Comm_free(&comm);
    }

    Domain(const Domain&) = delete;
    Domain& operator=(const Domain&) = delete;

    size_t index(size_t x, size_t y, size_t z) const {
        return z * planeStride + y * rowStride + x;
    }

    // Ghost edges/corners are unnecessary for the two seven-point stencils.
    void clampFaces(double* field) const {
        for (int side = 0; side < 2; ++side) {
            if (neighbor[side] == MPI_PROC_NULL) {
                const size_t source = side ? local[0] : 1;
                const size_t target = side ? local[0] + 1 : 0;
                for (size_t z = 1; z <= local[2]; ++z)
                    for (size_t y = 1; y <= local[1]; ++y)
                        field[index(target, y, z)] = field[index(source, y, z)];
            }
            if (neighbor[2 + side] == MPI_PROC_NULL) {
                const size_t source = side ? local[1] : 1;
                const size_t target = side ? local[1] + 1 : 0;
                for (size_t z = 1; z <= local[2]; ++z)
                    std::copy_n(field + index(1, source, z), local[0], field + index(1, target, z));
            }
            if (neighbor[4 + side] == MPI_PROC_NULL) {
                const size_t source = side ? local[2] : 1;
                const size_t target = side ? local[2] + 1 : 0;
                for (size_t y = 1; y <= local[1]; ++y)
                    std::copy_n(field + index(1, y, source), local[0], field + index(1, y, target));
            }
        }
    }
};

// Persistent requests amortize setup and transfer directly between field faces;
// there are no per-step allocations or explicit packing buffers.
class HaloExchange {
    const Domain& domain;
    double* field;
    MPI_Request requests[12];
    int count = 0;

public:
    HaloExchange(const Domain& d, std::vector<double>& values, int tagBase)
        : domain(d), field(values.data()) {
        for (int a = 0; a < 3; ++a) {
            for (int side = 0; side < 2; ++side) {
                const int peer = d.neighbor[2 * a + side];
                if (peer == MPI_PROC_NULL) continue;
                Extent sendStart = {1, 1, 1}, recvStart = {1, 1, 1};
                sendStart[a] = side ? d.local[a] : 1;
                recvStart[a] = side ? d.local[a] + 1 : 0;
                MPI_Recv_init(field + d.index(recvStart[0], recvStart[1], recvStart[2]),
                              1, d.face[a], peer, tagBase + 2 * a + (1 - side),
                              d.comm, &requests[count++]);
                MPI_Send_init(field + d.index(sendStart[0], sendStart[1], sendStart[2]),
                              1, d.face[a], peer, tagBase + 2 * a + side,
                              d.comm, &requests[count++]);
            }
        }
    }

    ~HaloExchange() {
        for (int i = 0; i < count; ++i) MPI_Request_free(&requests[i]);
    }

    HaloExchange(const HaloExchange&) = delete;
    HaloExchange& operator=(const HaloExchange&) = delete;

    void start() {
        domain.clampFaces(field);
        if (count) MPI_Startall(count, requests);
    }

    void finish() {
        if (count) MPI_Waitall(count, requests, MPI_STATUSES_IGNORE);
    }
};

inline double laplacian(const double* field, size_t i, size_t row, size_t plane) {
    // dx = dy = dz = 1 in the original benchmark. Keep its arithmetic order.
    const double twice = 2.0 * field[i];
    const double cxx = field[i + 1] + field[i - 1] - twice;
    const double cyy = field[i + row] + field[i - row] - twice;
    const double czz = field[i + plane] + field[i - plane] - twice;
    return cxx + cyy + czz;
}

// Separate rectangular interior/boundary regions keep the X loop branch-free
// and vectorizable. The chemical potential and update retain the original math.
template<bool Chemical>
void computeBlock(const Domain& d, const double* __restrict__ field,
                  const double* __restrict__ cold, double* __restrict__ output,
                  const Extent& begin, const Extent& end) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5, dt = 0.01, D = 1.0;
    for (size_t z = begin[2]; z < end[2]; ++z) {
        for (size_t y = begin[1]; y < end[1]; ++y) {
            const size_t offset = d.index(0, y, z);
            for (size_t x = begin[0]; x < end[0]; ++x) {
                const size_t i = offset + x;
                const double lap = laplacian(field, i, d.rowStride, d.planeStride);
                if constexpr (Chemical) {
                    const double cv = field[i];
                    output[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                              + 3.0 * cv + cv * cv * cv - gamma * lap;
                } else {
                    output[i] = cold[i] + dt * D * lap;
                }
            }
        }
    }
}

template<bool Chemical>
void computeBoundary(const Domain& d, const double* field, const double* cold, double* output) {
    Extent begin = {1, 1, 1}, end = {d.local[0] + 1, d.local[1] + 1, d.local[2] + 1};
    // Disjoint slabs cover the shell, including blocks just one cell thick.
    for (int a = 2; a >= 0; --a) {
        Extent slabEnd = end;
        slabEnd[a] = d.innerBegin[a];
        computeBlock<Chemical>(d, field, cold, output, begin, slabEnd);
        Extent slabBegin = begin;
        slabBegin[a] = d.innerEnd[a];
        computeBlock<Chemical>(d, field, cold, output, slabBegin, end);
        begin[a] = d.innerBegin[a];
        end[a] = d.innerEnd[a];
    }
}

void initializeConcentration(std::vector<double>& c, const Domain& d, const Extent& global) {
    const size_t volume = global[0] * global[1] * global[2];
    for (size_t z = 1; z <= d.local[2]; ++z) {
        for (size_t y = 1; y <= d.local[1]; ++y) {
            const size_t globalRow = ((d.origin[2] + z - 1) * global[1] + d.origin[1] + y - 1) * global[0];
            const size_t localRow = d.index(0, y, z);
            for (size_t x = 1; x <= d.local[0]; ++x) {
                const size_t linear_id = globalRow + d.origin[0] + x - 1;
                const double pseudo = (((linear_id + 1) * 1299709) % volume) / static_cast<double>(volume);
                c[localRow + x] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Gather only when requested. Receive each block directly into global XYZ order
// so the existing serial Kahan sum, samples, and byte hash remain unchanged.
void printDistributedResults(const std::vector<double>& c, const Domain& d,
                             const Extent& global, const int dims[3], int rank, int processes) {
    if (rank != 0) {
        MPI_Send(c.data() + d.index(1, 1, 1), 1, d.interior, 0, 12, d.comm);
        return;
    }
    std::vector<double> result(global[0] * global[1] * global[2]);
    std::vector<MPI_Request> requests(processes - 1);
    std::vector<MPI_Datatype> types(processes - 1);
    for (int peer = 1; peer < processes; ++peer) {
        int coords[3];
        Extent local, origin;
        MPI_Cart_coords(d.comm, peer, 3, coords);
        blockGeometry(global, dims, coords, local, origin);
        types[peer - 1] = blockType(local, global[0], global[0] * global[1]);
        const size_t offset = (origin[2] * global[1] + origin[1]) * global[0] + origin[0];
        MPI_Irecv(result.data() + offset, 1, types[peer - 1], peer, 12, d.comm, &requests[peer - 1]);
    }
    for (size_t z = 1; z <= d.local[2]; ++z) {
        for (size_t y = 1; y <= d.local[1]; ++y) {
            const size_t offset = ((d.origin[2] + z - 1) * global[1] + d.origin[1] + y - 1)
                                * global[0] + d.origin[0];
            std::copy_n(c.data() + d.index(1, y, z), d.local[0], result.data() + offset);
        }
    }
    if (processes > 1) MPI_Waitall(processes - 1, requests.data(), MPI_STATUSES_IGNORE);
    for (auto& type : types) MPI_Type_free(&type);
    print_results(result, "Concentration");
}

bool validateResult(const std::vector<double>& c, const Domain& d, int rank) {
    int invalid = 0, globalInvalid;
    double minimum = c[d.index(1, 1, 1)], maximum = minimum;
    for (size_t z = 1; z <= d.local[2]; ++z) {
        for (size_t y = 1; y <= d.local[1]; ++y) {
            const size_t offset = d.index(0, y, z);
            for (size_t x = 1; x <= d.local[0]; ++x) {
                const double value = c[offset + x];
                invalid |= !std::isfinite(value);
                minimum = std::min(minimum, value);
                maximum = std::max(maximum, value);
            }
        }
    }
    MPI_Allreduce(&invalid, &globalInvalid, 1, MPI_INT, MPI_MAX, d.comm);
    if (globalInvalid) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    double globalMin, globalMax;
    MPI_Allreduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, d.comm);
    MPI_Allreduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, d.comm);
    if (rank == 0) printf("Concentration range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 10.0 || globalMin < -10.0) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

bool parseDimension(const char* text, size_t& value) {
    char* end;
    errno = 0;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (*text == '-' || *text == '\0' || *end != '\0' || errno == ERANGE
        || parsed > std::numeric_limits<size_t>::max()) return false;
    value = static_cast<size_t>(parsed);
    return true;
}

int run(int argc, char** argv) {
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "-y") == 0 || strcmp(argv[i], "-z") == 0)
            && i + 1 < argc) {
            size_t& dimension = argv[i][1] == 'x' ? nx : (argv[i][1] == 'y' ? ny : nz);
            if (!parseDimension(argv[++i], dimension)) {
                if (worldRank == 0) fprintf(stderr, "Invalid grid dimension: %s\n", argv[i]);
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    const size_t limit = static_cast<size_t>(std::numeric_limits<MPI_Aint>::max()) / sizeof(double);
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit / ny || nx * ny > limit / nz) {
        if (worldRank == 0) fprintf(stderr, "Grid dimensions must be positive and their product must fit in memory\n");
        return 1;
    }
    const Extent global = {nx, ny, nz};
    const size_t gridSize = nx * ny * nz;
    int dims[3];
    const int active = chooseProcessGrid(global, worldSize, dims);
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED, worldRank, &workers);
    // Excess ranks have no nonempty Cartesian block and wait in MPI_Finalize.
    if (workers == MPI_COMM_NULL) return 0;
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(workers, 3, dims, periods, 0, &cart);
    MPI_Comm_free(&workers);
    Domain domain(cart, global, dims);
    int rank;
    MPI_Comm_rank(cart, &rank);
    if (rank == 0) {
        printf("Cahn-Hilliard Phase Separation Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Time steps: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d (process grid: %d x %d x %d)\n", active, dims[0], dims[1], dims[2]);
        printf("Initializing concentration field...\n");
    }
    std::vector<double> concentration[2] = {std::vector<double>(domain.storage), std::vector<double>(domain.storage)};
    std::vector<double> mu(domain.storage);
    initializeConcentration(concentration[0], domain, global);
    HaloExchange concentrationHalo[2] = {HaloExchange(domain, concentration[0], 0),
                                         HaloExchange(domain, concentration[1], 0)};
    HaloExchange muHalo(domain, mu, 6);
    if (rank == 0) printf("Running Cahn-Hilliard simulation...\n");
    MPI_Barrier(cart);
    const double start = MPI_Wtime();
    int current = 0;
    for (int t = 0; t < iterations; ++t) {
        const double* cold = concentration[current].data();
        double* cnew = concentration[1 - current].data();
        concentrationHalo[current].start();
        computeBlock<true>(domain, cold, nullptr, mu.data(), domain.innerBegin, domain.innerEnd);
        concentrationHalo[current].finish();
        computeBoundary<true>(domain, cold, nullptr, mu.data());
        muHalo.start();
        computeBlock<false>(domain, mu.data(), cold, cnew, domain.innerBegin, domain.innerEnd);
        muHalo.finish();
        computeBoundary<false>(domain, mu.data(), cold, cnew);
        current = 1 - current;
    }
    const double localTime = MPI_Wtime() - start;
    double elapsed;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
        const double cellUpdates = static_cast<double>(gridSize) * iterations;
        printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0);
    }
    if (printResults) printDistributedResults(concentration[current], domain, global, dims, rank, active);
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(concentration[current], domain, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int result = 1;
    try {
        result = run(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
