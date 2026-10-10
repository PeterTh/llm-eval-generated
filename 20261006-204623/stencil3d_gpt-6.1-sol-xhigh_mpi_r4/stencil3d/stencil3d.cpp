#include <algorithm>
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

#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

size_t gridSize(size_t nx, size_t ny, size_t nz) {
    const size_t limit = std::numeric_limits<ptrdiff_t>::max() / sizeof(Real);
    if (nx > limit / ny || nx * ny > limit / nz)
        throw std::length_error("Grid is too large");
    return nx * ny * nz;
}

// Prefer balanced blocks, accounting for uneven division and the extra cost
// of packing strided X faces. Keep all ranks whenever a nonempty tiling exists.
int chooseTopology(int processes, const int global[3], int dims[3]) {
    const size_t cells = gridSize(global[0], global[1], global[2]);
    for (int used = static_cast<int>(std::min(cells, size_t(processes))); used > 0; --used) {
        double best = std::numeric_limits<double>::infinity();
        for (int px = 1; px <= std::min(global[0], used); ++px) {
            if (used % px != 0) continue;
            const int remaining = used / px;
            for (int py = 1; py <= std::min(global[1], remaining); ++py) {
                if (remaining % py != 0) continue;
                const int pz = remaining / py;
                if (pz > global[2]) continue;
                const double x = double(global[0]) / px;
                const double y = double(global[1]) / py;
                const double z = double(global[2]) / pz;
                const double volume = double((global[0] - 1) / px + 1) *
                    ((global[1] - 1) / py + 1) * ((global[2] - 1) / pz + 1);
                const double faces = 2.0 * y * z * (1.0 - 1.0 / px) +
                    x * z * (1.0 - 1.0 / py) + x * y * (1.0 - 1.0 / pz);
                const double cost = volume + 8.0 * faces;
                if (cost < best) {
                    best = cost;
                    dims[0] = px;
                    dims[1] = py;
                    dims[2] = pz;
                }
            }
        }
        if (std::isfinite(best)) return used;
    }
    return 1;
}

void partition(int global, int parts, int coordinate, int& count, int& start) {
    const int base = global / parts;
    const int extra = global % parts;
    count = base + (coordinate < extra);
    start = coordinate * base + std::min(coordinate, extra);
}

struct Domain {
    MPI_Comm comm = MPI_COMM_NULL;
    int rank = 0;
    int n[3], start[3], neighbor[6];
    int lower[3], upper[3], coreLower[3], coreUpper[3];
    size_t pitch, plane;

    Domain(MPI_Comm active, const int dims[3], const int global[3]) {
        const int periods[3] = {0, 0, 0};
        MPI_Cart_create(active, 3, dims, periods, 0, &comm);
        MPI_Comm_rank(comm, &rank);
        int coordinates[3];
        MPI_Cart_coords(comm, rank, 3, coordinates);
        for (int axis = 0; axis < 3; ++axis) {
            partition(global[axis], dims[axis], coordinates[axis], n[axis], start[axis]);
            MPI_Cart_shift(comm, axis, 1, &neighbor[2 * axis], &neighbor[2 * axis + 1]);
            // Fixed global boundary cells are initialized in both buffers and
            // never written. Only MPI interfaces require a halo exchange.
            lower[axis] = 1 + (start[axis] == 0);
            upper[axis] = std::max(lower[axis], n[axis] + 1 -
                (start[axis] + n[axis] == global[axis]));
            coreLower[axis] = std::min(upper[axis], lower[axis] +
                (neighbor[2 * axis] != MPI_PROC_NULL));
            coreUpper[axis] = std::max(coreLower[axis], upper[axis] -
                (neighbor[2 * axis + 1] != MPI_PROC_NULL));
        }
        pitch = size_t(n[0]) + 2;
        plane = pitch * (size_t(n[1]) + 2);
    }

    ~Domain() { MPI_Comm_free(&comm); }
};

void initializeGrid(std::vector<Real>& grid, const Domain& d, const int global[3]) {
    for (int z = 1; z <= d.n[2]; ++z) {
        for (int y = 1; y <= d.n[1]; ++y) {
            const size_t globalRow = idx3(d.start[0], d.start[1] + y - 1,
                                         d.start[2] + z - 1, global[0], global[1]);
            Real* row = grid.data() + size_t(z) * d.plane + size_t(y) * d.pitch;
            for (int x = 1; x <= d.n[0]; ++x)
                row[x] = Real((globalRow + x - 1) % 19);
        }
    }
}

// Persistent, direct-to-grid face transfers avoid per-iteration setup and
// user packing buffers. Faces omit edges/corners, unused by a 7-point stencil.
class HaloExchange {
    MPI_Datatype faces[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    MPI_Request requests[2][12];
    int count = 0;

public:
    HaloExchange(std::vector<Real>& first, std::vector<Real>& second, const Domain& d) {
        const int sizes[3] = {d.n[2] + 2, d.n[1] + 2, d.n[0] + 2};
        const int starts[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; ++axis) {
            if (d.neighbor[2 * axis] == MPI_PROC_NULL &&
                d.neighbor[2 * axis + 1] == MPI_PROC_NULL) continue;
            int subsizes[3] = {d.n[2], d.n[1], d.n[0]};
            subsizes[2 - axis] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &faces[axis]);
            MPI_Type_commit(&faces[axis]);
        }
        Real* buffers[2] = {first.data(), second.data()};
        for (int buffer = 0; buffer < 2; ++buffer) {
            int request = 0;
            // Post all receives before sends.
            for (int sending = 0; sending < 2; ++sending) {
                for (int side = 0; side < 6; ++side) {
                    if (d.neighbor[side] == MPI_PROC_NULL) continue;
                    const int axis = side / 2;
                    int point[3] = {1, 1, 1};
                    point[axis] = (side % 2 == 0) ? (sending ? 1 : 0) :
                        (sending ? d.n[axis] : d.n[axis] + 1);
                    Real* address = buffers[buffer] + size_t(point[2]) * d.plane +
                        size_t(point[1]) * d.pitch + point[0];
                    if (sending) {
                        MPI_Send_init(address, 1, faces[axis], d.neighbor[side], side,
                                      d.comm, &requests[buffer][request++]);
                    } else {
                        MPI_Recv_init(address, 1, faces[axis], d.neighbor[side], side ^ 1,
                                      d.comm, &requests[buffer][request++]);
                    }
                }
            }
            count = request;
        }
    }

    ~HaloExchange() {
        for (int buffer = 0; buffer < 2; ++buffer)
            for (int i = 0; i < count; ++i) MPI_Request_free(&requests[buffer][i]);
        for (auto& face : faces)
            if (face != MPI_DATATYPE_NULL) MPI_Type_free(&face);
    }

    bool start(int buffer) {
        if (count) MPI_Startall(count, requests[buffer]);
        return count == 0;
    }
    bool progress(int buffer) {
        int complete = 0;
        MPI_Testall(count, requests[buffer], &complete, MPI_STATUSES_IGNORE);
        return complete != 0;
    }
    void wait(int buffer) {
        if (count) MPI_Waitall(count, requests[buffer], MPI_STATUSES_IGNORE);
    }
};

// Preserve the original addition order and division, without fast-math.
// Separate input/output pointers permit vectorizing the contiguous X loop.
void stencilBox(const Real* __restrict__ input, Real* __restrict__ output,
                const Domain& d, int x0, int x1, int y0, int y1, int z0, int z1) {
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) return;
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
            const size_t index = size_t(z) * d.plane + size_t(y) * d.pitch;
            const Real* row = input + index;
            const Real* front = row - d.pitch;
            const Real* back = row + d.pitch;
            const Real* bottom = row - d.plane;
            const Real* top = row + d.plane;
            Real* result = output + index;
            for (int x = x0; x < x1; ++x)
                result[x] = (row[x] + row[x - 1] + row[x + 1] + front[x] +
                             back[x] + bottom[x] + top[x]) / 7.0;
        }
    }
}

void stencilIteration(const std::vector<Real>& input, std::vector<Real>& output,
                      const Domain& d, HaloExchange& halo, int buffer) {
    bool complete = halo.start(buffer);
    const int* lo = d.lower;
    const int* hi = d.upper;
    const int* cl = d.coreLower;
    const int* ch = d.coreUpper;
    if (complete) {
        stencilBox(input.data(), output.data(), d, lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]);
        return;
    }
    // Periodic MPI polling enables overlap on MPI implementations that need
    // calls into MPI to progress transfers, without a progress thread.
    for (int z = cl[2]; z < ch[2];) {
        const int end = z + std::min(8, ch[2] - z);
        stencilBox(input.data(), output.data(), d, cl[0], ch[0], cl[1], ch[1], z, end);
        if (!complete) complete = halo.progress(buffer);
        z = end;
    }
    halo.wait(buffer);

    // Six disjoint boxes cover the remaining interface cells, including
    // subdomains only one or two cells wide. No cell is computed twice.
    stencilBox(input.data(), output.data(), d, lo[0], hi[0], lo[1], hi[1], lo[2], cl[2]);
    stencilBox(input.data(), output.data(), d, lo[0], hi[0], lo[1], hi[1], ch[2], hi[2]);
    stencilBox(input.data(), output.data(), d, lo[0], hi[0], lo[1], cl[1], cl[2], ch[2]);
    stencilBox(input.data(), output.data(), d, lo[0], hi[0], ch[1], hi[1], cl[2], ch[2]);
    stencilBox(input.data(), output.data(), d, lo[0], cl[0], cl[1], ch[1], cl[2], ch[2]);
    stencilBox(input.data(), output.data(), d, ch[0], hi[0], cl[1], ch[1], cl[2], ch[2]);
}

void gatherResults(const std::vector<Real>& grid, const Domain& d,
                   const int global[3], const int dims[3], int processes) {
    const int sizes[3] = {d.n[2] + 2, d.n[1] + 2, d.n[0] + 2};
    const int subsizes[3] = {d.n[2], d.n[1], d.n[0]};
    const int starts[3] = {1, 1, 1};
    MPI_Datatype owned;
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &owned);
    MPI_Type_commit(&owned);
    if (d.rank != 0) {
        MPI_Send(grid.data(), 1, owned, 0, 42, d.comm);
    } else {
        // Gathering is outside the benchmark and only needed for the exact,
        // order-dependent Kahan sum, samples and byte hash of the original.
        std::vector<Real> result(gridSize(global[0], global[1], global[2]));
        for (int z = 1; z <= d.n[2]; ++z)
            for (int y = 1; y <= d.n[1]; ++y)
                std::copy_n(grid.data() + size_t(z) * d.plane + size_t(y) * d.pitch + 1,
                            d.n[0], result.data() + idx3(d.start[0], d.start[1] + y - 1,
                                d.start[2] + z - 1, global[0], global[1]));
        const int globalSizes[3] = {global[2], global[1], global[0]};
        for (int rank = 1; rank < processes; ++rank) {
            int coordinates[3], counts[3], offsets[3];
            MPI_Cart_coords(d.comm, rank, 3, coordinates);
            for (int axis = 0; axis < 3; ++axis)
                partition(global[axis], dims[axis], coordinates[axis],
                          counts[2 - axis], offsets[2 - axis]);
            MPI_Datatype block;
            MPI_Type_create_subarray(3, globalSizes, counts, offsets, MPI_ORDER_C,
                                     MPI_DOUBLE, &block);
            MPI_Type_commit(&block);
            MPI_Recv(result.data(), 1, block, rank, 42, d.comm, MPI_STATUS_IGNORE);
            MPI_Type_free(&block);
        }
        print_results(result, "Grid");
    }
    MPI_Type_free(&owned);
}

bool validateResult(const std::vector<Real>& grid, const Domain& d) {
    Real minimum = std::numeric_limits<Real>::infinity();
    Real maximum = -minimum;
    int nonfinite = 0;
    for (int z = 1; z <= d.n[2]; ++z) {
        for (int y = 1; y <= d.n[1]; ++y) {
            const Real* row = grid.data() + size_t(z) * d.plane + size_t(y) * d.pitch;
            for (int x = 1; x <= d.n[0]; ++x) {
                nonfinite |= !std::isfinite(row[x]);
                minimum = std::min(minimum, row[x]);
                maximum = std::max(maximum, row[x]);
            }
        }
    }
    Real globalMin = 0, globalMax = 0;
    int globalNonfinite = 0;
    MPI_Reduce(&minimum, &globalMin, 1, MPI_DOUBLE, MPI_MIN, 0, d.comm);
    MPI_Reduce(&maximum, &globalMax, 1, MPI_DOUBLE, MPI_MAX, 0, d.comm);
    MPI_Reduce(&nonfinite, &globalNonfinite, 1, MPI_INT, MPI_MAX, 0, d.comm);
    if (d.rank != 0) return true;
    if (globalNonfinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", globalMin, globalMax);
    if (globalMax > 1e6 || globalMin < -1e6) {
        printf("Validation failed: values out of expected range\n");
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
    errno = 0;
    char* end = nullptr;
    const unsigned long long number = strtoull(text, &end, 10);
    if (errno || *end || number > std::numeric_limits<size_t>::max()) return false;
    value = size_t(number);
    return true;
}

int runBenchmark(int argc, char** argv, int rank, int processes) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if ((strcmp(argv[i], "-x") == 0 || strcmp(argv[i], "-y") == 0 ||
             strcmp(argv[i], "-z") == 0 || strcmp(argv[i], "-i") == 0) && i + 1 < argc) {
            const char option = argv[i][1];
            size_t value = 0;
            if (!parseNumber(argv[++i], value) ||
                value > size_t(option == 'i' ? INT_MAX : INT_MAX - 2)) {
                if (rank == 0) fprintf(stderr, "Invalid value for -%c: %s\n", option, argv[i]);
                return 1;
            }
            if (option == 'x') nx = value;
            else if (option == 'y') ny = value;
            else if (option == 'z') nz = value;
            else iterations = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0) {
        if (rank == 0) fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }
    const int global[3] = {int(nx), int(ny), int(nz)};
    int dims[3] = {1, 1, 1};
    const int used = chooseTopology(processes, global, dims);
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < used ? 0 : MPI_UNDEFINED, rank, &active);
    if (rank >= used) return 0;
    Domain domain(active, dims, global);
    MPI_Comm_free(&active);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d (topology: %d x %d x %d)\n", used, dims[0], dims[1], dims[2]);
        printf("Initializing grid...\n");
    }
    std::vector<Real> grid1(gridSize(domain.n[0] + 2, domain.n[1] + 2, domain.n[2] + 2));
    initializeGrid(grid1, domain, global);
    // The fixed physical boundary is identical in both buffers, eliminating
    // the original full-volume boundary scan on every iteration.
    std::vector<Real> grid2 = grid1;
    HaloExchange halo(grid1, grid2, domain);

    if (rank == 0) printf("Running stencil computation...\n");
    MPI_Barrier(domain.comm);
    const double start = MPI_Wtime();
    if (nx > 2 && ny > 2 && nz > 2) {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) stencilIteration(grid1, grid2, domain, halo, 0);
            else stencilIteration(grid2, grid1, domain, halo, 1);
        }
    }
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, domain.comm);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double cellUpdates = double(nx > 2 ? nx - 2 : 0) *
            double(ny > 2 ? ny - 2 : 0) * double(nz > 2 ? nz - 2 : 0) * iterations;
        const double mcups = duration > 0 ? cellUpdates / duration / 1e6 : 0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        gatherResults(finalGrid, domain, global, dims, used);
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, domain);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, processes = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int status = 0;
    try {
        status = runBenchmark(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    // Inactive ranks and validation failures take the same collective exit.
    int globalStatus = 0;
    MPI_Allreduce(&status, &globalStatus, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    MPI_Finalize();
    return globalStatus;
}
