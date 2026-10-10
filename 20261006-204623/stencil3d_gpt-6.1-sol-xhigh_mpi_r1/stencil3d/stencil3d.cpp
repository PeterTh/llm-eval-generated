#include <mpi.h>

#include <algorithm>
#include <cerrno>
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

inline constexpr size_t idx3(size_t x, size_t y, size_t z, size_t nx, size_t ny) noexcept {
    return (z * ny + y) * nx + x;
}

void initializeGrid(std::vector<Real>& grid) {
    for (size_t i = 0; i < grid.size(); ++i) grid[i] = static_cast<Real>(i % 19);
}

void splitExtent(int n, int parts, int coordinate, int& length, int& offset) {
    const int base = n / parts;
    const int remainder = n % parts;
    length = base + (coordinate < remainder);
    offset = coordinate * base + std::min(coordinate, remainder);
}

// Choose a geometry-aware Cartesian decomposition. Never create empty blocks;
// excess ranks wait outside the computation communicator on very small grids.
int chooseDecomposition(const int interior[3], int processes, int dims[3]) {
    const size_t cells = static_cast<size_t>(interior[0]) * interior[1] * interior[2];
    for (int count = static_cast<int>(std::min(cells, static_cast<size_t>(processes)));
         count > 0; --count) {
        double best = std::numeric_limits<double>::infinity();
        for (int pz = 1; pz <= std::min(count, interior[2]); ++pz) {
            if (count % pz != 0) continue;
            const int remaining = count / pz;
            for (int py = 1; py <= std::min(remaining, interior[1]); ++py) {
                if (remaining % py != 0) continue;
                const int px = remaining / py;
                if (px > interior[0]) continue;
                const double lx = interior[0] / px + (interior[0] % px != 0);
                const double ly = interior[1] / py + (interior[1] % py != 0);
                const double lz = interior[2] / pz + (interior[2] % pz != 0);
                // Balance the largest block and minimize halo traffic. Strided
                // X faces cost slightly more to pack than Y and Z faces.
                const double traffic = 1.25 * (px - 1.0) * interior[1] * interior[2]
                                     + 1.05 * (py - 1.0) * interior[0] * interior[2]
                                     + (pz - 1.0) * interior[0] * interior[1];
                const double cost = lx * ly * lz + 4.0 * traffic / count;
                if (cost < best) {
                    best = cost;
                    dims[0] = px;
                    dims[1] = py;
                    dims[2] = pz;
                }
            }
        }
        if (std::isfinite(best)) return count;
    }
    return 1;
}

struct Domain {
    int length[3];
    int offset[3];
    int neighbor[6];
    size_t row;
    size_t plane;

    Domain(const int interior[3], const int dims[3], MPI_Comm comm, int rank) {
        int coordinates[3];
        MPI_Cart_coords(comm, rank, 3, coordinates);
        for (int axis = 0; axis < 3; ++axis) {
            splitExtent(interior[axis], dims[axis], coordinates[axis],
                        length[axis], offset[axis]);
            MPI_Cart_shift(comm, axis, 1, &neighbor[2 * axis], &neighbor[2 * axis + 1]);
        }
        row = static_cast<size_t>(length[0]) + 2;
        plane = row * (static_cast<size_t>(length[1]) + 2);
    }

    size_t index(int x, int y, int z) const {
        return static_cast<size_t>(z) * plane + static_cast<size_t>(y) * row + x;
    }
};

void initializeLocal(std::vector<Real>& grid, const Domain& d, size_t nx, size_t ny) {
    for (int z = 0; z < d.length[2] + 2; ++z) {
        for (int y = 0; y < d.length[1] + 2; ++y) {
            const size_t global = idx3(d.offset[0], d.offset[1] + y,
                                      d.offset[2] + z, nx, ny);
            Real* row = grid.data() + d.index(0, y, z);
            for (int x = 0; x < d.length[0] + 2; ++x) {
                row[x] = static_cast<Real>((global + x) % 19);
            }
        }
    }
}

// Persistent requests are tied to the two fixed grid allocations. Face types
// exclude halo edges/corners, which the seven-point stencil never needs.
class HaloExchange {
    MPI_Datatype face[3] = {MPI_DATATYPE_NULL, MPI_DATATYPE_NULL, MPI_DATATYPE_NULL};
    std::vector<MPI_Request> requests[2];

public:
    HaloExchange(std::vector<Real>& a, std::vector<Real>& b, const Domain& d, MPI_Comm comm) {
        const int sizes[3] = {d.length[2] + 2, d.length[1] + 2, d.length[0] + 2};
        const int starts[3] = {0, 0, 0};
        for (int axis = 0; axis < 3; ++axis) {
            if (d.neighbor[2 * axis] == MPI_PROC_NULL &&
                d.neighbor[2 * axis + 1] == MPI_PROC_NULL) continue;
            int subsizes[3] = {d.length[2], d.length[1], d.length[0]};
            subsizes[2 - axis] = 1;
            MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                                     MPI_DOUBLE, &face[axis]);
            MPI_Type_commit(&face[axis]);
        }
        Real* buffers[2] = {a.data(), b.data()};
        for (int buffer = 0; buffer < 2; ++buffer) {
            requests[buffer].reserve(12);
            // Post receives before sends when starting each exchange.
            for (int sending = 0; sending < 2; ++sending) {
                for (int side = 0; side < 6; ++side) {
                    if (d.neighbor[side] == MPI_PROC_NULL) continue;
                    const int axis = side / 2;
                    const bool upper = side % 2 != 0;
                    int position[3] = {1, 1, 1};
                    position[axis] = sending ? (upper ? d.length[axis] : 1)
                                             : (upper ? d.length[axis] + 1 : 0);
                    Real* address = buffers[buffer] + d.index(position[0], position[1], position[2]);
                    MPI_Request request;
                    if (sending) {
                        MPI_Send_init(address, 1, face[axis], d.neighbor[side], side,
                                      comm, &request);
                    } else {
                        MPI_Recv_init(address, 1, face[axis], d.neighbor[side], side ^ 1,
                                      comm, &request);
                    }
                    requests[buffer].push_back(request);
                }
            }
        }
    }

    void start(int buffer) {
        auto& r = requests[buffer];
        if (!r.empty()) MPI_Startall(static_cast<int>(r.size()), r.data());
    }

    void finish(int buffer) {
        auto& r = requests[buffer];
        if (!r.empty()) MPI_Waitall(static_cast<int>(r.size()), r.data(), MPI_STATUSES_IGNORE);
    }

    ~HaloExchange() {
        for (auto& buffer : requests) {
            for (auto& request : buffer) MPI_Request_free(&request);
        }
        for (auto& type : face) {
            if (type != MPI_DATATYPE_NULL) MPI_Type_free(&type);
        }
    }
};

#if defined(__GNUC__) || defined(__clang__)
#define STENCIL_RESTRICT __restrict__
#else
#define STENCIL_RESTRICT
#endif

// Contiguous X traversal and independent input/output pointers allow SIMD
// without changing the original floating-point addition order.
void stencilBox(const Real* STENCIL_RESTRICT input, Real* STENCIL_RESTRICT output,
                size_t row, size_t plane, int x0, int x1, int y0, int y1, int z0, int z1) {
    if (x0 >= x1 || y0 >= y1 || z0 >= z1) return;
    for (int z = z0; z < z1; ++z) {
        for (int y = y0; y < y1; ++y) {
            const size_t base = static_cast<size_t>(z) * plane + static_cast<size_t>(y) * row;
            const Real* center = input + base;
            const Real* front = center - row;
            const Real* back = center + row;
            const Real* bottom = center - plane;
            const Real* top = center + plane;
            Real* result = output + base;
            for (int x = x0; x < x1; ++x) {
                result[x] = (center[x] + center[x - 1] + center[x + 1]
                           + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
            }
        }
    }
}

void runIterations(std::vector<Real>& a, std::vector<Real>& b, const Domain& d,
                   HaloExchange& halo, int iterations) {
    int begin[3], end[3];
    for (int axis = 0; axis < 3; ++axis) {
        begin[axis] = d.neighbor[2 * axis] == MPI_PROC_NULL ? 1 : 2;
        end[axis] = d.neighbor[2 * axis + 1] == MPI_PROC_NULL
                  ? d.length[axis] + 1 : d.length[axis];
        // A one-cell block may have no communication-independent interior.
        end[axis] = std::max(begin[axis], end[axis]);
    }
    const int x = d.length[0] + 1, y = d.length[1] + 1, z = d.length[2] + 1;
    for (int iter = 0; iter < iterations; ++iter) {
        const int buffer = iter & 1;
        const Real* input = buffer ? b.data() : a.data();
        Real* output = buffer ? a.data() : b.data();
        halo.start(buffer);
        stencilBox(input, output, d.row, d.plane,
                   begin[0], end[0], begin[1], end[1], begin[2], end[2]);
        halo.finish(buffer);
        // Six disjoint slabs cover everything outside the interior box.
        stencilBox(input, output, d.row, d.plane, 1, x, 1, y, 1, begin[2]);
        stencilBox(input, output, d.row, d.plane, 1, x, 1, y, end[2], z);
        stencilBox(input, output, d.row, d.plane, 1, x, 1, begin[1], begin[2], end[2]);
        stencilBox(input, output, d.row, d.plane, 1, x, end[1], y, begin[2], end[2]);
        stencilBox(input, output, d.row, d.plane,
                   1, begin[0], begin[1], end[1], begin[2], end[2]);
        stencilBox(input, output, d.row, d.plane,
                   end[0], x, begin[1], end[1], begin[2], end[2]);
    }
}

// Only -r gathers the grid. Receive directly into global subarrays to preserve
// the exact ordering used by the existing sum, samples and bytewise hash.
void printDistributedResults(const std::vector<Real>& local, const Domain& d,
                             const int global[3], const int dims[3], MPI_Comm comm, int rank) {
    if (rank != 0) {
        const int sizes[3] = {d.length[2] + 2, d.length[1] + 2, d.length[0] + 2};
        const int subsizes[3] = {d.length[2], d.length[1], d.length[0]};
        const int starts[3] = {1, 1, 1};
        MPI_Datatype interior;
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &interior);
        MPI_Type_commit(&interior);
        MPI_Send(local.data(), 1, interior, 0, 6, comm);
        MPI_Type_free(&interior);
        return;
    }
    std::vector<Real> grid(static_cast<size_t>(global[0]) * global[1] * global[2]);
    initializeGrid(grid);
    for (int z = 1; z <= d.length[2]; ++z) {
        for (int y = 1; y <= d.length[1]; ++y) {
            std::copy_n(local.data() + d.index(1, y, z), d.length[0],
                        grid.data() + idx3(d.offset[0] + 1, d.offset[1] + y,
                                           d.offset[2] + z, global[0], global[1]));
        }
    }
    int processes;
    MPI_Comm_size(comm, &processes);
    const int sizes[3] = {global[2], global[1], global[0]};
    for (int source = 1; source < processes; ++source) {
        int coordinates[3], subsizes[3], starts[3];
        MPI_Cart_coords(comm, source, 3, coordinates);
        for (int axis = 0; axis < 3; ++axis) {
            splitExtent(global[axis] - 2, dims[axis], coordinates[axis],
                        subsizes[2 - axis], starts[2 - axis]);
            ++starts[2 - axis];
        }
        MPI_Datatype block;
        MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C, MPI_DOUBLE, &block);
        MPI_Type_commit(&block);
        MPI_Recv(grid.data(), 1, block, source, 6, comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&block);
    }
    print_results(grid, "Grid");
}

bool validateResult(const std::vector<Real>& grid, const Domain& d,
                    const int global[3], MPI_Comm comm, int rank) {
    Real localMin = std::numeric_limits<Real>::infinity();
    Real localMax = -std::numeric_limits<Real>::infinity();
    int invalid = 0;
    for (int z = 0; z < d.length[2] + 2; ++z) {
        const bool boundaryZ = d.offset[2] + z == 0 || d.offset[2] + z == global[2] - 1;
        for (int y = 0; y < d.length[1] + 2; ++y) {
            const bool boundaryY = d.offset[1] + y == 0 || d.offset[1] + y == global[1] - 1;
            for (int x = 0; x < d.length[0] + 2; ++x) {
                const bool boundaryX = d.offset[0] + x == 0 || d.offset[0] + x == global[0] - 1;
                const bool owned = x >= 1 && x <= d.length[0] && y >= 1 && y <= d.length[1]
                                && z >= 1 && z <= d.length[2];
                // Internal halos can hold values from the preceding iteration.
                if (!owned && !boundaryX && !boundaryY && !boundaryZ) continue;
                const Real value = grid[d.index(x, y, z)];
                invalid |= !std::isfinite(value);
                localMin = std::min(localMin, value);
                localMax = std::max(localMax, value);
            }
        }
    }
    Real minVal, maxVal;
    int anyInvalid;
    MPI_Allreduce(&invalid, &anyInvalid, 1, MPI_INT, MPI_MAX, comm);
    MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (rank == 0) {
        if (anyInvalid) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
            if (maxVal > 1e6 || minVal < -1e6)
                printf("Validation failed: values out of expected range\n");
        }
    }
    return !anyInvalid && maxVal <= 1e6 && minVal >= -1e6;
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

int parseNumber(const char* argument) {
    char* end;
    errno = 0;
    const long value = std::strtol(argument, &end, 10);
    if (errno || end == argument || *end != '\0' || value < 0 ||
        value > std::numeric_limits<int>::max() - 2)
        throw std::invalid_argument("Expected a nonnegative integer within the MPI dimension limit");
    return static_cast<int>(value);
}

int benchmark(int argc, char** argv, int rank, int processes) {
    int global[3] = {128, 0, 0};
    int iterations = 10;
    bool validate = false, printResults = false;
    try {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
                global[0] = parseNumber(argv[++i]);
            } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
                global[1] = parseNumber(argv[++i]);
            } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
                global[2] = parseNumber(argv[++i]);
            } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
                iterations = parseNumber(argv[++i]);
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
        if (global[1] == 0) global[1] = global[0];
        if (global[2] == 0) global[2] = global[0];
        if (global[0] == 0) throw std::invalid_argument("Grid dimensions must be positive");
    } catch (const std::invalid_argument& error) {
        if (rank == 0) fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    size_t gridSize = 1;
    const size_t maxCells = std::min(std::numeric_limits<size_t>::max(),
                                   static_cast<size_t>(std::numeric_limits<MPI_Aint>::max())) / sizeof(Real);
    for (int dimension : global) {
        if (gridSize > maxCells / static_cast<size_t>(dimension)) {
            if (rank == 0) fprintf(stderr, "Grid size exceeds addressable storage\n");
            return 1;
        }
        gridSize *= dimension;
    }
    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    // With no interior, every cell is a fixed boundary and no updates occur.
    if (global[0] <= 2 || global[1] <= 2 || global[2] <= 2) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Running stencil computation...\n");
            printf("Computation time: 0 ms\n");
            printf("Performance: 0.000 MCellUpdates/s\n");
            if (printResults) {
                std::vector<Real> grid(gridSize);
                initializeGrid(grid);
                print_results(grid, "Grid");
            }
            if (validate) {
                printf("Validating result...\n");
                printf("Value range: [%.6f, %.6f]\n", 0.0,
                       static_cast<double>(std::min(gridSize - 1, size_t{18})));
                printf("Validation: PASSED\n");
            }
        }
        return 0;
    }
    const int interior[3] = {global[0] - 2, global[1] - 2, global[2] - 2};
    int dims[3];
    const int active = chooseDecomposition(interior, processes, dims);
    MPI_Comm work;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &work);
    int exitCode = 0;
    if (work != MPI_COMM_NULL) {
        const int periodic[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(work, 3, dims, periodic, 0, &cart);
        MPI_Comm_free(&work);
        {
            Domain domain(interior, dims, cart, rank);
            const size_t localSize = domain.plane * (static_cast<size_t>(domain.length[2]) + 2);
            std::vector<Real> grid1(localSize);
            initializeLocal(grid1, domain, global[0], global[1]);
            // Both buffers retain the physical boundaries forever, eliminating
            // the original full-volume boundary scan on every iteration.
            std::vector<Real> grid2 = grid1;
            HaloExchange halo(grid1, grid2, domain, cart);
            if (rank == 0) {
                printf("MPI ranks: %d (active: %d), decomposition: %d x %d x %d\n",
                       processes, active, dims[0], dims[1], dims[2]);
                printf("Running stencil computation...\n");
            }
            MPI_Barrier(cart);
            const double start = MPI_Wtime();
            runIterations(grid1, grid2, domain, halo, iterations);
            const double localTime = MPI_Wtime() - start;
            double elapsed = 0.0;
            MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            if (rank == 0) {
                const double cellUpdates = static_cast<double>(interior[0]) * interior[1]
                                         * interior[2] * iterations;
                printf("Computation time: %.3f ms\n", elapsed * 1000.0);
                printf("Performance: %.3f MCellUpdates/s\n",
                       elapsed > 0.0 ? cellUpdates / elapsed / 1e6 : 0.0);
            }
            const auto& finalGrid = (iterations & 1) ? grid2 : grid1;
            if (printResults) printDistributedResults(finalGrid, domain, global, dims, cart, rank);
            if (validate) {
                if (rank == 0) printf("Validating result...\n");
                const bool valid = validateResult(finalGrid, domain, global, cart, rank);
                if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                exitCode = valid ? 0 : 1;
            }
        }
        MPI_Comm_free(&cart);
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return exitCode;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);
    int result;
    try {
        result = benchmark(argc, argv, rank, processes);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
