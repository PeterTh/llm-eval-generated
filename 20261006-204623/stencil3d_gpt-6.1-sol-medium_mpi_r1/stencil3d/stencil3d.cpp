#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

using Real = double;

// Choose a geometry-aware process grid, allowing unused ranks when the
// process count cannot fit the grid. Favor low face area and balanced blocks.
std::array<int, 3> processGrid(const std::array<size_t, 3>& n, int ranks) {
    std::array<int, 3> best{1, 1, 1};
    int used = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= ranks && size_t(x) <= n[0]; ++x) {
        for (int y = 1; y <= ranks / x && size_t(y) <= n[1]; ++y) {
            int z = int(std::min(n[2], size_t(ranks / x / y)));
            int count = x * y * z;
            double lx = double((n[0] + x - 1) / x);
            double ly = double((n[1] + y - 1) / y);
            double lz = double((n[2] + z - 1) / z);
            double cost = lx * ly * lz +
                2.0 * ((x > 1 ? ly * lz : 0) + (y > 1 ? lx * lz : 0) +
                       (z > 1 ? lx * ly : 0));
            if (count > used || (count == used && cost < bestCost)) {
                used = count;
                bestCost = cost;
                best = {x, y, z};
            }
        }
    }
    return best;
}

struct Block {
    std::array<size_t, 3> n, offset;
    size_t row, plane;
    std::vector<Real> grids[2];
    std::vector<MPI_Datatype> types;
    std::vector<MPI_Request> requests[2];

    size_t index(size_t x, size_t y, size_t z) const {
        return z * plane + y * row + x;
    }

    Block(const std::array<size_t, 3>& global, const std::array<int, 3>& dims,
          MPI_Comm cart) {
        int rank, coords[3];
        MPI_Comm_rank(cart, &rank);
        MPI_Cart_coords(cart, rank, 3, coords);
        for (int d = 0; d < 3; ++d) {
            n[d] = global[d] / dims[d] + (size_t(coords[d]) < global[d] % dims[d]);
            offset[d] = size_t(coords[d]) * (global[d] / dims[d]) +
                        std::min(size_t(coords[d]), global[d] % dims[d]);
        }
        row = n[0] + 2;
        const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(Real);
        if (row > maxCells / (n[1] + 2) ||
            row * (n[1] + 2) > maxCells / (n[2] + 2))
            throw std::length_error("Local grid including halos is too large");
        plane = row * (n[1] + 2);
        for (auto& grid : grids) grid.resize(plane * (n[2] + 2));
        for (size_t z = 1; z <= n[2]; ++z)
            for (size_t y = 1; y <= n[1]; ++y)
                for (size_t x = 1; x <= n[0]; ++x) {
                    size_t gi = ((offset[2] + z - 1) * global[1] + offset[1] + y - 1) *
                                global[0] + offset[0] + x - 1;
                    // Both buffers retain fixed physical boundaries forever.
                    grids[0][index(x, y, z)] = grids[1][index(x, y, z)] = Real(gi % 19);
                }
        int sizes[3] = {int(n[2] + 2), int(n[1] + 2), int(n[0] + 2)};
        for (int d = 0; d < 3; ++d) {
            int neighbors[2];
            MPI_Cart_shift(cart, d, 1, &neighbors[0], &neighbors[1]);
            for (int side = 0; side < 2; ++side) {
                if (neighbors[side] == MPI_PROC_NULL) continue;
                int extent[3] = {int(n[2]), int(n[1]), int(n[0])};
                extent[2 - d] = 1;
                for (int receive = 0; receive < 2; ++receive) {
                    int starts[3] = {1, 1, 1};
                    starts[2 - d] = receive ? (side ? int(n[d] + 1) : 0)
                                                   : (side ? int(n[d]) : 1);
                    MPI_Datatype type;
                    MPI_Type_create_subarray(3, sizes, extent, starts, MPI_ORDER_C,
                                             MPI_DOUBLE, &type);
                    MPI_Type_commit(&type);
                    types.push_back(type);
                    for (int b = 0; b < 2; ++b) {
                        MPI_Request request;
                        if (receive)
                            MPI_Recv_init(grids[b].data(), 1, type, neighbors[side],
                                          2 * d + (1 - side), cart, &request);
                        else
                            MPI_Send_init(grids[b].data(), 1, type, neighbors[side],
                                          2 * d + side, cart, &request);
                        requests[b].push_back(request);
                    }
                }
            }
        }
    }

    ~Block() {
        for (auto& list : requests)
            for (auto& request : list) MPI_Request_free(&request);
        for (auto& type : types) MPI_Type_free(&type);
    }

    // Half-open box; x is contiguous for vectorization and memory bandwidth.
    void update(const Real* input, Real* output,
                const std::array<size_t, 3>& lo, const std::array<size_t, 3>& hi) const {
        for (size_t z = lo[2]; z < hi[2]; ++z)
            for (size_t y = lo[1]; y < hi[1]; ++y) {
                size_t base = index(0, y, z);
                for (size_t x = lo[0]; x < hi[0]; ++x) {
                    size_t i = base + x;
                    output[i] = (input[i] + input[i - 1] + input[i + 1] +
                                 input[i - row] + input[i + row] +
                                 input[i - plane] + input[i + plane]) / 7.0;
                }
            }
    }

    void run(int iterations, const std::array<size_t, 3>& global) {
        std::array<size_t, 3> lo, hi, innerLo, innerHi;
        for (int d = 0; d < 3; ++d) {
            hi[d] = n[d] + 1 - (offset[d] + n[d] == global[d]);
            lo[d] = std::min(hi[d], size_t(1 + (offset[d] == 0)));
            innerLo[d] = std::min(hi[d], lo[d] + (offset[d] != 0));
            innerHi[d] = std::max(innerLo[d], hi[d] - (offset[d] + n[d] != global[d]));
        }
        for (int iter = 0; iter < iterations; ++iter) {
            int b = iter % 2;
            auto& req = requests[b];
            if (!req.empty()) MPI_Startall(int(req.size()), req.data());
            const Real* input = grids[b].data();
            Real* output = grids[1 - b].data();
            update(input, output, innerLo, innerHi);
            if (!req.empty()) MPI_Waitall(int(req.size()), req.data(), MPI_STATUSES_IGNORE);
            // Six disjoint boxes cover the shell, including thin local blocks.
            auto boxLo = lo, boxHi = hi;
            for (int d = 2; d >= 0; --d) {
                boxHi[d] = innerLo[d];
                update(input, output, boxLo, boxHi);
                boxHi[d] = hi[d];
                boxLo[d] = innerHi[d];
                update(input, output, boxLo, boxHi);
                boxLo[d] = innerLo[d];
                boxHi[d] = innerHi[d];
            }
        }
    }

    std::vector<Real> pack(int b) const {
        std::vector<Real> packed(n[0] * n[1] * n[2]);
        size_t pos = 0;
        for (size_t z = 1; z <= n[2]; ++z)
            for (size_t y = 1; y <= n[1]; ++y) {
                std::copy_n(grids[b].data() + index(1, y, z), n[0], packed.data() + pos);
                pos += n[0];
            }
        return packed;
    }
};

// Chunk transfers to avoid MPI's int count limit. Gathering is only for -r.
void printGlobal(const Block& block, int b, const std::array<size_t, 3>& global,
                 const std::array<int, 3>& dims, MPI_Comm cart) {
    int rank, ranks;
    MPI_Comm_rank(cart, &rank);
    MPI_Comm_size(cart, &ranks);
    auto packed = block.pack(b);
    constexpr size_t chunk = 1 << 26;
    if (rank != 0) {
        for (size_t pos = 0; pos < packed.size(); pos += chunk)
            MPI_Send(packed.data() + pos, int(std::min(chunk, packed.size() - pos)),
                     MPI_DOUBLE, 0, 10, cart);
        return;
    }
    std::vector<Real> full(global[0] * global[1] * global[2]);
    for (int source = 0; source < ranks; ++source) {
        int coords[3];
        MPI_Cart_coords(cart, source, 3, coords);
        std::array<size_t, 3> n, offset;
        for (int d = 0; d < 3; ++d) {
            n[d] = global[d] / dims[d] + (size_t(coords[d]) < global[d] % dims[d]);
            offset[d] = size_t(coords[d]) * (global[d] / dims[d]) +
                        std::min(size_t(coords[d]), global[d] % dims[d]);
        }
        if (source != 0) {
            packed.resize(n[0] * n[1] * n[2]);
            for (size_t pos = 0; pos < packed.size(); pos += chunk)
                MPI_Recv(packed.data() + pos, int(std::min(chunk, packed.size() - pos)),
                         MPI_DOUBLE, source, 10, cart, MPI_STATUS_IGNORE);
        }
        size_t pos = 0;
        for (size_t z = 0; z < n[2]; ++z)
            for (size_t y = 0; y < n[1]; ++y) {
                size_t dest = ((offset[2] + z) * global[1] + offset[1] + y) * global[0] + offset[0];
                std::copy_n(packed.data() + pos, n[0], full.data() + dest);
                pos += n[0];
            }
    }
    print_results(full, "Grid");
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    // Subarray dimensions use MPI int extents; also reject overflowing allocations.
    const size_t limit = size_t(std::numeric_limits<int>::max()) - 2;
    const size_t sizeLimit = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || nx > limit || ny > limit || nz > limit ||
        iterations < 0 || nx > sizeLimit / ny || nx * ny > sizeLimit / nz) {
        if (worldRank == 0) fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        MPI_Finalize();
        return 1;
    }
    std::array<size_t, 3> global{nx, ny, nz};
    auto dims = processGrid(global, worldSize);
    int active = dims[0] * dims[1] * dims[2];
    MPI_Comm workers;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < active ? 0 : MPI_UNDEFINED,
                   worldRank, &workers);
    int status = 0;
    if (workers != MPI_COMM_NULL) {
        int periods[3] = {0, 0, 0};
        MPI_Comm cart;
        MPI_Cart_create(workers, 3, dims.data(), periods, 0, &cart);
        if (worldRank == 0) {
            printf("3D Stencil Benchmark\n");
            printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
            printf("Iterations: %d\n", iterations);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI processes: %d (%d x %d x %d)\n", active, dims[0], dims[1], dims[2]);
            printf("Initializing grid...\n");
        }
        try {
            Block block(global, dims, cart);
            if (worldRank == 0) printf("Running stencil computation...\n");
            MPI_Barrier(cart);
            double start = MPI_Wtime();
            block.run(iterations, global);
            double localTime = MPI_Wtime() - start, elapsed;
            MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
            if (worldRank == 0) {
                printf("Computation time: %ld ms\n", long(elapsed * 1000));
                double updates = double(nx > 2 ? nx - 2 : 0) *
                                 double(ny > 2 ? ny - 2 : 0) *
                                 double(nz > 2 ? nz - 2 : 0) * iterations;
                printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0 ? updates / elapsed / 1e6 : 0);
            }
            int b = iterations % 2;
            if (printResults) printGlobal(block, b, global, dims, cart);
            if (validate) {
                double localMin = std::numeric_limits<double>::infinity();
                double localMax = -localMin;
                int invalid = 0;
                const auto& grid = block.grids[b];
                for (size_t z = 1; z <= block.n[2]; ++z)
                    for (size_t y = 1; y <= block.n[1]; ++y)
                        for (size_t x = 1; x <= block.n[0]; ++x) {
                            double value = grid[block.index(x, y, z)];
                            invalid |= !std::isfinite(value);
                            localMin = std::min(localMin, value);
                            localMax = std::max(localMax, value);
                        }
                double minVal, maxVal;
                MPI_Allreduce(&invalid, &status, 1, MPI_INT, MPI_MAX, cart);
                MPI_Allreduce(&localMin, &minVal, 1, MPI_DOUBLE, MPI_MIN, cart);
                MPI_Allreduce(&localMax, &maxVal, 1, MPI_DOUBLE, MPI_MAX, cart);
                status |= minVal < -1e6 || maxVal > 1e6;
                if (worldRank == 0) {
                    printf("Validating result...\n");
                    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
                    printf("Validation: %s\n", status ? "FAILED" : "PASSED");
                }
            }
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&cart);
        MPI_Comm_free(&workers);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
