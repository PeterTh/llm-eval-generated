#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Extents = std::array<int, 3>;  // x, y, z
using Strides = std::array<size_t, 3>;

struct Box {
    Extents begin;
    Extents end;
};

Strides stridesFor(const Extents& size) {
    return {1, static_cast<size_t>(size[0]), static_cast<size_t>(size[0]) * size[1]};
}

size_t index(const Extents& point, const Strides& stride) {
    return point[0] + point[1] * stride[1] + point[2] * stride[2];
}

// Minimize internal face area, penalizing uneven partitions. Favor contiguous
// z faces on ties, then y faces. Use fewer ranks if the factorization will not fit.
Extents processGrid(const Extents& interior, int available) {
    const size_t cells = stridesFor(interior)[2] * interior[2];
    for (int ranks = static_cast<int>(std::min(cells, static_cast<size_t>(available)));
         ranks > 0; --ranks) {
        Extents best{};
        long double bestCost = std::numeric_limits<long double>::infinity();
        for (int x = 1; x <= std::min(ranks, interior[0]); ++x) {
            if (ranks % x != 0) continue;
            const int remaining = ranks / x;
            for (int y = 1; y <= std::min(remaining, interior[1]); ++y) {
                if (remaining % y != 0) continue;
                const int z = remaining / y;
                if (z > interior[2]) continue;

                const Extents parts{x, y, z};
                long double facesPerCell = 0.0;
                long double largestBlock = 1.0;
                for (int axis = 0; axis < 3; ++axis) {
                    facesPerCell += static_cast<long double>(parts[axis] - 1) / interior[axis];
                    largestBlock *= 1 + (interior[axis] - 1) / parts[axis];
                }
                const long double cost = (1.0L + 2.0L * facesPerCell) * largestBlock;
                if (cost < bestCost) {
                    bestCost = cost;
                    best = parts;
                }
            }
        }
        if (best[0] != 0) return best;
    }
    return {1, 1, 1};
}

struct Block {
    Extents origin;  // Global coordinates of local (0, 0, 0), including ghosts.
    Extents size;
    Box interior;
    Box owned;      // Interior plus any physical boundaries on this rank.
};

Block makeBlock(const Extents& global, const Extents& parts, const Extents& coords) {
    Block block{};
    for (int axis = 0; axis < 3; ++axis) {
        const int interior = global[axis] - 2;
        const int base = interior / parts[axis];
        const int extra = interior % parts[axis];
        const int count = base + (coords[axis] < extra);
        block.origin[axis] = base * coords[axis] + std::min(coords[axis], extra);
        block.size[axis] = count + 2;
        block.interior.begin[axis] = 1;
        block.interior.end[axis] = count + 1;
        block.owned.begin[axis] = coords[axis] == 0 ? 0 : 1;
        block.owned.end[axis] = coords[axis] == parts[axis] - 1 ? count + 2 : count + 1;
    }
    return block;
}

void initializeGrid(std::vector<Real>& grid, const Extents& size, const Extents& origin,
                    const Strides& globalStride) {
    const Strides stride = stridesFor(size);
    for (int z = 0; z < size[2]; ++z) {
        for (int y = 0; y < size[1]; ++y) {
            Real* row = grid.data() + index({0, y, z}, stride);
            const size_t start = index({origin[0], origin[1] + y, origin[2] + z}, globalStride);
            for (int x = 0; x < size[0]; ++x) {
                row[x] = static_cast<Real>((start + x) % 19);
            }
        }
    }
}

void stencil(const Real* __restrict input, Real* __restrict output,
             const Strides& stride, const Box& box) {
    if (box.begin[0] == box.end[0]) return;

    const size_t row = stride[1];
    const size_t plane = stride[2];
    for (int z = box.begin[2]; z < box.end[2]; ++z) {
        for (int y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t start = index({0, y, z}, stride);
            const Real* center = input + start;
            const Real* front = center - row;
            const Real* back = center + row;
            const Real* bottom = center - plane;
            const Real* top = center + plane;
            Real* out = output + start;
            for (int x = box.begin[0]; x < box.end[0]; ++x) {
                out[x] = (center[x] + center[x-1] + center[x+1] +
                          front[x] + back[x] + bottom[x] + top[x]) / 7.0;
            }
        }
    }
}

MPI_Datatype subarray(const Extents& size, const Extents& count) {
    const int sizes[] = {size[2], size[1], size[0]};
    const int counts[] = {count[2], count[1], count[0]};
    const int starts[] = {0, 0, 0};
    MPI_Datatype type;
    MPI_Type_create_subarray(3, sizes, counts, starts, MPI_ORDER_C, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct HaloExchange {
    Box core;
    std::array<Box, 6> shell;
    int shellCount = 0;
    std::array<std::array<MPI_Request, 12>, 2> requests;
    int requestCount = 0;

    HaloExchange(const Block& block, const Strides& stride, std::vector<Real> (&grids)[2],
                 MPI_Comm cart) : core(block.interior) {
        for (int axis = 0; axis < 3; ++axis) {
            int neighbors[2];
            MPI_Cart_shift(cart, axis, 1, &neighbors[0], &neighbors[1]);
            if (neighbors[0] == MPI_PROC_NULL && neighbors[1] == MPI_PROC_NULL) continue;

            Extents count;
            for (int d = 0; d < 3; ++d) count[d] = block.size[d] - 2;
            count[axis] = 1;
            MPI_Datatype face = subarray(block.size, count);
            for (int side = 0; side < 2; ++side) {
                const int neighbor = neighbors[side];
                if (neighbor == MPI_PROC_NULL) continue;

                Extents send = block.interior.begin;
                send[axis] = side == 0 ? 1 : block.interior.end[axis] - 1;
                Extents recv = send;
                recv[axis] += side == 0 ? -1 : 1;
                const int tag = 2 * axis + side;
                const int oppositeTag = 2 * axis + (1 - side);
                for (int slot = 0; slot < 2; ++slot) {
                    Real* data = grids[slot].data();
                    MPI_Recv_init(data + index(recv, stride), 1, face, neighbor, oppositeTag,
                                  cart, &requests[slot][requestCount]);
                    MPI_Send_init(data + index(send, stride), 1, face, neighbor, tag,
                                  cart, &requests[slot][requestCount + 1]);
                }
                requestCount += 2;

                // Peel disjoint faces from the shrinking core. A one-cell
                // block can depend on both neighbors along the same axis.
                if (core.begin[axis] == core.end[axis]) continue;
                Box faceBox = core;
                if (side == 0) {
                    faceBox.end[axis] = ++core.begin[axis];
                } else {
                    faceBox.begin[axis] = --core.end[axis];
                }
                shell[shellCount++] = faceBox;
            }
            MPI_Type_free(&face);
        }
    }

    ~HaloExchange() {
        for (auto& slot : requests) {
            for (int i = 0; i < requestCount; ++i) MPI_Request_free(&slot[i]);
        }
    }
};

std::vector<Real> gatherGrid(const std::vector<Real>& local, const Block& block,
                             const Extents& global, const Extents& parts, size_t cells,
                             int rank, int ranks, MPI_Comm cart) {
    const Strides localStride = stridesFor(block.size);
    Extents count;
    for (int axis = 0; axis < 3; ++axis) {
        count[axis] = block.owned.end[axis] - block.owned.begin[axis];
    }
    if (rank != 0) {
        MPI_Datatype type = subarray(block.size, count);
        MPI_Send(local.data() + index(block.owned.begin, localStride), 1, type, 0, 0, cart);
        MPI_Type_free(&type);
        return {};
    }

    std::vector<Real> grid(cells);
    const Strides globalStride = stridesFor(global);
    std::vector<MPI_Request> receives(ranks - 1);
    for (int peer = 1; peer < ranks; ++peer) {
        Extents coords;
        MPI_Cart_coords(cart, peer, 3, coords.data());
        const Block remote = makeBlock(global, parts, coords);
        Extents start;
        for (int axis = 0; axis < 3; ++axis) {
            start[axis] = remote.origin[axis] + remote.owned.begin[axis];
            count[axis] = remote.owned.end[axis] - remote.owned.begin[axis];
        }
        MPI_Datatype type = subarray(global, count);
        MPI_Irecv(grid.data() + index(start, globalStride), 1, type, peer, 0, cart,
                  &receives[peer - 1]);
        MPI_Type_free(&type);
    }

    const Box& owned = block.owned;
    for (int z = owned.begin[2]; z < owned.end[2]; ++z) {
        for (int y = owned.begin[1]; y < owned.end[1]; ++y) {
            const Extents start{owned.begin[0], y, z};
            const Extents dest{block.origin[0] + start[0], block.origin[1] + y, block.origin[2] + z};
            std::copy_n(local.data() + index(start, localStride), owned.end[0] - owned.begin[0],
                        grid.data() + index(dest, globalStride));
        }
    }
    MPI_Waitall(ranks - 1, receives.data(), MPI_STATUSES_IGNORE);
    return grid;
}

bool reportValidation(Real minimum, Real maximum, bool nonfinite) {
    if (nonfinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", minimum, maximum);
    if (maximum > 1e6 || minimum < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

bool validateGrid(const std::vector<Real>& local, const Block& block, int rank, MPI_Comm cart) {
    const Strides stride = stridesFor(block.size);
    const Box& owned = block.owned;
    const Real lowest = -std::numeric_limits<Real>::infinity();
    // Reduce the negated minimum, maximum and nonfinite flag together.
    std::array<Real, 3> stats{lowest, lowest, 0.0};
    for (int z = owned.begin[2]; z < owned.end[2]; ++z) {
        for (int y = owned.begin[1]; y < owned.end[1]; ++y) {
            const Real* row = local.data() + index({0, y, z}, stride);
            for (int x = owned.begin[0]; x < owned.end[0]; ++x) {
                const Real value = row[x];
                if (std::isfinite(value)) {
                    stats[0] = std::max(stats[0], -value);
                    stats[1] = std::max(stats[1], value);
                } else {
                    stats[2] = 1.0;
                }
            }
        }
    }

    std::array<Real, 3> global;
    MPI_Reduce(stats.data(), global.data(), 3, MPI_DOUBLE, MPI_MAX, 0, cart);
    return rank != 0 || reportValidation(-global[0], global[1], global[2] != 0.0);
}

void reportTiming(double elapsed, double updates) {
    printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000.0));
    printf("Performance: %.3f MCellUpdates/s\n", elapsed > 0.0 ? updates / elapsed / 1e6 : 0.0);
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

int runBenchmark(int argc, char** argv, int rank, int ranks) {
    Extents global{128, 0, 0};
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            global[0] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            global[1] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            global[2] = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
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

    size_t cells = 1;
    for (int size : global) {
        if (size <= 0 || cells > std::numeric_limits<size_t>::max() / size) {
            if (rank == 0) fprintf(stderr, "Invalid grid size\n");
            return 1;
        }
        cells *= size;
    }

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    const Extents interior{global[0] - 2, global[1] - 2, global[2] - 2};
    if (iterations <= 0 || *std::min_element(interior.begin(), interior.end()) <= 0) {
        if (rank != 0) return 0;
        // A negative odd iteration count selected the untouched second buffer
        // in the original benchmark. Otherwise an unchanged grid is initialized.
        const bool initialized = iterations >= 0 || iterations % 2 == 0;
        std::vector<Real> grid;
        if (printResults) {
            grid.resize(cells);
            if (initialized) initializeGrid(grid, global, {0, 0, 0}, stridesFor(global));
        }
        printf("Running stencil computation...\n");
        reportTiming(0.0, 0.0);
        if (printResults) print_results(grid, "Grid");
        if (!validate) return 0;
        printf("Validating result...\n");
        const Real maximum = initialized ? static_cast<Real>(std::min(cells - 1, size_t{18})) : 0.0;
        const bool valid = reportValidation(0.0, maximum, false);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }

    const Extents parts = processGrid(interior, ranks);
    const int activeRanks = parts[0] * parts[1] * parts[2];
    const int periods[3] = {0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, parts.data(), periods, 0, &cart);
    if (cart == MPI_COMM_NULL) return 0;

    Extents coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    const Block block = makeBlock(global, parts, coords);
    const Strides stride = stridesFor(block.size);
    std::vector<Real> grids[2];
    grids[0].resize(stride[2] * block.size[2]);
    initializeGrid(grids[0], block.size, block.origin, stridesFor(global));
    grids[1] = grids[0];  // Physical boundaries stay fixed in both buffers.

    if (rank == 0) printf("Running stencil computation...\n");
    double elapsed;
    {
        HaloExchange halo(block, stride, grids, cart);
        MPI_Barrier(cart);
        const double start = MPI_Wtime();
        // The first step already has its halo values from global initialization.
        stencil(grids[0].data(), grids[1].data(), stride, block.interior);
        for (int step = 1; step < iterations; ++step) {
            const int slot = step % 2;
            const Real* input = grids[slot].data();
            Real* output = grids[1 - slot].data();
            if (halo.requestCount == 0) {
                stencil(input, output, stride, block.interior);
            } else {
                MPI_Request* requests = halo.requests[slot].data();
                MPI_Startall(halo.requestCount, requests);
                stencil(input, output, stride, halo.core);
                MPI_Waitall(halo.requestCount, requests, MPI_STATUSES_IGNORE);
                for (int i = 0; i < halo.shellCount; ++i) {
                    stencil(input, output, stride, halo.shell[i]);
                }
            }
        }
        elapsed = MPI_Wtime() - start;
    }

    double maxElapsed;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        const double updates = static_cast<double>(interior[0]) * interior[1] * interior[2] * iterations;
        reportTiming(maxElapsed, updates);
    }

    const auto& finalGrid = grids[iterations % 2];
    if (printResults) {
        const auto grid = gatherGrid(finalGrid, block, global, parts, cells, rank, activeRanks, cart);
        if (rank == 0) print_results(grid, "Grid");
    }

    bool valid = true;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        valid = validateGrid(finalGrid, block, rank, cart);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    int ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status;
    try {
        status = runBenchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
