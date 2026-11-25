#include <mpi.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Extents = std::array<int, 3>;
using Grids = std::array<std::unique_ptr<Real[]>, 2>;

struct Box {
    Extents begin;
    Extents end;
};

static size_t volume(const Extents& size) {
    size_t result = 1;
    for (int length : size) {
        if (result > std::numeric_limits<size_t>::max() / length) {
            throw std::length_error("grid is too large");
        }
        result *= length;
    }
    return result;
}

static size_t index(const Extents& point, const Extents& size) {
    return (static_cast<size_t>(point[2]) * size[1] + point[1]) * size[0] + point[0];
}

// Use the most ranks that fit, then favor a balanced tile with less halo traffic.
// Keeping X long breaks ties in favor of contiguous stencil rows.
static Extents processGrid(const Extents& work, int ranks) {
    Extents best{1, 1, 1};
    int bestRanks = 0;
    double bestCost = std::numeric_limits<double>::infinity();
    for (int x = 1; x <= std::min(work[0], ranks); ++x) {
        for (int y = 1; y <= std::min(work[1], ranks / x); ++y) {
            const int z = std::min(work[2], ranks / x / y);
            const int used = x * y * z;
            if (used < bestRanks) {
                continue;
            }

            const Extents dims{x, y, z};
            double cells = 1.0;
            double faceCost = 0.0;
            for (int d = 0; d < 3; ++d) {
                const int tile = 1 + (work[d] - 1) / dims[d];
                cells *= tile;
                faceCost += 2.0 * (dims[d] - 1) / dims[d] / tile;
            }
            const double cost = cells * (1.0 + faceCost);
            if (used > bestRanks || cost < bestCost) {
                best = dims;
                bestRanks = used;
                bestCost = cost;
            }
        }
    }
    return best;
}

struct Block {
    Extents offset;
    Extents owned;
    Extents storage;
    Box update;
};

static Block makeBlock(const Extents& work, const Extents& dims,
                       const Extents& coords, bool hasInterior) {
    Block block{};
    for (int d = 0; d < 3; ++d) {
        const int base = work[d] / dims[d];
        const int extra = work[d] % dims[d];
        const int start = base * coords[d] + std::min(coords[d], extra);
        const int length = base + (coords[d] < extra);
        // Attach each fixed physical boundary to its adjacent interior tile.
        const bool low = hasInterior && coords[d] == 0;
        const bool high = hasInterior && coords[d] == dims[d] - 1;
        block.offset[d] = start + (hasInterior && !low);
        block.owned[d] = length + low + high;
        block.storage[d] = block.owned[d] + 2;
        block.update.begin[d] = 1 + low;
        block.update.end[d] = hasInterior ? 1 + low + length : 1;
    }
    return block;
}

static void initialize(const Block& block, const Extents& global,
                       Real* first, Real* second) {
    for (int z = 1; z <= block.owned[2]; ++z) {
        for (int y = 1; y <= block.owned[1]; ++y) {
            const size_t local = index({1, y, z}, block.storage);
            const size_t start = index({block.offset[0], block.offset[1] + y - 1,
                                        block.offset[2] + z - 1}, global);
            for (int x = 0; x < block.owned[0]; ++x) {
                first[local + x] = static_cast<Real>((start + x) % 19);
            }
            if (second == nullptr) {
                continue;
            }
            // Fixed boundary cells stay in both buffers for every iteration.
            if (z < block.update.begin[2] || z >= block.update.end[2] ||
                y < block.update.begin[1] || y >= block.update.end[1]) {
                std::copy_n(first + local, block.owned[0], second + local);
            } else {
                if (block.update.begin[0] == 2) {
                    second[local] = first[local];
                }
                if (block.update.end[0] == block.owned[0]) {
                    const size_t last = local + block.owned[0] - 1;
                    second[last] = first[last];
                }
            }
        }
    }
}

static MPI_Datatype subarray(const Extents& size, const Extents& start, const Extents& count) {
    // MPI's Fortran order makes the first dimension (X) contiguous.
    MPI_Datatype type;
    MPI_Type_create_subarray(3, size.data(), count.data(), start.data(),
                             MPI_ORDER_FORTRAN, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

struct HaloExchange {
    std::array<std::array<MPI_Request, 12>, 2> requests;
    int count = 0;
    Box core;

    HaloExchange(const Block& block, MPI_Comm comm, const Grids& grids, bool enabled)
        : core(block.update) {
        if (!enabled) {
            return;
        }
        for (int d = 0; d < 3; ++d) {
            int neighbors[2];
            MPI_Cart_shift(comm, d, 1, &neighbors[0], &neighbors[1]);
            for (int side = 0; side < 2; ++side) {
                if (neighbors[side] == MPI_PROC_NULL) {
                    continue;
                }

                Extents start = block.update.begin;
                Extents face;
                for (int axis = 0; axis < 3; ++axis) {
                    face[axis] = block.update.end[axis] - block.update.begin[axis];
                }
                face[d] = 1;
                start[d] = side == 0 ? block.update.begin[d] : block.update.end[d] - 1;
                const int sendTag = 2 * d + side;
                const int recvTag = 2 * d + (1 - side);
                MPI_Datatype send = subarray(block.storage, start, face);
                start[d] += side == 0 ? -1 : 1;
                MPI_Datatype recv = subarray(block.storage, start, face);

                for (int buffer = 0; buffer < 2; ++buffer) {
                    MPI_Recv_init(grids[buffer].get(), 1, recv, neighbors[side],
                                  recvTag, comm, &requests[buffer][count]);
                    MPI_Send_init(grids[buffer].get(), 1, send, neighbors[side],
                                  sendTag, comm, &requests[buffer][count + 1]);
                }
                count += 2;
                MPI_Type_free(&send);
                MPI_Type_free(&recv);

                if (side == 0) {
                    ++core.begin[d];
                } else {
                    core.end[d] = std::max(core.begin[d], core.end[d] - 1);
                }
            }
        }
    }

    ~HaloExchange() {
        for (auto& buffer : requests) {
            for (int i = 0; i < count; ++i) {
                MPI_Request_free(&buffer[i]);
            }
        }
    }
};

static void stencil(const Real* __restrict__ input, Real* __restrict__ output,
                    const Extents& size, const Box& region) {
    if (region.begin[0] == region.end[0] || region.begin[1] == region.end[1] ||
        region.begin[2] == region.end[2]) {
        return;
    }
    const ptrdiff_t row = size[0];
    const ptrdiff_t plane = row * size[1];
    const int width = region.end[0] - region.begin[0];
    for (int z = region.begin[2]; z < region.end[2]; ++z) {
        for (int y = region.begin[1]; y < region.end[1]; ++y) {
            const size_t start = index({region.begin[0], y, z}, size);
            const Real* in = input + start;
            Real* out = output + start;
            for (int x = 0; x < width; ++x) {
                out[x] = (in[x] + in[x - 1] + in[x + 1] +
                          in[x - row] + in[x + row] +
                          in[x - plane] + in[x + plane]) / 7.0;
            }
        }
    }
}

static double runIterations(const Grids& grids, const Block& block,
                            MPI_Comm comm, int iterations) {
    HaloExchange halo(block, comm, grids, iterations > 0);
    // Peel disjoint faces from the tile, leaving the halo-independent core.
    std::array<Box, 6> shell;
    Box remaining = block.update;
    for (int d = 2; d >= 0; --d) {
        shell[2 * d] = remaining;
        shell[2 * d].end[d] = halo.core.begin[d];
        remaining.begin[d] = halo.core.begin[d];
        shell[2 * d + 1] = remaining;
        shell[2 * d + 1].begin[d] = halo.core.end[d];
        remaining.end[d] = halo.core.end[d];
    }

    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const int buffer = iteration % 2;
        auto& requests = halo.requests[buffer];
        const Real* input = grids[buffer].get();
        Real* output = grids[1 - buffer].get();
        if (halo.count != 0) {
            MPI_Startall(halo.count, requests.data());
        }
        stencil(input, output, block.storage, halo.core);
        if (halo.count != 0) {
            MPI_Waitall(halo.count, requests.data(), MPI_STATUSES_IGNORE);
            for (const Box& face : shell) {
                stencil(input, output, block.storage, face);
            }
        }
    }
    return MPI_Wtime() - start;
}

static void printGrid(const Real* local, const Block& block,
                      const Extents& global, const Extents& work, const Extents& dims,
                      bool hasInterior, MPI_Comm comm, int rank) {
    constexpr int resultTag = 6;
    if (rank != 0) {
        MPI_Datatype owned = subarray(block.storage, {1, 1, 1}, block.owned);
        MPI_Send(local, 1, owned, 0, resultTag, comm);
        MPI_Type_free(&owned);
        return;
    }

    std::vector<Real> grid(volume(global));
    for (int z = 0; z < block.owned[2]; ++z) {
        for (int y = 0; y < block.owned[1]; ++y) {
            const size_t from = index({1, y + 1, z + 1}, block.storage);
            const size_t to = index({block.offset[0], block.offset[1] + y,
                                     block.offset[2] + z}, global);
            std::copy_n(local + from, block.owned[0], grid.data() + to);
        }
    }
    int ranks;
    MPI_Comm_size(comm, &ranks);
    for (int source = 1; source < ranks; ++source) {
        Extents coords;
        MPI_Cart_coords(comm, source, 3, coords.data());
        const Block remote = makeBlock(work, dims, coords, hasInterior);
        MPI_Datatype placement = subarray(global, remote.offset, remote.owned);
        MPI_Recv(grid.data(), 1, placement, source, resultTag, comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&placement);
    }
    print_results(grid, "Grid");
}

static bool validateGrid(const Real* grid, const Block& block,
                         MPI_Comm comm, int rank) {
    const Real infinity = std::numeric_limits<Real>::infinity();
    Real stats[3]{0.0, -infinity, -infinity};
    for (int z = 1; z <= block.owned[2]; ++z) {
        for (int y = 1; y <= block.owned[1]; ++y) {
            const Real* values = grid + index({1, y, z}, block.storage);
            for (int x = 0; x < block.owned[0]; ++x) {
                const Real value = values[x];
                if (std::isfinite(value)) {
                    stats[1] = std::max(stats[1], -value);
                    stats[2] = std::max(stats[2], value);
                } else {
                    stats[0] = 1.0;
                }
            }
        }
    }
    Real global[3];
    MPI_Allreduce(stats, global, 3, MPI_DOUBLE, MPI_MAX, comm);
    const Real minVal = -global[1] + 0.0;
    const Real maxVal = global[2];
    const bool valid = global[0] == 0.0 && minVal >= -1e6 && maxVal <= 1e6;
    if (rank == 0) {
        if (global[0] != 0.0) {
            printf("Validation failed: found NaN or Inf value\n");
        } else {
            printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
            if (!valid) {
                printf("Validation failed: values out of expected range\n");
            }
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    return valid;
}

static void printUsage(const char* progName) {
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

static int run(int argc, char** argv, int rank) {
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    if (std::any_of(global.begin(), global.end(), [](int n) { return n <= 0 || n > INT_MAX - 2; })) {
        if (rank == 0) {
            fprintf(stderr, "Grid dimensions must be between 1 and %d.\n", INT_MAX - 2);
        }
        return 1;
    }
    volume(global);

    const bool hasInterior = std::all_of(global.begin(), global.end(), [](int n) { return n > 2; });
    Extents work = global;
    if (hasInterior) {
        for (int& n : work) {
            n -= 2;
        }
    }
    int ranks;
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const Extents dims = processGrid(work, ranks);
    const Extents periods{0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), 0, &cart);
    if (cart == MPI_COMM_NULL) {
        return 0;
    }
    Extents coords;
    MPI_Cart_coords(cart, rank, 3, coords.data());
    const Block block = makeBlock(work, dims, coords, hasInterior);

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %d x %d x %d\n", global[0], global[1], global[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }
    const size_t localSize = volume(block.storage);
    const int steps = hasInterior ? std::max(iterations, 0) : 0;
    Grids grids{std::make_unique_for_overwrite<Real[]>(localSize), nullptr};
    if (steps > 0) {
        grids[1] = std::make_unique_for_overwrite<Real[]>(localSize);
    }
    if (iterations % 2 < 0) {
        // The original buffer selection returns the zero-filled buffer for negative odd counts.
        std::fill_n(grids[0].get(), localSize, 0.0);
    } else {
        initialize(block, global, grids[0].get(), grids[1].get());
    }

    if (rank == 0) {
        printf("Running stencil computation...\n");
    }
    const double elapsed = runIterations(grids, block, cart, steps);
    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, cart);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double updates = hasInterior ? static_cast<double>(volume(work)) * iterations : 0.0;
        const double mcups = duration > 0.0 ? updates / duration / 1e6 : 0.0;
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const Real* finalGrid = grids[steps % 2].get();
    if (printResults) {
        printGrid(finalGrid, block, global, work, dims, hasInterior, cart, rank);
    }
    if (validate && rank == 0) {
        printf("Validating result...\n");
    }
    const bool valid = !validate || validateGrid(finalGrid, block, cart, rank);
    MPI_Comm_free(&cart);
    return valid ? 0 : 1;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int status = 1;
    try {
        status = run(argc, argv, rank);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
