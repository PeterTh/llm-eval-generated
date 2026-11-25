#include <mpi.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;
using Index = std::array<size_t, 3>;
using ProcessGrid = std::array<int, 3>;

struct Box {
    Index begin;
    Index end;
};

struct Block {
    Index globalBegin;  // Global coordinates of the first owned cell
    Index shape;        // Local buffer dimensions, including halos
    Box owned;
    Box interior;       // Owned cells updated by the stencil
};

size_t offset(const Index& point, const Index& shape) {
    return (point[2] * shape[1] + point[1]) * shape[0] + point[0];
}

size_t volume(const Index& shape) {
    return shape[0] * shape[1] * shape[2];
}

ProcessGrid chooseProcessGrid(const Index& shape, int ranks) {
    Index interior;
    for (int axis = 0; axis < 3; ++axis) {
        interior[axis] = std::max<size_t>(3, shape[axis]) - 2;
    }

    ProcessGrid best{1, 1, 1};
    int bestRanks = 1;
    double bestCost = std::numeric_limits<double>::infinity();
    // Estimate the largest block's work plus face traffic. Allow a few idle
    // ranks when an elongated factorization would increase that cost.
    for (int x = 1; x <= ranks && size_t(x) <= interior[0]; ++x) {
        for (int y = 1; y <= ranks / x && size_t(y) <= interior[1]; ++y) {
            const int z = std::min<size_t>(ranks / x / y, interior[2]);
            const ProcessGrid parts{x, y, z};
            const int used = x * y * z;
            const double localX = (interior[0] + x - 1) / x;
            const double localY = (interior[1] + y - 1) / y;
            const double localZ = (interior[2] + z - 1) / z;
            const double cost = localX * localY * localZ
                + (2.0 * (x - 1) / x) * localY * localZ
                + (2.0 * (y - 1) / y) * localX * localZ
                + (2.0 * (z - 1) / z) * localX * localY;
            if (cost < bestCost || (cost == bestCost && used > bestRanks)) {
                best = parts;
                bestCost = cost;
                bestRanks = used;
            }
        }
    }
    return best;
}

Block makeBlock(const Index& shape, const ProcessGrid& parts, const ProcessGrid& coords) {
    Block block{};
    for (int axis = 0; axis < 3; ++axis) {
        const bool lowNeighbor = coords[axis] > 0;
        const bool highNeighbor = coords[axis] + 1 < parts[axis];
        const size_t interior = std::max<size_t>(3, shape[axis]) - 2;
        const size_t width = interior / parts[axis];
        const size_t extra = interior % parts[axis];
        const size_t coord = coords[axis];
        const size_t first = lowNeighbor ? 1 + coord * width + std::min(coord, extra) : 0;
        const size_t last = highNeighbor
            ? 1 + (coord + 1) * width + std::min(coord + 1, extra) : shape[axis];

        block.globalBegin[axis] = first;
        block.owned.begin[axis] = lowNeighbor;
        block.owned.end[axis] = block.owned.begin[axis] + last - first;
        block.shape[axis] = block.owned.end[axis] + highNeighbor;
        block.interior.begin[axis] = 1;
        block.interior.end[axis] = block.owned.end[axis] - !highNeighbor;
    }
    return block;
}

void initializeGrid(std::vector<Real>& first, std::vector<Real>& second,
                    const Block& block, const Index& globalShape, bool copyBoundary) {
    Index globalStart;
    for (int axis = 0; axis < 3; ++axis) {
        globalStart[axis] = block.globalBegin[axis] - block.owned.begin[axis];
    }
    const auto [nx, ny, nz] = block.shape;
    size_t local = 0;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t gy = globalStart[1] + y;
            const size_t gz = globalStart[2] + z;
            const size_t global = offset({globalStart[0], gy, gz}, globalShape);
            Real* row = first.data() + local;
            for (size_t x = 0; x < nx; ++x) {
                row[x] = (global + x) % 19;
            }
            if (copyBoundary) {
                if (gy == 0 || gy + 1 == globalShape[1] || gz == 0 || gz + 1 == globalShape[2]) {
                    std::copy_n(row, nx, second.data() + local);
                } else {
                    if (globalStart[0] == 0) second[local] = row[0];
                    if (globalStart[0] + nx == globalShape[0]) second[local + nx - 1] = row[nx - 1];
                }
            }
            local += nx;
        }
    }
}

void stencilIteration(const Real* __restrict input, Real* __restrict output,
                      const Index& shape, const Box& box) {
    const size_t nx = shape[0];
    const size_t plane = nx * shape[1];
    for (size_t z = box.begin[2]; z < box.end[2]; ++z) {
        for (size_t y = box.begin[1]; y < box.end[1]; ++y) {
            const size_t row = z * plane + y * nx;
            const Real* center = input + row;
            const Real* front = center - nx;
            const Real* back = center + nx;
            const Real* bottom = center - plane;
            const Real* top = center + plane;
            Real* result = output + row;
            for (size_t x = box.begin[0]; x < box.end[0]; ++x) {
                result[x] = (center[x] + center[x - 1] + center[x + 1]
                             + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
            }
        }
    }
}

MPI_Datatype makeBlockType(const Index& shape, const Box& box) {
    ProcessGrid sizes, lengths;
    const ProcessGrid starts{0, 0, 0};
    for (int axis = 0; axis < 3; ++axis) {
        sizes[axis] = static_cast<int>(shape[axis]);
        lengths[axis] = static_cast<int>(box.end[axis] - box.begin[axis]);
    }
    MPI_Datatype type;
    // Fortran order gives the X dimension unit stride.
    MPI_Type_create_subarray(3, sizes.data(), lengths.data(), starts.data(),
                             MPI_ORDER_FORTRAN, MPI_DOUBLE, &type);
    MPI_Type_commit(&type);
    return type;
}

class HaloExchange {
public:
    HaloExchange(const Block& block, MPI_Comm comm, std::vector<Real> (&grids)[2])
        : core(block.interior) {
        for (int axis = 0; axis < 3; ++axis) {
            int neighbors[2];
            MPI_Cart_shift(comm, axis, 1, &neighbors[0], &neighbors[1]);
            if (neighbors[0] == MPI_PROC_NULL && neighbors[1] == MPI_PROC_NULL) continue;

            Box face = block.interior;
            face.end[axis] = face.begin[axis] + 1;
            MPI_Datatype type = makeBlockType(block.shape, face);
            types[typeCount++] = type;
            for (int side = 0; side < 2; ++side) {
                if (neighbors[side] == MPI_PROC_NULL) continue;

                Index send = block.interior.begin;
                Index receive = send;
                send[axis] = side == 0 ? block.owned.begin[axis] : block.owned.end[axis] - 1;
                receive[axis] = side == 0 ? send[axis] - 1 : send[axis] + 1;
                for (int buffer = 0; buffer < 2; ++buffer) {
                    Real* data = grids[buffer].data();
                    MPI_Recv_init(data + offset(receive, block.shape), 1, type, neighbors[side],
                                  2 * axis + 1 - side, comm, &requests[buffer][requestCount]);
                    MPI_Send_init(data + offset(send, block.shape), 1, type, neighbors[side],
                                  2 * axis + side, comm, &requests[buffer][requestCount + 1]);
                }
                requestCount += 2;

                // Peel disjoint slabs off the interior until the remaining
                // cells can be updated without any received halo.
                if (core.begin[axis] == core.end[axis]) continue;
                Box edge = core;
                if (side == 0) {
                    edge.end[axis] = ++core.begin[axis];
                } else {
                    edge.begin[axis] = --core.end[axis];
                }
                edges[edgeCount++] = edge;
            }
        }
    }

    ~HaloExchange() {
        for (auto& buffer : requests) {
            for (int i = 0; i < requestCount; ++i) MPI_Request_free(&buffer[i]);
        }
        for (int i = 0; i < typeCount; ++i) MPI_Type_free(&types[i]);
    }

    void update(int buffer, const Real* input, Real* output, const Index& shape) {
        if (requestCount == 0) {
            stencilIteration(input, output, shape, core);
            return;
        }
        MPI_Request* active = requests[buffer].data();
        MPI_Startall(requestCount, active);

        int ready = 0;
        const size_t planeWork = (core.end[0] - core.begin[0]) * (core.end[1] - core.begin[1]);
        if (planeWork > 0 && core.begin[2] < core.end[2]) {
            // Enter MPI periodically so transports without asynchronous
            // progress can advance large messages while the core is computed.
            const size_t planesPerPoll = std::max<size_t>(1, 65536 / planeWork);
            Box chunk = core;
            while (chunk.begin[2] < core.end[2]) {
                chunk.end[2] = ready ? core.end[2] : std::min(core.end[2], chunk.begin[2] + planesPerPoll);
                stencilIteration(input, output, shape, chunk);
                chunk.begin[2] = chunk.end[2];
                if (!ready && chunk.begin[2] < core.end[2]) {
                    MPI_Testall(requestCount, active, &ready, MPI_STATUSES_IGNORE);
                }
            }
        }

        if (!ready) MPI_Waitall(requestCount, active, MPI_STATUSES_IGNORE);
        for (int edge = 0; edge < edgeCount; ++edge) {
            stencilIteration(input, output, shape, edges[edge]);
        }
    }

private:
    Box core;
    std::array<Box, 6> edges;
    std::array<MPI_Datatype, 3> types;
    std::array<MPI_Request, 12> requests[2];
    int edgeCount = 0;
    int typeCount = 0;
    int requestCount = 0;
};

void reportResult(const std::vector<Real>& local, const Block& block, const Index& globalShape,
                  const ProcessGrid& parts, MPI_Comm comm, int rank) {
    constexpr int resultTag = 6;
    if (rank != 0) {
        MPI_Datatype type = makeBlockType(block.shape, block.owned);
        MPI_Send(local.data() + offset(block.owned.begin, block.shape), 1, type, 0, resultTag, comm);
        MPI_Type_free(&type);
        return;
    }

    std::vector<Real> global(volume(globalShape));
    const size_t rowLength = block.owned.end[0] - block.owned.begin[0];
    for (size_t z = block.owned.begin[2]; z < block.owned.end[2]; ++z) {
        for (size_t y = block.owned.begin[1]; y < block.owned.end[1]; ++y) {
            const Index source{block.owned.begin[0], y, z};
            const Index dest{block.globalBegin[0],
                             block.globalBegin[1] + y - block.owned.begin[1],
                             block.globalBegin[2] + z - block.owned.begin[2]};
            std::copy_n(local.data() + offset(source, block.shape), rowLength,
                        global.data() + offset(dest, globalShape));
        }
    }

    const int ranks = parts[0] * parts[1] * parts[2];
    for (int peer = 1; peer < ranks; ++peer) {
        ProcessGrid coords;
        MPI_Cart_coords(comm, peer, 3, coords.data());
        const Block peerBlock = makeBlock(globalShape, parts, coords);
        MPI_Datatype type = makeBlockType(globalShape, peerBlock.owned);
        MPI_Recv(global.data() + offset(peerBlock.globalBegin, globalShape), 1, type,
                 peer, resultTag, comm, MPI_STATUS_IGNORE);
        MPI_Type_free(&type);
    }
    print_results(global, "Grid");
}

bool validateResult(const std::vector<Real>& local, const Block& block, MPI_Comm comm, int rank) {
    const Real infinity = std::numeric_limits<Real>::infinity();
    Real stats[3]{infinity, infinity, 1.0};
    for (size_t z = block.owned.begin[2]; z < block.owned.end[2]; ++z) {
        for (size_t y = block.owned.begin[1]; y < block.owned.end[1]; ++y) {
            const Real* row = local.data() + offset({0, y, z}, block.shape);
            for (size_t x = block.owned.begin[0]; x < block.owned.end[0]; ++x) {
                const Real value = row[x];
                if (!std::isfinite(value)) stats[2] = 0.0;
                stats[0] = std::min(stats[0], value);
                stats[1] = std::min(stats[1], -value);
            }
        }
    }

    Real global[3];
    MPI_Allreduce(stats, global, 3, MPI_DOUBLE, MPI_MIN, comm);
    if (global[2] == 0.0) {
        if (rank == 0) printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    const Real minVal = global[0];
    const Real maxVal = -global[1];
    if (rank == 0) printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        if (rank == 0) printf("Validation failed: values out of expected range\n");
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

struct Options {
    int nx = 128;
    int ny = 0;
    int nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
};

int parseOptions(int argc, char** argv, Options& options) {
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            options.nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            options.ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            options.nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            options.iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            options.validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            options.printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 2;
        }
    }
    if (options.ny == 0) options.ny = options.nx;
    if (options.nz == 0) options.nz = options.nx;
    if (options.nx <= 0 || options.ny <= 0 || options.nz <= 0) {
        printf("Grid dimensions must be positive\n");
        return 2;
    }
    return 0;
}

int runBenchmark(const Options& options, MPI_Comm comm, const ProcessGrid& parts) {
    int rank;
    ProcessGrid coords;
    MPI_Comm_rank(comm, &rank);
    MPI_Cart_coords(comm, rank, 3, coords.data());
    const Index shape{size_t(options.nx), size_t(options.ny), size_t(options.nz)};
    const Block block = makeBlock(shape, parts, coords);
    const int iterations = options.iterations;
    const bool hasInterior = options.nx > 2 && options.ny > 2 && options.nz > 2;

    if (rank == 0) {
        printf("3D Stencil Benchmark\n");
        printf("Grid size: %zu x %zu x %zu\n", shape[0], shape[1], shape[2]);
        printf("Iterations: %d\n", iterations);
        printf("Validation: %s\n", options.validate ? "enabled" : "disabled");
        printf("Initializing grid...\n");
    }

    const size_t localSize = volume(block.shape);
    std::vector<Real> grids[2]{std::vector<Real>(localSize), std::vector<Real>(localSize)};
    initializeGrid(grids[0], grids[1], block, shape, iterations > 0);
    double elapsed;
    if (rank == 0) printf("Running stencil computation...\n");
    if (hasInterior && iterations > 0) {
        HaloExchange halos(block, comm, grids);
        MPI_Barrier(comm);
        const double start = MPI_Wtime();
        // Initial halo values are generated directly from their global coordinates.
        stencilIteration(grids[0].data(), grids[1].data(), block.shape, block.interior);
        for (int iter = 1; iter < iterations; ++iter) {
            const int input = iter % 2;
            halos.update(input, grids[input].data(), grids[1 - input].data(), block.shape);
        }
        elapsed = MPI_Wtime() - start;
    } else {
        elapsed = 0.0;
    }

    double duration;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const double cellUpdates = hasInterior
            ? double(shape[0] - 2) * (shape[1] - 2) * (shape[2] - 2) * iterations : 0.0;
        const double mcups = duration > 0.0 ? cellUpdates / duration / 1e6 : 0.0;
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        printf("Performance: %.3f MCellUpdates/s\n", mcups);
    }

    const auto& result = grids[iterations % 2 != 0];
    if (options.printResults) reportResult(result, block, shape, parts, comm, rank);
    if (options.validate) {
        if (rank == 0) printf("Validating result...\n");
        const bool valid = validateResult(result, block, comm, rank);
        if (rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int settings[7];
    if (rank == 0) {
        Options options;
        settings[0] = parseOptions(argc, argv, options);
        settings[1] = options.nx;
        settings[2] = options.ny;
        settings[3] = options.nz;
        settings[4] = options.iterations;
        settings[5] = options.validate;
        settings[6] = options.printResults;
    }
    MPI_Bcast(settings, 7, MPI_INT, 0, MPI_COMM_WORLD);
    if (settings[0] != 0) {
        MPI_Finalize();
        return settings[0] == 1 ? 0 : 1;
    }

    const Options options{settings[1], settings[2], settings[3], settings[4],
                          settings[5] != 0, settings[6] != 0};
    ProcessGrid parts;
    if (rank == 0) {
        parts = chooseProcessGrid({size_t(options.nx), size_t(options.ny), size_t(options.nz)}, ranks);
    }
    MPI_Bcast(parts.data(), 3, MPI_INT, 0, MPI_COMM_WORLD);
    const ProcessGrid periods{0, 0, 0};
    MPI_Comm cart;
    MPI_Cart_create(MPI_COMM_WORLD, 3, parts.data(), periods.data(), 0, &cart);

    int status = 0;
    if (cart != MPI_COMM_NULL) {
        status = runBenchmark(options, cart, parts);
        MPI_Comm_free(&cart);
    }
    MPI_Finalize();
    return status;
}
