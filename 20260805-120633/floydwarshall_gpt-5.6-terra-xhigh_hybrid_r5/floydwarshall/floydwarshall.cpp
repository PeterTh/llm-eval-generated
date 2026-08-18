#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE_SIZE = 32;
constexpr int ROW_THREADS = 32;
constexpr int ROWS_PER_BLOCK = 8;

namespace {

int g_rank = 0;

#define MPI_CHECK(call)                                                                  \
    do {                                                                                 \
        const int mpiStatus = (call);                                                    \
        if (mpiStatus != MPI_SUCCESS) {                                                  \
            char mpiError[MPI_MAX_ERROR_STRING];                                         \
            int mpiErrorLength = 0;                                                      \
            MPI_Error_string(mpiStatus, mpiError, &mpiErrorLength);                      \
            std::fprintf(stderr, "MPI error on rank %d: %.*s\n", g_rank,               \
                         mpiErrorLength, mpiError);                                      \
            MPI_Abort(MPI_COMM_WORLD, mpiStatus);                                        \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t cudaStatus = (call);                                           \
        if (cudaStatus != cudaSuccess) {                                                 \
            std::fprintf(stderr, "CUDA error on rank %d at %s:%d: %s\n", g_rank,       \
                         __FILE__, __LINE__, cudaGetErrorString(cudaStatus));            \
            MPI_Abort(MPI_COMM_WORLD, static_cast<int>(cudaStatus));                     \
            std::abort();                                                                \
        }                                                                                \
    } while (0)

// The external representation is intentionally column-major to match the original
// benchmark.  GPUs use row-major storage internally so every warp accesses a contiguous
// destination span.
inline constexpr size_t externalIndex(const size_t source, const size_t destination,
                                      const size_t nodes) noexcept {
    return destination * nodes + source;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[externalIndex(i, i, numNodes)] = 0;
    }
}

// Phase 1: close the diagonal tile.  One CUDA block owns the tile, which gives a
// synchronization point after every intermediate vertex and exactly implements the
// dependency inside a Floyd-Warshall block.
__global__ void closeDiagonalTile(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path, const int nodes,
                                  const int localPivot, const int pivotStart,
                                  const int tileWidth) {
    __shared__ unsigned int distances[TILE_SIZE * TILE_SIZE];
    __shared__ unsigned int paths[TILE_SIZE * TILE_SIZE];

    const int column = threadIdx.x;
    const int row = threadIdx.y;
    const int tileIndex = row * TILE_SIZE + column;

    if (row < tileWidth && column < tileWidth) {
        const size_t matrixIndex = static_cast<size_t>(localPivot + row) * nodes +
                                   pivotStart + column;
        distances[tileIndex] = dist[matrixIndex];
        paths[tileIndex] = path[matrixIndex];
    }
    __syncthreads();

    for (int intermediate = 0; intermediate < tileWidth; ++intermediate) {
        if (row < tileWidth && column < tileWidth) {
            const unsigned int candidate = distances[row * TILE_SIZE + intermediate] +
                                           distances[intermediate * TILE_SIZE + column];
            if (candidate < distances[tileIndex]) {
                distances[tileIndex] = candidate;
                paths[tileIndex] = static_cast<unsigned int>(pivotStart + intermediate);
            }
        }
        __syncthreads();
    }

    if (row < tileWidth && column < tileWidth) {
        const size_t matrixIndex = static_cast<size_t>(localPivot + row) * nodes +
                                   pivotStart + column;
        dist[matrixIndex] = distances[tileIndex];
        path[matrixIndex] = paths[tileIndex];
    }
}

// Phase 2: update the pivot tile's complete row panel.  panel contains an immutable
// snapshot, preventing row-to-row races while the panel is closed with the diagonal tile.
__global__ void updatePivotRows(unsigned int* __restrict__ dist,
                                unsigned int* __restrict__ path,
                                const unsigned int* __restrict__ diagonal,
                                const unsigned int* __restrict__ panel, const int nodes,
                                const int localPivot, const int pivotStart,
                                const int tileWidth) {
    const int destination = blockIdx.x * blockDim.x + threadIdx.x;
    const int pivotRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (pivotRow >= tileWidth || destination >= nodes) {
        return;
    }

    const size_t resultIndex = static_cast<size_t>(localPivot + pivotRow) * nodes + destination;
    unsigned int best = panel[static_cast<size_t>(pivotRow) * nodes + destination];
    unsigned int bestPath = path[resultIndex];
    for (int intermediate = 0; intermediate < tileWidth; ++intermediate) {
        const unsigned int candidate = diagonal[pivotRow * TILE_SIZE + intermediate] +
                                       panel[static_cast<size_t>(intermediate) * nodes + destination];
        if (candidate < best) {
            best = candidate;
            bestPath = static_cast<unsigned int>(pivotStart + intermediate);
        }
    }
    dist[resultIndex] = best;
    path[resultIndex] = bestPath;
}

// Phase 3: update one local row's pivot-column segment.  The old segment is staged in
// shared memory because all of its output values depend on the same pre-update values.
__global__ void updatePivotColumns(unsigned int* __restrict__ dist,
                                   unsigned int* __restrict__ path,
                                   const unsigned int* __restrict__ diagonal,
                                   const int nodes, const int localRows,
                                   const int localPivot, const int pivotStart,
                                   const int tileWidth) {
    const int localRow = blockIdx.x;
    const int target = threadIdx.x;
    if (localRow >= localRows ||
        (localRow >= localPivot && localRow < localPivot + tileWidth)) {
        return;
    }

    __shared__ unsigned int oldColumn[TILE_SIZE];
    if (threadIdx.x < tileWidth) {
        oldColumn[threadIdx.x] =
            dist[static_cast<size_t>(localRow) * nodes + pivotStart + threadIdx.x];
    }
    __syncthreads();
    if (target >= tileWidth) {
        return;
    }

    unsigned int best = oldColumn[target];
    unsigned int bestPath =
        path[static_cast<size_t>(localRow) * nodes + pivotStart + target];
    for (int intermediate = 0; intermediate < tileWidth; ++intermediate) {
        const unsigned int candidate = oldColumn[intermediate] +
                                       diagonal[intermediate * TILE_SIZE + target];
        if (candidate < best) {
            best = candidate;
            bestPath = static_cast<unsigned int>(pivotStart + intermediate);
        }
    }
    const size_t resultIndex = static_cast<size_t>(localRow) * nodes + pivotStart + target;
    dist[resultIndex] = best;
    path[resultIndex] = bestPath;
}

// Phase 4: update all non-pivot rows and non-pivot columns.  Each warp advances along a
// contiguous destination segment, while the distributed pivot row panel stays resident
// on the device throughout the phase.
__global__ void updateRemainingTiles(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int* __restrict__ panel,
                                     const int nodes, const int localRows,
                                     const int localPivot, const int pivotStart,
                                     const int tileWidth) {
    const int destination = blockIdx.x * blockDim.x + threadIdx.x;
    const int localRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (localRow >= localRows || destination >= nodes ||
        (localRow >= localPivot && localRow < localPivot + tileWidth) ||
        (destination >= pivotStart && destination < pivotStart + tileWidth)) {
        return;
    }

    const size_t resultIndex = static_cast<size_t>(localRow) * nodes + destination;
    unsigned int best = dist[resultIndex];
    unsigned int bestPath = path[resultIndex];
    const size_t localRowStart = static_cast<size_t>(localRow) * nodes;
    for (int intermediate = 0; intermediate < tileWidth; ++intermediate) {
        const unsigned int candidate = dist[localRowStart + pivotStart + intermediate] +
                                       panel[static_cast<size_t>(intermediate) * nodes + destination];
        if (candidate < best) {
            best = candidate;
            bestPath = static_cast<unsigned int>(pivotStart + intermediate);
        }
    }
    dist[resultIndex] = best;
    path[resultIndex] = bestPath;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[externalIndex(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[externalIndex(i, j, numNodes)];
                const unsigned int distIK = dist[externalIndex(i, k, numNodes)];
                const unsigned int distKJ = dist[externalIndex(k, j, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool parseNodeCount(const char* argument, size_t* result) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(argument, &end, 10);
    if (errno != 0 || end == argument || *end != '\0' || value == 0 ||
        value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    *result = static_cast<size_t>(value);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel));

    int worldSize = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &g_rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    if (mpiThreadLevel < MPI_THREAD_FUNNELED) {
        if (g_rank == 0) {
            std::fprintf(stderr, "MPI does not provide the required MPI_THREAD_FUNNELED level\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool parseError = false;
    bool showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            parseError = !parseNodeCount(argv[++i], &numNodes);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (g_rank == 0) {
            if (parseError) {
                std::printf("Invalid command line arguments\n");
            }
            printUsage(argv[0]);
        }
        MPI_CHECK(MPI_Finalize());
        return parseError ? 1 : 0;
    }

    // MPI counts and CUDA kernel coordinates are signed ints.  More importantly, the
    // original program cannot practically allocate a larger dense matrix either.
    const size_t maxNodes = static_cast<size_t>(std::sqrt(static_cast<double>(INT_MAX)));
    if (numNodes > maxNodes) {
        if (g_rank == 0) {
            std::fprintf(stderr, "-n must not exceed %zu for this MPI/CUDA implementation\n", maxNodes);
        }
        MPI_CHECK(MPI_Finalize());
        return 1;
    }
    const int nodes = static_cast<int>(numNodes);
    const int numberOfTiles = (nodes + TILE_SIZE - 1) / TILE_SIZE;

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL,
                                  &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (g_rank == 0) {
            std::fprintf(stderr, "No CUDA device is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const int tileBegin = (g_rank * numberOfTiles) / worldSize;
    const int tileEnd = ((g_rank + 1) * numberOfTiles) / worldSize;
    const int localStart = tileBegin * TILE_SIZE;
    const int localEnd = std::min(nodes, tileEnd * TILE_SIZE);
    const int localRows = std::max(0, localEnd - localStart);
    const size_t localElements = static_cast<size_t>(localRows) * numNodes;
    const size_t allElements = numNodes * numNodes;

    std::vector<int> elementCounts(worldSize);
    std::vector<int> elementDisplacements(worldSize);
    for (int rank = 0; rank < worldSize; ++rank) {
        const int rankTileBegin = (rank * numberOfTiles) / worldSize;
        const int rankTileEnd = ((rank + 1) * numberOfTiles) / worldSize;
        const int rankStart = rankTileBegin * TILE_SIZE;
        const int rankEnd = std::min(nodes, rankTileEnd * TILE_SIZE);
        const size_t rankElements = static_cast<size_t>(std::max(0, rankEnd - rankStart)) * numNodes;
        if (rankElements > static_cast<size_t>(INT_MAX)) {
            if (g_rank == 0) {
                std::fprintf(stderr, "Local MPI message is too large\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        elementCounts[rank] = static_cast<int>(rankElements);
        elementDisplacements[rank] = rankStart * nodes;
    }

    if (g_rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Configuration: MPI ranks=%d, OpenMP threads=%d, CUDA tile=%d\n", worldSize,
                    omp_get_max_threads(), TILE_SIZE);
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDistance(localElements);
    std::vector<unsigned int> localPath(localElements);
    std::vector<unsigned int> rowMajorDistance;
    std::vector<unsigned int> rowMajorPath;
    if (g_rank == 0) {
        std::vector<unsigned int> externalDistance(allElements);
        initializeDistanceMatrix(externalDistance, numNodes, 1, MAX_DISTANCE);
        rowMajorDistance.resize(allElements);
        rowMajorPath.resize(allElements);

#pragma omp parallel for schedule(static)
        for (long long source = 0; source < nodes; ++source) {
            const size_t rowOffset = static_cast<size_t>(source) * numNodes;
            for (int destination = 0; destination < nodes; ++destination) {
                rowMajorDistance[rowOffset + destination] =
                    externalDistance[externalIndex(source, destination, numNodes)];
                rowMajorPath[rowOffset + destination] = static_cast<unsigned int>(destination);
            }
        }
    }

    MPI_CHECK(MPI_Scatterv(g_rank == 0 ? rowMajorDistance.data() : nullptr, elementCounts.data(),
                           elementDisplacements.data(), MPI_UNSIGNED, localDistance.data(),
                           static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Scatterv(g_rank == 0 ? rowMajorPath.data() : nullptr, elementCounts.data(),
                           elementDisplacements.data(), MPI_UNSIGNED, localPath.data(),
                           static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    rowMajorDistance.clear();
    rowMajorPath.clear();
    rowMajorDistance.shrink_to_fit();
    rowMajorPath.shrink_to_fit();

    unsigned int* deviceDistance = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* deviceDiagonal = nullptr;
    unsigned int* devicePanel = nullptr;
    unsigned int* hostPanel = nullptr;
    const size_t allocationElements = std::max(localElements, static_cast<size_t>(1));
    CUDA_CHECK(cudaMalloc(&deviceDistance, allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, allocationElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&deviceDiagonal, TILE_SIZE * TILE_SIZE * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePanel, static_cast<size_t>(TILE_SIZE) * numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaHostAlloc(&hostPanel, static_cast<size_t>(TILE_SIZE) * numNodes * sizeof(unsigned int),
                             cudaHostAllocPortable));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(deviceDistance, localDistance.data(), localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(), localElements * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    if (g_rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    for (int tile = 0; tile < numberOfTiles; ++tile) {
        const int pivotStart = tile * TILE_SIZE;
        const int tileWidth = std::min(TILE_SIZE, nodes - pivotStart);
        // This is the inverse of the floor-based tile partition above.  It also
        // correctly handles ranks with no tiles when there are more ranks than tiles.
        const int owner = ((tile + 1) * worldSize - 1) / numberOfTiles;
        const int ownerTileBegin = (owner * numberOfTiles) / worldSize;
        const int ownerLocalPivot = pivotStart - ownerTileBegin * TILE_SIZE;
        const int localPivot = g_rank == owner ? ownerLocalPivot : -TILE_SIZE;

        if (g_rank == owner) {
            const dim3 diagonalBlock(TILE_SIZE, TILE_SIZE);
            closeDiagonalTile<<<1, diagonalBlock>>>(deviceDistance, devicePath, nodes, ownerLocalPivot,
                                                     pivotStart, tileWidth);
            CUDA_CHECK(cudaGetLastError());

            const size_t pivotOffset = static_cast<size_t>(ownerLocalPivot) * nodes + pivotStart;
            CUDA_CHECK(cudaMemcpy2D(deviceDiagonal, TILE_SIZE * sizeof(unsigned int),
                                     deviceDistance + pivotOffset, nodes * sizeof(unsigned int),
                                     tileWidth * sizeof(unsigned int), tileWidth,
                                     cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy2D(devicePanel, nodes * sizeof(unsigned int),
                                     deviceDistance + static_cast<size_t>(ownerLocalPivot) * nodes,
                                     nodes * sizeof(unsigned int), nodes * sizeof(unsigned int), tileWidth,
                                     cudaMemcpyDeviceToDevice));

            const dim3 rowBlock(ROW_THREADS, ROWS_PER_BLOCK);
            const dim3 rowGrid((nodes + ROW_THREADS - 1) / ROW_THREADS,
                               (tileWidth + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
            updatePivotRows<<<rowGrid, rowBlock>>>(deviceDistance, devicePath, deviceDiagonal, devicePanel,
                                                    nodes, ownerLocalPivot, pivotStart, tileWidth);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy2D(hostPanel, nodes * sizeof(unsigned int),
                                     deviceDistance + static_cast<size_t>(ownerLocalPivot) * nodes,
                                     nodes * sizeof(unsigned int), nodes * sizeof(unsigned int), tileWidth,
                                     cudaMemcpyDeviceToHost));
        }

        MPI_CHECK(MPI_Bcast(hostPanel, tileWidth * nodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(devicePanel, hostPanel,
                              static_cast<size_t>(tileWidth) * nodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        // Every rank needs the closed diagonal tile for phase 3.  It is already part
        // of the broadcast pivot-row panel, so extract it locally instead of sending a
        // second MPI message.
        CUDA_CHECK(cudaMemcpy2D(deviceDiagonal, TILE_SIZE * sizeof(unsigned int),
                                 devicePanel + pivotStart, nodes * sizeof(unsigned int),
                                 tileWidth * sizeof(unsigned int), tileWidth,
                                 cudaMemcpyDeviceToDevice));

        if (localRows != 0) {
            updatePivotColumns<<<localRows, TILE_SIZE>>>(deviceDistance, devicePath, deviceDiagonal, nodes,
                                                         localRows, localPivot, pivotStart, tileWidth);
            CUDA_CHECK(cudaGetLastError());

            const dim3 remainingBlock(ROW_THREADS, ROWS_PER_BLOCK);
            const dim3 remainingGrid((nodes + ROW_THREADS - 1) / ROW_THREADS,
                                     (localRows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
            updateRemainingTiles<<<remainingGrid, remainingBlock>>>(
                deviceDistance, devicePath, devicePanel, nodes, localRows, localPivot, pivotStart, tileWidth);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double durationSeconds = MPI_Wtime() - start;

    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(localDistance.data(), deviceDistance,
                              localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFreeHost(hostPanel));
    CUDA_CHECK(cudaFree(devicePanel));
    CUDA_CHECK(cudaFree(deviceDiagonal));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDistance));

    std::vector<unsigned int> gatheredDistance;
    if (g_rank == 0) {
        gatheredDistance.resize(allElements);
    }
    MPI_CHECK(MPI_Gatherv(localDistance.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                          g_rank == 0 ? gatheredDistance.data() : nullptr, elementCounts.data(),
                          elementDisplacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD));

    int exitCode = 0;
    if (g_rank == 0) {
        std::vector<unsigned int> externalDistance(allElements);
#pragma omp parallel for schedule(static)
        for (long long source = 0; source < nodes; ++source) {
            const size_t rowOffset = static_cast<size_t>(source) * numNodes;
            for (int destination = 0; destination < nodes; ++destination) {
                externalDistance[externalIndex(source, destination, numNodes)] =
                    gatheredDistance[rowOffset + destination];
            }
        }

        const long long milliseconds = static_cast<long long>(durationSeconds * 1000.0);
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = durationSeconds > 0.0 ? operations / durationSeconds / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(externalDistance, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(externalDistance, numNodes)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_CHECK(MPI_Comm_free(&localComm));
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return exitCode;
}
