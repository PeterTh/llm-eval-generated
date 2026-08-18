#include <algorithm>
#include <cerrno>
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
constexpr unsigned int TILE_DIM = 32;

namespace {

int worldRank = 0;

[[noreturn]] void abortRun(const char* message) {
    std::fprintf(stderr, "Rank %d: %s\n", worldRank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkMpi(const int status, const char* expression, const char* file, const int line) {
    if (status == MPI_SUCCESS) {
        return;
    }

    char error[MPI_MAX_ERROR_STRING];
    int errorLength = 0;
    MPI_Error_string(status, error, &errorLength);
    std::fprintf(stderr, "Rank %d: MPI call %s failed at %s:%d: %.*s\n", worldRank,
                 expression, file, line, errorLength, error);
    MPI_Abort(MPI_COMM_WORLD, status);
    std::abort();
}

void checkCuda(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status == cudaSuccess) {
        return;
    }

    std::fprintf(stderr, "Rank %d: CUDA call %s failed at %s:%d: %s\n", worldRank,
                 expression, file, line, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

#define MPI_CHECK(call) checkMpi((call), #call, __FILE__, __LINE__)
#define CUDA_CHECK(call) checkCuda((call), #call, __FILE__, __LINE__)

// The original idx2(j, i, n) accesses source i and destination j.  Use that
// source-row layout directly throughout the distributed implementation.
inline constexpr size_t idx2(const size_t source, const size_t destination,
                             const size_t n) noexcept {
    return source * n + destination;
}

size_t blockCountForRank(const int rank, const size_t numBlocks, const int worldSize) {
    const size_t baseBlocks = numBlocks / static_cast<size_t>(worldSize);
    const size_t extraBlocks = numBlocks % static_cast<size_t>(worldSize);
    return baseBlocks + (static_cast<size_t>(rank) < extraBlocks ? 1 : 0);
}

size_t firstBlockForRank(const int rank, const size_t numBlocks, const int worldSize) {
    const size_t baseBlocks = numBlocks / static_cast<size_t>(worldSize);
    const size_t extraBlocks = numBlocks % static_cast<size_t>(worldSize);
    return baseBlocks * static_cast<size_t>(rank) +
           std::min(static_cast<size_t>(rank), extraBlocks);
}

int ownerOfBlock(const size_t block, const size_t numBlocks, const int worldSize) {
    const size_t baseBlocks = numBlocks / static_cast<size_t>(worldSize);
    const size_t extraBlocks = numBlocks % static_cast<size_t>(worldSize);
    const size_t largerBlockSize = baseBlocks + 1;
    const size_t blocksInLargerRanks = largerBlockSize * extraBlocks;

    if (block < blocksInLargerRanks) {
        return static_cast<int>(block / largerBlockSize);
    }
    // If baseBlocks is zero, every valid block was handled above.
    return static_cast<int>(extraBlocks + (block - blocksInLargerRanks) / baseBlocks);
}

size_t rowsInBlockRange(const size_t firstBlock, const size_t count, const size_t numNodes) {
    if (count == 0) {
        return 0;
    }
    const size_t firstRow = firstBlock * TILE_DIM;
    return std::min(numNodes - firstRow, count * static_cast<size_t>(TILE_DIM));
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    // Retain the original rand_r sequence so the graph is bit-for-bit identical
    // regardless of the number of MPI ranks.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
                                 range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t firstRow,
                               const size_t localRows, const size_t numNodes) {
    // The original initialization ultimately leaves path[source][destination]
    // equal to source.  This is independent for every local row and is the
    // OpenMP portion of the host-side hybrid implementation.
#pragma omp parallel for schedule(static)
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const unsigned int source = static_cast<unsigned int>(firstRow + localRow);
        unsigned int* const row = path.data() + localRow * numNodes;
        for (size_t destination = 0; destination < numNodes; ++destination) {
            row[destination] = source;
        }
    }
}

void sendUnsignedChunks(const unsigned int* data, const size_t count, const int destination,
                        const int tag) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count;) {
        const size_t chunk = std::min(maxChunk, count - offset);
        MPI_CHECK(MPI_Send(data + offset, static_cast<int>(chunk), MPI_UNSIGNED, destination, tag,
                           MPI_COMM_WORLD));
        offset += chunk;
    }
}

void receiveUnsignedChunks(unsigned int* data, const size_t count, const int source,
                           const int tag) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count;) {
        const size_t chunk = std::min(maxChunk, count - offset);
        MPI_CHECK(MPI_Recv(data + offset, static_cast<int>(chunk), MPI_UNSIGNED, source, tag,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += chunk;
    }
}

void broadcastUnsignedChunks(unsigned int* data, const size_t count, const int root) {
    constexpr size_t maxChunk = static_cast<size_t>(std::numeric_limits<int>::max());
    for (size_t offset = 0; offset < count;) {
        const size_t chunk = std::min(maxChunk, count - offset);
        MPI_CHECK(MPI_Bcast(data + offset, static_cast<int>(chunk), MPI_UNSIGNED, root,
                            MPI_COMM_WORLD));
        offset += chunk;
    }
}

void distributeRows(const std::vector<unsigned int>& globalDist,
                    std::vector<unsigned int>& localDist, const size_t numNodes,
                    const size_t numBlocks, const int worldSize) {
    constexpr int distributionTag = 27101;
    if (worldRank == 0) {
        if (!localDist.empty()) {
            std::copy_n(globalDist.data(), localDist.size(), localDist.data());
        }
        for (int rank = 1; rank < worldSize; ++rank) {
            const size_t firstBlock = firstBlockForRank(rank, numBlocks, worldSize);
            const size_t rows = rowsInBlockRange(firstBlock,
                                                  blockCountForRank(rank, numBlocks, worldSize),
                                                  numNodes);
            const size_t offset = firstBlock * static_cast<size_t>(TILE_DIM) * numNodes;
            sendUnsignedChunks(globalDist.data() + offset, rows * numNodes, rank, distributionTag);
        }
    } else {
        receiveUnsignedChunks(localDist.data(), localDist.size(), 0, distributionTag);
    }
}

void gatherRows(const std::vector<unsigned int>& localDist, std::vector<unsigned int>& globalDist,
                const size_t numNodes, const size_t numBlocks, const int worldSize) {
    constexpr int collectionTag = 27102;
    if (worldRank == 0) {
        if (!localDist.empty()) {
            std::copy_n(localDist.data(), localDist.size(), globalDist.data());
        }
        for (int rank = 1; rank < worldSize; ++rank) {
            const size_t firstBlock = firstBlockForRank(rank, numBlocks, worldSize);
            const size_t rows = rowsInBlockRange(firstBlock,
                                                  blockCountForRank(rank, numBlocks, worldSize),
                                                  numNodes);
            const size_t offset = firstBlock * static_cast<size_t>(TILE_DIM) * numNodes;
            receiveUnsignedChunks(globalDist.data() + offset, rows * numNodes, rank, collectionTag);
        }
    } else {
        sendUnsignedChunks(localDist.data(), localDist.size(), 0, collectionTag);
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    const size_t sampleNodes = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < sampleNodes; ++i) {
        for (size_t j = 0; j < sampleNodes; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];
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

size_t parseNumNodes(const char* value) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' ||
        parsed > std::numeric_limits<size_t>::max()) {
        abortRun("-n must be a non-negative integer that fits in size_t");
    }
    return static_cast<size_t>(parsed);
}

// Phase 1: close the pivot diagonal tile.  The explicit k loop preserves the
// strict-update path semantics of the original Floyd-Warshall implementation.
__global__ void closePivotTile(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                               const size_t localPivotRow, const size_t pivotRow,
                               const size_t numNodes, const unsigned int pivotRows) {
    __shared__ unsigned int distance[TILE_DIM][TILE_DIM];
    __shared__ unsigned int next[TILE_DIM][TILE_DIM];
    const unsigned int row = threadIdx.y;
    const unsigned int column = threadIdx.x;

    if (row < pivotRows && column < pivotRows) {
        const size_t index = (localPivotRow + row) * numNodes + pivotRow + column;
        distance[row][column] = dist[index];
        next[row][column] = path[index];
    } else {
        distance[row][column] = 0;
        next[row][column] = 0;
    }
    __syncthreads();

    for (unsigned int k = 0; k < pivotRows; ++k) {
        if (row < pivotRows && column < pivotRows) {
            const unsigned int alternative = distance[row][k] + distance[k][column];
            if (alternative < distance[row][column]) {
                distance[row][column] = alternative;
                next[row][column] = static_cast<unsigned int>(pivotRow) + k;
            }
        }
        __syncthreads();
    }

    if (row < pivotRows && column < pivotRows) {
        const size_t index = (localPivotRow + row) * numNodes + pivotRow + column;
        dist[index] = distance[row][column];
        path[index] = next[row][column];
    }
}

// Phase 2a: update the pivot block row after its diagonal tile is closed.
__global__ void updatePivotRow(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                               const unsigned int* __restrict__ diagonal,
                               const size_t localPivotRow, const size_t pivotRow,
                               const size_t numNodes, const size_t pivotBlock,
                               const unsigned int pivotRows) {
    const size_t destinationBlock = blockIdx.x;
    if (destinationBlock == pivotBlock) {
        return;
    }

    __shared__ unsigned int current[TILE_DIM][TILE_DIM];
    __shared__ unsigned int diagonalTile[TILE_DIM][TILE_DIM];
    const unsigned int row = threadIdx.y;
    const unsigned int column = threadIdx.x;
    const size_t destination = destinationBlock * TILE_DIM + column;

    if (row < pivotRows && destination < numNodes) {
        current[row][column] = dist[(localPivotRow + row) * numNodes + destination];
    } else {
        current[row][column] = 0;
    }
    if (row < pivotRows && column < pivotRows) {
        diagonalTile[row][column] = diagonal[row * TILE_DIM + column];
    } else {
        diagonalTile[row][column] = 0;
    }
    __syncthreads();

    for (unsigned int k = 0; k < pivotRows; ++k) {
        if (row < pivotRows && destination < numNodes) {
            const unsigned int alternative = diagonalTile[row][k] + current[k][column];
            if (alternative < current[row][column]) {
                current[row][column] = alternative;
                path[(localPivotRow + row) * numNodes + destination] =
                    static_cast<unsigned int>(pivotRow) + k;
            }
        }
        __syncthreads();
    }

    if (row < pivotRows && destination < numNodes) {
        dist[(localPivotRow + row) * numNodes + destination] = current[row][column];
    }
}

// Phase 2b: update every locally owned tile in the pivot block column.
__global__ void updatePivotColumn(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ diagonal,
                                  const size_t localFirstBlock,
                                  const size_t localBlockCount, const size_t localRows,
                                  const size_t pivotRow,
                                  const size_t numNodes, const size_t pivotBlock,
                                  const unsigned int pivotRows) {
    const size_t localBlock = blockIdx.x;
    if (localBlock >= localBlockCount || localFirstBlock + localBlock == pivotBlock) {
        return;
    }

    __shared__ unsigned int current[TILE_DIM][TILE_DIM];
    __shared__ unsigned int diagonalTile[TILE_DIM][TILE_DIM];
    const unsigned int row = threadIdx.y;
    const unsigned int column = threadIdx.x;
    const size_t localSource = localBlock * TILE_DIM + row;
    const size_t destination = pivotRow + column;

    if (localSource < localRows && destination < numNodes) {
        current[row][column] = dist[localSource * numNodes + destination];
    } else {
        current[row][column] = 0;
    }
    if (row < pivotRows && column < pivotRows) {
        diagonalTile[row][column] = diagonal[row * TILE_DIM + column];
    } else {
        diagonalTile[row][column] = 0;
    }
    __syncthreads();

    for (unsigned int k = 0; k < pivotRows; ++k) {
        if (localSource < localRows && destination < numNodes) {
            const unsigned int alternative = current[row][k] + diagonalTile[k][column];
            if (alternative < current[row][column]) {
                current[row][column] = alternative;
                path[localSource * numNodes + destination] = static_cast<unsigned int>(pivotRow) + k;
            }
        }
        __syncthreads();
    }

    if (localSource < localRows && destination < numNodes) {
        dist[localSource * numNodes + destination] = current[row][column];
    }
}

// Phase 3: update the remaining local tiles with the broadcast pivot panel.
__global__ void updateRemainingTiles(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int* __restrict__ pivotPanel,
                                     const size_t localFirstBlock,
                                     const size_t localBlockCount,
                                     const size_t localRows, const size_t pivotRow,
                                     const size_t numNodes, const size_t pivotBlock,
                                     const unsigned int pivotRows) {
    const size_t destinationBlock = blockIdx.x;
    const size_t localBlock = blockIdx.y;
    if (localBlock >= localBlockCount || destinationBlock == pivotBlock ||
        localFirstBlock + localBlock == pivotBlock) {
        return;
    }

    __shared__ unsigned int current[TILE_DIM][TILE_DIM];
    __shared__ unsigned int toPivot[TILE_DIM][TILE_DIM];
    __shared__ unsigned int fromPivot[TILE_DIM][TILE_DIM];
    const unsigned int row = threadIdx.y;
    const unsigned int column = threadIdx.x;
    const size_t localSource = localBlock * TILE_DIM + row;
    const size_t destination = destinationBlock * TILE_DIM + column;

    if (localSource < localRows && destination < numNodes) {
        current[row][column] = dist[localSource * numNodes + destination];
    } else {
        current[row][column] = 0;
    }
    if (localSource < localRows && column < pivotRows) {
        toPivot[row][column] = dist[localSource * numNodes + pivotRow + column];
    } else {
        toPivot[row][column] = 0;
    }
    if (row < pivotRows && destination < numNodes) {
        fromPivot[row][column] = pivotPanel[row * numNodes + destination];
    } else {
        fromPivot[row][column] = 0;
    }
    __syncthreads();

    for (unsigned int k = 0; k < pivotRows; ++k) {
        if (localSource < localRows && destination < numNodes) {
            const unsigned int alternative = toPivot[row][k] + fromPivot[k][column];
            if (alternative < current[row][column]) {
                current[row][column] = alternative;
                path[localSource * numNodes + destination] = static_cast<unsigned int>(pivotRow) + k;
            }
        }
        __syncthreads();
    }

    if (localSource < localRows && destination < numNodes) {
        dist[localSource * numNodes + destination] = current[row][column];
    }
}

void runDistributedFloydWarshall(
    unsigned int* deviceDist, unsigned int* devicePath, const size_t localFirstBlock,
    const size_t localBlockCount, const size_t localRows, const size_t numNodes,
    const size_t numBlocks, cudaStream_t computeStream, cudaStream_t transferStream,
    unsigned int* const deviceDiagonals[2], unsigned int* const hostDiagonals[2],
    unsigned int* const devicePanels[2], unsigned int* const hostPanels[2],
    cudaEvent_t diagonalReady[2], cudaEvent_t hostDiagonalConsumed[2],
    cudaEvent_t panelReady[2], cudaEvent_t hostPanelConsumed[2], const int worldSize) {
    const dim3 tileBlock(TILE_DIM, TILE_DIM);

    for (size_t pivotBlock = 0; pivotBlock < numBlocks; ++pivotBlock) {
        const int buffer = static_cast<int>(pivotBlock & 1U);
        const size_t pivotRow = pivotBlock * TILE_DIM;
        const unsigned int pivotRows = static_cast<unsigned int>(
            std::min(static_cast<size_t>(TILE_DIM), numNodes - pivotRow));
        const size_t panelElements = static_cast<size_t>(pivotRows) * numNodes;
        const int owner = ownerOfBlock(pivotBlock, numBlocks, worldSize);

        // The buffers used two pivot blocks ago must no longer be in a DMA read
        // before MPI can place the next diagonal/panel in them.
        CUDA_CHECK(cudaEventSynchronize(hostDiagonalConsumed[buffer]));
        CUDA_CHECK(cudaEventSynchronize(hostPanelConsumed[buffer]));

        if (worldRank == owner) {
            const size_t localPivotRow = pivotRow - localFirstBlock * TILE_DIM;
            // Phase 1 depends on all previous pivot panels on this owner.
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            closePivotTile<<<1, tileBlock, 0, computeStream>>>(
                deviceDist, devicePath, localPivotRow, pivotRow, numNodes, pivotRows);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(computeStream));

            // A 2-D copy extracts the diagonal tile without packing it on the CPU.
            CUDA_CHECK(cudaMemcpy2DAsync(hostDiagonals[buffer],
                                         TILE_DIM * sizeof(unsigned int),
                                         deviceDist + localPivotRow * numNodes + pivotRow,
                                         numNodes * sizeof(unsigned int),
                                         pivotRows * sizeof(unsigned int), pivotRows,
                                         cudaMemcpyDeviceToHost, transferStream));
            CUDA_CHECK(cudaStreamSynchronize(transferStream));
        }

        // Diagonal tiles are kept with a TILE_DIM pitch for the CUDA kernels.
        // Broadcast only their live rows, retaining that pitch on every rank.
        for (unsigned int row = 0; row < pivotRows; ++row) {
            broadcastUnsignedChunks(hostDiagonals[buffer] + row * TILE_DIM, pivotRows, owner);
        }

        // Phase 2 reads this diagonal while the next use of the same device slot
        // waits for phase 2 of pivotBlock-2 to complete.
        CUDA_CHECK(cudaStreamWaitEvent(transferStream, diagonalReady[buffer], 0));
        CUDA_CHECK(cudaMemcpy2DAsync(deviceDiagonals[buffer], TILE_DIM * sizeof(unsigned int),
                                     hostDiagonals[buffer], TILE_DIM * sizeof(unsigned int),
                                     pivotRows * sizeof(unsigned int), pivotRows,
                                     cudaMemcpyHostToDevice, transferStream));
        CUDA_CHECK(cudaEventRecord(hostDiagonalConsumed[buffer], transferStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, hostDiagonalConsumed[buffer], 0));

        if (worldRank == owner) {
            const size_t localPivotRow = pivotRow - localFirstBlock * TILE_DIM;
            updatePivotRow<<<dim3(static_cast<unsigned int>(numBlocks)), tileBlock, 0,
                             computeStream>>>(deviceDist, devicePath, deviceDiagonals[buffer],
                                               localPivotRow, pivotRow, numNodes, pivotBlock,
                                               pivotRows);
            CUDA_CHECK(cudaGetLastError());
        }
        if (localBlockCount != 0) {
            updatePivotColumn<<<dim3(static_cast<unsigned int>(localBlockCount)), tileBlock, 0,
                                computeStream>>>(deviceDist, devicePath,
                                                  deviceDiagonals[buffer], localFirstBlock,
                                                  localBlockCount, localRows, pivotRow, numNodes,
                                                  pivotBlock, pivotRows);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(diagonalReady[buffer], computeStream));

        if (worldRank == owner) {
            const size_t localPivotRow = pivotRow - localFirstBlock * TILE_DIM;
            // The panel must include phase 2a before every rank can do phase 3.
            CUDA_CHECK(cudaStreamSynchronize(computeStream));
            CUDA_CHECK(cudaMemcpyAsync(hostPanels[buffer],
                                       deviceDist + localPivotRow * numNodes,
                                       panelElements * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, transferStream));
            CUDA_CHECK(cudaStreamSynchronize(transferStream));
        }

        broadcastUnsignedChunks(hostPanels[buffer], panelElements, owner);

        // Double-buffering the pivot panels preserves overlap between a rank's
        // phase 3 work and communication for the next two pivot blocks.
        CUDA_CHECK(cudaStreamWaitEvent(transferStream, panelReady[buffer], 0));
        CUDA_CHECK(cudaMemcpyAsync(devicePanels[buffer], hostPanels[buffer],
                                   panelElements * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, transferStream));
        CUDA_CHECK(cudaEventRecord(hostPanelConsumed[buffer], transferStream));
        CUDA_CHECK(cudaStreamWaitEvent(computeStream, hostPanelConsumed[buffer], 0));
        if (localBlockCount != 0) {
            updateRemainingTiles<<<dim3(static_cast<unsigned int>(numBlocks),
                                        static_cast<unsigned int>(localBlockCount)),
                                   tileBlock, 0, computeStream>>>(
                deviceDist, devicePath, devicePanels[buffer], localFirstBlock, localBlockCount,
                localRows, pivotRow, numNodes, pivotBlock, pivotRows);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaEventRecord(panelReady[buffer], computeStream));
    }
}

void destroyCudaResources(unsigned int* deviceDist, unsigned int* devicePath,
                          unsigned int* const deviceDiagonals[2],
                          unsigned int* const hostDiagonals[2],
                          unsigned int* const devicePanels[2],
                          unsigned int* const hostPanels[2], cudaEvent_t diagonalReady[2],
                          cudaEvent_t hostDiagonalConsumed[2], cudaEvent_t panelReady[2],
                          cudaEvent_t hostPanelConsumed[2], cudaStream_t computeStream,
                          cudaStream_t transferStream) {
    cudaEventDestroy(diagonalReady[0]);
    cudaEventDestroy(diagonalReady[1]);
    cudaEventDestroy(hostDiagonalConsumed[0]);
    cudaEventDestroy(hostDiagonalConsumed[1]);
    cudaEventDestroy(panelReady[0]);
    cudaEventDestroy(panelReady[1]);
    cudaEventDestroy(hostPanelConsumed[0]);
    cudaEventDestroy(hostPanelConsumed[1]);
    cudaStreamDestroy(computeStream);
    cudaStreamDestroy(transferStream);
    cudaFree(deviceDist);
    cudaFree(devicePath);
    cudaFree(deviceDiagonals[0]);
    cudaFree(deviceDiagonals[1]);
    cudaFree(devicePanels[0]);
    cudaFree(devicePanels[1]);
    cudaFreeHost(hostDiagonals[0]);
    cudaFreeHost(hostDiagonals[1]);
    cudaFreeHost(hostPanels[0]);
    cudaFreeHost(hostPanels[1]);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));

    int worldSize = 0;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = parseNumNodes(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) {
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 0;
        } else {
            if (worldRank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_CHECK(MPI_Finalize());
            return 1;
        }
    }

    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes) ||
        numNodes > std::numeric_limits<size_t>::max() / TILE_DIM) {
        abortRun("matrix size exceeds the supported MPI count or address space");
    }

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank,
                                  MPI_INFO_NULL, &localComm));
    int localRank = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        abortRun("no CUDA device is available for this MPI rank");
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_CHECK(MPI_Comm_free(&localComm));

    const size_t numBlocks = (numNodes + TILE_DIM - 1) / TILE_DIM;
    const size_t localBlockCount = blockCountForRank(worldRank, numBlocks, worldSize);
    const size_t localFirstBlock = firstBlockForRank(worldRank, numBlocks, worldSize);
    const size_t firstRow = localFirstBlock * TILE_DIM;
    const size_t localRows = rowsInBlockRange(localFirstBlock, localBlockCount, numNodes);
    const size_t localElements = localRows * numNodes;

    if (worldRank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("MPI ranks: %d\n", worldSize);
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalDist;
    if (worldRank == 0) {
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
    }
    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    distributeRows(globalDist, localDist, numNodes, numBlocks, worldSize);
    initializeLocalPathMatrix(localPath, firstRow, localRows, numNodes);

    const size_t localAllocation = std::max(localElements, static_cast<size_t>(1));
    const size_t panelAllocation = std::max(numNodes * static_cast<size_t>(TILE_DIM),
                                            static_cast<size_t>(1));
    const size_t localBytes = localAllocation * sizeof(unsigned int);
    const size_t panelBytes = panelAllocation * sizeof(unsigned int);
    const size_t diagonalBytes = static_cast<size_t>(TILE_DIM) * TILE_DIM * sizeof(unsigned int);

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* deviceDiagonals[2] = {nullptr, nullptr};
    unsigned int* hostDiagonals[2] = {nullptr, nullptr};
    unsigned int* devicePanels[2] = {nullptr, nullptr};
    unsigned int* hostPanels[2] = {nullptr, nullptr};
    cudaStream_t computeStream = nullptr;
    cudaStream_t transferStream = nullptr;
    cudaEvent_t diagonalReady[2] = {nullptr, nullptr};
    cudaEvent_t hostDiagonalConsumed[2] = {nullptr, nullptr};
    cudaEvent_t panelReady[2] = {nullptr, nullptr};
    cudaEvent_t hostPanelConsumed[2] = {nullptr, nullptr};

    CUDA_CHECK(cudaMalloc(&deviceDist, localBytes));
    CUDA_CHECK(cudaMalloc(&devicePath, localBytes));
    CUDA_CHECK(cudaMalloc(&deviceDiagonals[0], diagonalBytes));
    CUDA_CHECK(cudaMalloc(&deviceDiagonals[1], diagonalBytes));
    CUDA_CHECK(cudaMalloc(&devicePanels[0], panelBytes));
    CUDA_CHECK(cudaMalloc(&devicePanels[1], panelBytes));
    CUDA_CHECK(cudaHostAlloc(&hostDiagonals[0], diagonalBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hostDiagonals[1], diagonalBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hostPanels[0], panelBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&hostPanels[1], panelBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking));
    for (int buffer = 0; buffer < 2; ++buffer) {
        CUDA_CHECK(cudaEventCreateWithFlags(&diagonalReady[buffer], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hostDiagonalConsumed[buffer], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&panelReady[buffer], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&hostPanelConsumed[buffer], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(diagonalReady[buffer], computeStream));
        CUDA_CHECK(cudaEventRecord(panelReady[buffer], computeStream));
        CUDA_CHECK(cudaEventRecord(hostDiagonalConsumed[buffer], transferStream));
        CUDA_CHECK(cudaEventRecord(hostPanelConsumed[buffer], transferStream));
    }
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpyAsync(deviceDist, localDist.data(), localElements * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, computeStream));
        CUDA_CHECK(cudaMemcpyAsync(devicePath, localPath.data(), localElements * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, computeStream));
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    if (worldRank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();

    // Implemented below after all CUDA/MPI resources have been initialized.
    runDistributedFloydWarshall(deviceDist, devicePath, localFirstBlock, localBlockCount,
                                localRows, numNodes, numBlocks, computeStream, transferStream,
                                deviceDiagonals, hostDiagonals, devicePanels, hostPanels,
                                diagonalReady, hostDiagonalConsumed, panelReady,
                                hostPanelConsumed, worldSize);
    CUDA_CHECK(cudaStreamSynchronize(computeStream));

    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));

    if (worldRank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = operations / (static_cast<double>(milliseconds) / 1000.0) / 1e9;
        std::printf("Performance: %.3f GOPS\n", gflops);
    }

    if (validate || printResults) {
        if (localElements != 0) {
            CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist, localElements * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }
        gatherRows(localDist, globalDist, numNodes, numBlocks, worldSize);
    }

    int exitStatus = 0;
    if (worldRank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }
    if (worldRank == 0 && validate) {
        std::printf("Validating result...\n");
        if (validateResult(globalDist, numNodes)) {
            std::printf("Validation: PASSED\n");
        } else {
            std::printf("Validation: FAILED\n");
            exitStatus = 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&exitStatus, 1, MPI_INT, 0, MPI_COMM_WORLD));

    destroyCudaResources(deviceDist, devicePath, deviceDiagonals, hostDiagonals, devicePanels,
                         hostPanels, diagonalReady, hostDiagonalConsumed, panelReady,
                         hostPanelConsumed, computeStream, transferStream);
    MPI_CHECK(MPI_Finalize());
    return exitStatus;
}
