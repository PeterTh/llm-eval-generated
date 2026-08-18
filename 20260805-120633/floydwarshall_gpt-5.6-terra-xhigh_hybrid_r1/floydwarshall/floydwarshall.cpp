#include <algorithm>
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
constexpr unsigned int ROWS_PER_BLOCK = 32;
constexpr unsigned int COLS_PER_BLOCK = 8;

// The original code stores a matrix column-major: element (source, destination)
// is at destination * numNodes + source.  Each MPI rank owns a contiguous range
// of source rows, which keeps every local column contiguous on the GPU.
inline constexpr size_t idx2(const size_t source, const size_t destination,
                             const size_t numNodes) noexcept {
    return destination * numNodes + source;
}

void checkCuda(const cudaError_t status, const char* expression, const int rank,
               const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "Rank %d: CUDA error at %s:%d in %s: %s\n", rank,
                     file, line, expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call, rank, __FILE__, __LINE__)

__global__ void gatherPivotRow(const unsigned int* const dist,
                               unsigned int* const pivot,
                               const size_t localRows,
                               const size_t localPivotRow,
                               const size_t numNodes) {
    const size_t destination =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (destination < numNodes) {
        pivot[destination] = dist[destination * localRows + localPivotRow];
    }
}

// One pivot is a global dependency, but all source/destination relaxations for
// that pivot are independent.  The x dimension maps to contiguous local source
// rows, yielding coalesced reads and writes in the column-major representation.
__global__ void relaxPivot(unsigned int* const dist, unsigned int* const path,
                           const unsigned int* const pivot,
                           const size_t localRows, const size_t numNodes,
                           const size_t pivotNode) {
    const size_t localSource =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t destination =
        static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    __shared__ unsigned int sourceToPivot[ROWS_PER_BLOCK];
    __shared__ unsigned int pivotToDestination[COLS_PER_BLOCK];

    if (threadIdx.y == 0 && localSource < localRows) {
        sourceToPivot[threadIdx.x] = dist[pivotNode * localRows + localSource];
    }
    if (threadIdx.x == 0 && destination < numNodes) {
        pivotToDestination[threadIdx.y] = pivot[destination];
    }
    __syncthreads();

    if (localSource < localRows && destination < numNodes) {
        const size_t position = destination * localRows + localSource;
        const unsigned int oldDistance = dist[position];
        const unsigned int newDistance = sourceToPivot[threadIdx.x] +
                                         pivotToDestination[threadIdx.y];
        if (newDistance < oldDistance) {
            dist[position] = newDistance;
            path[position] = static_cast<unsigned int>(pivotNode);
        }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // This intentionally remains bit-for-bit identical to the original
    // initialization, including rand_r's sequence and flattened ordering.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
                                  range * rand_r(&seed) /
                                  static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                               const size_t localRows,
                               const size_t numNodes) {
    // The original two nested assignments reduce to path[source][destination]
    // = destination.  The local layout is destination * localRows + source.
#pragma omp parallel for schedule(static)
    for (size_t element = 0; element < localRows * numNodes; ++element) {
        path[element] = static_cast<unsigned int>(element / localRows);
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n",
                        i, i);
            return false;
        }
    }

    // Keep the original sample and order of checks so validation has unchanged
    // semantics while avoiding an additional cubic pass.
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf("Validation failed: triangle inequality violated at "
                                "[%zu,%zu,%zu]\n", i, j, k);
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int argumentStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
                argumentStatus = 1;
            } else {
                numNodes = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            argumentStatus = 1;
        }
    }

    if (argumentStatus != 0 || numNodes == 0 || numNodes < static_cast<size_t>(worldSize) ||
        numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        numNodes > static_cast<size_t>(std::sqrt(std::numeric_limits<int>::max()))) {
        if (rank == 0) {
            std::printf("Invalid node count or too many MPI ranks for the matrix size\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    // Use a node-local rank so every node maps its ranks across its own GPUs.
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device is available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t extraRows = numNodes % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    const size_t firstLocalRow = static_cast<size_t>(rank) * baseRows +
                                 std::min(static_cast<size_t>(rank), extraRows);
    const size_t localElements = localRows * numNodes;

    std::vector<int> elementCounts(worldSize);
    std::vector<int> elementOffsets(worldSize);
    std::vector<size_t> rowStarts(worldSize);
    std::vector<size_t> rowCounts(worldSize);
    int runningOffset = 0;
    for (int process = 0; process < worldSize; ++process) {
        const size_t processRows = baseRows +
            (static_cast<size_t>(process) < extraRows ? 1 : 0);
        rowCounts[process] = processRows;
        rowStarts[process] = static_cast<size_t>(process) * baseRows +
                             std::min(static_cast<size_t>(process), extraRows);
        elementCounts[process] = static_cast<int>(processRows * numNodes);
        elementOffsets[process] = runningOffset;
        runningOffset += elementCounts[process];
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("MPI ranks: %d, CUDA devices per node: %d, OpenMP threads/rank: %d\n",
                    worldSize, deviceCount, omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> completeDistance;
    std::vector<unsigned int> packedDistance;
    if (rank == 0) {
        completeDistance.resize(numNodes * numNodes);
        initializeDistanceMatrix(completeDistance, numNodes, 1, MAX_DISTANCE);

        // MPI_Scatterv needs each rank's data contiguous; pack the same
        // column-major submatrices used by the device-resident distribution.
        packedDistance.resize(numNodes * numNodes);
#pragma omp parallel for collapse(2) schedule(static)
        for (int process = 0; process < worldSize; ++process) {
            for (size_t destination = 0; destination < numNodes; ++destination) {
                const size_t output = static_cast<size_t>(elementOffsets[process]) +
                                      destination * rowCounts[process];
                const size_t input = destination * numNodes + rowStarts[process];
                std::memcpy(&packedDistance[output], &completeDistance[input],
                            rowCounts[process] * sizeof(unsigned int));
            }
        }
        completeDistance.clear();
        completeDistance.shrink_to_fit();
    }

    std::vector<unsigned int> localDistance(localElements);
    std::vector<unsigned int> localPath(localElements);
    MPI_Scatterv(rank == 0 ? packedDistance.data() : nullptr, elementCounts.data(),
                 elementOffsets.data(), MPI_UNSIGNED, localDistance.data(),
                 static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    initializeLocalPathMatrix(localPath, localRows, numNodes);
    packedDistance.clear();
    packedDistance.shrink_to_fit();

    unsigned int* deviceDistance = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivot = nullptr;
    unsigned int* hostPivot = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceDistance, localElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, localElements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePivot, numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hostPivot, numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(deviceDistance, localDistance.data(),
                          localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(),
                          localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));

    if (rank == 0) {
        std::printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    const dim3 block(ROWS_PER_BLOCK, COLS_PER_BLOCK);
    const dim3 updateGrid(static_cast<unsigned int>((localRows + ROWS_PER_BLOCK - 1) /
                                                    ROWS_PER_BLOCK),
                          static_cast<unsigned int>((numNodes + COLS_PER_BLOCK - 1) /
                                                    COLS_PER_BLOCK));
    const unsigned int gatherBlockSize = 256;
    const dim3 gatherGrid(static_cast<unsigned int>((numNodes + gatherBlockSize - 1) /
                                                    gatherBlockSize));

    for (size_t pivotNode = 0; pivotNode < numNodes; ++pivotNode) {
        const int pivotOwner = static_cast<int>(
            std::upper_bound(rowStarts.begin(), rowStarts.end(), pivotNode) -
            rowStarts.begin()) - 1;

        if (rank == pivotOwner) {
            const size_t localPivotRow = pivotNode - firstLocalRow;
            gatherPivotRow<<<gatherGrid, gatherBlockSize>>>(
                deviceDistance, devicePivot, localRows, localPivotRow, numNodes);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(hostPivot, devicePivot, numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(hostPivot, static_cast<int>(numNodes), MPI_UNSIGNED, pivotOwner,
                  MPI_COMM_WORLD);
        // A synchronous boundary both protects the pinned MPI receive buffer
        // from the following broadcast and preserves the k-to-k dependency.
        // The next MPI_Bcast can still overlap the preceding GPU relaxation.
        CUDA_CHECK(cudaMemcpy(devicePivot, hostPivot, numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        relaxPivot<<<updateGrid, block>>>(deviceDistance, devicePath, devicePivot,
                                          localRows, numNodes, pivotNode);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const double localElapsed = MPI_Wtime() - startTime;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    const bool needCompleteResult = validate || printResults;
    if (needCompleteResult) {
        CUDA_CHECK(cudaMemcpy(localDistance.data(), deviceDistance,
                              localElements * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));

        std::vector<unsigned int> gatheredDistance;
        if (rank == 0) {
            gatheredDistance.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDistance.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                    rank == 0 ? gatheredDistance.data() : nullptr, elementCounts.data(),
                    elementOffsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            completeDistance.resize(numNodes * numNodes);
#pragma omp parallel for collapse(2) schedule(static)
            for (int process = 0; process < worldSize; ++process) {
                for (size_t destination = 0; destination < numNodes; ++destination) {
                    const size_t input = static_cast<size_t>(elementOffsets[process]) +
                                         destination * rowCounts[process];
                    const size_t output = destination * numNodes + rowStarts[process];
                    std::memcpy(&completeDistance[output], &gatheredDistance[input],
                                rowCounts[process] * sizeof(unsigned int));
                }
            }
        }
    }

    int returnCode = 0;
    if (rank == 0) {
        const long elapsedMilliseconds = static_cast<long>(std::llround(elapsed * 1000.0));
        std::printf("Computation time: %ld ms\n", elapsedMilliseconds);
        const double operations = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsed > 0.0 ? operations / elapsed / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(completeDistance, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(completeDistance, numNodes)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                returnCode = 1;
            }
        }
    }

    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFreeHost(hostPivot));
    CUDA_CHECK(cudaFree(devicePivot));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(deviceDistance));
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return returnCode;
}
