#include <algorithm>
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

// 32x32 tiles give the CUDA phases enough reuse while keeping all three phase
// working sets in shared memory.  Matrix data is source-row-major: d[i * n+j].
constexpr int TILE = 32;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void abortWithCudaError(const cudaError_t status, const char* operation) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        abortWithCudaError(status, operation);
    }
}

void checkKernel(const char* operation) {
    checkCuda(cudaGetLastError(), operation);
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Keep the original serial rand_r stream so that -r produces the same
    // deterministic graph as the baseline implementation.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) /
                                                       static_cast<double>(RAND_MAX));
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// This is the diagonal phase of blocked Floyd-Warshall.  A single CUDA block
// closes one pivot tile; the two explicit barriers preserve the k ordering.
__global__ void closeDiagonalTile(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const int n, const int localPivotRow,
                                  const int pivotStart, const int pivotWidth) {
    __shared__ unsigned int tileDist[TILE][TILE];
    __shared__ unsigned int tilePath[TILE][TILE];
    __shared__ unsigned int pivotColumn[TILE];
    __shared__ unsigned int pivotRow[TILE];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const bool valid = x < pivotWidth && y < pivotWidth;
    const size_t offset = static_cast<size_t>(localPivotRow + y) * n + pivotStart + x;

    if (valid) {
        tileDist[y][x] = dist[offset];
        tilePath[y][x] = path[offset];
    }
    __syncthreads();

    for (int kk = 0; kk < pivotWidth; ++kk) {
        if (x == 0 && y < pivotWidth) {
            pivotColumn[y] = tileDist[y][kk];
        }
        if (y == 0 && x < pivotWidth) {
            pivotRow[x] = tileDist[kk][x];
        }
        __syncthreads();

        if (valid) {
            const unsigned int candidate = pivotColumn[y] + pivotRow[x];
            if (candidate < tileDist[y][x]) {
                tileDist[y][x] = candidate;
                tilePath[y][x] = static_cast<unsigned int>(pivotStart + kk);
            }
        }
        __syncthreads();
    }

    if (valid) {
        dist[offset] = tileDist[y][x];
        path[offset] = tilePath[y][x];
    }
}

// Update the pivot block row.  Its input from the diagonal tile is already
// closed, while the pivot-row operand is staged per k to preserve dependencies.
__global__ void updatePivotRow(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const unsigned int* __restrict__ diagonal,
                               const int n, const int pivotStart,
                               const int pivotWidth, const int pivotTile,
                               const int localPivotRow) {
    if (static_cast<int>(blockIdx.x) == pivotTile) {
        return;
    }

    __shared__ unsigned int tileDist[TILE][TILE];
    __shared__ unsigned int tilePath[TILE][TILE];
    __shared__ unsigned int pivotRow[TILE];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int columnStart = static_cast<int>(blockIdx.x) * TILE;
    const int columnWidth = min(TILE, n - columnStart);
    const bool valid = x < columnWidth && y < pivotWidth;
    const size_t offset = static_cast<size_t>(localPivotRow + y) * n + columnStart + x;

    if (valid) {
        tileDist[y][x] = dist[offset];
        tilePath[y][x] = path[offset];
    }
    __syncthreads();

    for (int kk = 0; kk < pivotWidth; ++kk) {
        if (y == 0 && x < columnWidth) {
            pivotRow[x] = tileDist[kk][x];
        }
        __syncthreads();

        if (valid) {
            const unsigned int candidate = diagonal[y * pivotWidth + kk] + pivotRow[x];
            if (candidate < tileDist[y][x]) {
                tileDist[y][x] = candidate;
                tilePath[y][x] = static_cast<unsigned int>(pivotStart + kk);
            }
        }
        __syncthreads();
    }

    if (valid) {
        dist[offset] = tileDist[y][x];
        path[offset] = tilePath[y][x];
    }
}

// Update every locally owned tile in the pivot block column.  This runs on all
// ranks, including the owner (where the diagonal tile simply returns).
__global__ void updatePivotColumn(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ diagonal,
                                  const int n, const int localStart,
                                  const int localTiles, const int pivotStart,
                                  const int pivotWidth, const int pivotTile) {
    const int localTile = static_cast<int>(blockIdx.x);
    const int globalTile = localStart / TILE + localTile;
    if (localTile >= localTiles || globalTile == pivotTile) {
        return;
    }

    __shared__ unsigned int tileDist[TILE][TILE];
    __shared__ unsigned int tilePath[TILE][TILE];
    __shared__ unsigned int pivotColumn[TILE];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int localRow = localTile * TILE + y;
    const int rowWidth = min(TILE, n - (localStart + localTile * TILE));
    const bool valid = x < pivotWidth && y < rowWidth;
    const size_t offset = static_cast<size_t>(localRow) * n + pivotStart + x;

    if (valid) {
        tileDist[y][x] = dist[offset];
        tilePath[y][x] = path[offset];
    }
    __syncthreads();

    for (int kk = 0; kk < pivotWidth; ++kk) {
        if (x == 0 && y < rowWidth) {
            pivotColumn[y] = tileDist[y][kk];
        }
        __syncthreads();

        if (valid) {
            const unsigned int candidate = pivotColumn[y] + diagonal[kk * pivotWidth + x];
            if (candidate < tileDist[y][x]) {
                tileDist[y][x] = candidate;
                tilePath[y][x] = static_cast<unsigned int>(pivotStart + kk);
            }
        }
        __syncthreads();
    }

    if (valid) {
        dist[offset] = tileDist[y][x];
        path[offset] = tilePath[y][x];
    }
}

// Update the remaining tiles using the completed local pivot column and the
// MPI-broadcast completed pivot block row.  Each CUDA block owns one output
// tile, so its k loop has no cross-block dependency.
__global__ void updateRemainingTiles(unsigned int* __restrict__ dist,
                                     unsigned int* __restrict__ path,
                                     const unsigned int* __restrict__ pivotRows,
                                     const int n, const int localStart,
                                     const int localTiles, const int pivotStart,
                                     const int pivotWidth, const int pivotTile) {
    const int localTile = static_cast<int>(blockIdx.y);
    const int columnTile = static_cast<int>(blockIdx.x);
    const int globalTile = localStart / TILE + localTile;
    if (localTile >= localTiles || globalTile == pivotTile || columnTile == pivotTile) {
        return;
    }

    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int localRow = localTile * TILE + y;
    const int columnStart = columnTile * TILE;
    const int rowWidth = min(TILE, n - (localStart + localTile * TILE));
    const int columnWidth = min(TILE, n - columnStart);

    if (y < rowWidth && x < pivotWidth) {
        left[y][x] = dist[static_cast<size_t>(localRow) * n + pivotStart + x];
    }
    if (y < pivotWidth && x < columnWidth) {
        right[y][x] = pivotRows[static_cast<size_t>(y) * n + columnStart + x];
    }
    __syncthreads();

    if (y < rowWidth && x < columnWidth) {
        const size_t offset = static_cast<size_t>(localRow) * n + columnStart + x;
        unsigned int current = dist[offset];
        unsigned int currentPath = path[offset];
        for (int kk = 0; kk < pivotWidth; ++kk) {
            const unsigned int candidate = left[y][kk] + right[kk][x];
            if (candidate < current) {
                current = candidate;
                currentPath = static_cast<unsigned int>(pivotStart + kk);
            }
        }
        dist[offset] = current;
        path[offset] = currentPath;
    }
}

struct TileDistribution {
    int numTiles;
    int localStart;
    int localTiles;
    int localRows;
};

TileDistribution makeDistribution(const int n, const int rank, const int ranks) {
    const int numTiles = (n + TILE - 1) / TILE;
    const int base = numTiles / ranks;
    const int remainder = numTiles % ranks;
    const int localTiles = base + (rank < remainder ? 1 : 0);
    const int startTile = rank * base + std::min(rank, remainder);
    const int localStart = startTile * TILE;
    const int localRows = localTiles == 0 ? 0 : std::min(n - localStart, localTiles * TILE);
    return {numTiles, localStart, localTiles, localRows};
}

int ownerOfTile(const int tile, const int numTiles, const int ranks) {
    const int base = numTiles / ranks;
    const int remainder = numTiles % ranks;
    const int largerTiles = remainder * (base + 1);
    if (tile < largerTiles) {
        return tile / (base + 1);
    }
    return remainder + (tile - largerTiles) / base;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
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

int main(int argc, char** argv) {
    int mpiThreadLevel = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &mpiThreadLevel);

    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t parsedNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool argumentError = false;
    bool showHelp = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || value == 0 ||
                value > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                argumentError = true;
            } else {
                parsedNodes = static_cast<size_t>(value);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentError = true;
        }
    }

    // MPI_Scatterv/Gatherv use int element counts.  The root keeps the input
    // graph, so reject only sizes that cannot be represented by those calls.
    const size_t maxNodes = static_cast<size_t>(std::sqrt(static_cast<double>(INT_MAX)));
    if (parsedNodes > maxNodes) {
        argumentError = true;
    }

    if (showHelp || argumentError) {
        if (rank == 0) {
            if (argumentError) {
                std::printf("Invalid command line.\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentError ? 1 : 0;
    }

    const int n = static_cast<int>(parsedNodes);
    const TileDistribution distribution = makeDistribution(n, rank, ranks);
    const size_t localElements = static_cast<size_t>(distribution.localRows) * n;

    MPI_Comm localComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            std::fprintf(stderr, "No CUDA device is available. This benchmark requires CUDA.\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    checkCuda(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %d\n", n);
        std::printf("MPI ranks: %d, CUDA tile size: %d, OpenMP threads/rank: %d\n",
                    ranks, TILE, omp_get_max_threads());
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }

    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        globalDist.resize(static_cast<size_t>(n) * n);
        initializeDistanceMatrix(globalDist, n, 1, MAX_DISTANCE);
    }

    std::vector<int> counts(ranks);
    std::vector<int> displacements(ranks);
    for (int process = 0; process < ranks; ++process) {
        const TileDistribution processDistribution = makeDistribution(n, process, ranks);
        counts[process] = processDistribution.localRows * n;
        displacements[process] = processDistribution.localStart * n;
    }

    std::vector<unsigned int> hostDist(localElements);
    std::vector<unsigned int> hostPath(localElements);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(),
                 MPI_UNSIGNED, hostDist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Path rows are independent at initialization.  OpenMP is deliberately
    // used on every MPI rank while CUDA owns the compute-intensive phases.
#pragma omp parallel for schedule(static)
    for (int localRow = 0; localRow < distribution.localRows; ++localRow) {
        const unsigned int source = static_cast<unsigned int>(distribution.localStart + localRow);
        unsigned int* const pathRow = hostPath.data() + static_cast<size_t>(localRow) * n;
        for (int column = 0; column < n; ++column) {
            pathRow[column] = source;
        }
    }

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* deviceDiagonal = nullptr;
    unsigned int* devicePivotRows = nullptr;
    cudaStream_t stream = nullptr;
    const size_t localAllocation = std::max(localElements, static_cast<size_t>(1));
    const size_t pivotRowsCapacity = static_cast<size_t>(TILE) * n;

    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    checkCuda(cudaMalloc(&deviceDist, localAllocation * sizeof(unsigned int)), "cudaMalloc distance matrix");
    checkCuda(cudaMalloc(&devicePath, localAllocation * sizeof(unsigned int)), "cudaMalloc path matrix");
    checkCuda(cudaMalloc(&deviceDiagonal, static_cast<size_t>(TILE) * TILE * sizeof(unsigned int)),
              "cudaMalloc diagonal tile");
    checkCuda(cudaMalloc(&devicePivotRows, pivotRowsCapacity * sizeof(unsigned int)),
              "cudaMalloc pivot rows");

    if (localElements != 0) {
        checkCuda(cudaMemcpyAsync(deviceDist, hostDist.data(), localElements * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice, stream), "copy distance matrix to device");
        checkCuda(cudaMemcpyAsync(devicePath, hostPath.data(), localElements * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice, stream), "copy path matrix to device");
    }
    checkCuda(cudaStreamSynchronize(stream), "initial device upload");

    unsigned int* hostDiagonal = nullptr;
    unsigned int* hostPivotRows = nullptr;
    cudaEvent_t diagonalUploadComplete = nullptr;
    cudaEvent_t pivotRowsUploadComplete = nullptr;
    bool diagonalUploadPending = false;
    bool pivotRowsUploadPending = false;
    checkCuda(cudaMallocHost(&hostDiagonal, static_cast<size_t>(TILE) * TILE * sizeof(unsigned int)),
              "cudaMallocHost diagonal tile");
    checkCuda(cudaMallocHost(&hostPivotRows, pivotRowsCapacity * sizeof(unsigned int)),
              "cudaMallocHost pivot rows");
    checkCuda(cudaEventCreateWithFlags(&diagonalUploadComplete, cudaEventDisableTiming),
              "cudaEventCreate diagonal upload");
    checkCuda(cudaEventCreateWithFlags(&pivotRowsUploadComplete, cudaEventDisableTiming),
              "cudaEventCreate pivot-row upload");
    const dim3 tileThreads(TILE, TILE);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (int pivotTile = 0; pivotTile < distribution.numTiles; ++pivotTile) {
        // These pinned host buffers are reused by the following broadcasts.
        // Waiting only for their device upload permits the preceding phase-3
        // kernel to overlap MPI coordination for the next pivot tile.
        if (diagonalUploadPending) {
            checkCuda(cudaEventSynchronize(diagonalUploadComplete), "diagonal upload reuse");
        }
        const int pivotStart = pivotTile * TILE;
        const int pivotWidth = std::min(TILE, n - pivotStart);
        const int owner = ownerOfTile(pivotTile, distribution.numTiles, ranks);

        if (rank == owner) {
            const int localPivotRow = pivotStart - distribution.localStart;
            closeDiagonalTile<<<1, tileThreads, 0, stream>>>(deviceDist, devicePath, n,
                                                               localPivotRow, pivotStart, pivotWidth);
            checkKernel("diagonal tile kernel launch");
            checkCuda(cudaMemcpy2DAsync(hostDiagonal,
                                        static_cast<size_t>(pivotWidth) * sizeof(unsigned int),
                                        deviceDist + static_cast<size_t>(localPivotRow) * n + pivotStart,
                                        static_cast<size_t>(n) * sizeof(unsigned int),
                                        static_cast<size_t>(pivotWidth) * sizeof(unsigned int), pivotWidth,
                                        cudaMemcpyDeviceToHost, stream), "copy diagonal tile to host");
            checkCuda(cudaStreamSynchronize(stream), "diagonal tile completion");
        }

        MPI_Bcast(hostDiagonal, pivotWidth * pivotWidth, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        checkCuda(cudaMemcpyAsync(deviceDiagonal, hostDiagonal,
                                  static_cast<size_t>(pivotWidth) * pivotWidth * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice, stream), "copy diagonal tile to device");
        checkCuda(cudaEventRecord(diagonalUploadComplete, stream), "record diagonal upload");
        diagonalUploadPending = true;

        if (rank == owner) {
            const int localPivotRow = pivotStart - distribution.localStart;
            updatePivotRow<<<distribution.numTiles, tileThreads, 0, stream>>>(
                deviceDist, devicePath, deviceDiagonal, n, pivotStart, pivotWidth, pivotTile,
                localPivotRow);
            checkKernel("pivot row kernel launch");
        }
        if (distribution.localTiles != 0) {
            updatePivotColumn<<<distribution.localTiles, tileThreads, 0, stream>>>(
                deviceDist, devicePath, deviceDiagonal, n, distribution.localStart,
                distribution.localTiles, pivotStart, pivotWidth, pivotTile);
            checkKernel("pivot column kernel launch");
        }

        if (rank == owner) {
            const int localPivotRow = pivotStart - distribution.localStart;
            checkCuda(cudaMemcpyAsync(hostPivotRows,
                                      deviceDist + static_cast<size_t>(localPivotRow) * n,
                                      static_cast<size_t>(pivotWidth) * n * sizeof(unsigned int),
                                      cudaMemcpyDeviceToHost, stream), "copy pivot rows to host");
            checkCuda(cudaStreamSynchronize(stream), "pivot row completion");
        }

        if (pivotRowsUploadPending) {
            checkCuda(cudaEventSynchronize(pivotRowsUploadComplete), "pivot-row upload reuse");
        }
        MPI_Bcast(hostPivotRows, pivotWidth * n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        checkCuda(cudaMemcpyAsync(devicePivotRows, hostPivotRows,
                                  static_cast<size_t>(pivotWidth) * n * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice, stream), "copy pivot rows to device");
        checkCuda(cudaEventRecord(pivotRowsUploadComplete, stream), "record pivot-row upload");
        pivotRowsUploadPending = true;

        if (distribution.localTiles != 0) {
            const dim3 remainingGrid(distribution.numTiles, distribution.localTiles);
            updateRemainingTiles<<<remainingGrid, tileThreads, 0, stream>>>(
                deviceDist, devicePath, devicePivotRows, n, distribution.localStart,
                distribution.localTiles, pivotStart, pivotWidth, pivotTile);
            checkKernel("remaining tiles kernel launch");
        }
    }

    checkCuda(cudaStreamSynchronize(stream), "Floyd-Warshall computation");
    const double localDuration = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&localDuration, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localElements != 0) {
        checkCuda(cudaMemcpy(hostDist.data(), deviceDist, localElements * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "copy distance matrix from device");
    }
    MPI_Gatherv(hostDist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                rank == 0 ? globalDist.data() : nullptr, counts.data(), displacements.data(),
                MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(std::llround(duration * 1000.0));
        std::printf("Computation time: %lld ms\n", milliseconds);
        const double operations = static_cast<double>(n) * n * n;
        const double gops = duration > 0.0 ? operations / duration / 1.0e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(globalDist, n)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    checkCuda(cudaEventDestroy(pivotRowsUploadComplete), "cudaEventDestroy pivot-row upload");
    checkCuda(cudaEventDestroy(diagonalUploadComplete), "cudaEventDestroy diagonal upload");
    checkCuda(cudaFreeHost(hostPivotRows), "cudaFreeHost pivot rows");
    checkCuda(cudaFreeHost(hostDiagonal), "cudaFreeHost diagonal tile");
    checkCuda(cudaFree(devicePivotRows), "cudaFree pivot rows");
    checkCuda(cudaFree(deviceDiagonal), "cudaFree diagonal tile");
    checkCuda(cudaFree(devicePath), "cudaFree path matrix");
    checkCuda(cudaFree(deviceDist), "cudaFree distance matrix");
    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    MPI_Comm_free(&localComm);
    MPI_Finalize();
    return exitCode;
}
