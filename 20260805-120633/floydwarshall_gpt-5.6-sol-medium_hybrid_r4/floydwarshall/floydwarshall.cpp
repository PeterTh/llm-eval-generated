#include <algorithm>
#include <climits>
#include <cstdint>
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
constexpr int MPI_CHUNK_TAG = 731;

static_assert(sizeof(unsigned int) == sizeof(std::uint32_t),
              "This benchmark requires 32-bit unsigned int");

inline constexpr size_t idx2(const size_t i, const size_t j,
                             const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void abortAll(const char* message, int rank) {
    std::fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::abort();
}

void checkCuda(cudaError_t status, const char* operation, int rank) {
    if (status != cudaSuccess) {
        char message[512];
        std::snprintf(message, sizeof(message), "%s: %s", operation,
                      cudaGetErrorString(status));
        abortAll(message, rank);
    }
}

size_t rowsForRank(size_t n, int rank, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    return base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
}

size_t firstRowForRank(size_t n, int rank, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    return static_cast<size_t>(rank) * base +
           std::min(static_cast<size_t>(rank), remainder);
}

int ownerOfRow(size_t row, size_t n, int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    const size_t largeRows = (base + 1) * remainder;
    if (row < largeRows) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(remainder + (row - largeRows) / base);
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist,
                              const size_t numNodes,
                              const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(
                                 range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializeLocalPath(std::vector<unsigned int>& path, size_t firstRow,
                         size_t localRows, size_t n) {
    // The original path matrix initializes every entry in source row i to i.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i) {
        const unsigned int source = static_cast<unsigned int>(firstRow + i);
        for (size_t j = 0; j < n; ++j) {
            path[i * n + j] = source;
        }
    }
}

void sendLarge(const unsigned int* data, size_t count, int destination) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, MPI_UINT32_T, destination, MPI_CHUNK_TAG,
                 MPI_COMM_WORLD);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void receiveLarge(unsigned int* data, size_t count, int source) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, MPI_UINT32_T, source, MPI_CHUNK_TAG,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= static_cast<size_t>(chunk);
    }
}

void distributeRows(const std::vector<unsigned int>& global,
                    std::vector<unsigned int>& local, size_t n, int rank,
                    int ranks) {
    if (rank == 0) {
        std::copy_n(global.data(), local.size(), local.data());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t first = firstRowForRank(n, peer, ranks);
            const size_t count = rowsForRank(n, peer, ranks) * n;
            sendLarge(global.data() + first * n, count, peer);
        }
    } else {
        receiveLarge(local.data(), local.size(), 0);
    }
}

void gatherRows(const std::vector<unsigned int>& local,
                std::vector<unsigned int>& global, size_t n, int rank,
                int ranks) {
    if (rank == 0) {
        std::copy_n(local.data(), local.size(), global.data());
        for (int peer = 1; peer < ranks; ++peer) {
            const size_t first = firstRowForRank(n, peer, ranks);
            const size_t count = rowsForRank(n, peer, ranks) * n;
            receiveLarge(global.data() + first * n, count, peer);
        }
    } else {
        sendLarge(local.data(), local.size(), 0);
    }
}

template<int BX, int BY>
__global__ void floydWarshallStep(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivot,
                                  size_t n, size_t localRows,
                                  unsigned int k) {
    __shared__ unsigned int pivotTile[BX];
    __shared__ unsigned int dikTile[BY];

    for (size_t colBase = static_cast<size_t>(blockIdx.x) * BX;
         colBase < n;
         colBase += static_cast<size_t>(gridDim.x) * BX) {
        const size_t col = colBase + threadIdx.x;
        if (threadIdx.y == 0 && col < n) {
            pivotTile[threadIdx.x] = pivot[col];
        }
        __syncthreads();

        for (size_t rowBase = static_cast<size_t>(blockIdx.y) * BY;
             rowBase < localRows;
             rowBase += static_cast<size_t>(gridDim.y) * BY) {
            const size_t row = rowBase + threadIdx.y;
            if (threadIdx.x == 0 && row < localRows) {
                dikTile[threadIdx.y] = dist[row * n + k];
            }
            __syncthreads();

            if (row < localRows && col < n) {
                const size_t index = row * n + col;
                const unsigned int candidate =
                    dikTile[threadIdx.y] + pivotTile[threadIdx.x];
                if (candidate < dist[index]) {
                    dist[index] = candidate;
                    path[index] = k;
                }
            }
            __syncthreads();
        }
        __syncthreads();
    }
}

double floydWarshallHybrid(std::vector<unsigned int>& localDist,
                           std::vector<unsigned int>& localPath,
                           size_t n, size_t firstRow, size_t localRows,
                           int rank, int ranks, int device) {
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivot = nullptr;
    unsigned int* hostPivot = nullptr;
    const size_t localElements = localRows * n;
    const size_t allocationElements = std::max<size_t>(localElements, 1);

    checkCuda(cudaMalloc(&deviceDist, allocationElements * sizeof(unsigned int)),
              "cudaMalloc(distance block)", rank);
    checkCuda(cudaMalloc(&devicePath, allocationElements * sizeof(unsigned int)),
              "cudaMalloc(path block)", rank);
    checkCuda(cudaMalloc(&devicePivot, n * sizeof(unsigned int)),
              "cudaMalloc(pivot row)", rank);
    checkCuda(cudaMallocHost(&hostPivot, n * sizeof(unsigned int)),
              "cudaMallocHost(pivot row)", rank);
    if (localElements != 0) {
        checkCuda(cudaMemcpy(deviceDist, localDist.data(),
                             localElements * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "copy distances to GPU", rank);
        checkCuda(cudaMemcpy(devicePath, localPath.data(),
                             localElements * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "copy paths to GPU", rank);
    }

    cudaDeviceProp properties{};
    checkCuda(cudaGetDeviceProperties(&properties, device),
              "cudaGetDeviceProperties", rank);
    constexpr int BX = 32;
    constexpr int BY = 8;
    const unsigned int gridX = static_cast<unsigned int>(std::min<size_t>(
        (n + BX - 1) / BX, static_cast<size_t>(properties.maxGridSize[0])));
    const unsigned int gridY = static_cast<unsigned int>(std::max<size_t>(1,
        std::min<size_t>((localRows + BY - 1) / BY,
                         static_cast<size_t>(properties.maxGridSize[1]))));
    const dim3 block(BX, BY);
    const dim3 grid(gridX, gridY);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (size_t k = 0; k < n; ++k) {
        const int owner = ownerOfRow(k, n, ranks);
        if (rank == owner) {
            const size_t localK = k - firstRow;
            checkCuda(cudaMemcpy(hostPivot, deviceDist + localK * n,
                                 n * sizeof(unsigned int), cudaMemcpyDeviceToHost),
                      "copy pivot row from GPU", rank);
        }
        MPI_Bcast(hostPivot, static_cast<int>(n), MPI_UINT32_T, owner,
                  MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(devicePivot, hostPivot, n * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "copy pivot row to GPU", rank);
        if (localRows != 0) {
            floydWarshallStep<BX, BY><<<grid, block>>>(
                deviceDist, devicePath, devicePivot, n, localRows,
                static_cast<unsigned int>(k));
            checkCuda(cudaGetLastError(), "launch Floyd-Warshall CUDA kernel", rank);
        }
    }
    checkCuda(cudaDeviceSynchronize(), "finish Floyd-Warshall CUDA kernels", rank);
    MPI_Barrier(MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (localElements != 0) {
        checkCuda(cudaMemcpy(localDist.data(), deviceDist,
                             localElements * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "copy distances from GPU", rank);
        checkCuda(cudaMemcpy(localPath.data(), devicePath,
                             localElements * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost), "copy paths from GPU", rank);
    }
    cudaFreeHost(hostPivot);
    cudaFree(devicePivot);
    cudaFree(devicePath);
    cudaFree(deviceDist);
    return seconds;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(n, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int dij = dist[idx2(j, i, n)];
                const unsigned int dik = dist[idx2(k, i, n)];
                const unsigned int dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF && dik + dkj < dij) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        abortAll("MPI implementation does not provide MPI_THREAD_FUNNELED", rank);
    }

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool argumentsValid = true;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(INT_MAX)) {
                argumentsValid = false;
            } else {
                n = static_cast<size_t>(parsed);
            }
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            argumentsValid = false;
        }
    }
    if (n > std::numeric_limits<size_t>::max() / n ||
        n > std::numeric_limits<unsigned int>::max()) {
        argumentsValid = false;
    }
    if (showHelp || !argumentsValid) {
        if (rank == 0) {
            if (!argumentsValid) std::fprintf(stderr, "Invalid command-line arguments\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return argumentsValid ? 0 : 1;
    }

    MPI_Comm localCommunicator;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                        &localCommunicator);
    int localRank = 0;
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_free(&localCommunicator);
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (deviceCount == 0) abortAll("no CUDA-capable device is available", rank);
    const int device = localRank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice", rank);

    const size_t firstRow = firstRowForRank(n, rank, ranks);
    const size_t localRows = rowsForRank(n, rank, ranks);
    std::vector<unsigned int> globalDist;
    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", n);
        std::printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
                    ranks, omp_get_max_threads(), deviceCount);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
        globalDist.resize(n * n);
        initializeDistanceMatrix(globalDist, n, 1, MAX_DISTANCE);
    }

    std::vector<unsigned int> localDist(localRows * n);
    std::vector<unsigned int> localPath(localRows * n);
    distributeRows(globalDist, localDist, n, rank, ranks);
    initializeLocalPath(localPath, firstRow, localRows, n);
    globalDist.clear();
    globalDist.shrink_to_fit();

    if (rank == 0) std::printf("Computing shortest paths...\n");
    const double seconds = floydWarshallHybrid(localDist, localPath, n, firstRow,
                                                localRows, rank, ranks, device);

    if (validate || printResults) {
        if (rank == 0) globalDist.resize(n * n);
        gatherRows(localDist, globalDist, n, rank, ranks);
    }

    int returnCode = 0;
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(seconds * 1000.0);
        const double operations = static_cast<double>(n) * n * n;
        const double gops = seconds > 0.0 ? operations / seconds / 1.0e9 : 0.0;
        std::printf("Computation time: %lld ms\n", milliseconds);
        std::printf("Performance: %.3f GOPS\n", gops);
        if (printResults) print_results_int(globalDist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(globalDist, n);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            returnCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&returnCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return returnCode;
}
