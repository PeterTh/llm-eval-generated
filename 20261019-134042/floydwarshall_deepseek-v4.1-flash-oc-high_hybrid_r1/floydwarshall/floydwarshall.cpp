#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int* __restrict__ pivot,
                                    const size_t n,
                                    const size_t rows,
                                    const unsigned int k,
                                    const size_t rowsPerBlock,
                                    const int useShared) {
    extern __shared__ unsigned int sharedPivot[];

    const unsigned int* pivotRow = pivot;
    if (useShared) {
        for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
            sharedPivot[j] = pivot[j];
        }
        __syncthreads();
        pivotRow = sharedPivot;
    }

    size_t rowBegin = static_cast<size_t>(blockIdx.x) * rowsPerBlock;
    size_t rowEnd = rowBegin + rowsPerBlock;
    if (rowEnd > rows) {
        rowEnd = rows;
    }

    for (size_t r = rowBegin; r < rowEnd; ++r) {
        unsigned int* dRow = dist + r * n;
        unsigned int* pRow = path + r * n;
        const unsigned int dik = dRow[k];

        for (size_t j = threadIdx.x; j < n; j += blockDim.x) {
            const unsigned int newDist = dik + pivotRow[j];
            if (newDist < dRow[j]) {
                dRow[j] = newDist;
                pRow[j] = k;
            }
        }
    }
}

inline void generateLocalDistanceMatrix(std::vector<unsigned int>& dist,
                                        const size_t rowStart,
                                        const size_t localRows,
                                        const size_t numNodes,
                                        const unsigned int rangeMin,
                                        const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t skip = rowStart * numNodes;
    for (size_t i = 0; i < skip; ++i) {
        rand_r(&seed);
    }

    for (size_t r = 0; r < localRows; ++r) {
        const size_t globalRow = rowStart + r;
        unsigned int* row = dist.data() + r * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }
        row[globalRow] = 0;
    }
}

inline void initializeLocalPathMatrix(std::vector<unsigned int>& path,
                                      const size_t rowStart,
                                      const size_t localRows,
                                      const size_t numNodes) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < localRows; ++r) {
        const unsigned int globalRow = static_cast<unsigned int>(rowStart + r);
        unsigned int* row = path.data() + r * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = globalRow;
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    bool valid = true;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
#pragma omp critical
                        {
                            if (valid) {
                                printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                       i, j, k);
                                valid = false;
                            }
                        }
                    }
                }
            }
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
#pragma omp critical
            {
                if (valid) {
                    printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                    valid = false;
                }
            }
        }
    }

    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes == 0) {
        if (rank == 0) {
            printf("Number of nodes must be greater than zero\n");
        }
        MPI_Finalize();
        return 1;
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int ranksPerNode = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &ranksPerNode);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            printf("No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp deviceProp{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, device));

    int threadsPerRank = omp_get_num_procs() / ranksPerNode;
    if (threadsPerRank < 1) {
        threadsPerRank = 1;
    }
    threadsPerRank = std::min(threadsPerRank, omp_get_max_threads());
    omp_set_num_threads(threadsPerRank);

    const size_t rowStart = static_cast<size_t>((static_cast<unsigned long long>(rank) * numNodes) / numRanks);
    const size_t rowEnd = static_cast<size_t>((static_cast<unsigned long long>(rank + 1) * numNodes) / numRanks);
    const size_t localRows = rowEnd - rowStart;

    size_t cpuRows = 0;
    size_t gpuRows = 0;
    if (localRows > 0) {
        const size_t workPerRank = localRows * numNodes;
        size_t cpuPercent = 0;
        if (threadsPerRank >= 8 && workPerRank >= 1500000) {
            if (threadsPerRank >= 24) {
                cpuPercent = 25;
            } else if (threadsPerRank >= 16) {
                cpuPercent = 18;
            } else {
                cpuPercent = 10;
            }
        }
        cpuRows = (localRows * cpuPercent) / 100;
        if (cpuRows >= localRows) {
            cpuRows = localRows - 1;
        }
        gpuRows = localRows - cpuRows;
    }

    const size_t cpuThreadCount = std::max<size_t>(1, cpuRows);
    const size_t cpuThreads = std::min<size_t>(
        std::min<size_t>(static_cast<size_t>(threadsPerRank), cpuThreadCount),
        static_cast<size_t>(32));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);

    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    generateLocalDistanceMatrix(localDist, rowStart, localRows, numNodes, 1, MAX_DISTANCE);
    initializeLocalPathMatrix(localPath, rowStart, localRows, numNodes);

    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivot = nullptr;
    if (gpuRows > 0) {
        CUDA_CHECK(cudaMalloc(&deviceDist, gpuRows * numNodes * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&devicePath, gpuRows * numNodes * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&devicePivot, numNodes * sizeof(unsigned int)));

        CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data() + cpuRows * numNodes,
                              gpuRows * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, localPath.data() + cpuRows * numNodes,
                              gpuRows * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    const int threadsPerBlock = 256;
    size_t targetBlocks = static_cast<size_t>(deviceProp.multiProcessorCount) * 2;
    if (targetBlocks == 0) {
        targetBlocks = 1;
    }
    size_t rowsPerBlock = (gpuRows + targetBlocks - 1) / targetBlocks;
    if (rowsPerBlock < 1) {
        rowsPerBlock = 1;
    }
    if (rowsPerBlock > 16) {
        rowsPerBlock = 16;
    }
    const size_t gridSize = gpuRows > 0 ? (gpuRows + rowsPerBlock - 1) / rowsPerBlock : 0;

    const size_t sharedBytes = numNodes * sizeof(unsigned int);
    const bool useShared = sharedBytes <= 32 * 1024;
    const size_t dynamicShared = useShared ? sharedBytes : 0;

    std::vector<unsigned int> pivotRow(numNodes);

    if (gpuRows > 0) {
        floydWarshallKernel<<<1, 32, 0>>>(
            deviceDist, devicePath, devicePivot, numNodes, 0,
            0, 1, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    if (cpuRows > 0) {
#pragma omp parallel num_threads(static_cast<int>(cpuThreads))
        {
            for (size_t k = 0; k < numNodes; ++k) {
#pragma omp master
                {
                    const int owner = static_cast<int>(((static_cast<unsigned long long>(k) + 1) * numRanks - 1) / numNodes);

                    if (rank == owner) {
                        if (k >= rowStart && k < rowStart + cpuRows) {
                            const unsigned int* row = localDist.data() + (k - rowStart) * numNodes;
                            std::copy(row, row + numNodes, pivotRow.begin());
                        } else {
                            CUDA_CHECK(cudaMemcpy(pivotRow.data(),
                                                  deviceDist + (k - rowStart - cpuRows) * numNodes,
                                                  numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
                        }
                    }

                    MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

                    if (gpuRows > 0) {
                        CUDA_CHECK(cudaMemcpy(devicePivot, pivotRow.data(),
                                              numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
                        floydWarshallKernel<<<static_cast<unsigned int>(gridSize), threadsPerBlock, dynamicShared>>>(
                            deviceDist, devicePath, devicePivot, numNodes, gpuRows,
                            static_cast<unsigned int>(k), rowsPerBlock, useShared ? 1 : 0);
                    }
                }
#pragma omp barrier

#pragma omp for schedule(static)
                for (size_t r = 0; r < cpuRows; ++r) {
                    unsigned int* dRow = localDist.data() + r * numNodes;
                    unsigned int* pRow = localPath.data() + r * numNodes;
                    const unsigned int dik = dRow[k];

                    for (size_t j = 0; j < numNodes; ++j) {
                        const unsigned int newDist = dik + pivotRow[j];
                        if (newDist < dRow[j]) {
                            dRow[j] = newDist;
                            pRow[j] = static_cast<unsigned int>(k);
                        }
                    }
                }

#pragma omp master
                {
                    if (gpuRows > 0) {
                        CUDA_CHECK(cudaStreamSynchronize(0));
                    }
                }
#pragma omp barrier
            }
        }
    } else {
        for (size_t k = 0; k < numNodes; ++k) {
            const int owner = static_cast<int>(((static_cast<unsigned long long>(k) + 1) * numRanks - 1) / numNodes);

            if (rank == owner) {
                CUDA_CHECK(cudaMemcpy(pivotRow.data(),
                                      deviceDist + (k - rowStart - cpuRows) * numNodes,
                                      numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            }

            MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

            if (gpuRows > 0) {
                CUDA_CHECK(cudaMemcpy(devicePivot, pivotRow.data(),
                                      numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
                floydWarshallKernel<<<static_cast<unsigned int>(gridSize), threadsPerBlock, dynamicShared>>>(
                    deviceDist, devicePath, devicePivot, numNodes, gpuRows,
                    static_cast<unsigned int>(k), rowsPerBlock, useShared ? 1 : 0);
                CUDA_CHECK(cudaStreamSynchronize(0));
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double seconds = duration.count() / 1000.0;
        if (seconds <= 0.0) {
            seconds = 1e-9;
        }
        double gflops = ops / seconds / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        if (gpuRows > 0) {
            CUDA_CHECK(cudaMemcpy(localDist.data() + cpuRows * numNodes, deviceDist,
                                  gpuRows * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        const int sendCount = static_cast<int>(localRows * numNodes);
        std::vector<int> recvCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            recvCounts.resize(numRanks);
            displacements.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                const size_t rs = static_cast<size_t>((static_cast<unsigned long long>(r) * numNodes) / numRanks);
                const size_t re = static_cast<size_t>((static_cast<unsigned long long>(r + 1) * numNodes) / numRanks);
                recvCounts[r] = static_cast<int>((re - rs) * numNodes);
                displacements[r] = static_cast<int>(rs * numNodes);
            }
        }
        MPI_Gatherv(localDist.data(), sendCount, MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr,
                    rank == 0 ? recvCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    bool validationPassed = true;
    if (rank == 0) {
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            validationPassed = validateResult(fullDist, numNodes);
            if (validationPassed) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    if (gpuRows > 0) {
        CUDA_CHECK(cudaFree(devicePivot));
        CUDA_CHECK(cudaFree(devicePath));
        CUDA_CHECK(cudaFree(deviceDist));
    }

    int localResult = 0;
    if (rank == 0 && validate && !validationPassed) {
        localResult = 1;
    }

    MPI_Finalize();
    return localResult;
}
