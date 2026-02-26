#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

inline void checkCuda(const cudaError_t status, const char* context, MPI_Comm comm) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(status));
        MPI_Abort(comm, 1);
    }
}

inline int ownerOfRow(const size_t row, const size_t baseRows, const size_t remainder) {
    if (row < (baseRows + 1) * remainder) {
        return static_cast<int>(row / (baseRows + 1));
    }
    return static_cast<int>(remainder + (row - (baseRows + 1) * remainder) / baseRows);
}

__global__ void floydWarshallKernel(unsigned int* dist,
                                    unsigned int* path,
                                    const unsigned int* rowK,
                                    const size_t numNodes,
                                    const size_t localRows,
                                    const size_t k) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= localRows || j >= numNodes) {
        return;
    }

    const size_t idx = i * numNodes + j;
    const unsigned int distIJ = dist[idx];
    const unsigned int distIK = dist[i * numNodes + k];
    const unsigned int distKJ = rowK[j];
    const unsigned int newDist = distIK + distKJ;
    if (newDist < distIJ) {
        dist[idx] = newDist;
        path[idx] = static_cast<unsigned int>(k);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        unsigned int* row = path.data() + i * numNodes;
        const unsigned int value = static_cast<unsigned int>(i);
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = value;
        }
    }
}

void floydWarshall(unsigned int* distDevice,
                   unsigned int* pathDevice,
                   unsigned int* rowKDevice,
                   std::vector<unsigned int>& rowKHost,
                   const size_t numNodes,
                   const size_t localRows,
                   const size_t startRow,
                   const size_t baseRows,
                   const size_t remainder,
                   const int worldRank,
                   MPI_Comm comm) {
    const dim3 block(16, 16);
    const dim3 grid((numNodes + block.x - 1) / block.x, (localRows + block.y - 1) / block.y);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, baseRows, remainder);
        if (worldRank == owner) {
            const size_t localIndex = k - startRow;
            checkCuda(cudaMemcpy(rowKHost.data(),
                                 distDevice + localIndex * numNodes,
                                 numNodes * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost),
                      "rowK D2H", comm);
        }

        MPI_Bcast(rowKHost.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);
        checkCuda(cudaMemcpy(rowKDevice,
                             rowKHost.data(),
                             numNodes * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "rowK H2D", comm);

        if (localRows > 0) {
            floydWarshallKernel<<<grid, block>>>(distDevice, pathDevice, rowKDevice, numNodes, localRows, k);
            checkCuda(cudaGetLastError(), "kernel launch", comm);
            checkCuda(cudaDeviceSynchronize(), "kernel sync", comm);
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
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
    MPI_Init(&argc, &argv);

    int worldRank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int shouldExit = 0;
    int exitCode = 0;

    if (worldRank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                shouldExit = 1;
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    uint64_t numNodes64 = static_cast<uint64_t>(numNodes);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;

    MPI_Bcast(&numNodes64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    numNodes = static_cast<size_t>(numNodes64);
    validate = (validateInt != 0);
    printResults = (printResultsInt != 0);

    if (shouldExit) {
        MPI_Finalize();
        return exitCode;
    }

    if (worldRank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("MPI ranks: %d\n", worldSize);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t remainder = numNodes % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(worldRank) < remainder ? 1 : 0);
    const size_t startRow = baseRows * static_cast<size_t>(worldRank) +
                            std::min(static_cast<size_t>(worldRank), remainder);

    std::vector<unsigned int> distLocal(localRows * numNodes);
    std::vector<unsigned int> pathLocal(localRows * numNodes);
    std::vector<unsigned int> distFull;
    std::vector<unsigned int> pathFull;
    std::vector<int> counts;
    std::vector<int> displs;

    if (worldRank == 0) {
        distFull.resize(numNodes * numNodes);
        pathFull.resize(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(distFull, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(pathFull, numNodes);

        counts.resize(worldSize);
        displs.resize(worldSize);
        size_t offset = 0;
        for (int r = 0; r < worldSize; ++r) {
            const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
            counts[r] = static_cast<int>(rows * numNodes);
            displs[r] = static_cast<int>(offset);
            offset += rows * numNodes;
        }
    }

    MPI_Scatterv(worldRank == 0 ? distFull.data() : nullptr,
                 worldRank == 0 ? counts.data() : nullptr,
                 worldRank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 distLocal.data(),
                 static_cast<int>(distLocal.size()),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(worldRank == 0 ? pathFull.data() : nullptr,
                 worldRank == 0 ? counts.data() : nullptr,
                 worldRank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 pathLocal.data(),
                 static_cast<int>(pathLocal.size()),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "device count", MPI_COMM_WORLD);
    if (deviceCount <= 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int deviceId = worldRank % deviceCount;
    checkCuda(cudaSetDevice(deviceId), "set device", MPI_COMM_WORLD);

    const size_t localElems = distLocal.size();
    unsigned int* distDevice = nullptr;
    unsigned int* pathDevice = nullptr;
    unsigned int* rowKDevice = nullptr;

    if (localElems > 0) {
        checkCuda(cudaMalloc(&distDevice, localElems * sizeof(unsigned int)), "dist malloc", MPI_COMM_WORLD);
        checkCuda(cudaMalloc(&pathDevice, localElems * sizeof(unsigned int)), "path malloc", MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(distDevice,
                             distLocal.data(),
                             localElems * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "dist H2D", MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(pathDevice,
                             pathLocal.data(),
                             localElems * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "path H2D", MPI_COMM_WORLD);
    }

    checkCuda(cudaMalloc(&rowKDevice, numNodes * sizeof(unsigned int)), "rowK malloc", MPI_COMM_WORLD);
    std::vector<unsigned int> rowKHost(numNodes);

    if (worldRank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(distDevice,
                  pathDevice,
                  rowKDevice,
                  rowKHost,
                  numNodes,
                  localRows,
                  startRow,
                  baseRows,
                  remainder,
                  worldRank,
                  MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localElems > 0) {
        checkCuda(cudaMemcpy(distLocal.data(),
                             distDevice,
                             localElems * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost),
                  "dist D2H", MPI_COMM_WORLD);
    }

    if (distDevice) {
        cudaFree(distDevice);
    }
    if (pathDevice) {
        cudaFree(pathDevice);
    }
    if (rowKDevice) {
        cudaFree(rowKDevice);
    }

    MPI_Gatherv(distLocal.data(),
                static_cast<int>(distLocal.size()),
                MPI_UNSIGNED,
                worldRank == 0 ? distFull.data() : nullptr,
                worldRank == 0 ? counts.data() : nullptr,
                worldRank == 0 ? displs.data() : nullptr,
                MPI_UNSIGNED,
                0,
                MPI_COMM_WORLD);

    if (worldRank == 0) {
        const double ms = maxTime * 1000.0;
        printf("Computation time: %.3f ms\n", ms);

        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(distFull, "DistanceMatrix");
        }
    }

    int validationCode = 0;
    if (validate) {
        if (worldRank == 0) {
            printf("Validating result...\n");
            const bool valid = validateResult(distFull, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                validationCode = 0;
            } else {
                printf("Validation: FAILED\n");
                validationCode = 1;
            }
        }
        MPI_Bcast(&validationCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validationCode;
}
