#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

constexpr int BLOCK_I = 16;
constexpr int BLOCK_J = 16;

// Index calculation for flattened 2D array (column-major)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

struct RowPartition {
    size_t start = 0;
    size_t count = 0;
};

inline RowPartition getRowPartition(const size_t rank, const size_t size, const size_t n) {
    const size_t base = n / size;
    const size_t rem = n % size;
    const size_t count = base + (rank < rem ? 1 : 0);
    const size_t start = rank * base + std::min(rank, rem);
    return {start, count};
}

inline size_t ownerOfRow(const size_t k, const size_t size, const size_t n) {
    const size_t base = n / size;
    const size_t rem = n % size;
    if (base == 0) {
        return k;
    }
    const size_t cutoff = (base + 1) * rem;
    if (k < cutoff) {
        return k / (base + 1);
    }
    return rem + (k - cutoff) / base;
}

inline void checkCuda(const cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void fw_update(unsigned int* dist,
                          unsigned int* path,
                          const unsigned int* row_k,
                          const size_t n,
                          const size_t localRows,
                          const size_t k) {
    __shared__ unsigned int sharedRowK[BLOCK_J];
    __shared__ unsigned int sharedIK[BLOCK_I];

    const size_t j = blockIdx.x * BLOCK_J + threadIdx.x;
    const size_t i = blockIdx.y * BLOCK_I + threadIdx.y;

    if (threadIdx.y == 0 && j < n) {
        sharedRowK[threadIdx.x] = row_k[j];
    }
    if (threadIdx.x == 0 && i < localRows) {
        sharedIK[threadIdx.y] = dist[k * localRows + i];
    }
    __syncthreads();

    if (i < localRows && j < n) {
        const unsigned int distIJ = dist[j * localRows + i];
        const unsigned int distIK = sharedIK[threadIdx.y];
        const unsigned int distKJ = sharedRowK[threadIdx.x];
        const unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            dist[j * localRows + i] = newDist;
            path[j * localRows + i] = static_cast<unsigned int>(k);
        }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
        }
    }
}

void packLocalRows(const std::vector<unsigned int>& global,
                   std::vector<unsigned int>& local,
                   const size_t numNodes,
                   const size_t rowStart,
                   const size_t localRows) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < localRows; ++i) {
            local[j * localRows + i] = global[idx2(rowStart + i, j, numNodes)];
        }
    }
}

void unpackLocalRows(const std::vector<unsigned int>& local,
                     std::vector<unsigned int>& global,
                     const size_t numNodes,
                     const size_t rowStart,
                     const size_t localRows) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < localRows; ++i) {
            global[idx2(rowStart + i, j, numNodes)] = local[j * localRows + i];
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                showHelp = true;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                parseError = true;
                showHelp = true;
            }
        }
    }

    uint64_t numNodes64 = static_cast<uint64_t>(numNodes);
    MPI_Bcast(&numNodes64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(numNodes64);
    int validateFlag = validate ? 1 : 0;
    int printResultsFlag = printResults ? 1 : 0;
    int showHelpFlag = showHelp ? 1 : 0;
    int parseErrorFlag = parseError ? 1 : 0;
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelpFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&parseErrorFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validateFlag != 0;
    printResults = printResultsFlag != 0;
    showHelp = showHelpFlag != 0;
    parseError = parseErrorFlag != 0;

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (numNodes == 0 || numNodes64 > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Invalid number of nodes: %zu\n", numNodes);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", size);
        printf("OpenMP max threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount == 0) {
        if (rank == 0) {
            printf("No CUDA devices found\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    const int device = rank % deviceCount;
    checkCuda(cudaSetDevice(device), "cudaSetDevice");

    const RowPartition part = getRowPartition(static_cast<size_t>(rank),
                                             static_cast<size_t>(size),
                                             numNodes);
    const size_t localRows = part.count;
    const size_t rowStart = part.start;
    const size_t localCount = localRows * numNodes;
    if (localCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            printf("Problem size too large for MPI counts\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<unsigned int> localDist(localCount);
    std::vector<unsigned int> localPath(localCount);

    if (rank == 0) {
        std::vector<unsigned int> globalDist(numNodes * numNodes);
        std::vector<unsigned int> globalPath(numNodes * numNodes);

        printf("Initializing graph...\n");
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(globalPath, numNodes);

        for (int r = 0; r < size; ++r) {
            const RowPartition rPart = getRowPartition(static_cast<size_t>(r),
                                                       static_cast<size_t>(size),
                                                       numNodes);
            if (rPart.count == 0) {
                continue;
            }
            const size_t rCount = rPart.count * numNodes;
            if (rCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
                printf("Problem size too large for MPI counts\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            std::vector<unsigned int> sendDist(rCount);
            std::vector<unsigned int> sendPath(rCount);
            packLocalRows(globalDist, sendDist, numNodes, rPart.start, rPart.count);
            packLocalRows(globalPath, sendPath, numNodes, rPart.start, rPart.count);
            if (r == 0) {
                localDist.swap(sendDist);
                localPath.swap(sendPath);
            } else {
                MPI_Send(sendDist.data(),
                         static_cast<int>(sendDist.size()),
                         MPI_UNSIGNED,
                         r,
                         0,
                         MPI_COMM_WORLD);
                MPI_Send(sendPath.data(),
                         static_cast<int>(sendPath.size()),
                         MPI_UNSIGNED,
                         r,
                         1,
                         MPI_COMM_WORLD);
            }
        }
    } else if (localRows > 0) {
        MPI_Recv(localDist.data(),
                 static_cast<int>(localDist.size()),
                 MPI_UNSIGNED,
                 0,
                 0,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        MPI_Recv(localPath.data(),
                 static_cast<int>(localPath.size()),
                 MPI_UNSIGNED,
                 0,
                 1,
                 MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }

    unsigned int* dist_d = nullptr;
    unsigned int* path_d = nullptr;
    unsigned int* row_k_d = nullptr;

    if (localRows > 0) {
        checkCuda(cudaMalloc(&dist_d, localDist.size() * sizeof(unsigned int)), "cudaMalloc dist");
        checkCuda(cudaMalloc(&path_d, localPath.size() * sizeof(unsigned int)), "cudaMalloc path");
        checkCuda(cudaMalloc(&row_k_d, numNodes * sizeof(unsigned int)), "cudaMalloc row_k");
        checkCuda(cudaMemcpy(dist_d,
                             localDist.data(),
                             localDist.size() * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy dist");
        checkCuda(cudaMemcpy(path_d,
                             localPath.data(),
                             localPath.size() * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy path");
    } else {
        checkCuda(cudaMalloc(&row_k_d, numNodes * sizeof(unsigned int)), "cudaMalloc row_k");
    }

    std::vector<unsigned int> row_k(numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();

    for (size_t k = 0; k < numNodes; ++k) {
        const size_t owner = ownerOfRow(k, static_cast<size_t>(size), numNodes);
        if (static_cast<size_t>(rank) == owner) {
            const size_t iLocal = k - rowStart;
            checkCuda(cudaMemcpy2D(row_k.data(),
                                   sizeof(unsigned int),
                                   dist_d + iLocal,
                                   localRows * sizeof(unsigned int),
                                   sizeof(unsigned int),
                                   numNodes,
                                   cudaMemcpyDeviceToHost),
                      "cudaMemcpy2D row_k");
        }

        MPI_Bcast(row_k.data(),
                  static_cast<int>(numNodes),
                  MPI_UNSIGNED,
                  static_cast<int>(owner),
                  MPI_COMM_WORLD);

        checkCuda(cudaMemcpy(row_k_d,
                             row_k.data(),
                             numNodes * sizeof(unsigned int),
                             cudaMemcpyHostToDevice),
                  "cudaMemcpy row_k");

        if (localRows > 0) {
            const dim3 block(BLOCK_J, BLOCK_I);
            const dim3 grid((numNodes + BLOCK_J - 1) / BLOCK_J,
                            (localRows + BLOCK_I - 1) / BLOCK_I);
            fw_update<<<grid, block>>>(dist_d, path_d, row_k_d, numNodes, localRows, k);
            checkCuda(cudaGetLastError(), "fw_update launch");
        }
    }

    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);

    const double localMs =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start).count();
    double maxMs = 0.0;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (localRows > 0) {
        checkCuda(cudaMemcpy(localDist.data(),
                             dist_d,
                             localDist.size() * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost),
                  "cudaMemcpy dist back");
    }

    if (dist_d) {
        cudaFree(dist_d);
    }
    if (path_d) {
        cudaFree(path_d);
    }
    if (row_k_d) {
        cudaFree(row_k_d);
    }

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxMs);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> globalDist;
    if (printResults || validate) {
        if (rank == 0) {
            globalDist.assign(numNodes * numNodes, 0);
            unpackLocalRows(localDist, globalDist, numNodes, rowStart, localRows);
            for (int r = 1; r < size; ++r) {
                const RowPartition rPart = getRowPartition(static_cast<size_t>(r),
                                                           static_cast<size_t>(size),
                                                           numNodes);
                if (rPart.count == 0) {
                    continue;
                }
                const size_t rCount = rPart.count * numNodes;
                if (rCount > static_cast<size_t>(std::numeric_limits<int>::max())) {
                    printf("Problem size too large for MPI counts\n");
                    MPI_Abort(MPI_COMM_WORLD, 1);
                }
                std::vector<unsigned int> recvBuf(rCount);
                MPI_Recv(recvBuf.data(),
                         static_cast<int>(recvBuf.size()),
                         MPI_UNSIGNED,
                         r,
                         2,
                         MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                unpackLocalRows(recvBuf, globalDist, numNodes, rPart.start, rPart.count);
            }
        } else if (localRows > 0) {
            MPI_Send(localDist.data(),
                     static_cast<int>(localDist.size()),
                     MPI_UNSIGNED,
                     0,
                     2,
                     MPI_COMM_WORLD);
        }
    }

    if (rank == 0 && printResults) {
        print_results_int(globalDist, "DistanceMatrix");
    }

    if (rank == 0 && validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(globalDist, numNodes);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
        MPI_Finalize();
        return valid ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
