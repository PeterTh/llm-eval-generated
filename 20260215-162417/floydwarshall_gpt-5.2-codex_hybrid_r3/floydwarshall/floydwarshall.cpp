#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t startRow, const size_t localRows) {
#pragma omp parallel for schedule(static)
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const unsigned int row = static_cast<unsigned int>(startRow + localRow);
        unsigned int* rowPtr = path.data() + localRow * numNodes;
        for (size_t i = 0; i < numNodes; ++i) {
            rowPtr[i] = row;
        }
    }
}

__global__ void floydWarshallKernel(unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ path,
                                    const unsigned int* __restrict__ rowK,
                                    const size_t n,
                                    const size_t localRows,
                                    const unsigned int k) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < localRows && col < n) {
        const size_t kIndex = static_cast<size_t>(k);
        const size_t idx = row * n + col;
        const unsigned int distIJ = dist[idx];
        const unsigned int distIK = dist[row * n + kIndex];
        const unsigned int distKJ = rowK[col];
        const unsigned int newDist = distIK + distKJ;

        if (newDist < distIJ) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
}

inline void checkCuda(const cudaError_t result, const char* msg) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

inline int ownerForRow(const size_t k, const size_t base, const size_t rem) {
    if (base == 0) {
        return static_cast<int>(k);
    }
    const size_t cutoff = (base + 1) * rem;
    if (k < cutoff) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(rem + (k - cutoff) / base);
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

    int rank = 0;
    int worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = -1;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitCode = 0;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitCode != -1) {
        MPI_Finalize();
        return exitCode;
    }

    unsigned long long numNodes64 = static_cast<unsigned long long>(numNodes);
    int validateInt = validate ? 1 : 0;
    int printInt = printResults ? 1 : 0;
    MPI_Bcast(&numNodes64, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(numNodes64);
    validate = validateInt != 0;
    printResults = printInt != 0;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const size_t baseRows = numNodes / static_cast<size_t>(worldSize);
    const size_t remainder = numNodes % static_cast<size_t>(worldSize);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t startRow = baseRows * static_cast<size_t>(rank) + std::min(static_cast<size_t>(rank), remainder);

    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);
    std::vector<unsigned int> distFull;
    std::vector<int> counts;
    std::vector<int> displs;

    if (rank == 0) {
        distFull.resize(numNodes * numNodes);
        initializeDistanceMatrix(distFull, numNodes, 1, MAX_DISTANCE);

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

    MPI_Scatterv(rank == 0 ? distFull.data() : nullptr,
                 rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_UNSIGNED,
                 localRows ? localDist.data() : nullptr,
                 static_cast<int>(localRows * numNodes),
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    initializePathMatrix(localPath, numNodes, startRow, localRows);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_rowK = nullptr;

    if (localRows > 0) {
        int deviceCount = 0;
        checkCuda(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
        if (deviceCount == 0) {
            if (rank == 0) {
                fprintf(stderr, "No CUDA devices available.\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

        const size_t distBytes = localRows * numNodes * sizeof(unsigned int);
        checkCuda(cudaMalloc(&d_dist, distBytes), "cudaMalloc dist");
        checkCuda(cudaMalloc(&d_path, distBytes), "cudaMalloc path");
        checkCuda(cudaMalloc(&d_rowK, numNodes * sizeof(unsigned int)), "cudaMalloc rowK");
        checkCuda(cudaMemcpy(d_dist, localDist.data(), distBytes, cudaMemcpyHostToDevice), "cudaMemcpy dist");
        checkCuda(cudaMemcpy(d_path, localPath.data(), distBytes, cudaMemcpyHostToDevice), "cudaMemcpy path");
    }

    std::vector<unsigned int> rowK(numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerForRow(k, baseRows, remainder);
        if (owner == rank && localRows > 0) {
            const size_t localK = k - startRow;
            checkCuda(cudaMemcpy(rowK.data(),
                                 d_dist + localK * numNodes,
                                 numNodes * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost),
                      "cudaMemcpy rowK D2H");
        }

        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (localRows > 0) {
            checkCuda(cudaMemcpy(d_rowK, rowK.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice),
                      "cudaMemcpy rowK H2D");
            const dim3 block(16, 16);
            const dim3 grid((numNodes + block.x - 1) / block.x,
                            (localRows + block.y - 1) / block.y);
            floydWarshallKernel<<<grid, block>>>(d_dist, d_path, d_rowK, numNodes, localRows,
                                                 static_cast<unsigned int>(k));
            checkCuda(cudaGetLastError(), "kernel launch");
        }
    }

    if (localRows > 0) {
        checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / maxTime / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (localRows > 0) {
        const size_t distBytes = localRows * numNodes * sizeof(unsigned int);
        checkCuda(cudaMemcpy(localDist.data(), d_dist, distBytes, cudaMemcpyDeviceToHost), "cudaMemcpy dist D2H");
        checkCuda(cudaMemcpy(localPath.data(), d_path, distBytes, cudaMemcpyDeviceToHost), "cudaMemcpy path D2H");
        checkCuda(cudaFree(d_rowK), "cudaFree rowK");
        checkCuda(cudaFree(d_path), "cudaFree path");
        checkCuda(cudaFree(d_dist), "cudaFree dist");
    }

    int resultCode = 0;
    if (printResults || validate) {
        if (rank == 0 && distFull.size() != numNodes * numNodes) {
            distFull.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localRows ? localDist.data() : nullptr,
                    static_cast<int>(localRows * numNodes),
                    MPI_UNSIGNED,
                    rank == 0 ? distFull.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displs.data() : nullptr,
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        print_results_int(distFull, "DistanceMatrix");
    }

    if (validate) {
        int validInt = 1;
        if (rank == 0) {
            printf("Validating result...\n");
            const bool valid = validateResult(distFull, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
            validInt = valid ? 1 : 0;
        }
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        resultCode = validInt ? 0 : 1;
    }

    MPI_Finalize();
    return resultCode;
}
