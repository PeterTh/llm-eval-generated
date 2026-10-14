#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Fills rows [rowBegin, rowEnd) of the distance matrix. The random stream is consumed in the
// same global order as the serial original, so every rank generates the identical graph.
void initializeDistanceRows(std::vector<unsigned int>& dist, const size_t numNodes,
                            const size_t rowBegin, const size_t rowEnd,
                            const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t row = 0; row < numNodes; ++row) {
        for (size_t col = 0; col < numNodes; ++col) {
            const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (row >= rowBegin && row < rowEnd) {
                dist[(row - rowBegin) * numNodes + col] = value;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        dist[(i - rowBegin) * numNodes + i] = 0;
    }
}

// Path entries start out as their row index (same values as the serial initializePathMatrix)
__global__ void initPathKernel(unsigned int* __restrict__ path, const size_t rows,
                               const size_t n, const size_t rowOffset) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < rows && j < n) {
        path[i * n + j] = static_cast<unsigned int>(rowOffset + i);
    }
}

// One pivot step k applied to a block of rows. Column k and pivot row k are unchanged during
// step k (diagonal is zero, weights are non-negative), so all updates within the step are independent.
__global__ void floydWarshallStepKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                        const unsigned int* __restrict__ pivotRow,
                                        const size_t rows, const size_t n, const size_t k) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= rows || j >= n) {
        return;
    }

    const unsigned int distIK = dist[i * n + k];
    const unsigned int newDist = distIK + pivotRow[j];

    if (newDist < dist[i * n + j]) {
        dist[i * n + j] = newDist;
        path[i * n + j] = static_cast<unsigned int>(k);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool root = rank == 0;

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // Each rank drives the GPUs of its node that are assigned to it
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::vector<int> devices;
    if (deviceCount >= localSize) {
        for (int d = localRank; d < deviceCount; d += localSize) {
            devices.push_back(d);
        }
    } else {
        devices.push_back(localRank % deviceCount);
    }
    const int numLocalWorkers = static_cast<int>(devices.size());

    // Every GPU is a worker that owns a contiguous block of matrix rows
    std::vector<int> workersPerRank(numRanks);
    MPI_Allgather(&numLocalWorkers, 1, MPI_INT, workersPerRank.data(), 1, MPI_INT, MPI_COMM_WORLD);

    std::vector<int> firstWorker(numRanks + 1, 0);
    for (int r = 0; r < numRanks; ++r) {
        firstWorker[r + 1] = firstWorker[r] + workersPerRank[r];
    }
    const int numWorkers = firstWorker[numRanks];

    std::vector<int> workerRank(numWorkers);
    for (int r = 0; r < numRanks; ++r) {
        for (int w = firstWorker[r]; w < firstWorker[r + 1]; ++w) {
            workerRank[w] = r;
        }
    }

    std::vector<size_t> rowStart(numWorkers + 1);
    for (int w = 0; w <= numWorkers; ++w) {
        rowStart[w] = static_cast<size_t>(w) * numNodes / numWorkers;
    }

    const size_t rowBegin = rowStart[firstWorker[rank]];
    const size_t rowEnd = rowStart[firstWorker[rank + 1]];
    const size_t localRows = rowEnd - rowBegin;

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Host-side rows of the distance matrix for this rank
    std::vector<unsigned int> dist(localRows * numNodes);
    initializeDistanceRows(dist, numNodes, rowBegin, rowEnd, 1, MAX_DISTANCE);

    if (root) {
        printf("Computing shortest paths...\n");
    }

    // Pivot row k is broadcast from its owner and staged in buffer (k % 2). Alternating buffers
    // let the owner publish row k+1 while slower workers may still be copying row k.
    std::vector<unsigned int> pivotStage(2 * numNodes);
    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    omp_set_dynamic(0);
    #pragma omp parallel num_threads(numLocalWorkers)
    {
        const int t = omp_get_thread_num();
        const int worker = firstWorker[rank] + t;
        const size_t workerRowBegin = rowStart[worker];
        const size_t workerRows = rowStart[worker + 1] - workerRowBegin;
        const size_t hostOffset = (workerRowBegin - rowBegin) * numNodes;
        const size_t cells = std::max<size_t>(workerRows * numNodes, 1);

        const dim3 block(32, 8);
        const dim3 grid((numNodes + block.x - 1) / block.x, (workerRows + block.y - 1) / block.y);

        CUDA_CHECK(cudaSetDevice(devices[t]));

        unsigned int* dDist = nullptr;
        unsigned int* dPath = nullptr;
        unsigned int* dPivot = nullptr;
        CUDA_CHECK(cudaMalloc(&dDist, cells * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dPath, cells * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dPivot, std::max<size_t>(numNodes, 1) * sizeof(unsigned int)));

        if (workerRows > 0) {
            CUDA_CHECK(cudaMemcpy(dDist, dist.data() + hostOffset, workerRows * numNodes * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            initPathKernel<<<grid, block>>>(dPath, workerRows, numNodes, workerRowBegin);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        #pragma omp barrier
        if (t == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            start = std::chrono::high_resolution_clock::now();
        }

        for (size_t k = 0; k < numNodes; ++k) {
            unsigned int* stage = pivotStage.data() + (k % 2) * numNodes;

            // Worker that owns row k publishes it to the host staging buffer
            const int owner = static_cast<int>(std::upper_bound(rowStart.begin(), rowStart.end(), k)
                                               - rowStart.begin()) - 1;
            if (worker == owner) {
                CUDA_CHECK(cudaMemcpy(stage, dDist + (k - workerRowBegin) * numNodes,
                                      numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            }

            #pragma omp barrier
            if (t == 0) {
                MPI_Bcast(stage, static_cast<int>(numNodes), MPI_UNSIGNED, workerRank[owner], MPI_COMM_WORLD);
            }
            #pragma omp barrier

            if (workerRows > 0) {
                CUDA_CHECK(cudaMemcpy(dPivot, stage, numNodes * sizeof(unsigned int),
                                      cudaMemcpyHostToDevice));
                floydWarshallStepKernel<<<grid, block>>>(dDist, dPath, dPivot, workerRows, numNodes, k);
                CUDA_CHECK(cudaGetLastError());
            }
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        #pragma omp barrier
        if (t == 0) {
            MPI_Barrier(MPI_COMM_WORLD);
            end = std::chrono::high_resolution_clock::now();
        }
        #pragma omp barrier

        if (workerRows > 0) {
            CUDA_CHECK(cudaMemcpy(dist.data() + hostOffset, dDist, workerRows * numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        CUDA_CHECK(cudaFree(dDist));
        CUDA_CHECK(cudaFree(dPath));
        CUDA_CHECK(cudaFree(dPivot));
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int exitCode = 0;
    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    if (printResults || validate) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);

        std::vector<int> counts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>(rowStart[firstWorker[r + 1]] - rowStart[firstWorker[r]]);
            displs[r] = static_cast<int>(rowStart[firstWorker[r]]);
        }

        std::vector<unsigned int> fullDist(root ? numNodes * numNodes : 0);
        MPI_Gatherv(dist.data(), static_cast<int>(localRows), rowType, fullDist.data(),
                    counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);

        // Print results for external validation (integer hash-based)
        if (root && printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        // Validation
        if (root && validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
