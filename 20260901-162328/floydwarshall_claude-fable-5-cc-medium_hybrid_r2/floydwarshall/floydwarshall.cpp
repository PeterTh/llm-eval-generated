#include <algorithm>
#include <chrono>
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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // rand_r produces a sequential stream, so this loop stays serial to keep
    // the generated graph bit-identical to the original benchmark.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Every write in the original loop stores the row index of the target
    // element, so path[i][j] = i; this form is equivalent and race-free.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(j, i, numNodes)] = i;
        }
    }
}

// One Floyd-Warshall relaxation step over this rank's block of rows.
// dist/path hold localRows x n row-major blocks (global rows
// [rowOffset, rowOffset+localRows)), krow is the full pivot row k.
__global__ void fwRelaxKernel(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const unsigned int* __restrict__ krow,
                              const size_t localRows, const size_t n,
                              const unsigned int k) {
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;
    const unsigned int dkj = __ldg(&krow[j]);

    for (size_t li = blockIdx.y; li < localRows; li += gridDim.y) {
        const unsigned int dik = __ldg(&dist[li * n + k]);
        const unsigned int newDist = dik + dkj;
        const size_t ij = li * n + j;
        if (newDist < dist[ij]) {
            dist[ij] = newDist;
            path[ij] = k;
        }
    }
}

// Distributed Floyd-Warshall: rows are block-distributed over MPI ranks and
// each rank relaxes its block on the GPU. For every pivot k the owning rank
// downloads row k and broadcasts it to all ranks.
void floydWarshall(unsigned int* dDist, unsigned int* dPath,
                   const size_t localRows, const size_t rowOffset,
                   const size_t numNodes,
                   const std::vector<int>& rowStarts,
                   unsigned int* dKrow, unsigned int* hKrow) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int threads = 256;
    dim3 block(threads);
    dim3 grid((unsigned int)((numNodes + threads - 1) / threads),
              (unsigned int)std::min(localRows, (size_t)65535));

    int owner = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        // Advance owner to the rank holding global row k
        while (owner + 1 < size && (size_t)rowStarts[owner + 1] <= k) ++owner;

        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(hKrow, dDist + (k - rowStarts[owner]) * numNodes,
                                  numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hKrow, (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dKrow, hKrow, numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));

        if (localRows > 0) {
            fwRelaxKernel<<<grid, block>>>(dDist, dPath, dKrow, localRows, numNodes,
                                           (unsigned int)k);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    (void)rowOffset;
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
    bool valid = true;
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        #pragma omp critical
                        {
                            printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                   i, j, k);
                            valid = false;
                        }
                    }
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            if (rank == 0) printUsage(argv[0]);
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

    // Bind each rank to a GPU (round-robin over local devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                        MPI_INFO_NULL, &localComm);
    int localRank = 0;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA devices/node: %d\n",
               size, omp_get_max_threads(), deviceCount);
    }

    // Block distribution of rows over ranks
    std::vector<int> rowStarts(size + 1);
    for (int r = 0; r <= size; ++r) {
        rowStarts[r] = (int)((numNodes * (size_t)r) / (size_t)size);
    }
    const size_t rowOffset = rowStarts[rank];
    const size_t localRows = rowStarts[rank + 1] - rowStarts[rank];

    // Allocate full matrices on rank 0 (for init/output); every rank
    // initializes the full graph identically and keeps only its row block.
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Upload this rank's row block to the GPU
    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dKrow = nullptr;
    unsigned int* hKrow = nullptr;
    const size_t localBytes = std::max<size_t>(localRows, 1) * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dDist, localBytes));
    CUDA_CHECK(cudaMalloc(&dPath, localBytes));
    CUDA_CHECK(cudaMalloc(&dKrow, numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hKrow, numNodes * sizeof(unsigned int)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(dDist, dist.data() + rowOffset * numNodes,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data() + rowOffset * numNodes,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dDist, dPath, localRows, rowOffset, numNodes, rowStarts, dKrow, hKrow);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Download local blocks and gather the full matrices on rank 0
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data() + rowOffset * numNodes, dDist,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data() + rowOffset * numNodes, dPath,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = (int)((size_t)(rowStarts[r + 1] - rowStarts[r]) * numNodes);
        displs[r] = (int)((size_t)rowStarts[r] * numNodes);
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : dist.data() + rowOffset * numNodes,
                counts[rank], MPI_UNSIGNED, dist.data(), counts.data(),
                displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : path.data() + rowOffset * numNodes,
                counts[rank], MPI_UNSIGNED, path.data(), counts.data(),
                displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dKrow));
    CUDA_CHECK(cudaFreeHost(hKrow));

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
