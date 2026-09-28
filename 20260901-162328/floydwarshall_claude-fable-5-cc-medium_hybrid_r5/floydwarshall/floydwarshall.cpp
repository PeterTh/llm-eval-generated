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

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                    __FILE__, __LINE__);                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                           \
        }                                                                           \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Sequential: rand_r stream must match the original ordering exactly
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        // Distinct i values touch disjoint element pairs, so the inner loop is safe
        // to run in parallel (the i == j writes all store the same value j).
        #pragma omp parallel for
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// One thread per (local row, column) pair for the fixed pivot k.
// Row k and column k are invariant during iteration k, so reading the
// broadcast snapshot of row k gives results identical to the sequential code.
__global__ void fwKernel(unsigned int* __restrict__ dist,
                         unsigned int* __restrict__ path,
                         const unsigned int* __restrict__ rowK,
                         const size_t localRows, const size_t n, const unsigned int k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t li = blockIdx.y;
    if (j >= n || li >= localRows) return;

    const unsigned int distIK = dist[li * n + k];
    const unsigned int newDist = distIK + rowK[j];
    const size_t off = li * n + j;
    if (newDist < dist[off]) {
        dist[off] = newDist;
        path[off] = k;
    }
}

// Hybrid MPI + CUDA Floyd-Warshall: rows are block-distributed across ranks,
// each rank updates its row block on its GPU; the pivot row k is broadcast
// from its owner every iteration.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const int rank, const int nranks,
                   const std::vector<int>& counts, const std::vector<int>& displs) {
    const size_t n = numNodes;
    const size_t localRows = static_cast<size_t>(counts[rank]) / n;
    const size_t rowStart = static_cast<size_t>(displs[rank]) / n;

    // Scatter row blocks from rank 0
    std::vector<unsigned int> localDist(localRows * n);
    std::vector<unsigned int> localPath(localRows * n);
    MPI_Scatterv(dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), (int)(localRows * n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), (int)(localRows * n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dRowK = nullptr;
    unsigned int* hRowK = nullptr;
    const size_t blockBytes = localRows * n * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dDist, blockBytes));
    CUDA_CHECK(cudaMalloc(&dPath, blockBytes));
    CUDA_CHECK(cudaMalloc(&dRowK, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hRowK, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dDist, localDist.data(), blockBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, localPath.data(), blockBytes, cudaMemcpyHostToDevice));

    // Precompute which rank owns each pivot row
    std::vector<int> owner(n);
    for (int r = 0; r < nranks; ++r) {
        const size_t rs = static_cast<size_t>(displs[r]) / n;
        const size_t rc = static_cast<size_t>(counts[r]) / n;
        for (size_t i = rs; i < rs + rc; ++i) owner[i] = r;
    }

    const dim3 block(256);
    const dim3 grid((unsigned)((n + block.x - 1) / block.x), (unsigned)std::max<size_t>(localRows, 1));

    for (size_t k = 0; k < n; ++k) {
        // Owner extracts the current pivot row from its device block
        if (owner[k] == rank) {
            CUDA_CHECK(cudaMemcpy(hRowK, dDist + (k - rowStart) * n,
                                  n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (nranks > 1) {
            MPI_Bcast(hRowK, (int)n, MPI_UNSIGNED, owner[k], MPI_COMM_WORLD);
        }
        CUDA_CHECK(cudaMemcpy(dRowK, hRowK, n * sizeof(unsigned int), cudaMemcpyHostToDevice));

        if (localRows > 0) {
            fwKernel<<<grid, block>>>(dDist, dPath, dRowK, localRows, n, (unsigned int)k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, blockBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(localPath.data(), dPath, blockBytes, cudaMemcpyDeviceToHost));

    // Gather full result on rank 0
    MPI_Gatherv(localDist.data(), (int)(localRows * n), MPI_UNSIGNED,
                dist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), (int)(localRows * n), MPI_UNSIGNED,
                path.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dRowK));
    CUDA_CHECK(cudaFreeHost(hRowK));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    bool valid = true;

    // 1. Diagonal should be zero
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            valid = false;
        }
    }
    if (!valid) return false;

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    #pragma omp parallel for collapse(2)
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
                        valid = false;
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
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs visible: %d, OpenMP threads: %d\n",
               nranks, deviceCount, omp_get_max_threads());
    }

    // Block-distribute rows over ranks (element counts for Scatterv/Gatherv)
    std::vector<int> counts(nranks), displs(nranks);
    {
        const size_t base = numNodes / nranks;
        const size_t rem = numNodes % nranks;
        size_t row = 0;
        for (int r = 0; r < nranks; ++r) {
            const size_t rows = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            counts[r] = (int)(rows * numNodes);
            displs[r] = (int)(row * numNodes);
            row += rows;
        }
    }

    // Allocate full matrices on rank 0 only
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);

        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rank, nranks, counts, displs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

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

    MPI_Finalize();
    return exitCode;
}
