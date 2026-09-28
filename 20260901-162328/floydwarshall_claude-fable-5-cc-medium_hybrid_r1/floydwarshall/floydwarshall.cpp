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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // rand_r is an inherently sequential stream; keep it serial so the
    // generated graph is identical to the original code.
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
    // Same end state as the original doubly-written loop: path[idx2(j, i, n)] = i
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(j, i, numNodes)] = i;
        }
    }
}

// Relax all (i, j) pairs of this rank's row block against pivot row k.
// Within a fixed k, row k and column k are invariant (dist[k][k] == 0 and
// updates require a strict decrease), so this is exactly equivalent to the
// sequential triple loop.
__global__ void fwRelaxKernel(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const unsigned int* __restrict__ kRow,
                              const size_t n, const size_t k) {
    const size_t li = blockIdx.y;  // local row index within this rank's block
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const unsigned int distIK = dist[li * n + k];
    const unsigned int newDist = distIK + kRow[j];
    if (newDist < dist[li * n + j]) {
        dist[li * n + j] = newDist;
        path[li * n + j] = (unsigned int)k;
    }
}

// Hybrid MPI + CUDA Floyd-Warshall. Each rank owns a contiguous block of rows
// [rowStart, rowStart + localRows) resident on its GPU. For every pivot k the
// owning rank downloads the current row k, broadcasts it, and all ranks relax
// their block on the GPU.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t n = numNodes;

    // Block row distribution
    std::vector<size_t> rowsPerRank(size), rowOffset(size + 1, 0);
    for (int r = 0; r < size; ++r) {
        rowsPerRank[r] = n / size + ((size_t)r < n % size ? 1 : 0);
        rowOffset[r + 1] = rowOffset[r] + rowsPerRank[r];
    }
    const size_t rowStart = rowOffset[rank];
    const size_t localRows = rowsPerRank[rank];

    // Upload this rank's row block (dist and path) to the GPU
    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    unsigned int* dKRow = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, std::max(localRows * n, (size_t)1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max(localRows * n, (size_t)1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dKRow, std::max(n, (size_t)1) * sizeof(unsigned int)));

    unsigned int* hKRow = nullptr;  // pinned staging buffer for the pivot row
    CUDA_CHECK(cudaMallocHost(&hKRow, std::max(n, (size_t)1) * sizeof(unsigned int)));

    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(dDist, dist.data() + rowStart * n,
                              localRows * n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data() + rowStart * n,
                              localRows * n * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    const int blockSize = 256;
    const dim3 block(blockSize);
    const dim3 grid((unsigned int)((n + blockSize - 1) / blockSize),
                    (unsigned int)std::max(localRows, (size_t)1));

    // Owner of each pivot row, precomputed
    int owner = 0;
    for (size_t k = 0; k < n; ++k) {
        while ((size_t)owner + 1 <= (size_t)size && rowOffset[owner + 1] <= k) ++owner;

        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(hKRow, dDist + (k - rowStart) * n,
                                  n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(hKRow, (int)n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (localRows > 0) {
            CUDA_CHECK(cudaMemcpy(dKRow, hKRow, n * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            fwRelaxKernel<<<grid, block>>>(dDist, dPath, dKRow, n, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download local block and gather the full matrices on every rank
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data() + rowStart * n, dDist,
                              localRows * n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data() + rowStart * n, dPath,
                              localRows * n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = (int)(rowsPerRank[r] * n);
        displs[r] = (int)(rowOffset[r] * n);
    }
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, dist.data(),
                   counts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, path.data(),
                   counts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFreeHost(hKRow));
    CUDA_CHECK(cudaFree(dKRow));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dDist));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    bool valid = true;

    // 1. Diagonal should be zero
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
#pragma omp critical
            {
                printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                valid = false;
            }
        }
    }
    if (!valid) return false;

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
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
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Bind each rank to a GPU (round-robin over the node's devices)
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs per node: %d, OpenMP threads: %d\n",
               size, deviceCount, omp_get_max_threads());
    }

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize (every rank generates the identical deterministic graph)
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes);

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

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
