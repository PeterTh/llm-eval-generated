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

    // rand_r carries sequential state, so this loop must stay serial to keep
    // the exact same graph as the original benchmark.
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
    // Equivalent to the original double write pattern: path[idx2(i, j)] ends up
    // holding j for every entry (including the diagonal).
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
        }
    }
}

// One Floyd-Warshall relaxation step for intermediate node k, applied to this
// rank's block of rows. Row indices are logical rows i (dist[i*n + j]); rowK
// holds the full pivot row dist[k][*].
__global__ void fwRelaxKernel(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const unsigned int* __restrict__ rowK,
                              const size_t n, const unsigned int k) {
    const size_t li = blockIdx.y;                              // local row
    const size_t j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;  // column
    if (j >= n) return;

    const unsigned int distIK = dist[li * n + k];
    const unsigned int newDist = distIK + rowK[j];
    if (newDist < dist[li * n + j]) {
        dist[li * n + j] = newDist;
        path[li * n + j] = k;
    }
}

// Hybrid MPI + CUDA Floyd-Warshall: rows are block-distributed over the MPI
// ranks, each rank keeps its row block resident on its GPU, and for every k
// the owner of row k broadcasts that pivot row to all ranks before the CUDA
// relaxation kernel updates the local block.
void floydWarshall(unsigned int* d_dist, unsigned int* d_path,
                   unsigned int* h_rowK,  // pinned staging buffer, numNodes entries
                   const size_t numNodes, const size_t rowStart, const size_t localRows,
                   const int* rowOwner) {
    const int blockSize = 256;
    const dim3 grid((unsigned int)((numNodes + blockSize - 1) / blockSize),
                    (unsigned int)(localRows > 0 ? localRows : 1));
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    unsigned int* d_rowK = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rowK, numNodes * sizeof(unsigned int)));

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner[k];

        // The owner stages the current pivot row from its GPU block.
        if (rank == owner) {
            CUDA_CHECK(cudaMemcpy(h_rowK, d_dist + (k - rowStart) * numNodes,
                                  numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(h_rowK, (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (localRows > 0) {
            CUDA_CHECK(cudaMemcpy(d_rowK, h_rowK, numNodes * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            fwRelaxKernel<<<grid, blockSize>>>(d_dist, d_path, d_rowK, numNodes,
                                               (unsigned int)k);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_rowK));
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
        printf("Hybrid MPI+OpenMP+CUDA: %d MPI ranks, %d OpenMP threads/rank\n",
               numRanks, omp_get_max_threads());
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank to a GPU (round-robin over the node-local devices).
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_free(&nodeComm);
    CUDA_CHECK(cudaSetDevice(nodeRank % deviceCount));

    // Block distribution of rows over ranks.
    const size_t rowsBase = numNodes / (size_t)numRanks;
    const size_t rowsRem = numNodes % (size_t)numRanks;
    const size_t localRows = rowsBase + ((size_t)rank < rowsRem ? 1 : 0);
    const size_t rowStart = (size_t)rank * rowsBase + std::min((size_t)rank, rowsRem);

    std::vector<int> rowOwner(numNodes);
    std::vector<int> gatherCounts(numRanks), gatherDispls(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t rRows = rowsBase + ((size_t)r < rowsRem ? 1 : 0);
        const size_t rStart = (size_t)r * rowsBase + std::min((size_t)r, rowsRem);
        gatherCounts[r] = (int)(rRows * numNodes);
        gatherDispls[r] = (int)(rStart * numNodes);
        for (size_t i = rStart; i < rStart + rRows; ++i) rowOwner[i] = r;
    }

    // Initialize the full matrices on every rank (deterministic and cheap
    // relative to the O(n^3) solve), then upload the local row block.
    if (rank == 0) printf("Initializing graph...\n");
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* h_rowK = nullptr;
    const size_t localBytes = std::max<size_t>(localRows, 1) * numNodes * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&d_dist, localBytes));
    CUDA_CHECK(cudaMalloc(&d_path, localBytes));
    CUDA_CHECK(cudaMallocHost(&h_rowK, numNodes * sizeof(unsigned int)));
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_dist, dist.data() + rowStart * numNodes,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, path.data() + rowStart * numNodes,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(d_dist, d_path, h_rowK, numNodes, rowStart, localRows, rowOwner.data());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Bring the local blocks back and reassemble the full matrices on rank 0.
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data() + rowStart * numNodes, d_dist,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data() + rowStart * numNodes, d_path,
                              localRows * numNodes * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : dist.data() + rowStart * numNodes,
                gatherCounts[rank], MPI_UNSIGNED,
                dist.data(), gatherCounts.data(), gatherDispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : path.data() + rowStart * numNodes,
                gatherCounts[rank], MPI_UNSIGNED,
                path.data(), gatherCounts.data(), gatherDispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFreeHost(h_rowK));

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
