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

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", rank_val, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

static int rank_val = 0;

// CUDA kernel: each thread handles one (i_local, j) pair
// local_dist and local_path are row-major: [i_local * n + j]
// dist_k_row is the k-th row of the full distance matrix (row-major, n elements)
__global__ void fw_kernel(unsigned int* __restrict__ local_dist,
                          unsigned int* __restrict__ local_path,
                          const unsigned int* __restrict__ dist_k_row,
                          const int local_rows, const int n, const int k_val) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_local = blockIdx.y * blockDim.y + threadIdx.y;

    if (i_local < local_rows && j < n) {
        const unsigned int dist_ik = local_dist[i_local * n + k_val];
        const unsigned int dist_kj = dist_k_row[j];
        const unsigned int dist_ij = local_dist[i_local * n + j];
        const unsigned int newDist = dist_ik + dist_kj;

        if (newDist < dist_ij) {
            local_dist[i_local * n + j] = newDist;
            local_path[i_local * n + j] = k_val;
        }
    }
}

// Compute row distribution for 1D block-row decomposition
void computeRowDistribution(int numNodes, int numProcs, int rank,
                            int& rowStart, int& localRows) {
    int base = numNodes / numProcs;
    int rem = numNodes % numProcs;
    if (rank < rem) {
        rowStart = rank * (base + 1);
        localRows = base + 1;
    } else {
        rowStart = rem * (base + 1) + (rank - rem) * base;
        localRows = base;
    }
}

// Find which MPI rank owns row k
int rowOwner(int k, int numProcs, int numNodes) {
    int base = numNodes / numProcs;
    int rem = numNodes % numProcs;
    int boundary = rem * (base + 1);
    if (base == 0) {
        // More processes than rows; only first 'rem' processes have 1 row each
        if (k < rem) return k;
        return numProcs - 1; // shouldn't happen if k < numNodes
    }
    if (k < boundary) {
        return k / (base + 1);
    } else {
        return rem + (k - boundary) / base;
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
        dist[i * numNodes + i] = 0;
    }
}

void initializePathMatrixLocal(unsigned int* path, int rowStart, int localRows, int numNodes) {
    // Original: path[r*n+c] = r for all r,c (row-major)
    // Local portion: path_local[i_local*n + j] = rowStart + i_local
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < localRows; ++i) {
        for (int j = 0; j < numNodes; ++j) {
            path[i * numNodes + j] = static_cast<unsigned int>(rowStart + i);
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero (val=%u)\n",
                   i, i, dist[i * numNodes + i]);
            return false;
        }
    }

    // 2. Triangle inequality check (sample)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[i * numNodes + j];
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];

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
    // Initialize MPI with thread support for OpenMP
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int numProcs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rank_val = rank;

    // Set GPU device based on rank (one GPU per rank, round-robin)
    int numGPUs = 0;
    cudaGetDeviceCount(&numGPUs);
    if (numGPUs == 0) {
        fprintf(stderr, "[Rank %d] No CUDA devices found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int gpuId = rank % numGPUs;
    CUDA_CHECK(cudaSetDevice(gpuId));

    // Parse arguments on rank 0, broadcast to all
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 0);
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Broadcast parameters
    int params[3];
    if (rank == 0) {
        params[0] = static_cast<int>(numNodes);
        params[1] = validate ? 1 : 0;
        params[2] = printResults ? 1 : 0;
    }
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(params[0]);
    validate = params[1] != 0;
    printResults = params[2] != 0;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d, GPUs: %d, OpenMP threads: %d\n",
               numProcs, numGPUs, omp_get_max_threads());
    }

    // Compute row distribution
    int rowStart, localRows;
    computeRowDistribution(static_cast<int>(numNodes), numProcs, rank, rowStart, localRows);

    // Initialize distance matrix on rank 0
    std::vector<unsigned int> dist_full;
    if (rank == 0) {
        dist_full.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
    }

    // Compute scatter counts and displacements
    std::vector<int> scatterCounts(numProcs);
    std::vector<int> scatterDispls(numProcs);
    for (int p = 0; p < numProcs; ++p) {
        int rs, lr;
        computeRowDistribution(static_cast<int>(numNodes), numProcs, p, rs, lr);
        scatterCounts[p] = lr * static_cast<int>(numNodes);
        scatterDispls[p] = rs * static_cast<int>(numNodes);
    }

    // Allocate local host buffers
    size_t localSize = static_cast<size_t>(localRows) * numNodes;
    std::vector<unsigned int> local_dist_host(localSize);
    std::vector<unsigned int> local_path_host(localSize);

    // Scatter distance matrix
    MPI_Scatterv(rank == 0 ? dist_full.data() : nullptr,
                 scatterCounts.data(), scatterDispls.data(), MPI_UNSIGNED,
                 local_dist_host.data(), static_cast<int>(localSize), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Initialize local path matrix (OpenMP parallel)
    if (localRows > 0) {
        initializePathMatrixLocal(local_path_host.data(), rowStart, localRows,
                                  static_cast<int>(numNodes));
    }

    // Allocate GPU memory
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_krow = nullptr;

    if (localRows > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, localSize * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localSize * sizeof(unsigned int)));
    }
    CUDA_CHECK(cudaMalloc(&d_krow, numNodes * sizeof(unsigned int)));

    // Copy local data to GPU
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(d_dist, local_dist_host.data(),
                              localSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, local_path_host.data(),
                              localSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    // Pinned host memory for row k broadcast
    unsigned int* h_krow = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_krow, numNodes * sizeof(unsigned int)));

    // Free host path matrix to save memory (no longer needed on host)
    { std::vector<unsigned int>().swap(local_path_host); }

    // ===== Floyd-Warshall computation =====
    printf("[Rank %d] Computing shortest paths (rows %d-%d of %zu)...\n",
           rank, rowStart, rowStart + localRows - 1, numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // CUDA kernel launch configuration
    dim3 blockSize(32, 8);
    dim3 gridSize(1, 1);
    if (localRows > 0) {
        gridSize = dim3(
            (static_cast<int>(numNodes) + blockSize.x - 1) / blockSize.x,
            (localRows + blockSize.y - 1) / blockSize.y
        );
    }

    for (size_t k = 0; k < numNodes; ++k) {
        int owner = rowOwner(static_cast<int>(k), numProcs, static_cast<int>(numNodes));

        // Owner extracts row k from GPU
        if (rank == owner && localRows > 0) {
            int local_k = static_cast<int>(k) - rowStart;
            CUDA_CHECK(cudaMemcpy(h_krow, d_dist + local_k * numNodes,
                                  numNodes * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        // Broadcast row k to all processes
        MPI_Bcast(h_krow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (localRows > 0) {
            // Copy row k to GPU
            CUDA_CHECK(cudaMemcpy(d_krow, h_krow, numNodes * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

            // Launch CUDA kernel
            fw_kernel<<<gridSize, blockSize>>>(d_dist, d_path, d_krow,
                                               localRows, static_cast<int>(numNodes),
                                               static_cast<int>(k));
        }
    }

    // Synchronize GPU
    if (localRows > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Report timing
    long long maxDuration;
    long long localDuration = duration.count();
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", maxDuration);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather results to rank 0
    // Copy local dist from GPU to host
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(local_dist_host.data(), d_dist,
                              localSize * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    if (rank == 0) {
        dist_full.resize(numNodes * numNodes);
    }

    MPI_Gatherv(local_dist_host.data(), static_cast<int>(localSize), MPI_UNSIGNED,
                rank == 0 ? dist_full.data() : nullptr,
                scatterCounts.data(), scatterDispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Print results and validate on rank 0
    if (rank == 0) {
        if (printResults) {
            print_results_int(dist_full, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist_full, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFreeHost(h_krow));
    if (localRows > 0) {
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_path));
    }
    CUDA_CHECK(cudaFree(d_krow));

    MPI_Finalize();
    return 0;
}
