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

// Row-major index calculation for flattened 2D array (i = row, j = col)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return i * n + j;
}

// CUDA error check
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, -1); \
    } \
} while(0)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Parallelize initialization with OpenMP
    #pragma omp parallel
    {
        unsigned int local_seed = seed ^ (unsigned int)omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            dist[i] = rangeMin + (unsigned int)(range * rand_r(&local_seed) / (double)RAND_MAX);
        }
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Initialize path matrix in parallel
    #pragma omp parallel for collapse(2)
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            path[idx2(i, j, numNodes)] = j;
        }
        path[idx2(i, i, numNodes)] = i;
    }
}

// CUDA kernel: update local rows (row-major layout)
extern "C" __global__ void fw_update_kernel(unsigned int* local_dist, unsigned int* local_path,
                                             const unsigned int* rowK, const unsigned int n,
                                             const unsigned int k_global, const unsigned int local_n) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x; // column
    unsigned int i_local = blockIdx.y * blockDim.y + threadIdx.y; // local row index
    if (j >= n || i_local >= local_n) return;

    unsigned int idx = i_local * n + j;
    unsigned int distIJ = local_dist[idx];
    unsigned int distIK = local_dist[i_local * n + k_global];
    unsigned int distKJ = rowK[j];

    unsigned int newDist = distIK + distKJ;
    if (distIK < INF && distKJ < INF && newDist < distIJ) {
        local_dist[idx] = newDist;
        local_path[idx] = k_global;
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
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];
                
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
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only on rank 0 then broadcast)
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
                MPI_Finalize();
                return 0;
            }
        }
    }
    // Broadcast options
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Determine row distribution
    size_t base = numNodes / (size_t)world;
    size_t rem = numNodes % (size_t)world;
    size_t rowStart = rank * base + std::min((size_t)rank, rem);
    size_t local_n = base + (rank < (int)rem ? 1 : 0);

    // Prepare counts and displacements for scatter/gather (counts in number of elements)
    std::vector<int> sendcounts(world);
    std::vector<int> displs(world);
    for (int r = 0; r < world; ++r) {
        size_t rs = r * base + std::min((size_t)r, rem);
        size_t rn = base + (r < (int)rem ? 1 : 0);
        sendcounts[r] = static_cast<int>(rn * numNodes);
        displs[r] = static_cast<int>(rs * numNodes);
    }

    // Buffers: each rank holds only its local rows in local_dist/local_path; root will initialize full matrices
    std::vector<unsigned int> local_dist(local_n * numNodes);
    std::vector<unsigned int> local_path(local_n * numNodes);
    std::vector<unsigned int> full_dist; // used on root for init and for gathering result
    std::vector<unsigned int> full_path;

    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);
    }

    // Scatter rows to all ranks
    MPI_Scatterv(rank == 0 ? full_dist.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), static_cast<int>(local_dist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full_path.data() : nullptr, sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), static_cast<int>(local_path.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Prepare GPU buffers
    unsigned int *d_local_dist = nullptr;
    unsigned int *d_local_path = nullptr;
    unsigned int *d_rowK = nullptr;
    size_t local_bytes = local_n * numNodes * sizeof(unsigned int);
    size_t row_bytes = numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaSetDevice(0)); // assume single GPU per node/rank; adjust if multi-GPU environment
    CUDA_CHECK(cudaMalloc((void**)&d_local_dist, local_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_local_path, local_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_rowK, row_bytes));

    // Copy initial local data to device
    CUDA_CHECK(cudaMemcpy(d_local_dist, local_dist.data(), local_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_local_path, local_path.data(), local_bytes, cudaMemcpyHostToDevice));

    // Buffer for broadcasting row k
    std::vector<unsigned int> rowK(numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    double tstart = MPI_Wtime();

    // Main k loop: broadcast row k, update local rows on GPU, then allgather updated rows
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine owner rank for row k
        int owner = 0;
        // find r such that displs[r]/numNodes <= k < (displs[r]+sendcounts[r])/numNodes
        for (int r = 0; r < world; ++r) {
            int start_r = displs[r] / static_cast<int>(numNodes);
            int end_r = start_r + sendcounts[r] / static_cast<int>(numNodes);
            if (k >= (size_t)start_r && k < (size_t)end_r) {
                owner = r; break;
            }
        }

        // Owner fills rowK from its local_dist
        if (rank == owner) {
            size_t local_idx = k - rowStart; // local row index
            for (size_t j = 0; j < numNodes; ++j) rowK[j] = local_dist[local_idx * numNodes + j];
        }

        // Broadcast rowK from owner to all ranks
        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Copy rowK to device
        CUDA_CHECK(cudaMemcpy(d_rowK, rowK.data(), row_bytes, cudaMemcpyHostToDevice));

        // Launch kernel to update local rows
        dim3 block(32, 8);
        dim3 grid((numNodes + block.x - 1) / block.x, (local_n + block.y - 1) / block.y);
        fw_update_kernel<<<grid, block>>>(d_local_dist, d_local_path, d_rowK, static_cast<unsigned int>(numNodes), static_cast<unsigned int>(k), static_cast<unsigned int>(local_n));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy updated local rows back to host to participate in Allgatherv
        CUDA_CHECK(cudaMemcpy(local_dist.data(), d_local_dist, local_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(local_path.data(), d_local_path, local_bytes, cudaMemcpyDeviceToHost));

        // Allgather updated rows so next iteration has full matrix available
        if (rank == 0) {
            // reuse full_dist/full_path buffers
            MPI_Allgatherv(MPI_IN_PLACE, static_cast<int>(local_dist.size()), MPI_UNSIGNED,
                           full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
            MPI_Allgatherv(MPI_IN_PLACE, static_cast<int>(local_path.size()), MPI_UNSIGNED,
                           full_path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        } else {
            // non-root participants send local data and receive full matrix
            // use temporary global buffers
            if (full_dist.size() == 0) { full_dist.resize(numNodes * numNodes); full_path.resize(numNodes * numNodes); }
            MPI_Allgatherv(local_dist.data(), static_cast<int>(local_dist.size()), MPI_UNSIGNED,
                           full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
            MPI_Allgatherv(local_path.data(), static_cast<int>(local_path.size()), MPI_UNSIGNED,
                           full_path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        }

        // Extract local slice from full_dist/full_path for next iteration and copy to device
        for (size_t i = 0; i < local_n; ++i) {
            size_t global_row = rowStart + i;
            memcpy(&local_dist[i * numNodes], &full_dist[global_row * numNodes], numNodes * sizeof(unsigned int));
            memcpy(&local_path[i * numNodes], &full_path[global_row * numNodes], numNodes * sizeof(unsigned int));
        }
        CUDA_CHECK(cudaMemcpy(d_local_dist, local_dist.data(), local_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_local_path, local_path.data(), local_bytes, cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double tend = MPI_Wtime();
    double duration_s = tend - tstart;

    // Gather final result to root
    if (rank == 0) {
        // full_dist already holds final matrix on root due to Allgatherv
        printf("Computation time: %.3f ms\n", duration_s * 1000.0);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / duration_s / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) print_results_int(full_dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            if (valid) { printf("Validation: PASSED\n"); }
            else { printf("Validation: FAILED\n"); }
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_local_dist));
    CUDA_CHECK(cudaFree(d_local_path));
    CUDA_CHECK(cudaFree(d_rowK));

    MPI_Finalize();
    return 0;
}
