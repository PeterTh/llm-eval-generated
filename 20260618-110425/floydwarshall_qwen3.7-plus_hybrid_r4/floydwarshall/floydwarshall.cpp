// Hybrid MPI + OpenMP + CUDA Floyd-Warshall All-Pairs Shortest Path
// MPI: 1D row decomposition across ranks
// CUDA: GPU kernel for O(n^2) update per k iteration with shared memory
// OpenMP: parallel initialization

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
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: for each k, update all (i,j) pairs owned by this rank.
// Uses shared memory for the broadcast row-k and local column-k values.
// Threads map: threadIdx.x -> j (coalesced), threadIdx.y -> local_i.
__global__ void fw_kernel(unsigned int* __restrict__ local_dist,
                          unsigned int* __restrict__ local_path,
                          const unsigned int* __restrict__ d_row_k,
                          int local_n, int n, int k) {
    extern __shared__ unsigned int smem[];
    unsigned int* s_row_k = smem;          // n elements
    unsigned int* s_col_k = smem + n;      // blockDim.y elements

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_i = blockIdx.y * blockDim.y + threadIdx.y;

    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int block_size = blockDim.x * blockDim.y;

    // Cooperatively load broadcast row k into shared memory
    for (int idx = tid; idx < n; idx += block_size) {
        s_row_k[idx] = d_row_k[idx];
    }

    // Load column-k value for this block's local_i rows
    if (threadIdx.x == 0 && local_i < local_n) {
        s_col_k[threadIdx.y] = local_dist[local_i * n + k];
    }

    __syncthreads();

    if (j >= n || local_i >= local_n) return;

    const unsigned int dik = s_col_k[threadIdx.y];
    const unsigned int dkj = s_row_k[j];
    const unsigned int newDist = dik + dkj;

    const int idx = local_i * n + j;
    const unsigned int oldDist = local_dist[idx];

    if (newDist < oldDist) {
        local_dist[idx] = newDist;
        local_path[idx] = k;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    for (size_t i = 0; i < numNodes; ++i) {
        dist[i * numNodes + i] = 0;
    }
}

// Initialize local portion of path matrix.
// After the original nested-loop init, path[i*n+j] == i for all i,j.
void initializePathMatrixLocal(std::vector<unsigned int>& path, const size_t numNodes,
                                size_t row_offset, size_t local_n) {
    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < local_n; ++li) {
        unsigned int val = static_cast<unsigned int>(row_offset + li);
        for (size_t j = 0; j < numNodes; ++j) {
            path[li * numNodes + j] = val;
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

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
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

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
        printf("MPI ranks: %d, OpenMP threads: %d\n", num_ranks, omp_get_max_threads());
    }

    const int n = static_cast<int>(numNodes);

    // Block distribution of rows across MPI ranks
    const int base_n = n / num_ranks;
    const int remainder = n % num_ranks;
    const int local_n = base_n + (rank < remainder ? 1 : 0);
    const int row_offset = rank * base_n + std::min(rank, remainder);

    // Initialize full distance matrix on rank 0 (deterministic rand_r)
    std::vector<unsigned int> dist;
    if (rank == 0) {
        dist.resize(static_cast<size_t>(n) * n);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }

    // Compute send counts and displacements (in elements)
    std::vector<int> sendcounts(num_ranks);
    std::vector<int> displs(num_ranks);
    for (int r = 0; r < num_ranks; r++) {
        int rn = base_n + (r < remainder ? 1 : 0);
        sendcounts[r] = rn * n;
        displs[r] = (r * base_n + std::min(r, remainder)) * n;
    }

    // Scatter distance matrix
    std::vector<unsigned int> local_dist(static_cast<size_t>(local_n) * n);
    MPI_Scatterv(dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), local_n * n, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    if (rank == 0) {
        dist.clear();
        dist.shrink_to_fit();
    }

    // Initialize local path matrix using OpenMP
    std::vector<unsigned int> local_path(static_cast<size_t>(local_n) * n);
    initializePathMatrixLocal(local_path, numNodes, row_offset, local_n);

    // --- CUDA setup ---
    unsigned int *d_dist = nullptr, *d_path = nullptr, *d_row_k = nullptr;
    const bool has_work = (local_n > 0);

    if (has_work) {
        int num_devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&num_devices));
        if (num_devices == 0) {
            fprintf(stderr, "Rank %d: No CUDA devices found\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(rank % num_devices));

        CUDA_CHECK(cudaMalloc(&d_dist, static_cast<size_t>(local_n) * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, static_cast<size_t>(local_n) * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_row_k, static_cast<size_t>(n) * sizeof(unsigned int)));

        CUDA_CHECK(cudaMemcpy(d_dist, local_dist.data(),
                              static_cast<size_t>(local_n) * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, local_path.data(),
                              static_cast<size_t>(local_n) * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    // Host buffer for MPI broadcast of row k
    std::vector<unsigned int> h_row_k(n);

    // Kernel launch configuration
    const dim3 block(32, 16);  // 512 threads/block
    const dim3 grid((n + 31) / 32, (local_n + 15) / 16);
    const size_t smem_size = (n + 16) * sizeof(unsigned int);

    // --- Floyd-Warshall computation ---
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int k = 0; k < n; k++) {
        // Determine which rank owns row k (O(1) computation)
        int owner;
        if (k < remainder * (base_n + 1)) {
            owner = k / (base_n + 1);
        } else {
            owner = remainder + (k - remainder * (base_n + 1)) / base_n;
        }

        // Owner copies row k from GPU to host
        if (rank == owner && has_work) {
            int local_k = k - row_offset;
            CUDA_CHECK(cudaMemcpy(h_row_k.data(),
                                  d_dist + static_cast<size_t>(local_k) * n,
                                  static_cast<size_t>(n) * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        // Broadcast row k to all ranks
        MPI_Bcast(h_row_k.data(), n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Upload row k to GPU and launch update kernel
        if (has_work) {
            CUDA_CHECK(cudaMemcpy(d_row_k, h_row_k.data(),
                                  static_cast<size_t>(n) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            fw_kernel<<<grid, block, smem_size>>>(d_dist, d_path, d_row_k, local_n, n, k);
        }
    }

    // Copy final distance matrix back to host
    if (has_work) {
        CUDA_CHECK(cudaMemcpy(local_dist.data(), d_dist,
                              static_cast<size_t>(local_n) * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start).count();
    long global_duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &global_duration_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather results to rank 0
    std::vector<unsigned int> full_dist;
    if (rank == 0) {
        full_dist.resize(static_cast<size_t>(n) * n);
    }

    MPI_Gatherv(local_dist.data(), local_n * n, MPI_UNSIGNED,
                full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Free device memory
    if (has_work) {
        cudaFree(d_dist);
        cudaFree(d_path);
        cudaFree(d_row_k);
    }

    // Output and validation on rank 0
    if (rank == 0) {
        printf("Computation time: %ld ms\n", global_duration_ms);

        double ops = static_cast<double>(n) * n * n;
        double gops = ops / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gops);

        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Finalize();
    return 0;
}
