#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// CUDA block dimensions: x along rows (i), y along local columns (j_local)
#define FW_BLOCK_X 32
#define FW_BLOCK_Y 8

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Index calculation for flattened 2D array (column-major, same as original)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// CUDA kernel for Floyd-Warshall update
// Thread layout: threadIdx.x along rows (i), threadIdx.y along local columns (j_local)
// This ensures coalesced memory access for dist_local (column-major storage)
__global__ void fw_kernel(
    unsigned int* __restrict__ dist_local,
    unsigned int* __restrict__ path_local,
    const unsigned int* __restrict__ col_k_global,
    const size_t n,
    const size_t local_n_cols,
    const unsigned int k
) {
    extern __shared__ unsigned int s_col_k[];

    // Cooperatively load column k into shared memory
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int nthreads = blockDim.x * blockDim.y;
    for (int i = tid; i < (int)n; i += nthreads) {
        s_col_k[i] = col_k_global[i];
    }
    __syncthreads();

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j_local = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < n && j_local < local_n_cols) {
        size_t idx = j_local * n + i;
        unsigned int dist_ij = dist_local[idx];
        unsigned int dist_ik = s_col_k[i];
        // dist(k, j) is never modified during iteration k (since dist(k,k)=0),
        // so reading directly from dist_local is safe
        unsigned int dist_kj = __ldg(&dist_local[j_local * n + k]);
        unsigned int new_dist = dist_ik + dist_kj;
        if (new_dist < dist_ij) {
            dist_local[idx] = new_dist;
            path_local[idx] = k;
        }
    }
}

// Compute column-block distribution for a given rank
inline void compute_distribution(size_t n, int rank, int num_procs,
                                 size_t& start_col, size_t& local_n_cols) {
    size_t base_cols = n / num_procs;
    size_t extra = n % num_procs;
    if ((size_t)rank < extra) {
        local_n_cols = base_cols + 1;
        start_col = (size_t)rank * (base_cols + 1);
    } else {
        local_n_cols = base_cols;
        start_col = (size_t)rank * base_cols + extra;
    }
}

// Determine which MPI process owns column k
inline int get_owner(size_t k, size_t n, int num_procs) {
    size_t base_cols = n / num_procs;
    size_t extra = n % num_procs;
    if (extra == 0) {
        return (int)(k / base_cols);
    }
    size_t threshold = extra * (base_cols + 1);
    if (k < threshold) {
        return (int)(k / (base_cols + 1));
    } else {
        return (int)(extra + (k - threshold) / base_cols);
    }
}

// Initialize local portion of distance matrix with same RNG sequence as original
void initializeLocalDistanceMatrix(std::vector<unsigned int>& dist_local,
                                   const size_t numNodes,
                                   const size_t start_col,
                                   const size_t local_n_cols,
                                   const unsigned int rangeMin,
                                   const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Advance RNG to the start of our local columns (column-major linear order)
    size_t skip = start_col * numNodes;
    for (size_t i = 0; i < skip; ++i) {
        rand_r(&seed);
    }

    // Generate local columns
    size_t local_size = local_n_cols * numNodes;
    for (size_t i = 0; i < local_size; ++i) {
        dist_local[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero for local columns
    #pragma omp parallel for schedule(static)
    for (size_t j_local = 0; j_local < local_n_cols; ++j_local) {
        size_t j = start_col + j_local;
        dist_local[j_local * numNodes + j] = 0;
    }
}

// Initialize local portion of path matrix
// Original sets path(i,j) = j for all i,j
void initializeLocalPathMatrix(std::vector<unsigned int>& path_local,
                               const size_t numNodes,
                               const size_t start_col,
                               const size_t local_n_cols) {
    #pragma omp parallel for schedule(static)
    for (size_t j_local = 0; j_local < local_n_cols; ++j_local) {
        size_t j = start_col + j_local;
        size_t base = j_local * numNodes;
        for (size_t i = 0; i < numNodes; ++i) {
            path_local[base + i] = j;
        }
    }
}

// Hybrid MPI+OpenMP+CUDA Floyd-Warshall
void floydWarshall_hybrid(
    std::vector<unsigned int>& dist_local,
    std::vector<unsigned int>& path_local,
    const size_t numNodes,
    const size_t start_col,
    const size_t local_n_cols,
    const int rank,
    const int num_procs
) {
    if (local_n_cols == 0 || numNodes == 0) {
        // Still participate in MPI broadcasts
        std::vector<unsigned int> col_k_buf(numNodes);
        for (size_t k = 0; k < numNodes; ++k) {
            int owner_k = get_owner(k, numNodes, num_procs);
            MPI_Bcast(col_k_buf.data(), numNodes, MPI_UNSIGNED, owner_k, MPI_COMM_WORLD);
        }
        return;
    }

    // Select GPU (round-robin assignment)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus > 0) {
        CUDA_CHECK(cudaSetDevice(rank % num_gpus));
    }

    // Allocate GPU memory
    unsigned int *d_dist, *d_path, *d_col_k;
    size_t dist_bytes = local_n_cols * numNodes * sizeof(unsigned int);
    size_t col_bytes = numNodes * sizeof(unsigned int);

    CUDA_CHECK(cudaMalloc(&d_dist, dist_bytes));
    CUDA_CHECK(cudaMalloc(&d_path, dist_bytes));
    CUDA_CHECK(cudaMalloc(&d_col_k, col_bytes));

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_dist, dist_local.data(), dist_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path_local.data(), dist_bytes, cudaMemcpyHostToDevice));

    // Pinned host buffer for column k communication
    unsigned int* col_k_pinned;
    CUDA_CHECK(cudaMallocHost(&col_k_pinned, col_bytes));

    // Precompute grid dimensions
    dim3 block(FW_BLOCK_X, FW_BLOCK_Y);
    dim3 grid((unsigned int)((numNodes + FW_BLOCK_X - 1) / FW_BLOCK_X),
              (unsigned int)((local_n_cols + FW_BLOCK_Y - 1) / FW_BLOCK_Y));
    size_t smem_size = numNodes * sizeof(unsigned int);

    // Main Floyd-Warshall loop
    for (size_t k = 0; k < numNodes; ++k) {
        int owner_k = get_owner(k, numNodes, num_procs);

        // Owner extracts column k from GPU (contiguous in column-major storage)
        if (rank == owner_k) {
            size_t k_local = k - start_col;
            CUDA_CHECK(cudaMemcpy(col_k_pinned, d_dist + k_local * numNodes,
                                  col_bytes, cudaMemcpyDeviceToHost));
        }

        // Broadcast column k to all processes
        MPI_Bcast(col_k_pinned, (int)numNodes, MPI_UNSIGNED, owner_k, MPI_COMM_WORLD);

        // Copy column k to GPU
        CUDA_CHECK(cudaMemcpy(d_col_k, col_k_pinned, col_bytes, cudaMemcpyHostToDevice));

        // Launch FW kernel - dist(k,j) read directly from dist_local
        fw_kernel<<<grid, block, smem_size>>>(d_dist, d_path, d_col_k, numNodes, local_n_cols, (unsigned int)k);
    }

    // Copy results back to CPU
    CUDA_CHECK(cudaMemcpy(dist_local.data(), d_dist, dist_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path_local.data(), d_path, dist_bytes, cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_col_k));
    CUDA_CHECK(cudaFreeHost(col_k_pinned));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality check with OpenMP parallelism
    size_t check_limit = std::min(numNodes, static_cast<size_t>(10));
    bool valid = true;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < check_limit; ++i) {
        for (size_t j = 0; j < check_limit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        #pragma omp atomic write
                        valid = false;
                    }
                }
            }
        }
    }

    if (!valid) {
        printf("Validation failed: triangle inequality violated\n");
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

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (all processes for consistency)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = (size_t)atoi(argv[++i]);
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
        printf("MPI processes: %d\n", num_procs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute column distribution
    size_t start_col, local_n_cols;
    compute_distribution(numNodes, rank, num_procs, start_col, local_n_cols);

    // Allocate local matrices (column-major, local columns only)
    std::vector<unsigned int> dist_local(local_n_cols * numNodes);
    std::vector<unsigned int> path_local(local_n_cols * numNodes);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeLocalDistanceMatrix(dist_local, numNodes, start_col, local_n_cols, 1, MAX_DISTANCE);
    initializeLocalPathMatrix(path_local, numNodes, start_col, local_n_cols);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start_time = std::chrono::high_resolution_clock::now();

    floydWarshall_hybrid(dist_local, path_local, numNodes, start_col, local_n_cols, rank, num_procs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    // Gather full distance matrix on rank 0 for validation/output
    std::vector<int> recvcounts(num_procs);
    std::vector<int> displs(num_procs);
    for (int p = 0; p < num_procs; ++p) {
        size_t sc, lnc;
        compute_distribution(numNodes, p, num_procs, sc, lnc);
        recvcounts[p] = (int)(lnc * numNodes);
        displs[p] = (int)(sc * numNodes);
    }

    std::vector<unsigned int> full_dist;
    if (rank == 0) {
        full_dist.resize(numNodes * numNodes);
    }

    MPI_Gatherv(dist_local.data(), (int)(local_n_cols * numNodes), MPI_UNSIGNED,
                full_dist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    int result = 0;
    if (rank == 0) {
        long duration_ms = duration.count();
        printf("Computation time: %ld ms\n", duration_ms);

        double ops = (double)numNodes * numNodes * numNodes;
        double gops = ops / (duration_ms / 1000.0) / 1e9;
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
                result = 1;
            }
        }
    }

    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
