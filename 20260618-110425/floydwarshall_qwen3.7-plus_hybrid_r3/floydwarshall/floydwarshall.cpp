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

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: each thread updates one element D(a, b) for the k-th iteration
// Thread mapping: threadIdx.x -> row(a), threadIdx.y -> local column(bl)
// This ensures coalesced memory access since consecutive threads in a warp
// have consecutive 'a' values, accessing consecutive addresses in dist[bl*n + a]
__global__ void fw_kernel(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ g_col_k,
                          const size_t n,
                          const size_t num_local_cols,
                          const size_t k) {
    const size_t a  = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t bl = blockIdx.y * blockDim.y + threadIdx.y;

    if (a < n && bl < num_local_cols) {
        const size_t pos = bl * n + a;
        const unsigned int distAB = dist[pos];
        const unsigned int distAK = __ldg(&g_col_k[a]);
        const unsigned int distKB = dist[bl * n + k];
        const unsigned int newDist = distAK + distKB;
        if (newDist < distAB) {
            dist[pos] = newDist;
            path[pos] = static_cast<unsigned int>(k);
        }
    }
}

// Index calculation for flattened 2D array (column-major)
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    bool valid = true;

    // Check diagonal is zero (parallelized with OpenMP)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            fprintf(stderr, "Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            valid = false;
        }
    }

    if (!valid) return false;

    // Triangle inequality check (parallelized with OpenMP)
    const size_t check_limit = std::min(numNodes, static_cast<size_t>(10));
    #pragma omp parallel for collapse(2) schedule(static) reduction(&&:valid)
    for (size_t i = 0; i < check_limit; ++i) {
        for (size_t j = 0; j < check_limit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                if (!valid) continue;
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
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

    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Select CUDA device (round-robin for multi-GPU nodes)
    int num_gpus = 0;
    cudaGetDeviceCount(&num_gpus);
    if (num_gpus == 0) {
        if (rank == 0) fprintf(stderr, "Error: No CUDA devices found\n");
        MPI_Finalize();
        return 1;
    }
    cudaSetDevice(rank % num_gpus);

    // Parse arguments on rank 0, then broadcast
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

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
                MPI_Abort(MPI_COMM_WORLD, 0);
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Abort(MPI_COMM_WORLD, 1);
                return 1;
            }
        }
    }

    // Broadcast parameters to all processes
    unsigned long long buf_nodes = static_cast<unsigned long long>(numNodes);
    int params[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&buf_nodes, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(params, 2, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(buf_nodes);
    validate = params[0] != 0;
    printResults = params[1] != 0;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d, OpenMP threads: %d, CUDA devices: %d\n",
               nprocs, omp_get_max_threads(), num_gpus);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t n = numNodes;

    // Block distribution: each process owns a contiguous range of columns
    const size_t cols_per_proc = (n + static_cast<size_t>(nprocs) - 1) / static_cast<size_t>(nprocs);
    const size_t j_start = static_cast<size_t>(rank) * cols_per_proc;
    const size_t j_end = std::min(j_start + cols_per_proc, n);
    const size_t num_local_cols = (j_start < n) ? (j_end - j_start) : 0;

    if (rank == 0) printf("Initializing graph...\n");

    // Initialize local distance matrix
    // Each process generates the full random sequence to ensure identical values
    std::vector<unsigned int> local_dist(num_local_cols * n);
    {
        const unsigned int rangeMin = 1;
        const unsigned int rangeMax = MAX_DISTANCE;
        const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
        unsigned int seed = 42;

        for (size_t idx = 0; idx < n * n; ++idx) {
            unsigned int val = rangeMin + static_cast<unsigned int>(
                range * rand_r(&seed) / static_cast<double>(RAND_MAX));
            size_t col = idx / n;
            size_t row = idx % n;
            if (col >= j_start && col < j_end) {
                local_dist[(col - j_start) * n + row] = val;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t jl = 0; jl < num_local_cols; ++jl) {
        size_t j = j_start + jl;
        local_dist[jl * n + j] = 0;
    }

    // Initialize path matrix with OpenMP parallelization
    std::vector<unsigned int> local_path(num_local_cols * n);
    #pragma omp parallel for schedule(static)
    for (size_t jl = 0; jl < num_local_cols; ++jl) {
        unsigned int j = static_cast<unsigned int>(j_start + jl);
        size_t base = jl * n;
        for (size_t i = 0; i < n; ++i) {
            local_path[base + i] = j;
        }
    }

    // Allocate GPU memory and transfer initial data
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_col_k = nullptr;

    if (num_local_cols > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, num_local_cols * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, num_local_cols * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_col_k, n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_dist, local_dist.data(),
                              num_local_cols * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, local_path.data(),
                              num_local_cols * n * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }

    // Free CPU copies to save memory
    local_dist.clear();
    local_dist.shrink_to_fit();
    local_path.clear();
    local_path.shrink_to_fit();

    // Host buffer for column k broadcast
    std::vector<unsigned int> col_k(n);

    if (rank == 0) printf("Computing shortest paths...\n");

    // Synchronize all processes before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();

    // CUDA kernel configuration
    // threadIdx.x -> row dimension (a) for coalesced access
    // threadIdx.y -> local column dimension (bl)
    const dim3 block(32, 8); // 256 threads per block

    // Main Floyd-Warshall loop: iterate over all intermediate nodes k
    for (size_t k = 0; k < n; ++k) {
        // Determine which MPI process owns column k
        int owner = static_cast<int>(k / cols_per_proc);
        if (owner >= nprocs) owner = nprocs - 1;

        // Owner extracts column k from GPU memory
        if (rank == owner && num_local_cols > 0) {
            size_t local_col_idx = k - j_start;
            CUDA_CHECK(cudaMemcpy(col_k.data(), d_dist + local_col_idx * n,
                                  n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }

        // Broadcast column k to all processes
        MPI_Bcast(col_k.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Execute CUDA kernel on processes with local work
        if (num_local_cols > 0) {
            // Copy broadcast column to GPU
            CUDA_CHECK(cudaMemcpy(d_col_k, col_k.data(),
                                  n * sizeof(unsigned int), cudaMemcpyHostToDevice));

            // Launch kernel
            dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x),
                      static_cast<unsigned int>((num_local_cols + block.y - 1) / block.y));
            fw_kernel<<<grid, block>>>(d_dist, d_path, d_col_k, n, num_local_cols, k);
        }
    }

    // Ensure all GPU work is complete
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    double duration_ms = (end_time - start_time) * 1000.0;
    double global_duration_ms = 0.0;
    MPI_Reduce(&duration_ms, &global_duration_ms, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(global_duration_ms));
        double ops = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        double gflops = ops / (global_duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather results to rank 0 using MPI_Gatherv
    std::vector<int> recv_counts(nprocs);
    std::vector<int> recv_displs(nprocs);
    for (int p = 0; p < nprocs; ++p) {
        size_t p_j_start = static_cast<size_t>(p) * cols_per_proc;
        size_t p_j_end = std::min(p_j_start + cols_per_proc, n);
        size_t p_num_cols = (p_j_start < n) ? p_j_end - p_j_start : 0;
        recv_counts[p] = static_cast<int>(p_num_cols * n);
        recv_displs[p] = static_cast<int>(p_j_start * n);
    }

    // Copy local distance matrix from GPU to host
    std::vector<unsigned int> local_dist_host(num_local_cols * n);
    if (num_local_cols > 0) {
        CUDA_CHECK(cudaMemcpy(local_dist_host.data(), d_dist,
                              num_local_cols * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    std::vector<unsigned int> full_dist;
    if (rank == 0) full_dist.resize(n * n);

    MPI_Gatherv(local_dist_host.data(), static_cast<int>(num_local_cols * n), MPI_UNSIGNED,
                full_dist.data(), recv_counts.data(), recv_displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // Output and validation on rank 0
    int result = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, n);
            if (valid) {
                printf("Validation: PASSED\n");
                result = 0;
            } else {
                printf("Validation: FAILED\n");
                result = 1;
            }
        }
    }

    // Broadcast result so all processes exit with the same code
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Cleanup GPU resources
    if (num_local_cols > 0) {
        cudaFree(d_dist);
        cudaFree(d_path);
        cudaFree(d_col_k);
    }

    MPI_Finalize();
    return result;
}
