#include <algorithm>
#include <chrono>
#include <climits>
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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: each thread updates one (local_i, j) element for iteration k
// Uses shared memory for the broadcast k-row tile for fast access
__global__ void fw_kernel(unsigned int* __restrict__ d_dist,
                          unsigned int* __restrict__ d_path,
                          const unsigned int* __restrict__ d_k_row,
                          const int n, const int local_rows,
                          const unsigned int k) {
    extern __shared__ unsigned int s_k_row[];

    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int local_i = blockIdx.y * blockDim.y + threadIdx.y;

    // Cooperatively load k-row tile into shared memory
    const int lin = threadIdx.y * blockDim.x + threadIdx.x;
    if (lin < blockDim.x) {
        const int gj = blockIdx.x * blockDim.x + lin;
        s_k_row[lin] = (gj < n) ? d_k_row[gj] : 0u;
    }
    __syncthreads();

    if (local_i < local_rows && j < n) {
        const int ij = local_i * n + j;
        const unsigned int dik = d_dist[local_i * n + (int)k];
        const unsigned int dkj = s_k_row[threadIdx.x];
        const unsigned int dij = d_dist[ij];
        const unsigned int nd = dik + dkj;
        if (nd < dij) {
            d_dist[ij] = nd;
            d_path[ij] = k;
        }
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
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    bool valid = true;

    // 1. Diagonal should be zero (OpenMP parallel)
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

    // 2. Triangle inequality check on a sample (OpenMP parallel)
    const size_t check_size = std::min(numNodes, static_cast<size_t>(10));
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < check_size; ++i) {
        for (size_t j = 0; j < check_size; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

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
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Assign GPU based on node-local rank
    {
        MPI_Comm local_comm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                            MPI_INFO_NULL, &local_comm);
        int local_rank;
        MPI_Comm_rank(local_comm, &local_rank);
        int num_devices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&num_devices));
        if (num_devices > 0) {
            CUDA_CHECK(cudaSetDevice(local_rank % num_devices));
        }
        MPI_Comm_free(&local_comm);
    }

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

    const int n = static_cast<int>(numNodes);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d, OpenMP threads: %d\n", nprocs, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute row distribution across MPI ranks
    const int base_rows = n / nprocs;
    const int extra_rows = n % nprocs;

    std::vector<int> row_counts(nprocs), row_offsets(nprocs);
    std::vector<int> elem_counts(nprocs), elem_offsets(nprocs);
    for (int r = 0; r < nprocs; r++) {
        row_counts[r] = (r < extra_rows) ? (base_rows + 1) : base_rows;
        row_offsets[r] = (r < extra_rows)
            ? r * (base_rows + 1)
            : extra_rows * (base_rows + 1) + (r - extra_rows) * base_rows;
        elem_counts[r] = row_counts[r] * n;
        elem_offsets[r] = row_offsets[r] * n;
    }

    const int local_rows = row_counts[rank];

    // Precompute row-to-rank owner mapping
    std::vector<int> row_owner(n);
    for (int r = 0; r < nprocs; r++) {
        for (int i = 0; i < row_counts[r]; i++) {
            row_owner[row_offsets[r] + i] = r;
        }
    }

    // Initialize full matrices on rank 0
    std::vector<unsigned int> dist_full, path_full;
    if (rank == 0) {
        dist_full.resize(static_cast<size_t>(n) * n);
        path_full.resize(static_cast<size_t>(n) * n);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path_full, numNodes);
    }

    // Local CPU buffers for this rank's rows
    std::vector<unsigned int> local_dist(static_cast<size_t>(local_rows) * n);
    std::vector<unsigned int> local_path(static_cast<size_t>(local_rows) * n);

    // Scatter matrices from rank 0 to all ranks
    MPI_Scatterv(rank == 0 ? dist_full.data() : nullptr,
                 elem_counts.data(), elem_offsets.data(), MPI_UNSIGNED,
                 local_dist.data(), local_rows * n, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path_full.data() : nullptr,
                 elem_counts.data(), elem_offsets.data(), MPI_UNSIGNED,
                 local_path.data(), local_rows * n, MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);

    // Free full path on rank 0 (not needed for output)
    if (rank == 0) {
        path_full.clear();
        path_full.shrink_to_fit();
    }

    // Allocate GPU memory and transfer local data
    unsigned int *d_dist = nullptr, *d_path = nullptr, *d_k_row = nullptr;
    if (local_rows > 0) {
        const size_t local_sz = static_cast<size_t>(local_rows) * n * sizeof(unsigned int);
        CUDA_CHECK(cudaMalloc(&d_dist, local_sz));
        CUDA_CHECK(cudaMalloc(&d_path, local_sz));
        CUDA_CHECK(cudaMalloc(&d_k_row, static_cast<size_t>(n) * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_dist, local_dist.data(), local_sz, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, local_path.data(), local_sz, cudaMemcpyHostToDevice));
    }

    // Pinned host buffer for fast k-row GPU transfers
    unsigned int* h_k_row = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_k_row, static_cast<size_t>(n) * sizeof(unsigned int)));

    // Kernel launch configuration
    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
                    static_cast<unsigned>((local_rows + block.y - 1) / block.y));
    const size_t smem_size = block.x * sizeof(unsigned int);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Floyd-Warshall: outer k loop is sequential; inner (i,j) parallelized via MPI+CUDA
    for (int k = 0; k < n; k++) {
        const int owner = row_owner[k];

        // Owner copies its k-th row from GPU to pinned host buffer
        if (rank == owner && local_rows > 0) {
            const int local_k = k - row_offsets[owner];
            CUDA_CHECK(cudaMemcpy(h_k_row,
                                  d_dist + static_cast<size_t>(local_k) * n,
                                  static_cast<size_t>(n) * sizeof(unsigned int),
                                  cudaMemcpyDeviceToHost));
        }

        // Broadcast k-row to all ranks
        MPI_Bcast(h_k_row, n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        if (local_rows > 0) {
            // Transfer k-row to GPU
            CUDA_CHECK(cudaMemcpy(d_k_row, h_k_row,
                                  static_cast<size_t>(n) * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

            // Launch kernel to update all local rows
            fw_kernel<<<grid, block, smem_size>>>(
                d_dist, d_path, d_k_row, n, local_rows, static_cast<unsigned int>(k));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = static_cast<long long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count());
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX,
               0, MPI_COMM_WORLD);

    // Copy dist results back from GPU
    if (local_rows > 0) {
        CUDA_CHECK(cudaMemcpy(local_dist.data(), d_dist,
                              static_cast<size_t>(local_rows) * n * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }

    // Gather dist matrix to rank 0
    if (rank == 0) dist_full.resize(static_cast<size_t>(n) * n);
    MPI_Gatherv(local_dist.data(), local_rows * n, MPI_UNSIGNED,
                rank == 0 ? dist_full.data() : nullptr,
                elem_counts.data(), elem_offsets.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        // Calculate operations per second (Floyd-Warshall is O(n³))
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
                cudaFreeHost(h_k_row);
                if (d_dist) cudaFree(d_dist);
                if (d_path) cudaFree(d_path);
                if (d_k_row) cudaFree(d_k_row);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Cleanup
    if (d_dist) cudaFree(d_dist);
    if (d_path) cudaFree(d_path);
    if (d_k_row) cudaFree(d_k_row);
    if (h_k_row) cudaFreeHost(h_k_row);

    MPI_Finalize();
    return 0;
}
