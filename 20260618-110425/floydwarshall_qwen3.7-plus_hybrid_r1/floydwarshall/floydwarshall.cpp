#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <numeric>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define MPI_CHECK(call) do { \
    int mpi_err = (call); \
    if (mpi_err != MPI_SUCCESS) { \
        char err_str[MPI_MAX_ERROR_STRING]; \
        int len; \
        MPI_Error_string(mpi_err, err_str, &len); \
        fprintf(stderr, "MPI error at %s:%d: %s\n", \
                __FILE__, __LINE__, err_str); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// CUDA kernel: Floyd-Warshall update for local rows
// dist is stored in row-major order: D(i,j) at dist[local_i * n + j]
// pivot_col[i] = D(i, k) for all global rows i (size n)
// pivot_row[j] = D(k, j) for all columns j (size n)
__global__ void fw_update_kernel(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivot_col,
                                  const unsigned int* __restrict__ pivot_row,
                                  const unsigned int k,
                                  const size_t row_start,
                                  const size_t local_rows,
                                  const size_t n) {
    // 2D grid: x = column (j), y = local row (local_i)
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local_i = blockIdx.y * blockDim.y + threadIdx.y;

    if (local_i < local_rows && j < n) {
        size_t idx = local_i * n + j;
        unsigned int dist_ij = dist[idx];
        unsigned int dist_ik = pivot_col[row_start + local_i];
        unsigned int dist_kj = pivot_row[j];

        unsigned int new_dist = dist_ik + dist_kj;

        if (new_dist < dist_ij) {
            dist[idx] = new_dist;
            path[idx] = k;
        }
    }
}

// CUDA kernel: extract column k from local dist matrix
// D(i,k) is at dist[local_i * n + k], strided by n
__global__ void extract_column_kernel(const unsigned int* __restrict__ dist,
                                       unsigned int* __restrict__ col_buf,
                                       const size_t col_idx,
                                       const size_t local_rows,
                                       const size_t n) {
    size_t local_i = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_i < local_rows) {
        col_buf[local_i] = dist[local_i * n + col_idx];
    }
}

// CUDA kernel: extract a single row from local dist matrix
__global__ void extract_row_kernel(const unsigned int* __restrict__ dist,
                                    unsigned int* __restrict__ row_buf,
                                    const size_t local_row_idx,
                                    const size_t n) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) {
        row_buf[j] = dist[local_row_idx * n + j];
    }
}

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero
    for (size_t i = 0; i < numNodes; ++i) {
        dist[i * numNodes + i] = 0;
    }
}

void initializePathMatrixLocal(std::vector<unsigned int>& path, const size_t numNodes,
                                const size_t row_start, const size_t row_end) {
    size_t local_rows = row_end - row_start;
    path.resize(local_rows * numNodes);

    #pragma omp parallel for schedule(static)
    for (size_t li = 0; li < local_rows; ++li) {
        size_t i = row_start + li;
        for (size_t j = 0; j < numNodes; ++j) {
            path[li * numNodes + j] = static_cast<unsigned int>(i);
        }
    }
}

// Determine which rank owns a given global row
inline int get_row_owner(size_t row, size_t n, int numprocs) {
    int owner = static_cast<int>((row * numprocs) / n);
    if (owner >= numprocs) owner = numprocs - 1;
    return owner;
}

// Get local row range for a given rank
inline void get_local_range(int rank, size_t n, int numprocs,
                            size_t& row_start, size_t& row_end) {
    row_start = (static_cast<size_t>(rank) * n) / numprocs;
    row_end = (static_cast<size_t>(rank + 1) * n) / numprocs;
}

void floydWarshall_hybrid(std::vector<unsigned int>& local_dist,
                           std::vector<unsigned int>& local_path,
                           const size_t numNodes,
                           const size_t row_start,
                           const size_t row_end,
                           int rank, int numprocs) {
    const size_t local_rows = row_end - row_start;
    if (local_rows == 0) return;

    const size_t n = numNodes;
    const size_t local_size = local_rows * n;

    // Select GPU for this rank
    int num_gpus = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    int gpu_id = rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(gpu_id));

    // Allocate GPU memory
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_pivot_col = nullptr;
    unsigned int* d_pivot_row = nullptr;
    unsigned int* d_pivot_col_local = nullptr;

    CUDA_CHECK(cudaMalloc(&d_dist, local_size * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path, local_size * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_pivot_col, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_pivot_row, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_pivot_col_local, local_rows * sizeof(unsigned int)));

    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_dist, local_dist.data(), local_size * sizeof(unsigned int),
                           cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, local_path.data(), local_size * sizeof(unsigned int),
                           cudaMemcpyHostToDevice));

    // Host buffers for MPI communication
    std::vector<unsigned int> h_pivot_col(n);
    std::vector<unsigned int> h_pivot_row(n);
    std::vector<unsigned int> h_pivot_col_local(local_rows);

    // CUDA stream for async operations
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Thread block dimensions
    dim3 block_extract(256);
    dim3 grid_extract((local_rows + 255) / 256);

    dim3 block_fw(128, 4);
    dim3 grid_fw((n + block_fw.x - 1) / block_fw.x,
                 (local_rows + block_fw.y - 1) / block_fw.y);

    // Main FW loop
    for (size_t k = 0; k < n; ++k) {
        // Step 1: Extract pivot column from GPU
        extract_column_kernel<<<grid_extract, block_extract, 0, stream>>>(
            d_dist, d_pivot_col_local, k, local_rows, n);

        // Step 2: Copy pivot column local from GPU to host
        CUDA_CHECK(cudaMemcpyAsync(h_pivot_col_local.data(), d_pivot_col_local,
                                    local_rows * sizeof(unsigned int),
                                    cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Step 3: MPI_Allgather for pivot column
        MPI_Allgather(h_pivot_col_local.data(), static_cast<int>(local_rows), MPI_UNSIGNED,
                       h_pivot_col.data(), static_cast<int>(local_rows), MPI_UNSIGNED,
                       MPI_COMM_WORLD);

        // Step 4: MPI_Bcast for pivot row
        int owner = get_row_owner(k, n, numprocs);
        if (rank == owner) {
            size_t local_k = k - row_start;
            // Extract pivot row from GPU
            dim3 grid_row((n + 255) / 256);
            extract_row_kernel<<<grid_row, 256, 0, stream>>>(
                d_dist, d_pivot_row, local_k, n);
            CUDA_CHECK(cudaMemcpyAsync(h_pivot_row.data(), d_pivot_row,
                                        n * sizeof(unsigned int),
                                        cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        MPI_Bcast(h_pivot_row.data(), static_cast<int>(n), MPI_UNSIGNED, owner,
                   MPI_COMM_WORLD);

        // Step 5: Copy pivot data to GPU
        CUDA_CHECK(cudaMemcpyAsync(d_pivot_col, h_pivot_col.data(),
                                    n * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_pivot_row, h_pivot_row.data(),
                                    n * sizeof(unsigned int),
                                    cudaMemcpyHostToDevice, stream));

        // Step 6: Launch FW update kernel
        fw_update_kernel<<<grid_fw, block_fw, 0, stream>>>(
            d_dist, d_path, d_pivot_col, d_pivot_row,
            static_cast<unsigned int>(k), row_start, local_rows, n);
        
        // Synchronize before next iteration to ensure dist matrix is updated
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Copy results back to host
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaMemcpy(local_dist.data(), d_dist, local_size * sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(local_path.data(), d_path, local_size * sizeof(unsigned int),
                           cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_pivot_col));
    CUDA_CHECK(cudaFree(d_pivot_row));
    CUDA_CHECK(cudaFree(d_pivot_col_local));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    const size_t n = numNodes;

    // 1. Diagonal should be zero
    for (size_t i = 0; i < n; ++i) {
        if (dist[i * n + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero (value=%u)\n",
                   i, i, dist[i * n + i]);
            return false;
        }
    }

    // 2. Triangle inequality check (sampled)
    size_t check_limit = std::min(n, static_cast<size_t>(10));
    bool valid = true;

    #pragma omp parallel for schedule(dynamic) collapse(2) reduction(&&:valid)
    for (size_t i = 0; i < check_limit; ++i) {
        for (size_t j = 0; j < check_limit; ++j) {
            if (!valid) continue;
            for (size_t k = 0; k < n; ++k) {
                const unsigned int distIJ = dist[i * n + j];
                const unsigned int distIK = dist[i * n + k];
                const unsigned int distKJ = dist[k * n + j];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        valid = false;
                        break;
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
    // Initialize MPI
    MPI_CHECK(MPI_Init(&argc, &argv));

    int rank, numprocs;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &numprocs));

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 needs to parse, then broadcast)
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
            }
        }
    }

    // Broadcast parameters from rank 0
    struct {
        size_t numNodes;
        int validate;
        int printResults;
    } params;

    if (rank == 0) {
        params.numNodes = numNodes;
        params.validate = validate ? 1 : 0;
        params.printResults = printResults ? 1 : 0;
    }
    MPI_Bcast(&params, sizeof(params), MPI_BYTE, 0, MPI_COMM_WORLD);
    numNodes = params.numNodes;
    validate = params.validate != 0;
    printResults = params.printResults != 0;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d, OpenMP threads: %d\n", numprocs, omp_get_max_threads());
        int num_gpus = 0;
        cudaGetDeviceCount(&num_gpus);
        printf("CUDA GPUs: %d\n", num_gpus);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute local row range
    size_t row_start, row_end;
    get_local_range(rank, numNodes, numprocs, row_start, row_end);
    size_t local_rows = row_end - row_start;

    const size_t n = numNodes;

    // Initialize distance matrix on rank 0, then scatter
    std::vector<unsigned int> local_dist(local_rows * n);
    std::vector<unsigned int> local_path;

    if (rank == 0) {
        // Generate full distance matrix (sequential, matching original)
        printf("Initializing graph...\n");
        std::vector<unsigned int> full_dist(n * n);
        initializeDistanceMatrix(full_dist, n, 1, MAX_DISTANCE);

        // Scatter distance matrix rows
        if (numprocs == 1) {
            local_dist = std::move(full_dist);
        } else {
            // Use MPI_Scatterv for uneven distribution
            std::vector<int> sendcounts(numprocs);
            std::vector<int> displs(numprocs);
            for (int r = 0; r < numprocs; ++r) {
                size_t rs, re;
                get_local_range(r, n, numprocs, rs, re);
                sendcounts[r] = static_cast<int>((re - rs) * n);
                displs[r] = static_cast<int>(rs * n);
            }
            MPI_Scatterv(full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                          local_dist.data(), static_cast<int>(local_rows * n), MPI_UNSIGNED,
                          0, MPI_COMM_WORLD);
        }
    } else {
        if (local_rows > 0) {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED,
                          local_dist.data(), static_cast<int>(local_rows * n), MPI_UNSIGNED,
                          0, MPI_COMM_WORLD);
        }
    }

    // Initialize path matrix locally (each rank does its own rows, parallelized with OpenMP)
    initializePathMatrixLocal(local_path, n, row_start, row_end);

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall_hybrid(local_dist, local_path, n, row_start, row_end, rank, numprocs);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    long long duration_ms = 0;
    long long local_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    MPI_Reduce(&local_ms, &duration_ms, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);

        double ops = (double)n * n * n;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather results to rank 0
    std::vector<unsigned int> full_dist;
    if (rank == 0) {
        full_dist.resize(n * n);
    }

    if (numprocs == 1) {
        if (rank == 0) full_dist = local_dist;
    } else {
        if (rank == 0) {
            std::vector<int> recvcounts(numprocs);
            std::vector<int> recvdispls(numprocs);
            for (int r = 0; r < numprocs; ++r) {
                size_t rs, re;
                get_local_range(r, n, numprocs, rs, re);
                recvcounts[r] = static_cast<int>((re - rs) * n);
                recvdispls[r] = static_cast<int>(rs * n);
            }
            MPI_Gatherv(local_dist.data(), static_cast<int>(local_rows * n), MPI_UNSIGNED,
                         full_dist.data(), recvcounts.data(), recvdispls.data(), MPI_UNSIGNED,
                         0, MPI_COMM_WORLD);
        } else {
            MPI_Gatherv(local_dist.data(), static_cast<int>(local_rows * n), MPI_UNSIGNED,
                         nullptr, nullptr, nullptr, MPI_UNSIGNED,
                         0, MPI_COMM_WORLD);
        }
    }

    // Print results and validate on rank 0
    if (rank == 0) {
        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}
