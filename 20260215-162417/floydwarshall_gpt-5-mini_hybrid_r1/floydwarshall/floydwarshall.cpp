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

// Index calculation for flattened 2D array (element (i,j) stored as j*n + i)
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

// CUDA kernel: each thread updates one (i_local, j) element
#if defined(__CUDACC__)
extern "C" __global__ void fw_kernel(unsigned int* dist_local, const unsigned int* row_k,
                                      unsigned int* path_local, const size_t n,
                                      const size_t local_n, const size_t k_global) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = local_n * n;
    if (tid >= total) return;

    size_t i_local = tid % local_n;        // local i index
    size_t j = tid / local_n;             // global j index

    size_t idx = j * local_n + i_local;   // local storage layout

    unsigned int distIJ = dist_local[idx];
    unsigned int distIK = dist_local[k_global * local_n + i_local];
    unsigned int distKJ = row_k[j];

    unsigned int newDist = distIK + distKJ;
    if (newDist < distIJ) {
        dist_local[idx] = newDist;
        path_local[idx] = (unsigned int)k_global;
    }
}
#endif

static inline void cuda_check(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, -1);
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
    MPI_Init(&argc, &argv);

    int world_rank = 0, world_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (only rank 0 prints usage)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (world_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (world_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (world_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Partition rows among ranks
    size_t rows_per_rank = (numNodes + world_size - 1) / world_size;
    size_t row_start = world_rank * rows_per_rank;
    size_t row_end = std::min(numNodes, row_start + rows_per_rank);
    size_t local_n = (row_end > row_start) ? (row_end - row_start) : 0;

    // Allocate full matrix on host only as needed for initialization; each rank will own only its rows
    std::vector<unsigned int> dist_local(local_n * numNodes);
    std::vector<unsigned int> path_local(local_n * numNodes);

    // Initialize full matrices deterministically on rank 0 and scatter rows
    if (world_rank == 0) {
        std::vector<unsigned int> dist_full(numNodes * numNodes);
        std::vector<unsigned int> path_full(numNodes * numNodes);
        initializeDistanceMatrix(dist_full, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path_full, numNodes);

        // Scatter rows
        for (int r = 0; r < world_size; ++r) {
            size_t rs = r * rows_per_rank;
            size_t re = std::min(numNodes, rs + rows_per_rank);
            size_t rn = (re > rs) ? (re - rs) : 0;
            if (rn == 0) continue;
            if (r == 0) {
                // copy into local
                for (size_t j = 0; j < numNodes; ++j)
                    for (size_t i = 0; i < rn; ++i)
                        dist_local[j * rn + i] = dist_full[j * numNodes + (rs + i)];
                for (size_t j = 0; j < numNodes; ++j)
                    for (size_t i = 0; i < rn; ++i)
                        path_local[j * rn + i] = path_full[j * numNodes + (rs + i)];
            } else {
                // send to rank r
                std::vector<unsigned int> tmp_dist(rn * numNodes);
                std::vector<unsigned int> tmp_path(rn * numNodes);
                for (size_t j = 0; j < numNodes; ++j)
                    for (size_t i = 0; i < rn; ++i)
                        tmp_dist[j * rn + i] = dist_full[j * numNodes + (rs + i)];
                for (size_t j = 0; j < numNodes; ++j)
                    for (size_t i = 0; i < rn; ++i)
                        tmp_path[j * rn + i] = path_full[j * numNodes + (rs + i)];
                MPI_Send(tmp_dist.data(), (int)(rn * numNodes), MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
                MPI_Send(tmp_path.data(), (int)(rn * numNodes), MPI_UNSIGNED, r, 1, MPI_COMM_WORLD);
            }
        }
    } else {
        if (local_n > 0) {
            MPI_Recv(dist_local.data(), (int)(local_n * numNodes), MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(path_local.data(), (int)(local_n * numNodes), MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    // Setup CUDA device for each rank (round-robin mapping)
    int device_count = 0;
    cudaError_t cerr = cudaGetDeviceCount(&device_count);
    if (cerr != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cerr));
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    int device_id = 0;
    if (device_count > 0) device_id = world_rank % device_count;
    cuda_check(cudaSetDevice(device_id), "set device");

    // Allocate device buffers for local block and row_k
    unsigned int* d_dist_local = nullptr;
    unsigned int* d_path_local = nullptr;
    unsigned int* d_row_k = nullptr;

    if (local_n > 0) {
        cuda_check(cudaMalloc(&d_dist_local, sizeof(unsigned int) * local_n * numNodes), "alloc d_dist_local");
        cuda_check(cudaMalloc(&d_path_local, sizeof(unsigned int) * local_n * numNodes), "alloc d_path_local");
        // copy initial data
        cuda_check(cudaMemcpy(d_dist_local, dist_local.data(), sizeof(unsigned int) * local_n * numNodes, cudaMemcpyHostToDevice), "copy dist_local to device");
        cuda_check(cudaMemcpy(d_path_local, path_local.data(), sizeof(unsigned int) * local_n * numNodes, cudaMemcpyHostToDevice), "copy path_local to device");
    }
    cuda_check(cudaMalloc(&d_row_k, sizeof(unsigned int) * numNodes), "alloc d_row_k");

    // Host buffer for broadcasting row k
    std::vector<unsigned int> row_k(numNodes);

    if (world_rank == 0) printf("Computing shortest paths (MPI+OpenMP+CUDA hybrid)...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Main loop over k
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = (int)(k / rows_per_rank);
        // Owner prepares row_k from its local data
        if (world_rank == owner) {
            size_t local_k = k - row_start;
            for (size_t j = 0; j < numNodes; ++j) {
                // dist[idx2(j,k,numNodes)] corresponds to element (k,j) stored at j*numNodes + k in full layout
                // Our local layout stores element (k,j) at index j*local_n + local_k
                row_k[j] = dist_local[j * local_n + local_k];
            }
        }
        // Broadcast the row_k
        MPI_Bcast(row_k.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Copy row_k to device
        cuda_check(cudaMemcpy(d_row_k, row_k.data(), sizeof(unsigned int) * numNodes, cudaMemcpyHostToDevice), "copy row_k to device");

        if (local_n == 0) continue;

        // Launch CUDA kernel to update local block for this k
#if defined(__CUDACC__)
        size_t total = local_n * numNodes;
        int threads = 256;
        int blocks = (int)((total + threads - 1) / threads);
        fw_kernel<<<blocks, threads>>>(d_dist_local, d_row_k, d_path_local, numNodes, local_n, k - row_start);
        cuda_check(cudaGetLastError(), "kernel launch");
        cuda_check(cudaDeviceSynchronize(), "kernel sync");
#else
        // Fallback CPU parallel update using OpenMP (should not be used when CUDA is available)
        #pragma omp parallel for schedule(static)
        for (size_t i_local = 0; i_local < local_n; ++i_local) {
            for (size_t j = 0; j < numNodes; ++j) {
                size_t idx = j * local_n + i_local;
                unsigned int distIJ = dist_local[idx];
                unsigned int distIK = dist_local[(k - row_start) * local_n + i_local];
                unsigned int distKJ = row_k[j];
                unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    dist_local[idx] = newDist;
                    path_local[idx] = (unsigned int)k;
                }
            }
        }
#endif
        // Copy updated local block back to host for future owners to read
        cuda_check(cudaMemcpy(dist_local.data(), d_dist_local, sizeof(unsigned int) * local_n * numNodes, cudaMemcpyDeviceToHost), "copy dist_local to host");
        cuda_check(cudaMemcpy(path_local.data(), d_path_local, sizeof(unsigned int) * local_n * numNodes, cudaMemcpyDeviceToHost), "copy path_local to host");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (world_rank == 0) printf("Computation time: %ld ms\n", duration.count());

    // Gather results back to rank 0 for printing/validation
    if (world_rank == 0) {
        std::vector<unsigned int> dist_full(numNodes * numNodes);
        std::vector<unsigned int> path_full(numNodes * numNodes);

        // copy own block
        for (size_t j = 0; j < numNodes; ++j)
            for (size_t i = 0; i < local_n; ++i)
                dist_full[j * numNodes + (row_start + i)] = dist_local[j * local_n + i];
        for (size_t j = 0; j < numNodes; ++j)
            for (size_t i = 0; i < local_n; ++i)
                path_full[j * numNodes + (row_start + i)] = path_local[j * local_n + i];

        // receive from others
        for (int r = 1; r < world_size; ++r) {
            size_t rs = r * rows_per_rank;
            size_t re = std::min(numNodes, rs + rows_per_rank);
            size_t rn = (re > rs) ? (re - rs) : 0;
            if (rn == 0) continue;
            std::vector<unsigned int> tmp_dist(rn * numNodes);
            std::vector<unsigned int> tmp_path(rn * numNodes);
            MPI_Recv(tmp_dist.data(), (int)(rn * numNodes), MPI_UNSIGNED, r, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(tmp_path.data(), (int)(rn * numNodes), MPI_UNSIGNED, r, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for (size_t j = 0; j < numNodes; ++j)
                for (size_t i = 0; i < rn; ++i)
                    dist_full[j * numNodes + (rs + i)] = tmp_dist[j * rn + i];
            for (size_t j = 0; j < numNodes; ++j)
                for (size_t i = 0; i < rn; ++i)
                    path_full[j * numNodes + (rs + i)] = tmp_path[j * rn + i];
        }

        // Print performance
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) print_results_int(dist_full, "DistanceMatrix");

        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist_full, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

    } else {
        // send local blocks to rank 0
        if (local_n > 0) {
            MPI_Send(dist_local.data(), (int)(local_n * numNodes), MPI_UNSIGNED, 0, 2, MPI_COMM_WORLD);
            MPI_Send(path_local.data(), (int)(local_n * numNodes), MPI_UNSIGNED, 0, 3, MPI_COMM_WORLD);
        }
    }

    // Cleanup
    if (d_dist_local) cudaFree(d_dist_local);
    if (d_path_local) cudaFree(d_path_local);
    if (d_row_k) cudaFree(d_row_k);

    MPI_Finalize();
    return 0;
}
