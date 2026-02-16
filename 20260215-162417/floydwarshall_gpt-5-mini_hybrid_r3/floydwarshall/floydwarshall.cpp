#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Parallelization headers
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

// CUDA kernel: each thread updates one (i,j) element where i in [i_start,i_end)
extern "C" __global__ void fw_kernel_cuda(unsigned int* dist, const unsigned int* col_k, const unsigned int* row_k,
                                           size_t n, size_t i_start, size_t i_count) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = i_count * n;
    if (tid >= total) return;
    size_t local_i = tid / n; // 0..i_count-1
    size_t j = tid % n;
    size_t i = i_start + local_i;

    unsigned int distIK = col_k[i];
    unsigned int distKJ = row_k[j];
    if (distIK >= INF || distKJ >= INF) return;
    unsigned int newDist = distIK + distKJ;

    size_t index = j * n + i; // idx2(j,i,n)
    unsigned int old = dist[index];
    if (newDist < old) {
        dist[index] = newDist;
    }
}

void floydWarshall_hybrid(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes, int world_rank, int world_size) {
    // Distribute i indices across ranks
    size_t chunk = (numNodes + world_size - 1) / world_size;
    size_t i_start = std::min((size_t)world_rank * chunk, numNodes);
    size_t i_end = std::min(i_start + chunk, numNodes);
    size_t i_count = (i_end > i_start) ? (i_end - i_start) : 0;

    // Buffers for communication and device usage
    std::vector<unsigned int> dist_global(numNodes * numNodes);
    std::vector<unsigned int> col_k(numNodes);
    std::vector<unsigned int> row_k(numNodes);

    // Prepare CUDA device memory
    unsigned int* d_dist = nullptr;
    unsigned int* d_col = nullptr;
    unsigned int* d_row = nullptr;
    bool cuda_ok = true;
    if (cudaSuccess != cudaMalloc((void**)&d_dist, sizeof(unsigned int) * numNodes * numNodes)) cuda_ok = false;
    if (cudaSuccess != cudaMalloc((void**)&d_col, sizeof(unsigned int) * numNodes)) cuda_ok = false;
    if (cudaSuccess != cudaMalloc((void**)&d_row, sizeof(unsigned int) * numNodes)) cuda_ok = false;

    if (!cuda_ok) {
        // cleanup if partial
        if (d_dist) cudaFree(d_dist);
        if (d_col) cudaFree(d_col);
        if (d_row) cudaFree(d_row);
        d_dist = d_col = d_row = nullptr;
    } else {
        // copy initial matrix to device
        cudaMemcpy(d_dist, dist.data(), sizeof(unsigned int) * numNodes * numNodes, cudaMemcpyHostToDevice);
    }

    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // Ensure all ranks start this iteration with the same global dist
        MPI_Allreduce(dist.data(), dist_global.data(), static_cast<int>(numNodes * numNodes), MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
        // copy reduced global into local dist for use
        dist = dist_global;

        // Build col_k and row_k from global dist
        for (size_t i = 0; i < numNodes; ++i) {
            col_k[i] = dist[idx2(k, i, numNodes)]; // distIK
            row_k[i] = dist[idx2(i, k, numNodes)]; // distKJ with j=i
        }

        if (d_dist) {
            // copy updated global dist to device and col/row arrays
            cudaMemcpy(d_dist, dist.data(), sizeof(unsigned int) * numNodes * numNodes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_col, col_k.data(), sizeof(unsigned int) * numNodes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_row, row_k.data(), sizeof(unsigned int) * numNodes, cudaMemcpyHostToDevice);

            // launch kernel to update local i-range in parallel on GPU
            size_t total = i_count * numNodes;
            size_t threads = 256;
            size_t blocks = (total + threads - 1) / threads;
            fw_kernel_cuda<<<blocks, threads>>>(d_dist, d_col, d_row, numNodes, i_start, i_count);
            cudaDeviceSynchronize();

            // copy updated device dist back to host
            cudaMemcpy(dist.data(), d_dist, sizeof(unsigned int) * numNodes * numNodes, cudaMemcpyDeviceToHost);
        } else {
            // CPU/OpenMP fallback: update local i-range
            #pragma omp parallel for collapse(2) schedule(static)
            for (size_t local_i = 0; local_i < i_count; ++local_i) {
                size_t i = i_start + local_i;
                for (size_t j = 0; j < numNodes; ++j) {
                    unsigned int distIJ = dist[idx2(j, i, numNodes)];
                    unsigned int distIK = col_k[i];
                    unsigned int distKJ = row_k[j];
                    if (distIK < INF && distKJ < INF) {
                        unsigned int newDist = distIK + distKJ;
                        if (newDist < distIJ) {
                            dist[idx2(j, i, numNodes)] = newDist;
                            path[idx2(j, i, numNodes)] = static_cast<unsigned int>(k);
                        }
                    }
                }
            }
        }

        // After local update, combine updated matrices across ranks (element-wise minimum)
        MPI_Allreduce(dist.data(), dist_global.data(), static_cast<int>(numNodes * numNodes), MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
        // copy back reduced result into local dist for next iteration
        dist = dist_global;
    }

    // cleanup device memory
    if (d_dist) cudaFree(d_dist);
    if (d_col) cudaFree(d_col);
    if (d_row) cudaFree(d_row);
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
    // Initialize MPI environment unconditionally
    MPI_Init(&argc, &argv);
    int world_size = 1;
    int world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d\n", world_size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (do same initialization on all ranks so reductions are valid)
    if (world_rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall (hybrid parallel)
    if (world_rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall_hybrid(dist, path, numNodes, world_rank, world_size);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Print results for external validation (integer hash-based) only from rank 0
    if (printResults && world_rank == 0) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation only on rank 0
    if (validate && world_rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    MPI_Finalize();
    return 0;
}
