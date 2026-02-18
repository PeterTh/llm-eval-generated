#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// CUDA kernel for inner loop computation (only defined with CUDA)
#ifdef __CUDACC__
__global__ void floydWarshallKernel(unsigned int* dist, const unsigned int* k_row,
                                     const unsigned int* k_col, size_t numNodes,
                                     size_t block_start_i, size_t block_end_i) {
    size_t i = blockIdx.y * blockDim.y + threadIdx.y + block_start_i;
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < block_end_i && j < numNodes) {
        unsigned int distIJ = dist[j * numNodes + i];
        unsigned int distIK = k_row[i];
        unsigned int distKJ = k_col[j];
        
        if (distIK < 1000000000U && distKJ < 1000000000U) {
            unsigned int newDist = distIK + distKJ;
            if (newDist < distIJ) {
                dist[j * numNodes + i] = newDist;
            }
        }
    }
}
#endif

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

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Distribute rows across MPI ranks - these indices never change
    size_t rows_per_rank = (numNodes + size - 1) / size;
    size_t start_row = rank * rows_per_rank;
    size_t end_row = std::min((rank + 1) * rows_per_rank, numNodes);
    if (rank == size - 1) {
        end_row = numNodes;
    }
    size_t local_rows = end_row - start_row;
    
    // All ranks maintain the full distance matrix locally (column-major layout)
    std::vector<unsigned int> dist_full = dist;
    
    // GPU memory pointers (only used with CUDA)
    #ifdef __CUDACC__
    unsigned int* d_dist = nullptr;
    unsigned int* d_k_row = nullptr;
    unsigned int* d_k_col = nullptr;
    
    bool use_gpu = false;
    if (cudaMalloc(&d_dist, numNodes * numNodes * sizeof(unsigned int)) == cudaSuccess &&
        cudaMalloc(&d_k_row, numNodes * sizeof(unsigned int)) == cudaSuccess &&
        cudaMalloc(&d_k_col, numNodes * sizeof(unsigned int)) == cudaSuccess) {
        use_gpu = true;
        cudaMemcpy(d_dist, dist_full.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    #endif
    
    // Temporary buffers for MPI communication
    std::vector<unsigned int> k_row_global(numNodes);
    std::vector<unsigned int> k_col_global(numNodes);
    std::vector<unsigned int> local_rows_buffer(local_rows * numNodes);
    
    // Floyd-Warshall with hybrid parallelization
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns the k-th row
        int k_owner = k / rows_per_rank;
        if (k_owner >= size) k_owner = size - 1;
        
        // All ranks read from their local copy of dist_full
        #pragma omp parallel for simd schedule(static) num_threads(4)
        for (size_t i = 0; i < numNodes; ++i) {
            k_row_global[i] = dist_full[k * numNodes + i];
            k_col_global[i] = dist_full[i * numNodes + k];
        }
        
        // Broadcast k-th row and column from k_owner to all ranks
        MPI_Bcast(k_row_global.data(), numNodes, MPI_UNSIGNED, k_owner, MPI_COMM_WORLD);
        MPI_Bcast(k_col_global.data(), numNodes, MPI_UNSIGNED, k_owner, MPI_COMM_WORLD);
        
        #ifdef __CUDACC__
        if (use_gpu) {
            cudaMemcpy(d_k_row, k_row_global.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
            cudaMemcpy(d_k_col, k_col_global.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
            
            dim3 blockSize(16, 16);
            dim3 gridSize((numNodes + blockSize.x - 1) / blockSize.x,
                         (local_rows + blockSize.y - 1) / blockSize.y);
            
            floydWarshallKernel<<<gridSize, blockSize>>>(d_dist, d_k_row, d_k_col,
                                                        numNodes, start_row, end_row);
            cudaDeviceSynchronize();
        } else
        #endif
        {
            // CPU: parallelize with OpenMP - operate on rows [start_row, end_row)
            #pragma omp parallel for collapse(2) schedule(dynamic, 16) num_threads(8)
            for (size_t i = start_row; i < end_row; ++i) {
                for (size_t j = 0; j < numNodes; ++j) {
                    unsigned int distIJ = dist_full[j * numNodes + i];
                    unsigned int distIK = k_row_global[i];
                    unsigned int distKJ = k_col_global[j];
                    
                    unsigned int newDist = distIK + distKJ;
                    
                    if (newDist < distIJ) {
                        dist_full[j * numNodes + i] = newDist;
                        path[j * numNodes + i] = k;
                    }
                }
            }
        }
        
        // Copy GPU results back if using GPU
        #ifdef __CUDACC__
        if (use_gpu) {
            cudaMemcpy(dist_full.data() + start_row * numNodes, d_dist + start_row * numNodes, 
                      local_rows * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        }
        #endif
        
        // Extract local rows into contiguous buffer for communication (rows [start_row, end_row))
        #pragma omp parallel for schedule(static) num_threads(4)
        for (size_t buf_idx = 0; buf_idx < local_rows * numNodes; ++buf_idx) {
            size_t row_idx = start_row + buf_idx / numNodes;
            size_t col_idx = buf_idx % numNodes;
            local_rows_buffer[buf_idx] = dist_full[col_idx * numNodes + row_idx];
        }
        
        // Gather all rows to all ranks using Allgather
        std::vector<unsigned int> all_dist_buffer(numNodes * numNodes);
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size, 0);
        
        // Calculate sendcounts and displs
        for (int i = 0; i < size; ++i) {
            size_t rows_i = std::min((i + 1) * rows_per_rank, numNodes) - i * rows_per_rank;
            sendcounts[i] = rows_i * numNodes;
            if (i > 0) {
                displs[i] = displs[i-1] + sendcounts[i-1];
            }
        }
        
        MPI_Allgatherv(local_rows_buffer.data(), local_rows * numNodes, MPI_UNSIGNED,
                       all_dist_buffer.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        
        // Convert gathered buffer back to column-major layout
        #pragma omp parallel for schedule(static) num_threads(4)
        for (size_t buf_idx = 0; buf_idx < numNodes * numNodes; ++buf_idx) {
            size_t row_idx = buf_idx / numNodes;
            size_t col_idx = buf_idx % numNodes;
            dist_full[col_idx * numNodes + row_idx] = all_dist_buffer[buf_idx];
        }
        
        // Update GPU memory if using GPU
        #ifdef __CUDACC__
        if (use_gpu) {
            cudaMemcpy(d_dist, dist_full.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
        }
        #endif
    }
    
    #ifdef __CUDACC__
    if (use_gpu) {
        cudaMemcpy(dist_full.data(), d_dist, numNodes * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaFree(d_dist);
        cudaFree(d_k_row);
        cudaFree(d_k_col);
    }
    #endif
    
    dist = dist_full;
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (all ranks initialize the same data)
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
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
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    MPI_Finalize();
    return 0;
}
