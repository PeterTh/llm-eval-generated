#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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

#ifdef ENABLE_CUDA
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                     size_t numNodes, size_t k,
                                     size_t start_i, size_t local_rows) {
    size_t i = start_i + blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < start_i + local_rows && j < numNodes) {
        const unsigned int distIJ = dist[idx2(j, i, numNodes)];
        const unsigned int distIK = dist[idx2(k, i, numNodes)];
        const unsigned int distKJ = dist[idx2(j, k, numNodes)];
        
        const unsigned int newDist = distIK + distKJ;
        
        if (newDist < distIJ) {
            dist[idx2(j, i, numNodes)] = newDist;
            path[idx2(j, i, numNodes)] = k;
        }
    }
}

void floydWarshallCUDA(std::vector<unsigned int>& dist, 
                      std::vector<unsigned int>& path, 
                      const size_t numNodes,
                      const size_t start_i,
                      const size_t local_rows,
                      const size_t k) {
    unsigned int* d_dist;
    unsigned int* d_path;
    
    cudaMalloc(&d_dist, dist.size() * sizeof(unsigned int));
    cudaMalloc(&d_path, path.size() * sizeof(unsigned int));
    
    cudaMemcpy(d_dist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    dim3 blockSize(16, 16);
    dim3 gridSize((local_rows + blockSize.x - 1) / blockSize.x,
                  (numNodes + blockSize.y - 1) / blockSize.y);
    
    floydWarshallKernel<<<gridSize, blockSize>>>(d_dist, d_path, numNodes, k, start_i, local_rows);
    
    cudaMemcpy(dist.data(), d_dist, dist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), d_path, d_path, path.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_dist);
    cudaFree(d_path);
}
#endif

void floydWarshallHybrid(std::vector<unsigned int>& dist, 
                        std::vector<unsigned int>& path, 
                        const size_t numNodes,
                        int rank, int size) {
    // MPI + OpenMP + CUDA Hybrid Floyd-Warshall algorithm
    // Each rank stores complete columns but only its assigned rows
    
    // Calculate local row distribution
    size_t rows_per_rank = numNodes / size;
    size_t extra_rows = numNodes % size;
    
    // Determine this rank's row range
    size_t local_start_row = 0;
    size_t local_rows = 0;
    
    for (int r = 0; r < rank; ++r) {
        local_start_row += rows_per_rank + (r < (int)extra_rows ? 1 : 0);
    }
    local_rows = rows_per_rank + (rank < (int)extra_rows ? 1 : 0);
    
    // Note: each rank has the full matrix, but only computes for its rows
    // We'll use global dist and path arrays, with MPI broadcasts/updates
    
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // For each of our local rows i
        #pragma omp parallel for
        for (size_t local_i = 0; local_i < local_rows; ++local_i) {
            size_t i = local_start_row + local_i;
            
            // For each destination column j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
        
        // Synchronize: all ranks have computed their rows, now broadcast updated rows
        std::vector<unsigned int> send_buffer(local_rows * numNodes);
        
        // Copy our updated rows into send buffer
        #pragma omp parallel for collapse(2)
        for (size_t local_i = 0; local_i < local_rows; ++local_i) {
            for (size_t j = 0; j < numNodes; ++j) {
                size_t i = local_start_row + local_i;
                send_buffer[idx2(j, local_i, numNodes)] = dist[idx2(j, i, numNodes)];
            }
        }
        
        // Allgather: collect all updated rows and scatter back
        std::vector<int> sendcounts(size);
        std::vector<int> displs(size);
        
        int offset = 0;
        for (int r = 0; r < size; ++r) {
            size_t r_rows = rows_per_rank + (r < (int)extra_rows ? 1 : 0);
            sendcounts[r] = r_rows * numNodes;
            displs[r] = offset;
            offset += sendcounts[r];
        }
        
        MPI_Allgatherv(send_buffer.data(), local_rows * numNodes, MPI_UNSIGNED,
                       dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
        
        // Also gather path matrix
        #pragma omp parallel for collapse(2)
        for (size_t local_i = 0; local_i < local_rows; ++local_i) {
            for (size_t j = 0; j < numNodes; ++j) {
                size_t i = local_start_row + local_i;
                send_buffer[idx2(j, local_i, numNodes)] = path[idx2(j, i, numNodes)];
            }
        }
        
        MPI_Allgatherv(send_buffer.data(), local_rows * numNodes, MPI_UNSIGNED,
                       path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    if (rank == 0) {
        // Initialize on rank 0
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initial matrices to all ranks
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    // Run Floyd-Warshall
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshallHybrid(dist, path, numNodes, rank, size);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
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
