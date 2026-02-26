#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#if HAVE_CUDA
#include <cuda_runtime.h>
#endif

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

#if HAVE_CUDA
// CUDA kernel declaration
extern "C" {
    void floydWarshall_cuda_kernel(unsigned int* dist_dev, unsigned int* dist_row_k_dev, 
                                   unsigned int* dist_col_k_dev, size_t numNodes, 
                                   size_t row_start, size_t row_count, size_t k);
}
#endif

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
    // MPI initialization
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Only rank 0 does the computation for correctness
    // Other ranks synchronize at barriers
    
    // Determine if GPU is available (only on rank 0)
    bool use_cuda = false;
#if HAVE_CUDA
    use_cuda = (rank == 0);
#endif
    
    unsigned int* dist_dev = nullptr;
    unsigned int* dist_row_k_dev = nullptr;
    unsigned int* dist_col_k_dev = nullptr;
    
#if HAVE_CUDA
    if (use_cuda) {
        // Allocate GPU memory for full matrix
        cudaMalloc(&dist_dev, numNodes * numNodes * sizeof(unsigned int));
        cudaMalloc(&dist_row_k_dev, numNodes * sizeof(unsigned int));
        cudaMalloc(&dist_col_k_dev, numNodes * sizeof(unsigned int));
        cudaMemcpy(dist_dev, dist.data(), numNodes * numNodes * sizeof(unsigned int), 
                  cudaMemcpyHostToDevice);
    }
#endif
    
    // Synchronize all processes before computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Classic Floyd-Warshall algorithm
    for (size_t k = 0; k < numNodes; ++k) {
        if (rank == 0) {
            if (use_cuda && dist_dev) {
#if HAVE_CUDA
                // Update GPU data
                cudaMemcpy(dist_dev, dist.data(), numNodes * numNodes * sizeof(unsigned int),
                          cudaMemcpyHostToDevice);
                
                // Call CUDA kernel for all rows
                floydWarshall_cuda_kernel(dist_dev, dist_row_k_dev, dist_col_k_dev, 
                                         numNodes, 0, numNodes, k);
                
                // Copy results back
                cudaMemcpy(dist.data(), dist_dev, numNodes * numNodes * sizeof(unsigned int),
                           cudaMemcpyDeviceToHost);
#endif
            } else {
                // CPU path with OpenMP parallelization for all rows
                #pragma omp parallel for collapse(2) schedule(static)
                for (size_t i = 0; i < numNodes; ++i) {
                    for (size_t j = 0; j < numNodes; ++j) {
                        const unsigned int distIJ = dist[i * numNodes + j];
                        const unsigned int distIK = dist[i * numNodes + k];
                        const unsigned int distKJ = dist[k * numNodes + j];
                        
                        const unsigned int newDist = distIK + distKJ;
                        
                        if (newDist < distIJ) {
                            dist[i * numNodes + j] = newDist;
                            path[i * numNodes + j] = k;
                        }
                    }
                }
            }
        }
        
        // Broadcast updated distances to all processes
        MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
#if HAVE_CUDA
    // Cleanup GPU memory
    if (dist_dev) {
        cudaFree(dist_dev);
        cudaFree(dist_row_k_dev);
        cudaFree(dist_col_k_dev);
    }
#endif
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
    
    int rank = 0, size = 1;
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else if (rank == 0) {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", size);
        printf("Number of OpenMP threads: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices (full on all processes for correctness)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initialized data to all processes
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    MPI_Barrier(MPI_COMM_WORLD);
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
            }
        }
    }
    
    MPI_Finalize();
    
    if (rank == 0 && validate) {
        return validateResult(dist, numNodes) ? 0 : 1;
    }
    
    return 0;
}
