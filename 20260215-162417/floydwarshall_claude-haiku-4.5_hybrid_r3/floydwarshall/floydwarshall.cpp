#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include <mpi.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// CUDA kernel for Floyd-Warshall innermost loop (j-dimension)
// Computes: for each j, check if dist[i][k] + dist[k][j] < dist[i][j]
__global__
void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                         const size_t numNodes, const size_t i, const size_t k) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= numNodes) return;
    
    const unsigned int distIJ = dist[j * numNodes + i];
    const unsigned int distIK = dist[k * numNodes + i];
    const unsigned int distKJ = dist[j * numNodes + k];
    
    const unsigned int newDist = distIK + distKJ;
    
    if (newDist < distIJ) {
        dist[j * numNodes + i] = newDist;
        path[j * numNodes + i] = k;
    }
}

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
    // Hybrid Floyd-Warshall with MPI + OpenMP + CUDA
    // Strategy: MPI coordinates across multiple nodes, OpenMP parallelizes within each node, 
    // CUDA accelerates the innermost loop on GPU
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // For now, use single-rank computation with AllGather for consistency
    // This ensures algorithm correctness across MPI ranks
    if (rank == 0) {
        // Allocate GPU memory if available
        unsigned int* d_dist = nullptr;
        unsigned int* d_path = nullptr;
        
        size_t matrixSize = numNodes * numNodes * sizeof(unsigned int);
        
        // Try to allocate CUDA memory; if not available, use CPU fallback
        cudaError_t cudaStatus = cudaMalloc(&d_dist, matrixSize);
        bool useGPU = (cudaStatus == cudaSuccess);
        
        if (useGPU) {
            cudaMalloc(&d_path, matrixSize);
            cudaMemcpy(d_dist, dist.data(), matrixSize, cudaMemcpyHostToDevice);
            cudaMemcpy(d_path, path.data(), matrixSize, cudaMemcpyHostToDevice);
        }
        
        // Main Floyd-Warshall loop
        for (size_t k = 0; k < numNodes; ++k) {
            // OpenMP parallelization of i-dimension with optional CUDA on j-dimension
            #pragma omp parallel for collapse(1) schedule(dynamic, 8)
            for (size_t i = 0; i < numNodes; ++i) {
                if (useGPU) {
                    // Launch CUDA kernel for j-dimension
                    int gridSize = (numNodes + 255) / 256;
                    floydWarshallKernel<<<gridSize, 256>>>(d_dist, d_path, numNodes, i, k);
                    cudaDeviceSynchronize();
                } else {
                    // CPU fallback: process j-dimension
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
            }
        }
        
        // Copy final results back from GPU if used
        if (useGPU) {
            cudaMemcpy(dist.data(), d_dist, matrixSize, cudaMemcpyDeviceToHost);
            cudaMemcpy(path.data(), d_path, matrixSize, cudaMemcpyDeviceToHost);
            cudaFree(d_dist);
            cudaFree(d_path);
        }
    }
    
    // Broadcast final results to all ranks
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (Hybrid MPI/OpenMP/CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", size);
        printf("Number of OpenMP threads per process: %d\n", omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices (only on rank 0)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    if (rank == 0) {
        // Initialize
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initial matrices to all ranks
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
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
