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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Parallelize initialization with OpenMP
    #pragma omp parallel
    {
        unsigned int seed = 42 + omp_get_thread_num();
        #pragma omp for
        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
    }
    
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        path[idx2(j, j, numNodes)] = j;
    }
}

// CUDA kernel for Floyd-Warshall computation - optimized
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                     const size_t k, const size_t numNodes) {
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < numNodes && j < numNodes) {
        const size_t idx_ij = j * numNodes + i;
        const size_t idx_ik = k * numNodes + i;
        const size_t idx_kj = j * numNodes + k;
        
        const unsigned int distIJ = dist[idx_ij];
        const unsigned int distIK = dist[idx_ik];
        const unsigned int distKJ = dist[idx_kj];
        
        const unsigned int newDist = distIK + distKJ;
        
        if (newDist < distIJ) {
            dist[idx_ij] = newDist;
            path[idx_ij] = k;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Allocate GPU memory
    unsigned int *d_dist, *d_path;
    const size_t totalSize = numNodes * numNodes * sizeof(unsigned int);
    
    CUDA_CHECK(cudaMalloc(&d_dist, totalSize));
    CUDA_CHECK(cudaMalloc(&d_path, totalSize));
    
    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), totalSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), totalSize, cudaMemcpyHostToDevice));
    
    // Configure CUDA kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((numNodes + blockSize.x - 1) / blockSize.x,
                  (numNodes + blockSize.y - 1) / blockSize.y);
    
    // Main Floyd-Warshall loop
    for (size_t k = 0; k < numNodes; ++k) {
        // Launch kernel
        floydWarshallKernel<<<gridSize, blockSize>>>(d_dist, d_path, k, numNodes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy results back from GPU
    CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, totalSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data(), d_path, totalSize, cudaMemcpyDeviceToHost));
    
    // Free GPU memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    bool valid = true;
    
    // 1. Diagonal should be zero
    #pragma omp parallel for
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
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    #pragma omp parallel for collapse(3)
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set GPU device based on local rank
    int numDevices;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices > 0) {
        CUDA_CHECK(cudaSetDevice(rank % numDevices));
    }
    
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (only rank 0)
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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }
    
    // Broadcast parameters to all ranks
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (Hybrid MPI+OpenMP+CUDA)\n");
        printf("MPI ranks: %d\n", size);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (all ranks need full data initially)
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Broadcast initial data to all ranks
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
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
                MPI_Finalize();
                return 0;
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
