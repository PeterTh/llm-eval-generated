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
    } while(0)

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax, int rank) {
    unsigned int seed = 42 + rank;
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

// CUDA kernel for Floyd-Warshall inner loop computation
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path, 
                                    const size_t numNodes, const size_t k) {
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

// OpenMP CPU kernel for fallback
void floydWarshallCPU(unsigned int* dist, unsigned int* path,
                      const size_t numNodes, const size_t k) {
    #pragma omp parallel for collapse(2) schedule(dynamic, 32)
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
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
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check if CUDA is available
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    bool useCUDA = (err == cudaSuccess && deviceCount > 0);
    
    if (useCUDA && rank == 0) {
        printf("Using CUDA acceleration with %d device(s)\n", deviceCount);
    }
    
    // Each MPI rank gets its own GPU if available
    if (useCUDA) {
        int device = rank % deviceCount;
        CUDA_CHECK(cudaSetDevice(device));
    }
    
    unsigned int *d_dist = nullptr, *d_path = nullptr;
    const size_t matrixBytes = numNodes * numNodes * sizeof(unsigned int);
    
    if (useCUDA) {
        // Allocate and initialize device memory
        CUDA_CHECK(cudaMalloc(&d_dist, matrixBytes));
        CUDA_CHECK(cudaMalloc(&d_path, matrixBytes));
        CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, path.data(), matrixBytes, cudaMemcpyHostToDevice));
    }
    
    // Setup CUDA parameters
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Main Floyd-Warshall loop
    for (size_t k = 0; k < numNodes; ++k) {
        if (useCUDA) {
            // Execute on GPU
            floydWarshallKernel<<<gridDim, blockDim>>>(d_dist, d_path, numNodes, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            // Execute on CPU with OpenMP
            floydWarshallCPU(dist.data(), path.data(), numNodes, k);
        }
        
        // For multi-node MPI: synchronize every few iterations for correctness
        if (size > 1 && k % 50 == 49) {
            if (useCUDA) {
                CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matrixBytes, cudaMemcpyDeviceToHost));
            }
            MPI_Allreduce(MPI_IN_PLACE, dist.data(), numNodes * numNodes,
                         MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
            if (useCUDA) {
                CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matrixBytes, cudaMemcpyHostToDevice));
            }
        }
    }
    
    // Final copy back and synchronization
    if (useCUDA) {
        CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matrixBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), d_path, matrixBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_dist));
        CUDA_CHECK(cudaFree(d_path));
    }
    
    // Final synchronization across MPI ranks
    if (size > 1) {
        MPI_Allreduce(MPI_IN_PLACE, dist.data(), numNodes * numNodes,
                     MPI_UNSIGNED, MPI_MIN, MPI_COMM_WORLD);
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("MPI Processes: %d\n", size);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    
    // All ranks initialize with same seed (rank 0's seed) for consistency
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, 0);
    initializePathMatrix(path, numNodes);
    
    // Barrier to ensure all ranks are ready
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
