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

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while(0)

// CUDA kernel for Floyd-Warshall
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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
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
        }
    }
    
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        path[idx2(i, i, numNodes)] = i;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get number of GPUs and assign one to each MPI rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    int deviceId = rank % deviceCount;
    CUDA_CHECK(cudaSetDevice(deviceId));
    
    // Allocate device memory
    unsigned int* d_dist;
    unsigned int* d_path;
    const size_t matrixSize = numNodes * numNodes * sizeof(unsigned int);
    
    CUDA_CHECK(cudaMalloc(&d_dist, matrixSize));
    CUDA_CHECK(cudaMalloc(&d_path, matrixSize));
    
    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matrixSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data(), matrixSize, cudaMemcpyHostToDevice));
    
    // Configure CUDA kernel launch parameters
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                 (numNodes + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    // Floyd-Warshall algorithm with hybrid parallelization
    // Strategy: Distribute k-iterations among MPI ranks in blocks
    // Each rank processes a block of k values on its GPU
    const size_t kPerRank = (numNodes + size - 1) / size;
    
    for (size_t kBlock = 0; kBlock < size; ++kBlock) {
        const size_t kStart = kBlock * kPerRank;
        const size_t kEnd = std::min(kStart + kPerRank, numNodes);
        
        // Each rank processes its assigned block of k values
        if (kBlock == (size_t)rank && kStart < numNodes) {
            for (size_t k = kStart; k < kEnd; ++k) {
                // Launch CUDA kernel with OpenMP enabled device
                floydWarshallKernel<<<gridDim, blockDim>>>(d_dist, d_path, numNodes, k);
                CUDA_CHECK(cudaGetLastError());
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Copy updated data back to host
            CUDA_CHECK(cudaMemcpy(dist.data(), d_dist, matrixSize, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(path.data(), d_path, matrixSize, cudaMemcpyDeviceToHost));
        }
        
        // Broadcast the updated matrices to all ranks after this block
        MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, kBlock, MPI_COMM_WORLD);
        MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, kBlock, MPI_COMM_WORLD);
        
        // Update device memory with broadcasted data for next iteration
        if (kBlock != (size_t)rank || kStart >= numNodes) {
            CUDA_CHECK(cudaMemcpy(d_dist, dist.data(), matrixSize, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_path, path.data(), matrixSize, cudaMemcpyHostToDevice));
        }
    }
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
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
        printf("MPI ranks: %d\n", size);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (only rank 0 initializes, then broadcasts)
    if (rank == 0) {
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
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (validate && rank == 0) {
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
    
    MPI_Finalize();
    return 0;
}
