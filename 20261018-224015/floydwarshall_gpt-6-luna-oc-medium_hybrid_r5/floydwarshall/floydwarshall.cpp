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

__global__ void floydStep(unsigned int* dist, unsigned int* path, size_t n, size_t k,
                          size_t firstRow, size_t lastRow) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = firstRow + blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= lastRow || col >= n) return;
    const size_t ij = col * n + row;
    const unsigned int candidate = dist[k * n + row] + dist[col * n + k];
    if (candidate < dist[ij]) {
        dist[ij] = candidate;
        path[ij] = static_cast<unsigned int>(k);
    }
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
    #pragma omp parallel for schedule(static)
    for (size_t index = 0; index < numNodes * numNodes; ++index) {
        path[index] = static_cast<unsigned int>(index / numNodes);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes, int rank, int ranks) {
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    cudaSetDevice(rank % [] { int count=1; cudaGetDeviceCount(&count); return count; }());
    cudaMalloc(&deviceDist, dist.size() * sizeof(unsigned int));
    cudaMalloc(&devicePath, path.size() * sizeof(unsigned int));
    cudaMemcpy(deviceDist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePath, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    const size_t first = numNodes * rank / ranks;
    const size_t last = numNodes * (rank + 1) / ranks;
    for (size_t k = 0; k < numNodes; ++k) {
        const dim3 block(32, 8);
        const dim3 grid((numNodes + block.x - 1) / block.x,
                        (last - first + block.y - 1) / block.y);
        floydStep<<<grid, block>>>(deviceDist, devicePath, numNodes, k, first, last);
        cudaDeviceSynchronize();
        cudaMemcpy(dist.data() + first * numNodes, deviceDist + first * numNodes,
                   (last-first)*numNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(path.data() + first * numNodes, devicePath + first * numNodes,
                   (last-first)*numNodes*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        // Row partitions can differ by rank; broadcast each owner's contiguous row range.
        for (int owner=0; owner<ranks; ++owner) {
            const size_t begin=numNodes*owner/ranks, end=numNodes*(owner+1)/ranks;
            MPI_Bcast(dist.data()+begin*numNodes, static_cast<int>((end-begin)*numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            MPI_Bcast(path.data()+begin*numNodes, static_cast<int>((end-begin)*numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        }
        cudaMemcpy(deviceDist, dist.data(), dist.size()*sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(devicePath, path.data(), path.size()*sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    cudaFree(deviceDist); cudaFree(devicePath);
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
    int rank=0, ranks=1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (rank == 0) printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    if (rank == 0) printf("Number of nodes: %zu\n", numNodes);
    if (rank == 0) printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);
    
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
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
