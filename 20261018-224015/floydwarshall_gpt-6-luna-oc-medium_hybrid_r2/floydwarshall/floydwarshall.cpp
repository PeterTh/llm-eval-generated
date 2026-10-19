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

__global__ void fwUpdate(unsigned int* dist, unsigned int* path, size_t n, size_t k,
                         size_t firstRow, size_t rowCount) {
    const size_t offset = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rowCount * n;
    if (offset >= total) return;
    const size_t i = firstRow + offset / n;
    const size_t j = offset % n;
    const unsigned int a = dist[k * n + i];
    const unsigned int b = dist[j * n + k];
    const size_t p = j * n + i;
    const unsigned int candidate = a + b;
    if (candidate < dist[p]) { dist[p] = candidate; path[p] = static_cast<unsigned int>(k); }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const size_t first = numNodes * rank / ranks;
    const size_t last = numNodes * (rank + 1) / ranks;
    const size_t rows = last - first;
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    cudaMalloc(&deviceDist, dist.size() * sizeof(unsigned int));
    cudaMalloc(&devicePath, path.size() * sizeof(unsigned int));
    cudaMemcpy(deviceDist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(devicePath, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    MPI_Datatype rowSlice;
    MPI_Type_vector(static_cast<int>(rows), static_cast<int>(numNodes), static_cast<int>(numNodes), MPI_UNSIGNED, &rowSlice);
    MPI_Type_commit(&rowSlice);
    std::vector<int> counts(ranks, 1), displacements(ranks);
    for (int r = 0; r < ranks; ++r) displacements[r] = static_cast<int>(numNodes * r / ranks);
    constexpr int threads = 256;
    for (size_t k = 0; k < numNodes; ++k) {
        if (rows) fwUpdate<<<static_cast<unsigned>((rows * numNodes + threads - 1) / threads), threads>>>(deviceDist, devicePath, numNodes, k, first, rows);
        cudaDeviceSynchronize();
        cudaMemcpy(dist.data(), deviceDist, dist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        MPI_Allgatherv(dist.data() + first, 1, rowSlice, dist.data(), counts.data(), displacements.data(), rowSlice, MPI_COMM_WORLD);
        cudaMemcpy(deviceDist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(path.data(), devicePath, path.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        MPI_Allgatherv(path.data() + first, 1, rowSlice, path.data(), counts.data(), displacements.data(), rowSlice, MPI_COMM_WORLD);
        cudaMemcpy(devicePath, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    }
    MPI_Type_free(&rowSlice);
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
    int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank);
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
    if (rank == 0) { printf("Number of nodes: %zu\n", numNodes); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
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
    
    floydWarshall(dist, path, numNodes);
    
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
