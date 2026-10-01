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

__global__ void fwKernel(unsigned int* dist, unsigned int* path,
                         const unsigned int* pivot, size_t n,
                         size_t first, size_t count, size_t k) {
    const size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count * n) return;
    const size_t i = first + q / n;
    const size_t j = q % n;
    const unsigned int candidate = dist[k * n + i] + pivot[j];
    const size_t at = j * n + i;
    if (candidate < dist[at]) {
        dist[at] = candidate;
        path[at] = static_cast<unsigned int>(k);
    }
}

static void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
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
    #pragma omp parallel for schedule(static)
    for (long long q = 0; q < static_cast<long long>(numNodes * numNodes); ++q)
        path[static_cast<size_t>(q)] = static_cast<unsigned int>(static_cast<size_t>(q) % numNodes);
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes, int rank, int ranks) {
    const size_t first = numNodes * static_cast<size_t>(rank) / ranks;
    const size_t last = numNodes * static_cast<size_t>(rank + 1) / ranks;
    const size_t count = last - first;
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    cudaCheck(cudaMalloc(&dDist, dist.size() * sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&dPath, path.size() * sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&dPivot, numNodes * sizeof(unsigned int)));
    cudaCheck(cudaMemcpy(dDist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dPath, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    std::vector<unsigned int> pivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = static_cast<int>(((k + 1) * static_cast<size_t>(ranks) - 1) / numNodes);
        if (rank == owner)
            for (size_t j = 0; j < numNodes; ++j) pivot[j] = dist[idx2(j, k, numNodes)];
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dPivot, pivot.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (count) {
            const size_t total = count * numNodes;
            fwKernel<<<static_cast<unsigned int>((total + 255) / 256), 256>>>(dDist, dPath, dPivot, numNodes, first, count, k);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaDeviceSynchronize());
            // Keep this rank's rows current on the host for pivot broadcasts.
            for (size_t j = 0; j < numNodes; ++j)
                if (count) cudaCheck(cudaMemcpy(dist.data() + j * numNodes + first,
                    dDist + j * numNodes + first, count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
    }
    for (size_t j = 0; j < numNodes; ++j)
        if (count) cudaCheck(cudaMemcpy(path.data() + j * numNodes + first,
            dPath + j * numNodes + first, count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    cudaFree(dDist); cudaFree(dPath); cudaFree(dPivot);
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices <= 0) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaCheck(cudaSetDevice(rank % devices));
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
    
    if (rank == 0) {
      printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
      printf("Number of nodes: %zu\n", numNodes);
      printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
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
    double localSeconds = std::chrono::duration<double>(end - start).count(), maxSeconds = 0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    auto duration = std::chrono::milliseconds(static_cast<long long>(maxSeconds * 1000));
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        for (size_t i = 0; i < numNodes; ++i) {
            const int owner = static_cast<int>(((i + 1) * static_cast<size_t>(ranks) - 1) / numNodes);
            std::vector<unsigned int> row(numNodes);
            if (rank == owner) for (size_t j = 0; j < numNodes; ++j) row[j] = dist[j * numNodes + i];
            MPI_Bcast(row.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            if (rank == 0) for (size_t j = 0; j < numNodes; ++j) dist[j * numNodes + i] = row[j];
        }
        if (rank == 0) print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        // Validation is performed on rank zero after collecting distributed rows.
        for (size_t i = 0; i < numNodes; ++i) {
            const int owner = static_cast<int>(((i + 1) * static_cast<size_t>(ranks) - 1) / numNodes);
            std::vector<unsigned int> row(numNodes);
            if (rank == owner) for (size_t j = 0; j < numNodes; ++j) row[j] = dist[j * numNodes + i];
            MPI_Bcast(row.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            if (rank == 0) for (size_t j = 0; j < numNodes; ++j) dist[j * numNodes + i] = row[j];
        }
    }
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
    
    MPI_Finalize();
    return 0;
}
