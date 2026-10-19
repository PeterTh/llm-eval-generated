#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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
    #pragma omp parallel for
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

__global__ void updateRows(unsigned int* dist, unsigned int* path,
                           const unsigned int* pivot, size_t n, size_t rows,
                           size_t k) {
    const size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t cells = rows * n;
    if (q >= cells) return;
    const size_t r = q / n, j = q % n;
    const unsigned int candidate = dist[r * n + k] + pivot[j];
    if (candidate < dist[q]) {
        dist[q] = candidate;
        path[q] = static_cast<unsigned int>(k);
    }
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, size_t start, size_t rows, int rank, int ranks) {
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    checkCuda(cudaMalloc(&dDist, dist.size() * sizeof(unsigned int)));
    checkCuda(cudaMalloc(&dPath, path.size() * sizeof(unsigned int)));
    checkCuda(cudaMalloc(&dPivot, n * sizeof(unsigned int)));
    checkCuda(cudaMemcpy(dDist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(dPath, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    std::vector<unsigned int> pivot(n);
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k * static_cast<size_t>(ranks) / n);
        if (rank == owner) {
            const size_t local = k - start;
            checkCuda(cudaMemcpy(pivot.data(), dDist + local * n, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(dPivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        const size_t cells = rows * n;
        if (cells) {
            updateRows<<<static_cast<unsigned int>((cells + 255) / 256), 256>>>(dDist, dPath, dPivot, n, rows, k);
            checkCuda(cudaGetLastError());
        }
    }
    checkCuda(cudaDeviceSynchronize());
    checkCuda(cudaMemcpy(dist.data(), dDist, dist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(path.data(), dPath, path.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
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
    
    if (numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) fprintf(stderr, "Invalid number of nodes\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    const size_t startRow = numNodes * static_cast<size_t>(rank) / ranks;
    const size_t endRow = numNodes * static_cast<size_t>(rank + 1) / ranks;
    const size_t localRows = endRow - startRow;
    std::vector<unsigned int> dist(localRows * numNodes), path(localRows * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    std::vector<unsigned int> fullDist, fullPath;
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t first = numNodes * static_cast<size_t>(r) / ranks;
        const size_t last = numNodes * static_cast<size_t>(r + 1) / ranks;
        counts[r] = static_cast<int>((last - first) * numNodes);
        displacements[r] = static_cast<int>(first * numNodes);
    }
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes); fullPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullPath.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 path.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    checkCuda(cudaSetDevice(rank % deviceCount));
    floydWarshall(dist, path, numNodes, startRow, localRows, rank, ranks);
    std::vector<unsigned int> resultDist, resultPath;
    if (rank == 0) { resultDist.resize(numNodes * numNodes); resultPath.resize(numNodes * numNodes); }
    MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? resultDist.data() : nullptr,
                counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(path.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? resultPath.data() : nullptr,
                counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    if (rank != 0) { MPI_Finalize(); return 0; }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        print_results_int(resultDist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(resultDist, numNodes);
        
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
