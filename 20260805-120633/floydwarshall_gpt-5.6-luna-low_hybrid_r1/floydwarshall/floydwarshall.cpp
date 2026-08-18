#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
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
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

__global__ void floydKernel(unsigned int* dist, unsigned int* path,
                            const unsigned int* rowK, size_t localRows,
                            size_t n, size_t firstRow, size_t k) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t li = blockIdx.y * blockDim.y + threadIdx.y;
    if (j >= n || li >= localRows) return;
    const size_t i = firstRow + li;
    const unsigned int old = dist[idx2(j, i, n)];
    const unsigned int candidate = dist[idx2(k, i, n)] + rowK[j];
    if (candidate < old) {
        dist[idx2(j, i, n)] = candidate;
        path[idx2(j, i, n)] = static_cast<unsigned int>(k);
    }
}

void checkCuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, size_t first, size_t rows, int rank,
                   const std::vector<int>& counts, const std::vector<int>& displs) {
    unsigned int *dDist = nullptr, *dPath = nullptr, *dRow = nullptr;
    checkCuda(cudaMalloc(&dDist, rows * n * sizeof(unsigned int)), "dist alloc");
    checkCuda(cudaMalloc(&dPath, rows * n * sizeof(unsigned int)), "path alloc");
    checkCuda(cudaMalloc(&dRow, n * sizeof(unsigned int)), "row alloc");
    checkCuda(cudaMemcpy(dDist, dist.data(), rows*n*sizeof(unsigned int), cudaMemcpyHostToDevice), "dist upload");
    checkCuda(cudaMemcpy(dPath, path.data(), rows*n*sizeof(unsigned int), cudaMemcpyHostToDevice), "path upload");
    std::vector<unsigned int> row(n);
    dim3 block(32, 8), grid((n + 31) / 32, (rows + 7) / 8);
    for (size_t k = 0; k < n; ++k) {
        int owner = 0;
        while (owner + 1 < static_cast<int>(counts.size()) && k >= static_cast<size_t>(displs[owner + 1])) ++owner;
        size_t begin = static_cast<size_t>(displs[owner]);
        if (rank == owner) checkCuda(cudaMemcpy(row.data(), dDist + (k-begin), n*sizeof(unsigned int), cudaMemcpyDeviceToHost), "row download");
        MPI_Bcast(row.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        checkCuda(cudaMemcpy(dRow, row.data(), n*sizeof(unsigned int), cudaMemcpyHostToDevice), "row upload");
        floydKernel<<<grid, block>>>(dDist, dPath, dRow, rows, n, first, k);
        checkCuda(cudaGetLastError(), "kernel"); checkCuda(cudaDeviceSynchronize(), "kernel sync");
    }
    checkCuda(cudaMemcpy(dist.data(), dDist, rows*n*sizeof(unsigned int), cudaMemcpyDeviceToHost), "dist download");
    checkCuda(cudaMemcpy(path.data(), dPath, rows*n*sizeof(unsigned int), cudaMemcpyDeviceToHost), "path download");
    cudaFree(dDist); cudaFree(dPath); cudaFree(dRow);
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
    
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX)) { if (rank == 0) fprintf(stderr, "Invalid node count\n"); MPI_Finalize(); return 1; }
    std::vector<int> counts(world), displs(world);
    for (int r = 0; r < world; ++r) { counts[r] = static_cast<int>(numNodes / world + (r < static_cast<int>(numNodes % world))); displs[r] = static_cast<int>(r * (numNodes / world) + std::min<size_t>(r, numNodes % world)); }
    const size_t first = static_cast<size_t>(displs[rank]), rows = static_cast<size_t>(counts[rank]);
    std::vector<unsigned int> globalDist, globalPath;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n", numNodes, validate ? "enabled" : "disabled");
        globalDist.resize(numNodes * numNodes); globalPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(globalPath, numNodes);
    }
    std::vector<unsigned int> dist(rows * numNodes), path(rows * numNodes);
    std::vector<int> elemCounts(world), elemDispls(world);
    for (int r = 0; r < world; ++r) { elemCounts[r] = counts[r] * static_cast<int>(numNodes); elemDispls[r] = displs[r] * static_cast<int>(numNodes); }
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, elemCounts.data(), elemDispls.data(), MPI_UNSIGNED, dist.data(), elemCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalPath.data() : nullptr, elemCounts.data(), elemDispls.data(), MPI_UNSIGNED, path.data(), elemCounts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int devices = 0; checkCuda(cudaGetDeviceCount(&devices), "device count"); if (!devices) { fprintf(stderr, "No CUDA device on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 3); } cudaSetDevice(rank % devices);
    if (rank == 0) printf("Computing shortest paths...\n");
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, first, rows, rank, counts, displs);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    MPI_Gatherv(dist.data(), elemCounts[rank], MPI_UNSIGNED, rank == 0 ? globalDist.data() : nullptr, elemCounts.data(), elemDispls.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        if (rank == 0) print_results_int(globalDist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = rank == 0 && validateResult(globalDist, numNodes);
        
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
