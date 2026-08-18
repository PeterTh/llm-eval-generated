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

__global__ void floydKernel(unsigned int* dist, unsigned int* path,
                            const unsigned int* pivot, size_t n, size_t localN,
                            size_t k) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= localN || j >= n) return;
    const size_t at = j * localN + i;
    const unsigned int candidate = dist[k * localN + i] + pivot[j];
    if (candidate < dist[at]) {
        dist[at] = candidate;
        path[at] = static_cast<unsigned int>(k);
    }
}

static void cudaCheck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    #pragma omp parallel for
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
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
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
    int rank = 0, ranks = 1;
    MPI_Init(&argc, &argv);
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
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", numNodes, validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    const size_t base = numNodes / static_cast<size_t>(ranks);
    const size_t rem = numNodes % static_cast<size_t>(ranks);
    const size_t first = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
    const size_t localN = base + (static_cast<size_t>(rank) < rem ? 1 : 0);
    std::vector<unsigned int> dist(numNodes * localN), path(numNodes * localN);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    unsigned int seed = 42;
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) for (size_t li = 0; li < localN; ++li) {
        const size_t i = first + li;
        unsigned int x = seed ^ static_cast<unsigned int>((j * numNodes + i) * 747796405u + 2891336453u);
        x ^= x >> 16; x *= 2246822519u; x ^= x >> 13;
        dist[j * localN + li] = (i == j) ? 0u : 1u + x % (MAX_DISTANCE);
        path[j * localN + li] = static_cast<unsigned int>(j);
    }
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    cudaCheck(cudaMalloc(&dDist, dist.size() * sizeof(unsigned int)), "dist allocation");
    cudaCheck(cudaMalloc(&dPath, path.size() * sizeof(unsigned int)), "path allocation");
    cudaCheck(cudaMalloc(&dPivot, numNodes * sizeof(unsigned int)), "pivot allocation");
    cudaCheck(cudaMemcpy(dDist, dist.data(), dist.size()*sizeof(unsigned int), cudaMemcpyHostToDevice), "dist upload");
    cudaCheck(cudaMemcpy(dPath, path.data(), path.size()*sizeof(unsigned int), cudaMemcpyHostToDevice), "path upload");
    std::vector<unsigned int> pivot(numNodes);
    auto start = std::chrono::high_resolution_clock::now();
    
    for (size_t k = 0; k < numNodes; ++k) {
        int source = 0;
        for (int r = 0; r < ranks; ++r) {
            const size_t rf = static_cast<size_t>(r)*base + std::min(static_cast<size_t>(r), rem);
            const size_t rn = base + (static_cast<size_t>(r) < rem ? 1 : 0);
            if (k >= rf && k < rf + rn) { source = r; break; }
        }
        if (rank == source) {
            const size_t localK = k - (static_cast<size_t>(source)*base + std::min(static_cast<size_t>(source), rem));
            for (size_t j = 0; j < numNodes; ++j)
                cudaCheck(cudaMemcpy(&pivot[j], dDist + j*localN + localK, sizeof(unsigned int), cudaMemcpyDeviceToHost), "pivot download");
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, source, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dPivot, pivot.data(), numNodes*sizeof(unsigned int), cudaMemcpyHostToDevice), "pivot upload");
        dim3 block(32, 8), grid((localN + block.x - 1)/block.x, (numNodes + block.y - 1)/block.y);
        floydKernel<<<grid, block>>>(dDist, dPath, dPivot, numNodes, localN, k);
        cudaCheck(cudaGetLastError(), "kernel launch");
    }
    cudaCheck(cudaDeviceSynchronize(), "kernel completion");
    cudaCheck(cudaMemcpy(dist.data(), dDist, dist.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost), "dist download");
    cudaCheck(cudaMemcpy(path.data(), dPath, path.size()*sizeof(unsigned int), cudaMemcpyDeviceToHost), "path download");
    cudaFree(dDist); cudaFree(dPath); cudaFree(dPivot);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        std::vector<unsigned int> full(numNodes*numNodes);
        std::vector<int> counts(ranks), displs(ranks);
        for (int r=0;r<ranks;++r) { size_t rn=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(rn*numNodes); displs[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem))*numNodes); }
        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED, full.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) print_results_int(full, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        std::vector<unsigned int> full(numNodes*numNodes);
        std::vector<int> counts(ranks), displs(ranks);
        for (int r=0;r<ranks;++r) { size_t rn=base+(static_cast<size_t>(r)<rem); counts[r]=static_cast<int>(rn*numNodes); displs[r]=static_cast<int>((static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem))*numNodes); }
        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED, full.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        bool valid = rank == 0 && validateResult(full, numNodes);
        int allValid = valid ? 1 : 0; MPI_Bcast(&allValid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        valid = allValid != 0;
        
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
