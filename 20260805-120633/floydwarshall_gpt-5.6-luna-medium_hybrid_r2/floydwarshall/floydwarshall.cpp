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
__host__ __device__ inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Keep the original serial RNG stream exactly intact.  The diagonal pass below
    // is independent and is parallelized with OpenMP.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(numNodes); ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

__global__ void floydUpdate(unsigned int* dist, unsigned int* path, size_t n,
                            size_t firstSource, size_t sourceCount, size_t k) {
    const size_t localSource = blockIdx.x * blockDim.x + threadIdx.x;
    if (localSource >= sourceCount) return;
    const size_t i = firstSource + localSource;
    const unsigned int distIK = dist[idx2(k, i, n)];
    for (size_t j = 0; j < n; ++j) {
        const unsigned int candidate = distIK + dist[idx2(j, k, n)];
        const size_t p = idx2(j, i, n);
        if (candidate < dist[p]) {
            dist[p] = candidate;
            path[p] = static_cast<unsigned int>(k);
        }
    }
}

// MPI exchanges contiguous source ranges. These kernels convert to/from the
// benchmark's column-major (source-contiguous) representation on the device.
__global__ void packSources(const unsigned int* matrix, unsigned int* packed,
                            size_t n, size_t firstSource, size_t sourceCount) {
    const size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = n * sourceCount;
    if (q >= total) return;
    const size_t j = q / sourceCount;
    const size_t local = q % sourceCount;
    packed[q] = matrix[idx2(j, firstSource + local, n)];
}

__global__ void unpackSources(unsigned int* matrix, const unsigned int* packed,
                              size_t n, size_t firstSource, size_t sourceCount) {
    const size_t q = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = n * sourceCount;
    if (q >= total) return;
    const size_t j = q / sourceCount;
    const size_t local = q % sourceCount;
    matrix[idx2(j, firstSource + local, n)] = packed[q];
}

static void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t n, const int rank, const int ranks) {
    const size_t first = (n * static_cast<size_t>(rank)) / static_cast<size_t>(ranks);
    const size_t last = (n * static_cast<size_t>(rank + 1)) / static_cast<size_t>(ranks);
    const size_t local = last - first;
    const size_t elements = n * n;
    const size_t localElements = n * local;
    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "querying CUDA devices");
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA accelerator is available for MPI rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(rank % deviceCount), "selecting CUDA device");
    int* counts = new int[ranks];
    int* displacements = new int[ranks];
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = (n * static_cast<size_t>(r)) / static_cast<size_t>(ranks);
        const size_t end = (n * static_cast<size_t>(r + 1)) / static_cast<size_t>(ranks);
        counts[r] = static_cast<int>(n * (end - begin));
        displacements[r] = static_cast<int>(n * begin);
    }

    unsigned int *dDist = nullptr, *dPath = nullptr, *dLocal = nullptr, *dGlobal = nullptr;
    cudaCheck(cudaMalloc(&dDist, elements * sizeof(unsigned int)), "allocating distance matrix");
    cudaCheck(cudaMalloc(&dPath, elements * sizeof(unsigned int)), "allocating path matrix");
    cudaCheck(cudaMalloc(&dLocal, std::max<size_t>(1, localElements) * sizeof(unsigned int)), "allocating exchange buffer");
    cudaCheck(cudaMalloc(&dGlobal, elements * sizeof(unsigned int)), "allocating receive buffer");
    cudaCheck(cudaMemcpy(dDist, dist.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice), "copying distance matrix");
    cudaCheck(cudaMemcpy(dPath, path.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice), "copying path matrix");
    std::vector<unsigned int> localPacked(std::max<size_t>(1, localElements));
    std::vector<unsigned int> localPathPacked(std::max<size_t>(1, localElements));
    std::vector<unsigned int> globalPacked(elements);

    constexpr int threads = 256;
    for (size_t k = 0; k < n; ++k) {
        if (local != 0) {
            floydUpdate<<<static_cast<unsigned int>((local + threads - 1) / threads), threads>>>(dDist, dPath, n, first, local, k);
            cudaCheck(cudaGetLastError(), "launching Floyd-Warshall kernel");
            cudaCheck(cudaDeviceSynchronize(), "completing Floyd-Warshall kernel");
        }
        if (localElements != 0) {
            const unsigned int blocks = static_cast<unsigned int>((localElements + threads - 1) / threads);
            packSources<<<blocks, threads>>>(dDist, dLocal, n, first, local);
            cudaCheck(cudaGetLastError(), "packing distance matrix");
            cudaCheck(cudaMemcpy(localPacked.data(), dLocal, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copying distance exchange");
            packSources<<<blocks, threads>>>(dPath, dLocal, n, first, local);
            cudaCheck(cudaGetLastError(), "packing path matrix");
            cudaCheck(cudaMemcpy(localPathPacked.data(), dLocal, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copying path exchange");
        }
        // MPI receives contiguous source ranges, so this works with ordinary MPI
        // implementations without requiring CUDA-aware MPI.
        MPI_Allgatherv(localPacked.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                       globalPacked.data(), counts, displacements, MPI_UNSIGNED, MPI_COMM_WORLD);
        for (int r = 0; r < ranks; ++r) {
            const size_t begin = static_cast<size_t>(displacements[r]) / n;
            const size_t rows = static_cast<size_t>(counts[r]) / n;
            if (rows == 0) continue;
            cudaCheck(cudaMemcpy(dGlobal, globalPacked.data() + displacements[r], counts[r] * sizeof(unsigned int), cudaMemcpyHostToDevice), "copying gathered distances");
            const unsigned int blocks = static_cast<unsigned int>((static_cast<size_t>(counts[r]) + threads - 1) / threads);
            unpackSources<<<blocks, threads>>>(dDist, dGlobal, n, begin, rows);
        }
        MPI_Allgatherv(localPathPacked.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                       globalPacked.data(), counts, displacements, MPI_UNSIGNED, MPI_COMM_WORLD);
        for (int r = 0; r < ranks; ++r) {
            const size_t begin = static_cast<size_t>(displacements[r]) / n;
            const size_t rows = static_cast<size_t>(counts[r]) / n;
            if (rows == 0) continue;
            cudaCheck(cudaMemcpy(dGlobal, globalPacked.data() + displacements[r], counts[r] * sizeof(unsigned int), cudaMemcpyHostToDevice), "copying gathered paths");
            const unsigned int blocks = static_cast<unsigned int>((static_cast<size_t>(counts[r]) + threads - 1) / threads);
            unpackSources<<<blocks, threads>>>(dPath, dGlobal, n, begin, rows);
        }
        cudaCheck(cudaDeviceSynchronize(), "updating gathered distances");
    }
    cudaCheck(cudaMemcpy(dist.data(), dDist, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copying final distances");
    cudaCheck(cudaMemcpy(path.data(), dPath, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost), "copying final paths");
    cudaFree(dDist); cudaFree(dPath); cudaFree(dLocal); cudaFree(dGlobal);
    delete[] counts; delete[] displacements;
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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n", ranks, omp_get_max_threads());
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
    const double start = MPI_Wtime();
    
    floydWarshall(dist, path, numNodes, rank, ranks);
    
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    if (rank == 0) {
        double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        if (rank == 0) print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
        if (valid) {
            if (rank == 0) printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Finalize();
    return 0;
}
