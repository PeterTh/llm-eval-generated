#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#ifdef USE_MPI
#include <mpi.h>
#endif
#ifdef USE_OPENMP
#include <omp.h>
#endif
#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

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

// CUDA kernel for a single k iteration
#ifdef USE_CUDA
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path, const unsigned int* kRow, size_t numNodes, size_t k) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < numNodes && j < numNodes) {
        size_t idx = j * numNodes + i;
        unsigned int distIJ = dist[idx];
        unsigned int distIK = kRow[i];
        unsigned int distKJ = dist[j * numNodes + k];
        unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
}
#endif

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
#ifdef USE_MPI
    int mpi_rank = 0, mpi_size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
    size_t rows_per_proc = numNodes / mpi_size;
    size_t row_start = mpi_rank * rows_per_proc;
    size_t row_end = (mpi_rank == mpi_size - 1) ? numNodes : row_start + rows_per_proc;
#else
    size_t row_start = 0, row_end = numNodes;
#endif

#ifdef USE_CUDA
    unsigned int* d_dist;
    unsigned int* d_path;
    unsigned int* d_kRow;
    cudaMalloc(&d_dist, numNodes * numNodes * sizeof(unsigned int));
    cudaMalloc(&d_path, numNodes * numNodes * sizeof(unsigned int));
    cudaMalloc(&d_kRow, numNodes * sizeof(unsigned int));
    cudaMemcpy(d_dist, dist.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, path.data(), numNodes * numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
    std::vector<unsigned int> kRow(numNodes);
#endif

    for (size_t k = 0; k < numNodes; ++k) {
#ifdef USE_MPI
        // Broadcast k-th row from owner to all
        size_t owner = k / rows_per_proc;
        if (mpi_rank == owner) {
            memcpy(&kRow[0], &dist[k * numNodes], numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(&kRow[0], numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
#else
        std::vector<unsigned int> kRow(numNodes);
        memcpy(&kRow[0], &dist[k * numNodes], numNodes * sizeof(unsigned int));
#endif
#ifdef USE_CUDA
        cudaMemcpy(d_kRow, kRow.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
        dim3 block(16, 16);
        dim3 grid((numNodes + block.x - 1) / block.x, (numNodes + block.y - 1) / block.y);
        floydWarshallKernel<<<grid, block>>>(d_dist, d_path, d_kRow, numNodes, k);
        cudaDeviceSynchronize();
#else
        #pragma omp parallel for schedule(static)
        for (size_t i = row_start; i < row_end; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = kRow[i];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                const unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
#endif
#ifdef USE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
#ifdef USE_CUDA
    cudaMemcpy(dist.data(), d_dist, numNodes * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(path.data(), d_path, numNodes * numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
    cudaFree(d_path);
    cudaFree(d_kRow);
#endif
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
#ifdef USE_MPI
    MPI_Init(&argc, &argv);
#endif
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
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
#ifdef USE_MPI
            MPI_Finalize();
#endif
            return 1;
        }
    }
    
    int mpi_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
#endif
    if (mpi_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    // Initialize
    if (mpi_rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    // Run Floyd-Warshall
    if (mpi_rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (mpi_rank == 0) {
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
            }
        }
    }
#ifdef USE_MPI
    MPI_Finalize();
#endif
    return 0;
}
