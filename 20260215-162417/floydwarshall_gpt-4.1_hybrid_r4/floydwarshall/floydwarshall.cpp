#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

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

#ifdef __CUDACC__
extern "C" __global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path, const unsigned int* kRow, size_t numNodes, size_t k) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < numNodes && j < numNodes) {
        unsigned int distIJ = dist[j * numNodes + i];
        unsigned int distIK = kRow[i];
        unsigned int distKJ = dist[j * numNodes + k];
        unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            dist[j * numNodes + i] = newDist;
            path[j * numNodes + i] = k;
        }
    }
}
#endif

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    int mpi_rank, mpi_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    size_t rows_per_proc = numNodes / mpi_size;
    size_t extra = numNodes % mpi_size;
    size_t start_row = mpi_rank * rows_per_proc + (mpi_rank < extra ? mpi_rank : extra);
    size_t end_row = start_row + rows_per_proc + (mpi_rank < extra ? 1 : 0);
    size_t local_rows = end_row - start_row;

    // Allocate device memory
    unsigned int *d_dist, *d_path, *d_kRow;
    cudaMalloc(&d_dist, numNodes * local_rows * sizeof(unsigned int));
    cudaMalloc(&d_path, numNodes * local_rows * sizeof(unsigned int));
    cudaMalloc(&d_kRow, numNodes * sizeof(unsigned int));

    // Copy local block to device
    cudaMemcpy(d_dist, &dist[start_row * numNodes], numNodes * local_rows * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_path, &path[start_row * numNodes], numNodes * local_rows * sizeof(unsigned int), cudaMemcpyHostToDevice);

    std::vector<unsigned int> kRow(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        // Each process prepares the k-th row if it owns it
        if (k >= start_row && k < end_row) {
            memcpy(kRow.data(), &dist[k * numNodes], numNodes * sizeof(unsigned int));
        }
        // Broadcast k-th row to all processes
        int owner = 0;
        size_t acc = 0;
        for (int r = 0; r < mpi_size; ++r) {
            size_t rows = numNodes / mpi_size + (r < (int)extra ? 1 : 0);
            if (k >= acc && k < acc + rows) {
                owner = r;
                break;
            }
            acc += rows;
        }
        MPI_Bcast(kRow.data(), numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        // Copy k-th row to device
        cudaMemcpy(d_kRow, kRow.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice);
        // Launch CUDA kernel for local block
        dim3 block(16, 16);
        dim3 grid((numNodes + block.x - 1) / block.x, (local_rows + block.y - 1) / block.y);
        #ifdef __CUDACC__
        floydWarshallKernel<<<grid, block>>>(d_dist, d_path, d_kRow, numNodes, k);
        cudaDeviceSynchronize();
        #endif
    }
    // Copy results back
    cudaMemcpy(&dist[start_row * numNodes], d_dist, numNodes * local_rows * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&path[start_row * numNodes], d_path, numNodes * local_rows * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d_dist);
    cudaFree(d_path);
    cudaFree(d_kRow);
    // Gather results from all processes
    std::vector<unsigned int> dist_global(numNodes * numNodes);
    std::vector<unsigned int> path_global(numNodes * numNodes);
    MPI_Allgather(&dist[start_row * numNodes], numNodes * local_rows, MPI_UNSIGNED, dist_global.data(), numNodes * local_rows, MPI_UNSIGNED, MPI_COMM_WORLD);
    MPI_Allgather(&path[start_row * numNodes], numNodes * local_rows, MPI_UNSIGNED, path_global.data(), numNodes * local_rows, MPI_UNSIGNED, MPI_COMM_WORLD);
    dist.swap(dist_global);
    path.swap(path_global);
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
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (mpi_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (mpi_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (mpi_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    // Initialize
    if (mpi_rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    // Broadcast initial matrices to all processes
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    // Run Floyd-Warshall
    if (mpi_rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(dist, path, numNodes);
    MPI_Barrier(MPI_COMM_WORLD);
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
                MPI_Finalize();
                return 0;
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
