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

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // For each source node i
        for (size_t i = 0; i < numNodes; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
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
    MPI_Init(&argc, &argv);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
    
    if (world_rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate matrices (distributed row blocks)
    size_t rows_per_rank = numNodes / world_size;
    size_t extra = numNodes % world_size;
    size_t my_rows = rows_per_rank + (world_rank < extra ? 1 : 0);
    size_t my_row_start = world_rank * rows_per_rank + std::min((size_t)world_rank, extra);

    std::vector<unsigned int> dist(my_rows * numNodes);
    std::vector<unsigned int> path(my_rows * numNodes);

    // Only rank 0 initializes the full matrix
    std::vector<unsigned int> full_dist, full_path;
    if (world_rank == 0) {
        full_dist.resize(numNodes * numNodes);
        full_path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(full_path, numNodes);
    }

    // Scatter the distance and path matrices
    std::vector<int> sendcounts(world_size), displs(world_size);
    for (int r = 0, off = 0; r < world_size; ++r) {
        size_t r_rows = rows_per_rank + (r < extra ? 1 : 0);
        sendcounts[r] = r_rows * numNodes;
        displs[r] = off;
        off += sendcounts[r];
    }
    MPI_Scatterv(full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 dist.data(), my_rows * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(full_path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                 path.data(), my_rows * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    // Run Floyd-Warshall (hybrid parallel)
    if (world_rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Hybrid parallel region (OpenMP + CUDA offload)
    for (size_t k = 0; k < numNodes; ++k) {
        // Broadcast k-th row to all ranks
        std::vector<unsigned int> k_row(numNodes);
        if (my_row_start <= k && k < my_row_start + my_rows) {
            memcpy(k_row.data(), &dist[(k - my_row_start) * numNodes], numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(k_row.data(), numNodes, MPI_UNSIGNED, (int)std::min(k / rows_per_rank, (size_t)world_size-1), MPI_COMM_WORLD);

        // OpenMP parallel for over my_rows
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < my_rows; ++i) {
            // CUDA kernel offload for innermost loop
            // (see below for kernel launch)
            floydwarshall_cuda_kernel(&dist[i * numNodes], k_row.data(), &path[i * numNodes], numNodes, k);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather results to rank 0
    if (world_rank == 0) {
        full_dist.swap(dist);
        full_path.swap(path);
    }
    MPI_Gatherv(dist.data(), my_rows * numNodes, MPI_UNSIGNED,
                full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(path.data(), my_rows * numNodes, MPI_UNSIGNED,
                full_path.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        if (printResults) {
            print_results_int(full_dist, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full_dist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    MPI_Finalize();
    return 0;
}

// CUDA kernel for innermost loop (to be implemented below)
void floydwarshall_cuda_kernel(unsigned int* dist_row, const unsigned int* k_row, unsigned int* path_row, size_t numNodes, size_t k) {
    // Launch CUDA kernel for the innermost loop (j)
    // Each thread handles one j
    // This is a stub; actual kernel implementation is below
    floydwarshall_cuda_kernel_impl<<<(numNodes+255)/256, 256>>>(dist_row, k_row, path_row, numNodes, k);
    cudaDeviceSynchronize();
}

__global__ void floydwarshall_cuda_kernel_impl(unsigned int* dist_row, const unsigned int* k_row, unsigned int* path_row, size_t numNodes, size_t k) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < numNodes) {
        unsigned int distIJ = dist_row[j];
        unsigned int distIK = k_row[j];
        unsigned int distKJ = dist_row[k];
        unsigned int newDist = distIK + distKJ;
        if (newDist < distIJ) {
            dist_row[j] = newDist;
            path_row[j] = k;
        }
    }
}

