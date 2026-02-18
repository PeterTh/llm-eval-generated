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
// Original: idx2(j, i, n) used for dist[i][j] implies row-major
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CHECK_CUDA(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // Serial initialization to preserve exact values
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Serial initialization
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    bool valid = true;
    #pragma omp parallel for reduction(&&:valid)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            valid = false;
        }
    }
    
    if (!valid) {
        printf("Validation failed: diagonal elements are not all zero\n");
        return false;
    }
    
    // 2. Triangle inequality
    // Check a sample of paths
    bool triangle_valid = true;
    size_t limit = std::min(numNodes, static_cast<size_t>(10));
    
    #pragma omp parallel for collapse(2) reduction(&&:triangle_valid)
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        triangle_valid = false;
                    }
                }
            }
        }
    }
    
    if (!triangle_valid) {
        printf("Validation failed: triangle inequality violated in sample\n");
        return false;
    }
    
    return true;
}

__global__ void floyd_kernel(unsigned int* dist, unsigned int* path, const unsigned int* k_row_dist, 
                             const int numNodes, const int local_rows, const int k, const int row_offset) {
    int local_i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (local_i < local_rows && j < numNodes) {
        // Access dist[local_i][j] -> dist[local_i * numNodes + j]
        int idx = local_i * numNodes + j;
        
        unsigned int distIJ = dist[idx];
        
        // dist[i][k]
        // i = row_offset + local_i
        // But we have local chunk, so we need dist[local_i][k]
        unsigned int distIK = dist[local_i * numNodes + k];
        
        // dist[k][j] is in k_row_dist[j]
        unsigned int distKJ = k_row_dist[j];
        
        unsigned int newDist = distIK + distKJ;
        
        if (newDist < distIJ) {
            dist[idx] = newDist;
            path[idx] = k;
        }
    }
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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
        }
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall Hybrid Benchmark (MPI + OpenMP + CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI Ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Distribution parameters
    size_t rows_per_proc = numNodes / size;
    size_t remainder = numNodes % size;
    
    // Count and displacement for scatter/gather
    std::vector<int> counts(size);
    std::vector<int> displs(size);
    
    size_t current_disp = 0;
    for (int i = 0; i < size; ++i) {
        size_t rows = rows_per_proc + (i < (int)remainder ? 1 : 0);
        counts[i] = rows * numNodes; // Elements (ints)
        displs[i] = current_disp; // Elements (ints)
        current_disp += rows * numNodes;
    }
    
    size_t my_rows = rows_per_proc + (rank < (int)remainder ? 1 : 0);
    size_t my_elements = my_rows * numNodes;
    size_t my_row_offset = displs[rank] / numNodes;

    // Host memory
    std::vector<unsigned int> h_dist;
    std::vector<unsigned int> h_path;
    
    std::vector<unsigned int> local_dist(my_elements);
    std::vector<unsigned int> local_path(my_elements);
    std::vector<unsigned int> k_row_host(numNodes);

    // Initialization (Rank 0 only)
    if (rank == 0) {
        h_dist.resize(numNodes * numNodes);
        h_path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(h_dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(h_path, numNodes);
    }
    
    // Scatter data
    MPI_Scatterv(rank == 0 ? h_dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 local_dist.data(), my_elements, MPI_UNSIGNED, 
                 0, MPI_COMM_WORLD);
                 
    MPI_Scatterv(rank == 0 ? h_path.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 local_path.data(), my_elements, MPI_UNSIGNED, 
                 0, MPI_COMM_WORLD);
                 
    // Device memory
    unsigned int *d_dist, *d_path, *d_k_row;
    CHECK_CUDA(cudaMalloc(&d_dist, my_elements * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_path, my_elements * sizeof(unsigned int)));
    CHECK_CUDA(cudaMalloc(&d_k_row, numNodes * sizeof(unsigned int)));
    
    CHECK_CUDA(cudaMemcpy(d_dist, local_dist.data(), my_elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_path, local_path.data(), my_elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    dim3 block(32, 16);
    dim3 grid((numNodes + block.x - 1) / block.x, (my_rows + block.y - 1) / block.y);

    for (size_t k = 0; k < numNodes; ++k) {
        // Determine owner of row k
        int owner = -1;
        for (int r = 0; r < size; ++r) {
            size_t start_r = displs[r] / numNodes;
            size_t count_r = counts[r] / numNodes;
            if (k >= start_r && k < start_r + count_r) {
                owner = r;
                break;
            }
        }
        
        if (rank == owner) {
            size_t local_k = k - (displs[rank] / numNodes);
            CHECK_CUDA(cudaMemcpy(k_row_host.data(), d_dist + local_k * numNodes, numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        
        MPI_Bcast(k_row_host.data(), numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        
        CHECK_CUDA(cudaMemcpy(d_k_row, k_row_host.data(), numNodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
        
        if (my_rows > 0) {
            floyd_kernel<<<grid, block>>>(d_dist, d_path, d_k_row, numNodes, my_rows, k, my_row_offset);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy results back
    CHECK_CUDA(cudaMemcpy(local_dist.data(), d_dist, my_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Gather results
    MPI_Gatherv(local_dist.data(), my_elements, MPI_UNSIGNED,
                rank == 0 ? h_dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
                
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        if (printResults) {
            print_results_int(h_dist, "DistanceMatrix");
        }
        
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(h_dist, numNodes);
             if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
    }
    
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_path));
    CHECK_CUDA(cudaFree(d_k_row));
    
    MPI_Finalize();
    return 0;
}
