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

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
}

// CUDA Kernel for Floyd-Warshall update
__global__ void floydWarshallKernel(unsigned int* dist, const unsigned int* kRow, 
                                    const size_t numNodes, const size_t numLocalRows, const size_t rowOffset, const size_t k) {
    const size_t i_local = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i_local < numLocalRows && j < numNodes) {
        // We are processing local row i_local, which corresponds to global row (rowOffset + i_local)
        // We need dist[i][j], dist[i][k], and dist[k][j].
        // dist[i][j] is at local index i_local * numNodes + j.
        // dist[i][k] is at local index i_local * numNodes + k.
        // dist[k][j] is in kRow[j].
        
        unsigned int distIJ = dist[i_local * numNodes + j];
        unsigned int distIK = dist[i_local * numNodes + k];
        unsigned int distKJ = kRow[j];
        
        if (distIK != INF && distKJ != INF) {
             unsigned int newDist = distIK + distKJ;
             if (newDist < distIJ) {
                 dist[i_local * numNodes + j] = newDist;
             }
        }
    }
}

// Initialize part of the matrix owned by this rank
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t rowOffset, const size_t numLocalRows) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    // We generate all numbers to maintain consistency with the serial version's sequence
    // Optimization: Depending on N, this might be slow, but correctness is prioritized.
    // Ideally, we'd skip ahead in the PRNG stream, but rand_r doesn't support that easily.
    
    for (size_t i = 0; i < numNodes; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
             unsigned int val = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
             
             if (i >= rowOffset && i < rowOffset + numLocalRows) {
                 // Store in local buffer
                 // Local row index: i - rowOffset
                 dist[(i - rowOffset) * numNodes + j] = val;
             }
        }
    }

    // Set diagonal to zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (i >= rowOffset && i < rowOffset + numLocalRows) {
            dist[(i - rowOffset) * numNodes + i] = 0;
        }
    }
}


void floydWarshall(std::vector<unsigned int>& dist, 
                   const size_t numNodes, const int rank, const int size,
                   const size_t rowOffset, const size_t numLocalRows) {
    
    // Allocate device memory
    unsigned int *d_dist;
    unsigned int *d_kRow;
    
    size_t distSize = numLocalRows * numNodes * sizeof(unsigned int);
    size_t rowSize = numNodes * sizeof(unsigned int);
    
    CHECK_CUDA(cudaMalloc(&d_dist, distSize));
    CHECK_CUDA(cudaMalloc(&d_kRow, rowSize));
    
    // Copy local distance matrix to device
    CHECK_CUDA(cudaMemcpy(d_dist, dist.data(), distSize, cudaMemcpyHostToDevice));
    
    std::vector<unsigned int> kRowHost(numNodes);
    
    // CUDA block/grid dimensions
    dim3 blockSize(32, 32);
    dim3 gridSize((numNodes + blockSize.x - 1) / blockSize.x, 
                  (numLocalRows + blockSize.y - 1) / blockSize.y);

    for (size_t k = 0; k < numNodes; ++k) {
        // Determine who owns row k
        // Block distribution logic:
        // Rank r owns [start, end)
        // If k is in this range, rank r is owner.
        
        int root = -1;
        // Re-calculate root for k to be safe and sure
        size_t base = numNodes / size;
        size_t rem = numNodes % size;
        size_t current_start = 0;
        for(int r=0; r<size; ++r) {
            size_t count = base + (r < (int)rem ? 1 : 0);
            if (k >= current_start && k < current_start + count) {
                root = r;
                break;
            }
            current_start += count;
        }

        if (root == rank) {
            // I am the owner. Copy row k from device to host.
            // My local index for row k is (k - rowOffset).
            size_t local_k = k - rowOffset;
            CHECK_CUDA(cudaMemcpy(kRowHost.data(), d_dist + local_k * numNodes, rowSize, cudaMemcpyDeviceToHost));
        }
        
        // Broadcast row k
        MPI_Bcast(kRowHost.data(), numNodes, MPI_UNSIGNED, root, MPI_COMM_WORLD);
        
        // Copy broadcasted row to device
        CHECK_CUDA(cudaMemcpy(d_kRow, kRowHost.data(), rowSize, cudaMemcpyHostToDevice));
        
        // Launch kernel
        // Note: Check if gridSize.y is 0 (if numLocalRows is 0 for some rank)
        if (numLocalRows > 0) {
            floydWarshallKernel<<<gridSize, blockSize>>>(d_dist, d_kRow, numNodes, numLocalRows, rowOffset, k);
            CHECK_CUDA(cudaGetLastError());
        }
    }
    
    // Copy back results
    CHECK_CUDA(cudaMemcpy(dist.data(), d_dist, distSize, cudaMemcpyDeviceToHost));
    
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_kRow));
}

// Helper to gather all data to rank 0
void gatherResults(std::vector<unsigned int>& local_dist, std::vector<unsigned int>& global_dist,
                   const size_t numNodes, const int rank, const int size,
                   const size_t numLocalRows) {
    
    if (rank == 0) {
        global_dist.resize(numNodes * numNodes);
    }
    
    // Prepare counts and displacements for Gatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    if (rank == 0) {
        size_t current_offset = 0;
        size_t base = numNodes / size;
        size_t rem = numNodes % size;
        for(int r=0; r<size; ++r) {
            size_t count = base + (r < (int)rem ? 1 : 0);
            recvcounts[r] = count * numNodes; // in elements
            displs[r] = current_offset * numNodes;
            current_offset += count;
        }
    }
    
    MPI_Gatherv(local_dist.data(), numLocalRows * numNodes, MPI_UNSIGNED,
                global_dist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Assign device
    int num_devices = 0;
    cudaGetDeviceCount(&num_devices);
    if (num_devices > 0) {
        int device_id = rank % num_devices;
        cudaSetDevice(device_id);
    }

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
            if (rank == 0) {
                printf("Usage: %s [options]\n", argv[0]);
                printf("Options:\n");
                printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
                printf("  -v           Enable validation\n");
                printf("  -r           Print results for external validation\n");
                printf("  -h           Show this help message\n");
            }
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
    
    // Calculate partitioning
    size_t base = numNodes / size;
    size_t rem = numNodes % size;
    size_t numLocalRows = base + (rank < (int)rem ? 1 : 0);
    size_t rowOffset = rank * base + (rank < (int)rem ? rank : rem);

    // Allocate matrices
    std::vector<unsigned int> dist(numLocalRows * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    // All ranks call initialization to sync RNG (inefficient but correct)
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, rowOffset, numLocalRows);
    
    // Run Floyd-Warshall
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, numNodes, rank, size, rowOffset, numLocalRows);
    
    auto end = std::chrono::high_resolution_clock::now();
    long long local_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long long duration_ms = 0;
    MPI_Reduce(&local_duration_ms, &duration_ms, 1, MPI_LONG_LONG_INT, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Computation time: %lld ms\n", duration_ms);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Gather results for validation or printing
    if (printResults || validate) {
        std::vector<unsigned int> global_dist;
        gatherResults(dist, global_dist, numNodes, rank, size, numLocalRows);
        
        if (rank == 0) {
             if (printResults) {
                print_results_int(global_dist, "DistanceMatrix");
             }
             
             if (validate) {
                printf("Validating result...\n");
                bool valid = true;
                
                // 1. Diagonal should be zero
                for (size_t i = 0; i < numNodes; ++i) {
                    if (global_dist[i * numNodes + i] != 0) {
                        printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                        valid = false;
                        break;
                    }
                }
                
                if (valid) {
                    // 2. Triangle inequality
                    // Check a sample of paths in parallel using OpenMP
                    #pragma omp parallel for reduction(&&:valid)
                    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
                        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
                            for (size_t k = 0; k < numNodes; ++k) {
                                const unsigned int distIJ = global_dist[i * numNodes + j];
                                const unsigned int distIK = global_dist[i * numNodes + k];
                                const unsigned int distKJ = global_dist[k * numNodes + j];
                                
                                if (distIK < INF && distKJ < INF) {
                                    if (distIK + distKJ < distIJ) {
                                        valid = false;
                                    }
                                }
                            }
                        }
                    }
                }

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
             }
        }
    }
    
    MPI_Finalize();
    return 0;
}
