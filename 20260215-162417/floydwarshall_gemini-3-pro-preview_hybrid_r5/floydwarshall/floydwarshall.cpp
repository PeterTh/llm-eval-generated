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

// CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            MPI_Abort(MPI_COMM_WORLD, 1); \
        } \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

__global__ void floydWarshallKernel(unsigned int* d_dist, unsigned int* d_path, 
                                    const unsigned int* d_kRow_dist, 
                                    const size_t numNodes, const size_t localRows, 
                                    const size_t startRow, const size_t k) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < localRows && j < numNodes) {
        size_t global_i = startRow + i;
        
        // d_dist is local chunk, size localRows * numNodes
        // Indices in d_dist: row i (local) is i * numNodes + j
        size_t idx = i * numNodes + j;
        
        unsigned int distIK = d_dist[i * numNodes + k]; // Local (i, k)
        unsigned int distKJ = d_kRow_dist[j];           // Broadcasted k-th row (k, j)
        unsigned int distIJ = d_dist[idx];              // Local (i, j)

        unsigned int newDist = distIK + distKJ;

        if (newDist < distIJ) {
            d_dist[idx] = newDist;
            d_path[idx] = (unsigned int)k;
        }
    }
}


// Forward declarations
void printUsage(const char* progName);
bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes);

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        unsigned int seed = 42 + (unsigned int)i; // Deterministic seed per element for parallel init
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for collapse(2)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j; 
            path[idx2(j, i, numNodes)] = i;
        }
    }
    // Set diagonal (redundant but safe)
     #pragma omp parallel for
     for (size_t j = 0; j < numNodes; ++j) {
        path[idx2(j, j, numNodes)] = j;
     }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes,
                   int rank, int size) {
    
    // 1D Row Block Distribution
    size_t rowsPerRank = numNodes / size;
    size_t remainder = numNodes % size;
    size_t startRow = rank * rowsPerRank + std::min((size_t)rank, remainder);
    size_t localRows = rowsPerRank + (static_cast<size_t>(rank) < remainder ? 1 : 0);

    // Allocate device memory
    unsigned int *d_dist, *d_path, *d_kRow_dist;
    CUDA_CHECK(cudaMalloc(&d_dist, localRows * numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path, localRows * numNodes * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_kRow_dist, numNodes * sizeof(unsigned int)));

    // Copy local portion to device
    // Since dist/path on host are full matrices on rank 0 (or all if we duplicate), 
    // we need to be careful. The input 'dist' here is the FULL matrix on all nodes 
    // (based on original code structure which I'm modifying).
    // To save memory, we should have only allocated the local part, but I'll assume 
    // the vector passed in is valid. If it's the full vector, we copy our slice.
    
    // Wait, if I change the signature/allocation, validation might break.
    // The original main allocates full 'dist'. I will use that.
    
    // Correct logic:
    // 1. Copy OUR rows from host dist to device d_dist.
    //    The input `dist` is a flat vector. Row `i` starts at `i * numNodes` (if idx2 is i*n+j).
    //    idx2(j, i, n) = i * n + j. Yes, Row-Major.
    
    CUDA_CHECK(cudaMemcpy(d_dist, &dist[startRow * numNodes], 
                         localRows * numNodes * sizeof(unsigned int), 
                         cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, &path[startRow * numNodes], 
                         localRows * numNodes * sizeof(unsigned int), 
                         cudaMemcpyHostToDevice));

    std::vector<unsigned int> kRowBuffer(numNodes);

    // Loop k
    for (size_t k = 0; k < numNodes; ++k) {
        // Owner of row k broadcasts it
        int owner = -1;
        // Find owner: simplest is iterate ranks (or calculate if regular)
        // Since distribution is slightly irregular (remainder), let's calc.
        // Fast calc:
        // if k < remainder * (rowsPerRank + 1) -> owner = k / (rowsPerRank + 1)
        // else -> owner = remainder + (k - remainder * (rowsPerRank + 1)) / rowsPerRank
        
        if (k < remainder * (rowsPerRank + 1)) {
            owner = k / (rowsPerRank + 1);
        } else {
            owner = remainder + (k - remainder * (rowsPerRank + 1)) / rowsPerRank;
        }

        if (rank == owner) {
            // My local index for global row k
            size_t local_k = k - (startRow); // startRow is start of THIS rank
            
            // On GPU, we need to copy row 'k' to d_kRow_dist
            // But we can't easily copy device->device across MPI without CUDA-aware MPI.
            // Safe fallback: Device -> Host -> Bcast -> Device
            
            CUDA_CHECK(cudaMemcpy(kRowBuffer.data(), 
                                  d_dist + local_k * numNodes, // pointer arithmetic
                                  numNodes * sizeof(unsigned int), 
                                  cudaMemcpyDeviceToHost));
        }

        MPI_Bcast(kRowBuffer.data(), numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Copy broadcasted row to device
        CUDA_CHECK(cudaMemcpy(d_kRow_dist, kRowBuffer.data(), 
                             numNodes * sizeof(unsigned int), 
                             cudaMemcpyHostToDevice));

        // Launch Kernel
        dim3 block(32, 16);
        dim3 grid((numNodes + block.x - 1) / block.x, 
                  (localRows + block.y - 1) / block.y);
        
        floydWarshallKernel<<<grid, block>>>(d_dist, d_path, d_kRow_dist, 
                                             numNodes, localRows, startRow, k);
        // No sync needed here, implicit in next copy/bcast or loop
    }
    
    // Copy back result
    CUDA_CHECK(cudaMemcpy(&dist[startRow * numNodes], d_dist, 
                         localRows * numNodes * sizeof(unsigned int), 
                         cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&path[startRow * numNodes], d_path, 
                         localRows * numNodes * sizeof(unsigned int), 
                         cudaMemcpyDeviceToHost));
    
    // Gather results to Rank 0 for validation/output if needed
    // The main function expects `dist` to be fully populated on the measuring node (likely 0).
    // So we use MPI_Allgatherv or just Gatherv to 0.
    
    // We need displacements and counts for Gatherv
    std::vector<int> recvcounts(size);
    std::vector<int> displs(size);
    
    for(int r=0; r<size; ++r) {
        size_t r_rows = rowsPerRank + (static_cast<size_t>(r) < remainder ? 1 : 0);
        recvcounts[r] = r_rows * numNodes; // elements
        // Displ is sum of previous counts
        displs[r] = (r == 0) ? 0 : displs[r-1] + recvcounts[r-1]; 
    }
    
    // MPI_Allgatherv to update everyone's full matrix (simplest for validation correctness)
    // In a real huge benchmark, we wouldn't do this, but here validation runs on the full matrix.
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                   dist.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);
                   
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                   path.data(), recvcounts.data(), displs.data(), MPI_UNSIGNED, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_kRow_dist));
}

// Keep validateResult, printUsage as is...
bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    bool diagFailed = false;
    #pragma omp parallel for shared(diagFailed)
    for (size_t i = 0; i < numNodes; ++i) {
        if (!diagFailed && dist[idx2(i, i, numNodes)] != 0) {
            #pragma omp critical
            {
                if (!diagFailed) {
                    printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                    diagFailed = true;
                }
            }
        }
    }
    if (diagFailed) return false;
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    bool triFailed = false;
    const size_t checkLimit = std::min(numNodes, static_cast<size_t>(10));
    
    #pragma omp parallel for collapse(3) shared(triFailed)
    for (size_t i = 0; i < checkLimit; ++i) {
        for (size_t j = 0; j < checkLimit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                if (triFailed) continue;
                
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        #pragma omp critical
                        {
                            if (!triFailed) {
                                printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                                       i, j, k);
                                triFailed = true;
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (triFailed) return false;
    
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
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Set CUDA device
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices > 0) {
        cudaSetDevice(rank % numDevices);
    } else {
        if (rank == 0) printf("No CUDA devices found!\n");
        MPI_Finalize();
        return 1;
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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } 
        // Note: Ignoring unknown options silently to avoid MPI spam or use rank 0 check
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI+CUDA)\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI Size: %d, CUDA Devices per node: %d\n", size, numDevices);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    
    // Allocate matrices (everyone allocates full for simplicity of gathering at end)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (replicate on all or broadcast)
    // To ensure consistency, Rank 0 inits and Bcasts.
    if (rank == 0) {
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initial state
    // For very large N this is slow, but consistent with request for "correctness".
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD); // Sync before timing
    
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, size);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) {
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
    
    MPI_Finalize();
    return 0;
}
