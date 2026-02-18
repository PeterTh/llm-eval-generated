#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>

#ifdef HAVE_CUDA
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)
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
    
    #pragma omp parallel for
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        unsigned int local_seed = seed + i;
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&local_seed) / (double)RAND_MAX);
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
    
    #pragma omp parallel for
    for (size_t j = 0; j < numNodes; ++j) {
        path[idx2(j, j, numNodes)] = j;
    }
}

#ifdef HAVE_CUDA
// CUDA kernel for Floyd-Warshall iteration
__global__ void floydWarshallKernel(unsigned int* dist, unsigned int* path,
                                     const unsigned int* rowK,
                                     const size_t n, const size_t k,
                                     const size_t rowStart, const size_t numLocalRows) {
    // Each thread handles one (i, j) pair
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t localI = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (j >= n || localI >= numLocalRows) return;
    
    size_t i = rowStart + localI;
    
    // Load values
    const size_t idx_ij = localI * n + j;  // Local index for dist
    const unsigned int distIJ = dist[idx_ij];
    const unsigned int distIK = dist[localI * n + k];  // dist[i][k]
    const unsigned int distKJ = rowK[j];  // dist[k][j]
    
    // Compute new distance
    const unsigned int newDist = distIK + distKJ;
    
    // Update if shorter path found
    if (newDist < distIJ) {
        dist[idx_ij] = newDist;
        path[idx_ij] = k;
    }
}

void floydWarshallHybrid(std::vector<unsigned int>& dist, 
                         std::vector<unsigned int>& path, 
                         const size_t numNodes,
                         int rank, int size) {
    // Calculate row distribution
    const size_t rowsPerRank = numNodes / size;
    const size_t extraRows = numNodes % size;
    const size_t rowStart = rank * rowsPerRank + std::min((size_t)rank, extraRows);
    const size_t numLocalRows = rowsPerRank + (rank < (int)extraRows ? 1 : 0);
    
    // Allocate GPU memory
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_rowK = nullptr;
    
    const size_t localSize = numLocalRows * numNodes;
    
    CUDA_CHECK(cudaMalloc(&d_dist, localSize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_path, localSize * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_rowK, numNodes * sizeof(unsigned int)));
    
    // Copy local data to GPU
    CUDA_CHECK(cudaMemcpy(d_dist, dist.data() + rowStart * numNodes,
                          localSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_path, path.data() + rowStart * numNodes,
                          localSize * sizeof(unsigned int), cudaMemcpyHostToDevice));
    
    // Allocate buffer for row k
    std::vector<unsigned int> rowK(numNodes);
    
    // Configure CUDA kernel
    dim3 blockSize(16, 16);
    dim3 gridSize((numNodes + blockSize.x - 1) / blockSize.x,
                  (numLocalRows + blockSize.y - 1) / blockSize.y);
    
    // Floyd-Warshall iterations
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k
        size_t ownerRank = 0;
        size_t cumulativeRows = 0;
        for (int r = 0; r < size; ++r) {
            size_t rankRows = rowsPerRank + (r < (int)extraRows ? 1 : 0);
            if (k < cumulativeRows + rankRows) {
                ownerRank = r;
                break;
            }
            cumulativeRows += rankRows;
        }
        
        // Broadcast row k from owner
        if (rank == (int)ownerRank) {
            // Copy row k from GPU to host
            size_t localK = k - rowStart;
            CUDA_CHECK(cudaMemcpy(rowK.data(), d_dist + localK * numNodes,
                                  numNodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        
        MPI_Bcast(rowK.data(), numNodes, MPI_UNSIGNED, ownerRank, MPI_COMM_WORLD);
        
        // Copy row k to GPU
        CUDA_CHECK(cudaMemcpy(d_rowK, rowK.data(), numNodes * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        
        // Launch kernel
        floydWarshallKernel<<<gridSize, blockSize>>>(d_dist, d_path, d_rowK,
                                                      numNodes, k, rowStart, numLocalRows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Copy results back
    CUDA_CHECK(cudaMemcpy(dist.data() + rowStart * numNodes, d_dist,
                          localSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(path.data() + rowStart * numNodes, d_path,
                          localSize * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_dist));
    CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_rowK));
}
#else
// CPU fallback using MPI + OpenMP
void floydWarshallHybrid(std::vector<unsigned int>& dist, 
                         std::vector<unsigned int>& path, 
                         const size_t numNodes,
                         int rank, int size) {
    // Calculate row distribution
    const size_t rowsPerRank = numNodes / size;
    const size_t extraRows = numNodes % size;
    const size_t rowStart = rank * rowsPerRank + std::min((size_t)rank, extraRows);
    const size_t numLocalRows = rowsPerRank + (rank < (int)extraRows ? 1 : 0);
    
    // Allocate buffer for row k
    std::vector<unsigned int> rowK(numNodes);
    
    // Floyd-Warshall iterations
    for (size_t k = 0; k < numNodes; ++k) {
        // Determine which rank owns row k
        size_t ownerRank = 0;
        size_t cumulativeRows = 0;
        for (int r = 0; r < size; ++r) {
            size_t rankRows = rowsPerRank + (r < (int)extraRows ? 1 : 0);
            if (k < cumulativeRows + rankRows) {
                ownerRank = r;
                break;
            }
            cumulativeRows += rankRows;
        }
        
        // Broadcast row k from owner
        if (rank == (int)ownerRank) {
            size_t localK = k - rowStart;
            for (size_t j = 0; j < numNodes; ++j) {
                rowK[j] = dist[(rowStart + localK) * numNodes + j];
            }
        }
        
        MPI_Bcast(rowK.data(), numNodes, MPI_UNSIGNED, ownerRank, MPI_COMM_WORLD);
        
        // Update local rows using OpenMP
        #pragma omp parallel for collapse(2)
        for (size_t localI = 0; localI < numLocalRows; ++localI) {
            for (size_t j = 0; j < numNodes; ++j) {
                size_t i = rowStart + localI;
                const size_t idx_ij = i * numNodes + j;
                const unsigned int distIJ = dist[idx_ij];
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = rowK[j];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx_ij] = newDist;
                    path[idx_ij] = k;
                }
            }
        }
    }
}
#endif

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
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    #ifdef HAVE_CUDA
    // Set GPU device based on rank
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices found\n");
        }
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    #endif
    
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
                printUsage(argv[0]);
            }
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
        printf("Parallelization: MPI + OpenMP");
        #ifdef HAVE_CUDA
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        printf(" + CUDA\n");
        printf("CUDA devices: %d\n", deviceCount);
        #else
        printf("\n");
        #endif
        printf("MPI ranks: %d\n", size);
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices (all ranks need full matrices for initialization and result gathering)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize (only rank 0, then broadcast)
    if (rank == 0) {
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    
    // Broadcast initial data to all ranks
    MPI_Bcast(dist.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(path.data(), numNodes * numNodes, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Synchronize before starting computation
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshallHybrid(dist, path, numNodes, rank, size);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    // Gather results back to rank 0
    // Calculate row distribution
    const size_t rowsPerRank = numNodes / size;
    const size_t extraRows = numNodes % size;
    
    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    
    for (int r = 0; r < size; ++r) {
        size_t rankRows = rowsPerRank + (r < (int)extraRows ? 1 : 0);
        recvCounts[r] = rankRows * numNodes;
        size_t rankStart = r * rowsPerRank + std::min((size_t)r, extraRows);
        displs[r] = rankStart * numNodes;
    }
    
    // Gather dist matrix
    std::vector<unsigned int> distFull;
    if (rank == 0) {
        distFull.resize(numNodes * numNodes);
    }
    
    const size_t rowStart = rank * rowsPerRank + std::min((size_t)rank, extraRows);
    
    MPI_Gatherv(dist.data() + rowStart * numNodes, recvCounts[rank], MPI_UNSIGNED,
                rank == 0 ? distFull.data() : nullptr, recvCounts.data(), 
                displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Gather path matrix
    std::vector<unsigned int> pathFull;
    if (rank == 0) {
        pathFull.resize(numNodes * numNodes);
    }
    
    MPI_Gatherv(path.data() + rowStart * numNodes, recvCounts[rank], MPI_UNSIGNED,
                rank == 0 ? pathFull.data() : nullptr, recvCounts.data(), 
                displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(distFull, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(distFull, numNodes);
            
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
