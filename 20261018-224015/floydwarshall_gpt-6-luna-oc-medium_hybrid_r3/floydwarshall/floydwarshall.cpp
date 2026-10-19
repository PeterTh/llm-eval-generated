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

__global__ void relaxColumns(unsigned int* d, unsigned int* p, const unsigned int* pivot,
                             size_t n, size_t firstColumn, size_t columns, size_t k) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t local = blockIdx.y;
    if (j >= n || local >= columns) return;
    size_t i = firstColumn + local;
    unsigned int candidate = d[k + local * n] + pivot[j];
    size_t at = j + local * n;
    if (candidate < d[at]) { d[at] = candidate; p[at] = static_cast<unsigned int>(k); }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path, size_t n,
                   int rank, int ranks, int deviceCount) {
    size_t first = n * static_cast<size_t>(rank) / ranks;
    size_t last = n * static_cast<size_t>(rank + 1) / ranks;
    size_t columns = last - first;
    std::vector<unsigned int> localDist(n * columns), localPath(n * columns), pivot(n);
    #pragma omp parallel for
    for (size_t i = 0; i < columns; ++i)
        for (size_t j = 0; j < n; ++j) {
            localDist[j + i*n] = dist[j + (first+i)*n];
            localPath[j + i*n] = path[j + (first+i)*n];
        }
    if (!deviceCount) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    cudaSetDevice(rank % deviceCount);
    unsigned int *dd=nullptr, *dp=nullptr, *dv=nullptr;
    cudaMalloc(&dd, std::max<size_t>(1,n*columns)*sizeof(unsigned int));
    cudaMalloc(&dp, std::max<size_t>(1,n*columns)*sizeof(unsigned int));
    cudaMalloc(&dv, n*sizeof(unsigned int));
    cudaMemcpy(dd, localDist.data(), n*columns*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(dp, localPath.data(), n*columns*sizeof(unsigned int), cudaMemcpyHostToDevice);
    for (size_t k=0;k<n;++k) {
        int owner = static_cast<int>(((k+1)*static_cast<size_t>(ranks)-1)/n);
        if (rank == owner) {
            size_t offset = k - first;
            cudaMemcpy(pivot.data(), localDist.data()+offset*n, n*sizeof(unsigned int), cudaMemcpyHostToHost);
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        cudaMemcpy(dv,pivot.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice);
        if (columns) {
            dim3 grid((n+255)/256, columns);
            relaxColumns<<<grid,256>>>(dd,dp,dv,n,first,columns,k);
            cudaDeviceSynchronize();
            cudaMemcpy(localDist.data(),dd,n*columns*sizeof(unsigned int),cudaMemcpyDeviceToHost);
        }
    }
    if (columns) { cudaMemcpy(localDist.data(),dd,n*columns*sizeof(unsigned int),cudaMemcpyDeviceToHost); cudaMemcpy(localPath.data(),dp,n*columns*sizeof(unsigned int),cudaMemcpyDeviceToHost); }
    for (size_t i=0;i<columns;++i) for(size_t j=0;j<n;++j) { dist[j+(first+i)*n]=localDist[j+i*n]; path[j+(first+i)*n]=localPath[j+i*n]; }
    cudaFree(dd); cudaFree(dp); cudaFree(dv);
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
    int rank=0, ranks=1, deviceCount=0;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    cudaGetDeviceCount(&deviceCount);
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
    
    if (rank == 0) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n"); printf("Number of nodes: %zu\n", numNodes); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rank, ranks, deviceCount);
    size_t ownedFirst = numNodes * static_cast<size_t>(rank) / ranks;
    size_t ownedLast = numNodes * static_cast<size_t>(rank + 1) / ranks;
    for (size_t c=0;c<numNodes;++c) if (c < ownedFirst || c >= ownedLast)
        std::fill(dist.begin()+c*numNodes, dist.begin()+(c+1)*numNodes, 0u);
    MPI_Allreduce(MPI_IN_PLACE, dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize(); return 0;
}
