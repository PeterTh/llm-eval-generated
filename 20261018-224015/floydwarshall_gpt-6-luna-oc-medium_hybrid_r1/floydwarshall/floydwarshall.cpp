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

__global__ void updateRows(unsigned int* dist, unsigned int* path, size_t n,
                           size_t firstRow, size_t rowCount, size_t k,
                           const unsigned int* pivotRow, const unsigned int* pivotCol) {
    const size_t i = firstRow + blockIdx.y;
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.y >= rowCount || j >= n) return;
    const unsigned int candidate = pivotCol[i] + pivotRow[j];
    const size_t p = j * n + i;
    if (candidate < dist[p]) { dist[p] = candidate; path[p] = static_cast<unsigned int>(k); }
}

static void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
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
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j)
        for (size_t i = 0; i < numNodes; ++i)
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes, int rank, int ranks) {
    const size_t n=numNodes, first=n*rank/ranks, last=n*(rank+1)/ranks, rows=last-first;
    unsigned int *dd=nullptr,*dp=nullptr,*dr=nullptr,*dc=nullptr;
    cudaCheck(cudaMalloc(&dd,n*n*sizeof(unsigned int))); cudaCheck(cudaMalloc(&dp,n*n*sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&dr,n*sizeof(unsigned int))); cudaCheck(cudaMalloc(&dc,n*sizeof(unsigned int)));
    cudaCheck(cudaMemcpy(dd,dist.data(),n*n*sizeof(unsigned int),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dp,path.data(),n*n*sizeof(unsigned int),cudaMemcpyHostToDevice));
    std::vector<unsigned int> pivotRow(n),pivotCol(n);
    for(size_t k=0;k<n;++k) {
        if(rank==0) {
            for(size_t j=0;j<n;++j) pivotRow[j]=dist[idx2(j,k,n)];
            for(size_t i=0;i<n;++i) pivotCol[i]=dist[idx2(k,i,n)];
        }
        MPI_Bcast(pivotRow.data(),static_cast<int>(n),MPI_UNSIGNED,0,MPI_COMM_WORLD);
        MPI_Bcast(pivotCol.data(),static_cast<int>(n),MPI_UNSIGNED,0,MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dr,pivotRow.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dc,pivotCol.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice));
        dim3 block(256), grid((n+255)/256,rows);
        if(rows) { updateRows<<<grid,block>>>(dd,dp,n,first,rows,k,dr,dc); cudaCheck(cudaGetLastError()); }
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaMemcpy(dist.data()+first*n,dd+first*n,rows*n*sizeof(unsigned int),cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemcpy(path.data()+first*n,dp+first*n,rows*n*sizeof(unsigned int),cudaMemcpyDeviceToHost));
        // Gather contiguous row partitions from all ranks.
        std::vector<int> counts(ranks), offsets(ranks);
        for(int r=0;r<ranks;++r){size_t a=n*r/ranks,b=n*(r+1)/ranks;counts[r]=static_cast<int>((b-a)*n);offsets[r]=static_cast<int>(a*n);}
        std::vector<unsigned int> localD(dist.begin()+first*n,dist.begin()+last*n), localP(path.begin()+first*n,path.begin()+last*n);
        MPI_Allgatherv(localD.data(),counts[rank],MPI_UNSIGNED,dist.data(),counts.data(),offsets.data(),MPI_UNSIGNED,MPI_COMM_WORLD);
        MPI_Allgatherv(localP.data(),counts[rank],MPI_UNSIGNED,path.data(),counts.data(),offsets.data(),MPI_UNSIGNED,MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dd,dist.data(),n*n*sizeof(unsigned int),cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(dp,path.data(),n*n*sizeof(unsigned int),cudaMemcpyHostToDevice));
    }
    cudaFree(dd); cudaFree(dp); cudaFree(dr); cudaFree(dc);
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
    MPI_Init(&argc,&argv);
    int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int devices=0; cudaCheck(cudaGetDeviceCount(&devices)); if(devices>0) cudaCheck(cudaSetDevice(rank%devices));
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
    
    if(rank==0){ printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n"); printf("Number of nodes: %zu\n", numNodes); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
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
    
    floydWarshall(dist, path, numNodes,rank,ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if(rank==0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if(rank==0) printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank==0) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (validate && rank==0) {
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
