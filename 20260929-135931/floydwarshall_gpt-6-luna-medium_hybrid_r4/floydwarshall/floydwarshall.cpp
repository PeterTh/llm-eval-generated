#include <algorithm>
#include <chrono>
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

// Matrices are row-major on the GPU. Each rank owns a contiguous range of rows;
// after each pivot, updated rows are exchanged so every GPU has the pivot data.
__global__ void fw_rows(unsigned int* dist, unsigned int* path,
                        const unsigned int* pivotRow, size_t n,
                        size_t rowBegin, size_t rowCount, size_t k) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= rowCount) return;
    const size_t i = rowBegin + y;
    const unsigned int candidate = dist[i * n + k] + pivotRow[x];
    const size_t pos = i * n + x;
    if (candidate < dist[pos]) {
        dist[pos] = candidate;
        path[pos] = static_cast<unsigned int>(k);
    }
}

static void cudaCheck(cudaError_t e) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n) {
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[i * n + i] = 0;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) if (dist[i * n + i] != 0) {
        printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i); return false;
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[i * n + k] < INF && dist[k * n + j] < INF &&
                    dist[i * n + k] + dist[k * n + j] < dist[i * n + j]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i,j,k); return false;
                }
    return true;
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of nodes in the graph (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, printResults = false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1<argc) n = static_cast<size_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(INT_MAX)) { if (!rank) fprintf(stderr,"Invalid node count\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    const size_t rowBegin = n * rank / ranks, rowEnd = n * (rank+1) / ranks;
    const size_t localRows = rowEnd-rowBegin;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r=0; r<ranks; ++r) { offsets[r] = static_cast<int>((n*r/ranks)*n); counts[r] = static_cast<int>((n*(r+1)/ranks-n*r/ranks)*n); }
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n", n, validate?"enabled":"disabled");
    }
    std::vector<unsigned int> dist(n*n), path(n*n);
    if (!rank) initializeDistanceMatrix(dist,n);
    MPI_Bcast(dist.data(), static_cast<int>(n*n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    for (size_t i=0;i<n;++i) for(size_t j=0;j<n;++j) path[i*n+j]=static_cast<unsigned int>(i);
    int deviceCount=0; cudaCheck(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaCheck(cudaSetDevice(rank % deviceCount));
    unsigned int *dDist=nullptr,*dPath=nullptr,*dPivot=nullptr;
    cudaCheck(cudaMalloc(&dDist,n*n*sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&dPath,n*n*sizeof(unsigned int)));
    cudaCheck(cudaMalloc(&dPivot,n*sizeof(unsigned int)));
    cudaCheck(cudaMemcpy(dDist,dist.data(),n*n*sizeof(unsigned int),cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(dPath,path.data(),n*n*sizeof(unsigned int),cudaMemcpyHostToDevice));
    if (!rank) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start=MPI_Wtime();
    // OpenMP provides the host-side parallel region; a single host thread submits
    // the ordered CUDA pivot kernels while MPI synchronizes the distributed rows.
#pragma omp parallel
    {
#pragma omp master
        {
            for (size_t k=0;k<n;++k) {
                std::vector<unsigned int> pivot(n);
                if (rowBegin <= k && k < rowEnd)
                    cudaCheck(cudaMemcpy(pivot.data(), dDist + k*n, n*sizeof(unsigned int), cudaMemcpyDeviceToHost));
                const int pivotOwner = static_cast<int>(((k + 1) * static_cast<size_t>(ranks) - 1) / n);
                MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, pivotOwner, MPI_COMM_WORLD);
                cudaCheck(cudaMemcpy(dPivot,pivot.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice));
                if (localRows) {
                    dim3 block(32,8), grid((n+31)/32,(localRows+7)/8);
                    fw_rows<<<grid,block>>>(dDist,dPath,dPivot,n,rowBegin,localRows,k);
                    cudaCheck(cudaGetLastError());
                    cudaCheck(cudaDeviceSynchronize());
                    cudaCheck(cudaMemcpy(dist.data()+rowBegin*n,dDist+rowBegin*n,localRows*n*sizeof(unsigned int),cudaMemcpyDeviceToHost));
                }
                MPI_Allgatherv(localRows ? dist.data()+rowBegin*n : nullptr, counts[rank], MPI_UNSIGNED,
                               dist.data(),counts.data(),offsets.data(),MPI_UNSIGNED,MPI_COMM_WORLD);
                cudaCheck(cudaMemcpy(dDist,dist.data(),n*n*sizeof(unsigned int),cudaMemcpyHostToDevice));
            }
        }
    }
    const double elapsed=MPI_Wtime()-start;
    double maxElapsed=0; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(path.data(),dPath,n*n*sizeof(unsigned int),cudaMemcpyDeviceToHost));
    cudaFree(dDist); cudaFree(dPath); cudaFree(dPivot);
    if (!rank) {
        const long ms=static_cast<long>(maxElapsed*1000.0);
        printf("Computation time: %ld ms\n",ms);
        printf("Performance: %.3f GOPS\n", (n ? static_cast<double>(n)*n*n/(maxElapsed*1e9) : 0.0));
        if (printResults) print_results_int(dist,"DistanceMatrix");
        if (validate) { printf("Validating result...\n"); bool ok=validateResult(dist,n); printf("Validation: %s\n",ok?"PASSED":"FAILED"); MPI_Finalize(); return ok?0:1; }
    }
    MPI_Finalize(); return 0;
}
