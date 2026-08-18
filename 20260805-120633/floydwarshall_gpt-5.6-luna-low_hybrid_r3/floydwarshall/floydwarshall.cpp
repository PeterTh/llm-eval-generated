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

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

__global__ void fw_step(unsigned int* d, unsigned int* p, const unsigned int* pivot,
                        size_t rows, size_t n, size_t k) {
    size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= rows || j >= n) return;
    unsigned int a = d[j * rows + i];
    unsigned int b = d[k * rows + i];
    unsigned int c = pivot[j];
    unsigned int candidate = b + c;
    if (candidate < a) {
        d[j * rows + i] = candidate;
        p[j * rows + i] = static_cast<unsigned int>(k);
    }
}

__global__ void extract_pivot(const unsigned int* d, unsigned int* pivot,
                              size_t rows, size_t localK, size_t n) {
    size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < n) pivot[j] = d[j * rows + localK];
}

static void check_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

static void initializeGlobal(std::vector<unsigned int>& d, size_t n) {
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE) + 1.0;
    for (size_t x = 0; x < n * n; ++x)
        d[x] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    #pragma omp parallel for
    for (long long i = 0; i < static_cast<long long>(n); ++i) d[idx2(i, i, n)] = 0;
}

static bool validateResult(const std::vector<unsigned int>& d, size_t n) {
    for (size_t i = 0; i < n; ++i) if (d[idx2(i, i, n)] != 0) return false;
    const size_t sample = std::min(n, size_t(10));
    for (size_t i = 0; i < sample; ++i) for (size_t j = 0; j < sample; ++j)
        for (size_t k = 0; k < n; ++k) {
            unsigned int a=d[idx2(k,i,n)], b=d[idx2(j,k,n)], c=d[idx2(j,i,n)];
            if (a < INF && b < INF && a + b < c) return false;
        }
    return true;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, nranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    size_t n = 512; bool validate = false, printResults = false;
    for (int a=1; a<argc; ++a) {
        if (!strcmp(argv[a], "-n") && a+1<argc) n = std::strtoull(argv[++a], nullptr, 10);
        else if (!strcmp(argv[a], "-v")) validate=true;
        else if (!strcmp(argv[a], "-r")) printResults=true;
        else if (!strcmp(argv[a], "-h")) { if(rank==0) printf("Usage: %s [-n nodes] [-v] [-r]\n",argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) fprintf(stderr,"Unknown option: %s\n",argv[a]); MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(INT_MAX)) { if(rank==0) fprintf(stderr,"Invalid node count\n"); MPI_Finalize(); return 1; }
    size_t first = (n * static_cast<size_t>(rank)) / nranks;
    size_t last = (n * static_cast<size_t>(rank+1)) / nranks, rows = last-first;
    std::vector<int> counts(nranks), displs(nranks);
    int packedOffset=0;
    for(int r=0;r<nranks;++r) { size_t s=n*r/nranks, e=n*(r+1)/nranks; counts[r]=static_cast<int>((e-s)*n); displs[r]=packedOffset; packedOffset += counts[r]; }
    std::vector<unsigned int> global, packed;
    if(rank==0) { global.resize(n*n); initializeGlobal(global,n); packed.resize(n*n); }
    std::vector<unsigned int> local(rows*n), path(rows*n);
    if(rank==0) for(int r=0;r<nranks;++r) { size_t s=n*r/nranks, rr=n*(r+1)/nranks-s; for(size_t j=0;j<n;++j) for(size_t i=0;i<rr;++i) packed[displs[r]+j*rr+i]=global[j*n+s+i]; }
    MPI_Scatterv(rank==0?packed.data():nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 local.data(), static_cast<int>(rows*n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    #pragma omp parallel for
    for(long long x=0;x<static_cast<long long>(rows*n);++x) path[x]=static_cast<unsigned int>(first + (x % rows));
    int deviceCount=0; check_cuda(cudaGetDeviceCount(&deviceCount),"cudaGetDeviceCount");
    check_cuda(cudaSetDevice(rank % deviceCount),"cudaSetDevice");
    unsigned int *dd,*dp,*pivot; check_cuda(cudaMalloc(&dd,local.size()*sizeof(unsigned int)),"malloc dist"); check_cuda(cudaMalloc(&dp,path.size()*sizeof(unsigned int)),"malloc path"); check_cuda(cudaMalloc(&pivot,n*sizeof(unsigned int)),"malloc pivot");
    check_cuda(cudaMemcpy(dd,local.data(),local.size()*sizeof(unsigned int),cudaMemcpyHostToDevice),"copy dist"); check_cuda(cudaMemcpy(dp,path.data(),path.size()*sizeof(unsigned int),cudaMemcpyHostToDevice),"copy path");
    std::vector<unsigned int> row(n); MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime();
    dim3 block(32,8), grid((n+31)/32,(rows+7)/8);
    for(size_t k=0;k<n;++k) {
        int owner=std::min(static_cast<int>((k*nranks)/n), nranks-1);
        if(rank==owner) {
            size_t ownerFirst = n * static_cast<size_t>(owner) / nranks;
            extract_pivot<<<(n+255)/256,256>>>(dd,pivot,rows,k-ownerFirst,n);
            check_cuda(cudaGetLastError(),"extract pivot");
            check_cuda(cudaMemcpy(row.data(),pivot,n*sizeof(unsigned int),cudaMemcpyDeviceToHost),"copy pivot");
        }
        MPI_Bcast(row.data(),static_cast<int>(n),MPI_UNSIGNED,owner,MPI_COMM_WORLD);
        check_cuda(cudaMemcpy(pivot,row.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice),"broadcast pivot");
        fw_step<<<grid,block>>>(dd,dp,pivot,rows,n,k); check_cuda(cudaGetLastError(),"kernel");
    }
    check_cuda(cudaDeviceSynchronize(),"synchronize"); double elapsed=MPI_Wtime()-start;
    check_cuda(cudaMemcpy(local.data(),dd,local.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost),"copy result"); cudaFree(dd); cudaFree(dp); cudaFree(pivot);
    std::vector<unsigned int> gathered, result; if(rank==0) { gathered.resize(n*n); result.resize(n*n); }
    MPI_Gatherv(local.data(),static_cast<int>(rows*n),MPI_UNSIGNED,rank==0?gathered.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    if(rank==0) for(int r=0;r<nranks;++r) { size_t s=n*r/nranks, rr=n*(r+1)/nranks-s; for(size_t j=0;j<n;++j) for(size_t i=0;i<rr;++i) result[j*n+s+i]=gathered[displs[r]+j*rr+i]; }
    double maxTime; MPI_Reduce(&elapsed,&maxTime,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nComputation time: %.3f ms\nPerformance: %.3f GOPS\n",n,maxTime*1000,(n*n*n)/(maxTime*1e9)); if(printResults) print_results_int(result,"DistanceMatrix"); if(validate) printf("Validation: %s\n",validateResult(result,n)?"PASSED":"FAILED"); }
    MPI_Finalize(); return 0;
}
