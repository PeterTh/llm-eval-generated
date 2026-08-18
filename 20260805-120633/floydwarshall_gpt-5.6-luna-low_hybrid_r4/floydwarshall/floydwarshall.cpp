#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000, MAX_DISTANCE = 200;

__global__ void fw_update(unsigned int *d, unsigned int *p, const unsigned int *row,
                          size_t rows, size_t n, size_t first, size_t k) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t count = rows * n;
    if (x >= count) return;
    size_t li = x / n, j = x % n;
    unsigned int old = d[li * n + j];
    unsigned int candidate = row[j] + d[li * n + k];
    if (candidate < old) { d[li * n + j] = candidate; p[li * n + j] = (unsigned int)k; }
}

static void check_cuda(cudaError_t e, const char *what) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

static void make_graph(std::vector<unsigned int>& d, size_t n, unsigned int lo, unsigned int hi) {
    unsigned int seed = 42; double range = double(hi - lo) + 1.0;
    // Keep the benchmark's original deterministic rand_r stream.
    for (size_t x = 0; x < d.size(); ++x)
        d[x] = lo + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    #pragma omp parallel for
    for (long long i = 0; i < (long long)n; ++i) d[(size_t)i * n + i] = 0;
}

static bool validate(const std::vector<unsigned int>& d, size_t n) {
    for (size_t i=0; i<n; ++i) if (d[i*n+i] != 0) return false;
    for (size_t i=0; i<std::min(n,size_t(10)); ++i) for (size_t j=0; j<std::min(n,size_t(10)); ++j)
        for (size_t k=0; k<n; ++k) if (d[i*n+k] < INF && d[k*n+j] < INF && d[i*n+k]+d[k*n+j] < d[i*n+j]) return false;
    return true;
}

static void usage(const char *p) { printf("Usage: %s [-n nodes] [-v] [-r] [-h]\n", p); }

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); int rank, nr; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nr);
    size_t n=512; bool valid=false, results=false;
    for (int a=1;a<argc;++a) { if (!strcmp(argv[a],"-n") && a+1<argc) n=strtoull(argv[++a],nullptr,10); else if(!strcmp(argv[a],"-v")) valid=true; else if(!strcmp(argv[a],"-r")) results=true; else if(!strcmp(argv[a],"-h")){if(rank==0)usage(argv[0]); MPI_Finalize(); return 0;} else {if(rank==0)usage(argv[0]); MPI_Finalize(); return 1;} }
    int devs=0; check_cuda(cudaGetDeviceCount(&devs), "device count"); check_cuda(cudaSetDevice(rank % std::max(devs,1)), "set device");
    std::vector<int> counts(nr), displs(nr); for(int r=0;r<nr;++r){ size_t b=n*r/n, e=n*(r+1)/nr; counts[r]=(int)((e-b)*n); displs[r]=(int)(b*n); }
    size_t first=n*rank/n, rows=n*(rank+1)/nr-first, localCount=rows*n;
    std::vector<unsigned int> global, local(localCount), path(localCount);
    if(rank==0){ global.resize(n*n); make_graph(global,n,1,MAX_DISTANCE); }
    MPI_Scatterv(rank==0?global.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,local.data(),(int)localCount,MPI_UNSIGNED,0,MPI_COMM_WORLD);
    #pragma omp parallel for
    for(long long x=0;x<(long long)localCount;++x) path[(size_t)x]=(unsigned int)(x%n);
    unsigned int *dd=nullptr,*pp=nullptr,*dr=nullptr; check_cuda(cudaMalloc(&dd,localCount*sizeof(unsigned int)),"dist alloc"); check_cuda(cudaMalloc(&pp,localCount*sizeof(unsigned int)),"path alloc"); check_cuda(cudaMalloc(&dr,n*sizeof(unsigned int)),"row alloc");
    check_cuda(cudaMemcpy(dd,local.data(),localCount*sizeof(unsigned int),cudaMemcpyHostToDevice),"dist upload"); check_cuda(cudaMemcpy(pp,path.data(),localCount*sizeof(unsigned int),cudaMemcpyHostToDevice),"path upload");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now(); std::vector<unsigned int> row(n);
    for(size_t k=0;k<n;++k){ int owner=(int)(k*nr/n); if(rank==owner) check_cuda(cudaMemcpy(row.data(),dd+(k-first)*n,n*sizeof(unsigned int),cudaMemcpyDeviceToHost),"row download"); MPI_Bcast(row.data(),(int)n,MPI_UNSIGNED,owner,MPI_COMM_WORLD); check_cuda(cudaMemcpy(dr,row.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice),"row upload"); fw_update<<<(localCount+255)/256,256>>>(dd,pp,dr,rows,n,first,k); check_cuda(cudaGetLastError(),"kernel"); check_cuda(cudaDeviceSynchronize(),"kernel sync"); }
    check_cuda(cudaMemcpy(local.data(),dd,localCount*sizeof(unsigned int),cudaMemcpyDeviceToHost),"result download"); check_cuda(cudaFree(dd),"free"); cudaFree(pp); cudaFree(dr);
    MPI_Gatherv(local.data(),(int)localCount,MPI_UNSIGNED,rank==0?global.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,0,MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    if(rank==0){ double sec=std::chrono::duration<double>(end-start).count(); printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nComputation time: %.3f ms\nPerformance: %.3f GOPS\n",n,sec*1000,(double)n*n*n/sec/1e9); if(results) print_results_int(global,"DistanceMatrix"); if(valid) printf("Validation: %s\n",validate(global,n)?"PASSED":"FAILED"); }
    int ok=1; if(rank==0&&valid) ok=validate(global,n); MPI_Bcast(&ok,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return ok?0:1;
}
