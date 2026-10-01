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
// Matrices use [column][row] indexing so each source row is contiguous.
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j*n+i; }

__global__ void updateRows(unsigned int* dist, unsigned int* path,
                           const unsigned int* pivot, size_t n, size_t first,
                           size_t rows, size_t k) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= n || y >= rows) return;
    const size_t i = first + y;
    const unsigned int dik = dist[k*n+i];
    const unsigned int dkj = pivot[x];
    const unsigned int candidate = dik + dkj;
    const size_t at = x*rows+y;
    if (candidate < dist[at]) {
        dist[at] = candidate;
        path[at] = static_cast<unsigned int>(k);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& a, size_t n) {
    unsigned int seed=42;
    for(size_t q=0;q<n*n;++q)
        a[q]=1u+(unsigned int)(201.0*rand_r(&seed)/(double)RAND_MAX);
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<static_cast<long long>(n); ++i) a[idx2(i,i,n)]=0;
}

bool validateResult(const std::vector<unsigned int>& d, size_t n) {
    for(size_t i=0;i<n;++i) if(d[idx2(i,i,n)]!=0) return false;
    for(size_t i=0;i<std::min(n,size_t(10));++i)
      for(size_t j=0;j<std::min(n,size_t(10));++j)
       for(size_t k=0;k<n;++k)
        if(d[idx2(k,i,n)]<INF && d[idx2(j,k,n)]<INF &&
           d[idx2(k,i,n)]+d[idx2(j,k,n)]<d[idx2(j,i,n)]) return false;
    return true;
}

void printUsage(const char* p) { printf("Usage: %s [-n num] [-v] [-r] [-h]\n",p); }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t n=512; bool validate=false, printResults=false;
    for(int a=1;a<argc;++a) {
        if(!strcmp(argv[a],"-n")&&a+1<argc) n=static_cast<size_t>(atoi(argv[++a]));
        else if(!strcmp(argv[a],"-v")) validate=true;
        else if(!strcmp(argv[a],"-r")) printResults=true;
        else if(!strcmp(argv[a],"-h")){ if(rank==0)printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0){printf("Unknown option: %s\n",argv[a]);printUsage(argv[0]);} MPI_Finalize(); return 1; }
    }
    int devices=0; cudaGetDeviceCount(&devices);
    if(devices<=0){ if(rank==0) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaSetDevice(rank%devices);
    const size_t base=n/static_cast<size_t>(size), rem=n%static_cast<size_t>(size);
    const size_t rows=base+(static_cast<size_t>(rank)<rem), first=static_cast<size_t>(rank)*base+std::min(static_cast<size_t>(rank),rem);
    std::vector<int> counts(size), displs(size);
    for(int r=0;r<size;++r){size_t rr=base+(static_cast<size_t>(r)<rem); size_t ff=static_cast<size_t>(r)*base+std::min(static_cast<size_t>(r),rem); counts[r]=static_cast<int>(rr*n);displs[r]=static_cast<int>(ff*n);}
    std::vector<unsigned int> full(n*n), local(rows*n), pathLocal(rows*n), pivot(n);
    if(rank==0) initializeDistanceMatrix(full,n);
    // Pack source rows for the row-block decomposition used by the device kernel.
    std::vector<unsigned int> packed;
    if(rank==0) {
        packed.resize(n*n);
        for(size_t r=0;r<n;++r) for(size_t j=0;j<n;++j) packed[j*n+r]=full[idx2(j,r,n)];
    }
    MPI_Scatterv(rank==0?packed.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,
                 local.data(),static_cast<int>(rows*n),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for(long long q=0;q<static_cast<long long>(rows*n);++q){size_t j=q/rows, i=first+q%rows;pathLocal[q]=static_cast<unsigned int>(j); (void)i;}
    unsigned int *dd=nullptr,*dp=nullptr,*dr=nullptr;
    cudaMalloc(&dd,local.size()*sizeof(unsigned int)); cudaMalloc(&dp,pathLocal.size()*sizeof(unsigned int)); cudaMalloc(&dr,n*sizeof(unsigned int));
    cudaMemcpy(dd,local.data(),local.size()*sizeof(unsigned int),cudaMemcpyHostToDevice);
    cudaMemcpy(dp,pathLocal.data(),pathLocal.size()*sizeof(unsigned int),cudaMemcpyHostToDevice);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start=std::chrono::high_resolution_clock::now();
    for(size_t k=0;k<n;++k){
        int owner=0;
        while(owner+1<size && k>=static_cast<size_t>(displs[owner+1])/n) ++owner;
        // A row is contiguous locally; extract it on its owner then broadcast.
        if(rank==owner){cudaMemcpy(local.data(),dd,local.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost); const size_t y=k-first; for(size_t j=0;j<n;++j)pivot[j]=local[j*rows+y];}
        MPI_Bcast(pivot.data(),static_cast<int>(n),MPI_UNSIGNED,owner,MPI_COMM_WORLD);
        cudaMemcpy(dr,pivot.data(),n*sizeof(unsigned int),cudaMemcpyHostToDevice);
        dim3 block(32,8), grid((n+31)/32,(rows+7)/8);
        updateRows<<<grid,block>>>(dd,dp,dr,n,first,rows,k);
        cudaDeviceSynchronize();
    }
    cudaMemcpy(local.data(),dd,local.size()*sizeof(unsigned int),cudaMemcpyDeviceToHost);
    auto end=std::chrono::high_resolution_clock::now();
    std::vector<unsigned int> gathered;
    if(rank==0) gathered.resize(n*n);
    MPI_Gatherv(local.data(),static_cast<int>(rows*n),MPI_UNSIGNED,rank==0?gathered.data():nullptr,counts.data(),displs.data(),MPI_UNSIGNED,0,MPI_COMM_WORLD);
    if(rank==0){
      for(size_t i=0;i<n;++i) for(size_t j=0;j<n;++j) full[idx2(j,i,n)]=gathered[j*n+i];
      auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
      printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",n,validate?"enabled":"disabled");
      printf("Computation time: %ld ms\nPerformance: %.3f GOPS\n",ms,(double)n*n*n/(ms/1000.0)/1e9);
      if(printResults)print_results_int(full,"DistanceMatrix");
      if(validate){printf("Validating result...\n"); bool ok=validateResult(full,n);printf("Validation: %s\n",ok?"PASSED":"FAILED"); if(!ok) MPI_Abort(MPI_COMM_WORLD,1);}
    }
    cudaFree(dd);cudaFree(dp);cudaFree(dr); MPI_Finalize(); return 0;
}
