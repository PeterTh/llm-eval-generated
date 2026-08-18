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

static void cuda_check(cudaError_t e, const char *where) {
    if (e != cudaSuccess) { fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); }
}

// One CUDA thread computes one owned row of the current Cholesky column.
__global__ void cholesky_column(double *a, size_t n, size_t k, size_t first, size_t last, int *bad) {
    size_t i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= last || i < k) return;
    double s = 0.0;
    for (size_t q = 0; q < k; ++q) s += a[i*n+q] * a[k*n+q];
    a[i*n+k] = (a[i*n+k] - s) / a[k*n+k];
}

__global__ void cholesky_diagonal(double *a, size_t n, size_t k, int *bad) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double s=0.0; for(size_t q=0;q<k;++q) s+=a[k*n+q]*a[k*n+q];
        double d=a[k*n+k]-s; if(d<=0.0) atomicExch(bad,1); else a[k*n+k]=sqrt(d);
    }
}

static std::vector<int> counts_for(size_t n, int p) {
    std::vector<int> c(p); size_t base=n/(size_t)p, rem=n%(size_t)p;
    for (int r=0;r<p;++r) c[r]=(int)(base+(r<(int)rem));
    return c;
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int nranks) {
    int devices=0; cuda_check(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) { if (!rank) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cuda_check(cudaSetDevice(rank % devices), "cudaSetDevice");
    double *dA=nullptr; int *dBad=nullptr;
    cuda_check(cudaMalloc(&dA,n*n*sizeof(double)), "cudaMalloc(matrix)");
    cuda_check(cudaMalloc(&dBad,sizeof(int)), "cudaMalloc(status)");
    cuda_check(cudaMemcpy(dA,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice), "H2D");
    auto counts=counts_for(n,nranks); std::vector<int> displs(nranks); for(int r=1;r<nranks;++r) displs[r]=displs[r-1]+counts[r-1];
    size_t first=(size_t)displs[rank], last=first+(size_t)counts[rank];
    std::vector<double> column(n);
    bool ok=true;
    for (size_t k=0;k<n;++k) {
        cuda_check(cudaMemset(dBad,0,sizeof(int)), "status reset");
        if (k >= first && k < last) cholesky_diagonal<<<1,1>>>(dA,n,k,dBad);
        cuda_check(cudaGetLastError(), "diagonal kernel"); cuda_check(cudaDeviceSynchronize(), "diagonal sync");
        double diagonal=0.0;
        if (k >= first && k < last) cuda_check(cudaMemcpy(&diagonal,dA+k*n+k,sizeof(double),cudaMemcpyDeviceToHost), "diagonal D2H");
        int owner=0; while(owner+1<nranks && k>=(size_t)(displs[owner]+counts[owner])) ++owner;
        MPI_Bcast(&diagonal,1,MPI_DOUBLE,owner,MPI_COMM_WORLD);
        cuda_check(cudaMemcpy2D(dA+k,n*sizeof(double),&diagonal,sizeof(double),sizeof(double),1,cudaMemcpyHostToDevice), "diagonal H2D");
        size_t work=last>k ? last-k : 0;
        if (work) {
            size_t update_first=std::max(first,k+1);
            if (update_first<last) cholesky_column<<<(unsigned)((last-update_first+255)/256),256>>>(dA,n,k,update_first,last,dBad);
        }
        cuda_check(cudaGetLastError(), "column kernel"); cuda_check(cudaDeviceSynchronize(), "column sync");
        int bad=0; cuda_check(cudaMemcpy(&bad,dBad,sizeof(int),cudaMemcpyDeviceToHost), "status D2H");
        MPI_Allreduce(MPI_IN_PLACE,&bad,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); if(bad){ok=false;break;}
        if (counts[rank]) cuda_check(cudaMemcpy2D(column.data()+first, sizeof(double), dA+first*n+k, n*sizeof(double), sizeof(double), counts[rank], cudaMemcpyDeviceToHost), "column D2H");
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DOUBLE,column.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        cuda_check(cudaMemcpy2D(dA+k,n*sizeof(double),column.data(),sizeof(double),sizeof(double),n,cudaMemcpyHostToDevice), "column H2D");
    }
    cuda_check(cudaMemcpy(A.data(),dA,n*n*sizeof(double),cudaMemcpyDeviceToHost), "result D2H");
    #pragma omp parallel for schedule(static)
    for (long long i=0;i<(long long)n;++i) for(size_t j=(size_t)i+1;j<n;++j) A[(size_t)i*n+j]=0.0;
    cudaFree(dBad); cudaFree(dA); return ok;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n); unsigned int seed=42;
    #pragma omp parallel for schedule(static)
    for (long long i=0;i<(long long)n*n;++i) { unsigned int s=seed+(unsigned)i*2654435761u; B[i]=(rand_r(&s)/(double)RAND_MAX)-0.5; }
    #pragma omp parallel for schedule(static)
    for (long long i=0;i<(long long)n;++i) for(size_t j=0;j<n;++j) { double s=0; for(size_t k=0;k<n;++k) s+=B[(size_t)i*n+k]*B[j*n+k]; A[(size_t)i*n+j]=s+(i==(long long)j?n:0); }
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& O,size_t n) {
    double ma=0,mr=0;
    #pragma omp parallel for reduction(max:ma,mr) schedule(static)
    for(long long i=0;i<(long long)n*n;++i){size_t r=(size_t)i/n,c=(size_t)i%n; double s=0; for(size_t k=0;k<n;++k)s+=L[r*n+k]*L[c*n+k]; double e=fabs(s-O[i]); ma=std::max(ma,e); mr=std::max(mr,e/(fabs(O[i])+1e-10));}
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",ma,mr); return mr<=1e-6;
}

void printUsage(const char* p){printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n",p);}

int main(int argc,char** argv){ MPI_Init(&argc,&argv); int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size); size_t n=512; bool val=false,pr=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-n")&&i+1<argc)n=strtoull(argv[++i],nullptr,10);else if(!strcmp(argv[i],"-v"))val=true;else if(!strcmp(argv[i],"-r"))pr=true;else if(!strcmp(argv[i],"-h")){if(!rank)printUsage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)printf("Unknown option: %s\n",argv[i]);MPI_Finalize();return 1;}}
    std::vector<double>A(n*n),O; if(!rank){printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,val?"enabled":"disabled"); generatePositiveDefiniteMatrix(A,n); if(val)O=A;}
    MPI_Bcast(A.data(),(int)(n*n),MPI_DOUBLE,0,MPI_COMM_WORLD); if(val&&rank)O=A; MPI_Barrier(MPI_COMM_WORLD); if(!rank)printf("Computing Cholesky decomposition...\n"); auto st=std::chrono::high_resolution_clock::now(); bool ok=choleskyDecomposition(A,n,rank,size); MPI_Allreduce(MPI_IN_PLACE,&ok,1,MPI_C_BOOL,MPI_LAND,MPI_COMM_WORLD); auto en=std::chrono::high_resolution_clock::now();
    if(!rank){auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(en-st).count(); if(!ok){printf("Cholesky decomposition failed\n");MPI_Finalize();return 1;} printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",ms,(double)n*n*n/3.0/(ms/1000.0)/1e9); if(pr)print_results(A,"CholeskyL"); if(val){printf("Validating result...\n");bool good=validateCholesky(A,O,n);printf("Validation: %s\n",good?"PASSED":"FAILED");MPI_Finalize();return good?0:1;}}
    MPI_Finalize(); return ok?0:1;
}
