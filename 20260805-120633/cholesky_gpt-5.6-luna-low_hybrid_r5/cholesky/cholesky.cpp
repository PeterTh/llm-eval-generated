#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "../common/results_output.hpp"

#define CUDA_OK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); } } while(0)

__global__ void factor_pivot(double* row, const double* a, size_t n, size_t k) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double s = 0.0;
    for (size_t q=0; q<k; ++q) s += row[q]*row[q];
    row[k] = sqrt(row[k]-s);
    for (size_t j=k+1; j<n; ++j) {
        s=0.0; for (size_t q=0; q<k; ++q) s += row[q]*a[j*n+q];
        row[j]=(row[j]-s)/row[k];
    }
}

__global__ void update_rows(double* a, const double* pivot, size_t n, size_t k, size_t first, size_t rows) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t i = first + k + 1 + x;
    if (i >= first + rows) return;
    size_t li = i - first;
    double lik = a[li*n+k] / pivot[k];
    a[li*n+k] = lik;
    for (size_t j=k+1; j<n; ++j) a[li*n+j] -= lik*pivot[j];
}

static void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n); unsigned int seed=42;
    for (long long i=0; i<(long long)(n*n); ++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n; ++i) for(size_t j=0;j<n;++j) {
        double s=0; for(size_t k=0;k<n;++k) s+=B[i*n+k]*B[j*n+k]; A[i*n+j]=s;
    }
    #pragma omp parallel for
    for (long long i=0;i<(long long)n;++i) A[i*n+i]+=n;
}

static bool validateCholesky(const std::vector<double>& L,const std::vector<double>& A,size_t n) {
    double maxe=0,re=0;
    #pragma omp parallel for reduction(max:maxe,re) schedule(static)
    for(long long z=0;z<(long long)(n*n);++z){ size_t i=z/n,j=z%n; double s=0; for(size_t k=0;k<n;++k)s+=L[i*n+k]*L[j*n+k]; double e=fabs(s-A[z]); maxe=std::max(maxe,e); re=std::max(re,e/(fabs(A[z])+1e-10)); }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxe,re); return re<=1e-6;
}

static void usage(const char* p){printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n",p);}

int main(int argc,char** argv){
    MPI_Init(&argc,&argv); int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t n=512; bool validate=false, printResults=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-n")&&i+1<argc)n=strtoull(argv[++i],nullptr,10);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))printResults=true;else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)printf("Unknown option: %s\n",argv[i]);MPI_Finalize();return 1;}}
    if(!rank){printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,validate?"enabled":"disabled");}
    std::vector<double> full, original; if(!rank){full.resize(n*n);generatePositiveDefiniteMatrix(full,n);if(validate)original=full;}
    std::vector<int> counts(size),displs(size); for(int r=0;r<size;++r){size_t b=r*n/size,e=(r+1)*n/size;counts[r]=(int)((e-b)*n);displs[r]=(int)(b*n);}
    size_t first=(size_t)rank*n/size, rows=(size_t)(rank+1)*n/size-first; std::vector<double> local(rows*n);
    MPI_Scatterv(rank?nullptr:full.data(),counts.data(),displs.data(),MPI_DOUBLE,local.data(),counts[rank],MPI_DOUBLE,0,MPI_COMM_WORLD);
    double *dA=nullptr,*dP=nullptr; CUDA_OK(cudaMalloc(&dA,local.size()*sizeof(double)));CUDA_OK(cudaMalloc(&dP,n*sizeof(double)));CUDA_OK(cudaMemcpy(dA,local.data(),local.size()*sizeof(double),cudaMemcpyHostToDevice));
    MPI_Barrier(MPI_COMM_WORLD); if(!rank)printf("Computing Cholesky decomposition...\n"); auto start=std::chrono::high_resolution_clock::now();
    std::vector<double> pivot(n); int threads=256;
    for(size_t k=0;k<n;++k){int owner=(int)(k*size/n); size_t ownerFirst=(size_t)owner*n/size; if(rank==owner){CUDA_OK(cudaMemcpy(pivot.data(),dA+(k-ownerFirst)*n,n*sizeof(double),cudaMemcpyDeviceToHost));CUDA_OK(cudaMemcpy(dP,pivot.data(),n*sizeof(double),cudaMemcpyHostToDevice));factor_pivot<<<1,1>>>(dP,dA,n,k);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());CUDA_OK(cudaMemcpy(pivot.data(),dP,n*sizeof(double),cudaMemcpyDeviceToHost));} MPI_Bcast(pivot.data(),(int)n,MPI_DOUBLE,owner,MPI_COMM_WORLD); CUDA_OK(cudaMemcpy(dP,pivot.data(),n*sizeof(double),cudaMemcpyHostToDevice)); if(first<=k&&k<first+rows)CUDA_OK(cudaMemcpy(dA+(k-first)*n,pivot.data(),n*sizeof(double),cudaMemcpyHostToDevice)); if(first+rows>k+1) {size_t begin=std::max(first,k+1); size_t count=first+rows-begin; update_rows<<<(count+255)/256,256>>>(dA,dP,n,k,first,rows);CUDA_OK(cudaGetLastError());CUDA_OK(cudaDeviceSynchronize());}}
    CUDA_OK(cudaMemcpy(local.data(),dA,local.size()*sizeof(double),cudaMemcpyDeviceToHost));cudaFree(dA);cudaFree(dP); MPI_Gatherv(local.data(),counts[rank],MPI_DOUBLE,rank?nullptr:full.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    if(!rank){for(size_t i=0;i<n;++i)for(size_t j=i+1;j<n;++j)full[i*n+j]=0.0;auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start).count();printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",ms,(double)n*n*n/3.0/(ms/1000.0)/1e9);if(printResults)print_results(full,"CholeskyL");if(validate){printf("Validating result...\nValidation: %s\n",validateCholesky(full,original,n)?"PASSED":"FAILED");}}
    MPI_Finalize(); return 0;
}
