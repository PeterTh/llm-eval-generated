#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

// A blocked, right-looking Cholesky.  Every MPI rank owns a contiguous part of
// the active block rows; the replicated panel is exchanged with MPI after each
// phase.  CUDA performs the panel factor, triangular solve and rank-b update.
constexpr int BLOCK = 32;

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void potrf_block(double* a, int n, int p, int bs, int* bad) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < bs; ++i) {
        double d = a[(p + i) * n + p + i];
        for (int k = 0; k < i; ++k) { double x = a[(p+i)*n+p+k]; d -= x*x; }
        if (d <= 0.0) { *bad = 1; return; }
        a[(p+i)*n+p+i] = sqrt(d);
        for (int r = i + 1; r < bs; ++r) {
            double x = a[(p+r)*n+p+i];
            for (int k = 0; k < i; ++k) x -= a[(p+r)*n+p+k]*a[(p+i)*n+p+k];
            a[(p+r)*n+p+i] = x / a[(p+i)*n+p+i];
        }
    }
}

__global__ void trsm_rows(double* a, int n, int p, int bs, int first, int last) {
    int r = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= last) return;
    for (int j = 0; j < bs; ++j) {
        double x = a[r*n+p+j];
        for (int k = 0; k < j; ++k) x -= a[r*n+p+k] * a[(p+j)*n+p+k];
        a[r*n+p+j] = x / a[(p+j)*n+p+j];
    }
}

__global__ void update_rows(double* a, int n, int p, int bs, int first, int last) {
    int j = p + bs + blockIdx.x * blockDim.x + threadIdx.x;
    int i = first + blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= last || j > i) return;
    double x = a[i*n+j];
    for (int k = 0; k < bs; ++k) x -= a[i*n+p+k] * a[j*n+p+k];
    a[i*n+j] = x;
}

static void range_for_rank(int first, int last, int rank, int ranks, int& lo, int& hi) {
    int rows = last - first;
    lo = first + (rows * rank) / ranks;
    hi = first + (rows * (rank + 1)) / ranks;
}

static bool cholesky_hybrid(std::vector<double>& a, int n, int rank, int ranks) {
    double* d = nullptr; int* bad = nullptr;
    CUDA_CHECK(cudaMalloc(&d, size_t(n)*n*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&bad, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d, a.data(), size_t(n)*n*sizeof(double), cudaMemcpyHostToDevice));
    for (int p = 0; p < n; p += BLOCK) {
        const int bs = std::min(BLOCK, n-p);
        if (rank == 0) { CUDA_CHECK(cudaMemset(bad, 0, sizeof(int))); potrf_block<<<1,1>>>(d,n,p,bs,bad); CUDA_CHECK(cudaGetLastError()); }
        int failed = 0;
        if (rank == 0) CUDA_CHECK(cudaMemcpy(&failed,bad,sizeof(int),cudaMemcpyDeviceToHost));
        MPI_Bcast(&failed,1,MPI_INT,0,MPI_COMM_WORLD);
        if (failed) { if(rank==0) std::printf("Error: Matrix is not positive definite at diagonal element %d\n",p); cudaFree(bad); cudaFree(d); return false; }
        // Publish the completed diagonal block before distributed panel solves.
        if (rank == 0) CUDA_CHECK(cudaMemcpy(a.data()+size_t(p)*n, d+size_t(p)*n, size_t(bs)*n*sizeof(double), cudaMemcpyDeviceToHost));
        MPI_Bcast(a.data()+size_t(p)*n, bs*n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d+size_t(p)*n, a.data()+size_t(p)*n, size_t(bs)*n*sizeof(double), cudaMemcpyHostToDevice));
        int lo, hi; range_for_rank(p+bs,n,rank,ranks,lo,hi);
        if (lo < hi) { trsm_rows<<<(hi-lo+255)/256,256>>>(d,n,p,bs,lo,hi); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
        // Row ownership is contiguous, therefore one Allgatherv exchanges whole rows efficiently.
        std::vector<int> counts(ranks), displs(ranks);
        for(int q=0;q<ranks;++q) { int x,y; range_for_rank(p+bs,n,q,ranks,x,y); counts[q]=(y-x)*n; displs[q]=x*n; }
        if (lo < hi) CUDA_CHECK(cudaMemcpy(a.data()+size_t(lo)*n,d+size_t(lo)*n,size_t(hi-lo)*n*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,a.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        if (p+bs < n) CUDA_CHECK(cudaMemcpy(d+size_t(p+bs)*n,a.data()+size_t(p+bs)*n,size_t(n-p-bs)*n*sizeof(double),cudaMemcpyHostToDevice));
        if (lo < hi && p+bs < n) { dim3 t(16,16), g((n-(p+bs)+15)/16,(hi-lo+15)/16); update_rows<<<g,t>>>(d,n,p,bs,lo,hi); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); }
        if (lo < hi) CUDA_CHECK(cudaMemcpy(a.data()+size_t(lo)*n,d+size_t(lo)*n,size_t(hi-lo)*n*sizeof(double),cudaMemcpyDeviceToHost));
        MPI_Allgatherv(MPI_IN_PLACE,0,MPI_DATATYPE_NULL,a.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        // The next diagonal block may have been updated by another rank.
        if (p + bs < n)
            CUDA_CHECK(cudaMemcpy(d+size_t(p+bs)*n, a.data()+size_t(p+bs)*n,
                                  size_t(n-p-bs)*n*sizeof(double), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaFree(bad)); CUDA_CHECK(cudaFree(d)); return true;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, int n) {
    std::vector<double> b(size_t(n)*n); unsigned seed=42;
    for(double& x:b) x=(rand_r(&seed)/(double)RAND_MAX)-.5;
    #pragma omp parallel for schedule(static)
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) { double s=0; for(int k=0;k<n;++k) s+=b[size_t(i)*n+k]*b[size_t(j)*n+k]; a[size_t(i)*n+j]=s+(i==j?n:0); }
}

static bool validateCholesky(const std::vector<double>& l,const std::vector<double>& original,int n) {
    double maxerr=0, maxrel=0;
    #pragma omp parallel for reduction(max:maxerr,maxrel) schedule(static)
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) { double s=0; for(int k=0;k<=std::min(i,j);++k) s+=l[size_t(i)*n+k]*l[size_t(j)*n+k]; double e=std::fabs(s-original[size_t(i)*n+j]); maxerr=std::max(maxerr,e); maxrel=std::max(maxrel,e/(std::fabs(original[size_t(i)*n+j])+1e-10)); }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxerr,maxrel); return maxrel<=1e-6;
}
static void usage(const char* p) { std::printf("Usage: %s [-n num] [-v] [-r] [-h]\n",p); }
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int n=512; bool validate=false, results=false;
    for(int i=1;i<argc;++i) { if(!std::strcmp(argv[i],"-n")&&i+1<argc)n=std::atoi(argv[++i]); else if(!std::strcmp(argv[i],"-v"))validate=true; else if(!std::strcmp(argv[i],"-r"))results=true; else if(!std::strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;} else {if(!rank)usage(argv[0]);MPI_Finalize();return 1;} }
    if(n<=0){if(!rank)std::printf("Matrix size must be positive\n");MPI_Finalize();return 1;}
    int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices)); if(!devices){if(!rank)std::fprintf(stderr,"No CUDA device available\n");MPI_Abort(MPI_COMM_WORLD,2);} CUDA_CHECK(cudaSetDevice(rank%devices));
    std::vector<double>a(size_t(n)*n), original; if(!rank){std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nMPI ranks: %d, OpenMP threads/rank: %d\n",n,n,ranks,omp_get_max_threads());generatePositiveDefiniteMatrix(a,n);if(validate)original=a;} MPI_Bcast(a.data(),n*n,MPI_DOUBLE,0,MPI_COMM_WORLD); if(validate){if(rank)original.resize(size_t(n)*n);MPI_Bcast(original.data(),n*n,MPI_DOUBLE,0,MPI_COMM_WORLD);}
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now(); bool ok=cholesky_hybrid(a,n,rank,ranks); MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    int rc=ok?0:1; if(!rank&&ok){double sec=std::chrono::duration<double>(end-start).count();std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n",sec*1e3,(double(n)*n*n/3e9)/sec);if(results)print_results(a,"CholeskyL");if(validate){bool good=validateCholesky(a,original,n);std::printf("Validation: %s\n",good?"PASSED":"FAILED");rc=good?0:1;}} MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return rc;
}
