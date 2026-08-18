#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// A replicated matrix plus block-row ownership avoids a process-grid transpose
// at every panel.  Only the owner writes a block row; completed tiles are then
// broadcast so every rank has the panels required by later steps.
constexpr int BLOCK = 64;

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

__global__ void potrf_block(double* a, int n, int k, int bs, int* failed) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < bs; ++i) {
        double s = a[(k + i) * n + k + i];
        for (int p = 0; p < i; ++p) { double x = a[(k + i) * n + k + p]; s -= x * x; }
        if (s <= 0.0) { *failed = 1; return; }
        a[(k + i) * n + k + i] = sqrt(s);
        for (int r = i + 1; r < bs; ++r) {
            double v = a[(k + r) * n + k + i];
            for (int p = 0; p < i; ++p) v -= a[(k + r) * n + k + p] * a[(k + i) * n + k + p];
            a[(k + r) * n + k + i] = v / a[(k + i) * n + k + i];
        }
    }
}

__global__ void trsm_block(double* a, int n, int row, int k, int rb, int kb) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rb) return;
    for (int j = 0; j < kb; ++j) {
        double v = a[(row + r) * n + k + j];
        for (int p = 0; p < j; ++p) v -= a[(row + r) * n + k + p] * a[(k + j) * n + k + p];
        a[(row + r) * n + k + j] = v / a[(k + j) * n + k + j];
    }
}

__global__ void syrk_update(double* a, int n, int row, int col, int k, int rb, int cb, int kb) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= rb || c >= cb) return;
    double v = a[(row + r) * n + col + c];
    for (int p = 0; p < kb; ++p) v -= a[(row + r) * n + k + p] * a[(col + c) * n + k + p];
    a[(row + r) * n + col + c] = v;
}

static void generatePositiveDefiniteMatrix(std::vector<double>& a, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) b[i] = rand_r(&seed) / static_cast<double>(RAND_MAX) - .5;
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.;
            #pragma omp simd reduction(+:sum)
            for (size_t p = 0; p < n; ++p) sum += b[i*n+p] * b[j*n+p];
            a[i*n+j] = sum;
        }
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) a[i*n+i] += n;
}

static bool distributedCholesky(std::vector<double>& a, int n, int rank, int ranks) {
    double* da = nullptr; int *df = nullptr, failed = 0;
    CUDA_CHECK(cudaMalloc(&da, a.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&df, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), a.size()*sizeof(double), cudaMemcpyHostToDevice));
    const dim3 threads(16, 16);
    for (int k = 0; k < n; k += BLOCK) {
        const int kb = std::min(BLOCK, n-k), owner = (k/BLOCK) % ranks;
        if (rank == owner) { CUDA_CHECK(cudaMemset(df, 0, sizeof(int))); potrf_block<<<1,1>>>(da,n,k,kb,df); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(&failed,df,sizeof(int),cudaMemcpyDeviceToHost)); }
        MPI_Bcast(&failed, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (failed) { cudaFree(df); cudaFree(da); return false; }
        // Diagonal tile is strided: broadcast it one row at a time.
        for (int r=0; r<kb; ++r) { if (rank==owner) CUDA_CHECK(cudaMemcpy(a.data()+static_cast<size_t>(k+r)*n+k, da+static_cast<size_t>(k+r)*n+k, kb*sizeof(double), cudaMemcpyDeviceToHost)); MPI_Bcast(a.data()+static_cast<size_t>(k+r)*n+k, kb, MPI_DOUBLE, owner, MPI_COMM_WORLD); CUDA_CHECK(cudaMemcpy(da+static_cast<size_t>(k+r)*n+k, a.data()+static_cast<size_t>(k+r)*n+k, kb*sizeof(double), cudaMemcpyHostToDevice)); }
        for (int row=k+kb; row<n; row+=BLOCK) {
            const int rb=std::min(BLOCK,n-row), rowOwner=(row/BLOCK)%ranks;
            if(rank==rowOwner) { trsm_block<<<(rb+127)/128,128>>>(da,n,row,k,rb,kb); CUDA_CHECK(cudaGetLastError()); }
            for(int r=0;r<rb;++r) { if(rank==rowOwner) CUDA_CHECK(cudaMemcpy(a.data()+static_cast<size_t>(row+r)*n+k,da+static_cast<size_t>(row+r)*n+k,kb*sizeof(double),cudaMemcpyDeviceToHost)); MPI_Bcast(a.data()+static_cast<size_t>(row+r)*n+k,kb,MPI_DOUBLE,rowOwner,MPI_COMM_WORLD); CUDA_CHECK(cudaMemcpy(da+static_cast<size_t>(row+r)*n+k,a.data()+static_cast<size_t>(row+r)*n+k,kb*sizeof(double),cudaMemcpyHostToDevice)); }
        }
        for (int row=k+kb; row<n; row+=BLOCK) {
            const int rb=std::min(BLOCK,n-row), rowOwner=(row/BLOCK)%ranks;
            if (rank==rowOwner) for (int col=k+kb; col<=row; col+=BLOCK) { const int cb=std::min(BLOCK,n-col); syrk_update<<<dim3((cb+15)/16,(rb+15)/16),threads>>>(da,n,row,col,k,rb,cb,kb); CUDA_CHECK(cudaGetLastError()); }
            for (int col=k+kb; col<=row; col+=BLOCK) { const int cb=std::min(BLOCK,n-col); for(int r=0;r<rb;++r) { if(rank==rowOwner) CUDA_CHECK(cudaMemcpy(a.data()+static_cast<size_t>(row+r)*n+col,da+static_cast<size_t>(row+r)*n+col,cb*sizeof(double),cudaMemcpyDeviceToHost)); MPI_Bcast(a.data()+static_cast<size_t>(row+r)*n+col,cb,MPI_DOUBLE,rowOwner,MPI_COMM_WORLD); CUDA_CHECK(cudaMemcpy(da+static_cast<size_t>(row+r)*n+col,a.data()+static_cast<size_t>(row+r)*n+col,cb*sizeof(double),cudaMemcpyHostToDevice)); } }
        }
    }
    CUDA_CHECK(cudaFree(df)); CUDA_CHECK(cudaFree(da));
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<n;++i) for(int j=i+1;j<n;++j) a[i*n+j]=0.;
    return true;
}

static bool validateCholesky(const std::vector<double>& l,const std::vector<double>& a,size_t n) {
    double maxRel=0., maxAbs=0.;
    #pragma omp parallel for reduction(max:maxRel,maxAbs) schedule(static)
    for(long long i=0;i<(long long)n;++i) for(size_t j=0;j<n;++j) { double s=0.; for(size_t k=0;k<=std::min<size_t>(i,j);++k) s+=l[i*n+k]*l[j*n+k]; double e=std::fabs(s-a[i*n+j]); maxAbs=std::max(maxAbs,e); maxRel=std::max(maxRel,e/(std::fabs(a[i*n+j])+1e-10)); }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxAbs,maxRel); return maxRel<=1e-6;
}

int main(int argc,char** argv) {
    int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t ns=512; bool validate=false, results=false;
    for(int i=1;i<argc;++i) { if(!std::strcmp(argv[i],"-n")&&i+1<argc) ns=std::strtoull(argv[++i],nullptr,10); else if(!std::strcmp(argv[i],"-v")) validate=true; else if(!std::strcmp(argv[i],"-r")) results=true; else if(!std::strcmp(argv[i],"-h")) { if(!rank) std::printf("Usage: %s [-n size] [-v] [-r] [-h]\n",argv[0]); MPI_Finalize(); return 0; } else { if(!rank) std::fprintf(stderr,"Unknown option: %s\n",argv[i]); MPI_Finalize(); return 1; } }
    if(ns==0 || ns>static_cast<size_t>(std::numeric_limits<int>::max())) { if(!rank) std::fprintf(stderr,"Invalid matrix size\n"); MPI_Finalize(); return 1; }
    const int n=static_cast<int>(ns); int devices=0; CUDA_CHECK(cudaGetDeviceCount(&devices)); if(!devices) { if(!rank) std::fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,3); } CUDA_CHECK(cudaSetDevice(rank%devices));
    std::vector<double> a(ns*ns), orig; if(!rank) { std::printf("Cholesky Decomposition Benchmark (MPI ranks: %d, OpenMP threads: %d, CUDA devices: %d)\nMatrix size: %zu x %zu\n",ranks,omp_get_max_threads(),devices,ns,ns); generatePositiveDefiniteMatrix(a,ns); if(validate) orig=a; } MPI_Bcast(a.data(),static_cast<int>(ns*ns),MPI_DOUBLE,0,MPI_COMM_WORLD); if(validate && rank) orig=a;
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime(); bool ok=distributedCholesky(a,n,rank,ranks); double elapsed=MPI_Wtime()-start, maxElapsed=0.; MPI_Reduce(&elapsed,&maxElapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    int success=ok?1:0, allSuccess=0; MPI_Allreduce(&success,&allSuccess,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD); int rc=0; if(!allSuccess) { if(!rank) std::fprintf(stderr,"Cholesky decomposition failed\n"); rc=1; } else if(!rank) { std::printf("Computation time: %.3f ms\nPerformance: %.3f GFLOPS\n",maxElapsed*1000.,(double(ns)*ns*ns/3.)/(maxElapsed*1e9)); if(results) print_results(a,"CholeskyL"); if(validate) { bool valid=validateCholesky(a,orig,ns); std::printf("Validation: %s\n",valid?"PASSED":"FAILED"); rc=valid?0:1; } } MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return rc;
}
