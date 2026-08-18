#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include "../common/results_output.hpp"

// One CUDA thread owns one row of the current panel.  The dot product is
// deliberately kept in the thread: this preserves the left-looking
// Cholesky dependency while allowing all independent rows to run together.
__global__ void panel_kernel(double* a, size_t n, size_t k,
                             size_t first, size_t last, int* bad) {
    const size_t i = first + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= last || i < k) return;
    double s = 0.0;
    for (size_t q = 0; q < k; ++q) s += a[i*n+q] * a[k*n+q];
    if (i == k) {
        const double d = a[k*n+k] - s;
        if (d <= 0.0) { *bad = 1; return; }
        a[k*n+k] = sqrt(d);
    } else {
        a[i*n+k] = (a[i*n+k] - s) / a[k*n+k];
    }
}

static bool cuda_ok(cudaError_t e, const char* what, int rank) {
    if (e == cudaSuccess) return true;
    if (rank == 0) fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    return false;
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, MPI_Comm comm) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(comm, &rank); MPI_Comm_size(comm, &nranks);
    cudaSetDevice(rank % std::max(1, [] { int x=0; cudaGetDeviceCount(&x); return x; }()));
    double* dA = nullptr; int* dBad = nullptr; int bad = 0;
    if (!cuda_ok(cudaMalloc(&dA, A.size()*sizeof(double)), "cudaMalloc(matrix)", rank) ||
        !cuda_ok(cudaMalloc(&dBad, sizeof(int)), "cudaMalloc(status)", rank)) return false;
    cudaMemcpy(dA, A.data(), A.size()*sizeof(double), cudaMemcpyHostToDevice);
    std::vector<double> column(n), local_column(n);
    const size_t first = (n * rank) / nranks, last = (n * (rank+1)) / nranks;
    constexpr int threads = 256;
    for (size_t k = 0; k < n; ++k) {
        cudaMemset(dBad, 0, sizeof(int));
        const int blocks = last > k ? static_cast<int>((last - std::max(k, first) + threads - 1) / threads) : 0;
        if (last > k) panel_kernel<<<blocks, threads>>>(dA, n, k, std::max(k, first), last, dBad);
        if (!cuda_ok(cudaGetLastError(), "panel kernel", rank) ||
            !cuda_ok(cudaDeviceSynchronize(), "panel synchronization", rank)) { bad=1; break; }
        std::fill(local_column.begin(), local_column.end(), 0.0);
        if (last > k) cudaMemcpy2D(local_column.data()+std::max(k, first), sizeof(double),
                                   dA + std::max(k, first)*n + k, n*sizeof(double),
                                   sizeof(double), last-std::max(k, first), cudaMemcpyDeviceToHost);
        MPI_Allreduce(local_column.data(), column.data(), static_cast<int>(n), MPI_DOUBLE, MPI_SUM, comm);
        if (rank == 0 && column[k] <= 0.0) bad=1;
        MPI_Bcast(&bad, 1, MPI_INT, 0, comm);
        if (bad) break;
        cudaMemcpy2D(dA+k, n*sizeof(double), column.data()+k, sizeof(double),
                     sizeof(double), n-k, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(A.data(), dA, A.size()*sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dBad);
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n; ++i)
        for (size_t j=static_cast<size_t>(i)+1; j<n; ++j) A[static_cast<size_t>(i)*n+j]=0.0;
    return bad == 0;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n); unsigned int seed=42;
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n*n; ++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n; ++i) for (size_t j=0;j<n;++j) {
        double s=0; for(size_t k=0;k<n;++k) s+=B[i*n+k]*B[j*n+k]; A[i*n+j]=s;
    }
    #pragma omp parallel for
    for(long long i=0;i<(long long)n;++i) A[i*n+i]+=n;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& O,size_t n){
    double maxe=0,re=0;
    #pragma omp parallel for reduction(max:maxe,re) schedule(static)
    for(long long i=0;i<(long long)n*n;++i){size_t r=i/n,c=i%n; double s=0;for(size_t k=0;k<n;++k)s+=L[r*n+k]*L[c*n+k];double e=fabs(s-O[i]);maxe=std::max(maxe,e);re=std::max(re,e/(fabs(O[i])+1e-10));}
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxe,re); return re<=1e-6;
}
void printUsage(const char* p){printf("Usage: %s [-n num] [-v] [-r] [-h]\n",p);}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv); int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    size_t n=512; bool val=false, results=false;
    for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-n")&&i+1<argc)n=strtoull(argv[++i],nullptr,10);else if(!strcmp(argv[i],"-v"))val=true;else if(!strcmp(argv[i],"-r"))results=true;else if(!strcmp(argv[i],"-h")){if(!rank)printUsage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)printUsage(argv[0]);MPI_Finalize();return 1;}}
    std::vector<double>A(n*n), O; if(!rank){printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\n",n,n);generatePositiveDefiniteMatrix(A,n);if(val)O=A;}
    MPI_Bcast(A.data(),static_cast<int>(A.size()),MPI_DOUBLE,0,MPI_COMM_WORLD); if(val) O=A;
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now(); bool ok=choleskyDecomposition(A,n,MPI_COMM_WORLD); MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    long long ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(), worst=0; MPI_Reduce(&ms,&worst,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD);
    if(!rank){printf("Computation time: %lld ms\nPerformance: %.3f GFLOPS\n",worst,(double)n*n*n/3e9/(worst/1000.0));if(results)print_results(A,"CholeskyL");if(val)printf("Validation: %s\n",validateCholesky(A,O,n)?"PASSED":"FAILED");}
    MPI_Finalize(); return ok?0:1;
}
