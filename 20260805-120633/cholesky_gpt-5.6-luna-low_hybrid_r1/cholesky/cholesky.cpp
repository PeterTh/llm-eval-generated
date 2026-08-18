#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>
#include "../common/results_output.hpp"

// One CUDA launch advances one Cholesky panel.  The panel itself is inherently
// sequential, while all rows below its diagonal are independent.
__global__ void chol_panel(double* a, size_t n, size_t j) {
    const size_t i = j + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double s = 0.0;
    for (size_t k = 0; k < j; ++k) s += a[i*n+k] * a[j*n+k];
    a[i*n+j] = (a[i*n+j] - s) / a[j*n+j];
}

static void cuda_check(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "CUDA error in %s: %s\n", where, cudaGetErrorString(e)); std::abort(); }
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank) {
    double* d = nullptr;
    cuda_check(cudaMalloc(&d, n*n*sizeof(double)), "cudaMalloc");
    cuda_check(cudaMemcpy(d, A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice), "H2D");
    for (size_t j = 0; j < n; ++j) {
        // Diagonal panel factorization is done by the host after copying only
        // the current row. This avoids a global device synchronization per dot.
        cuda_check(cudaMemcpy(A.data()+j*n, d+j*n, (j+1)*sizeof(double), cudaMemcpyDeviceToHost), "panel D2H");
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (size_t k = 0; k < j; ++k) sum += A[j*n+k] * A[j*n+k];
        const double v = A[j*n+j] - sum;
        if (v <= 0.0) { if (rank == 0) std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j); cudaFree(d); return false; }
        A[j*n+j] = std::sqrt(v);
        cuda_check(cudaMemcpy(d+j*n+j, A.data()+j*n+j, sizeof(double), cudaMemcpyHostToDevice), "diagonal H2D");
        const int threads = 256;
        chol_panel<<<(n-j-1+threads-1)/threads, threads>>>(d, n, j);
        cuda_check(cudaGetLastError(), "chol_panel");
        cuda_check(cudaDeviceSynchronize(), "chol_panel sync");
    }
    cuda_check(cudaMemcpy(A.data(), d, n*n*sizeof(double), cudaMemcpyDeviceToHost), "D2H");
    cuda_check(cudaFree(d), "cudaFree");
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n; ++i)
        for (size_t j=i+1; j<n; ++j) A[(size_t)i*n+j]=0.0;
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n); unsigned int seed=42;
    // Keep the seeded stream identical to the reference implementation.
    for (size_t i=0; i<n*n; ++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    #pragma omp parallel for schedule(static)
    for (long long i=0; i<(long long)n; ++i) for (size_t j=0;j<n;++j) {
        double s=0; for(size_t k=0;k<n;++k) s+=B[(size_t)i*n+k]*B[j*n+k];
        A[(size_t)i*n+j]=s+(i==(long long)j ? (double)n : 0.0);
    }
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& O,size_t n) {
    double ma=0, mr=0;
    #pragma omp parallel for reduction(max:ma,mr) schedule(static)
    for(long long i=0;i<(long long)n*n;++i){ size_t r=i/n,c=i%n; double s=0; for(size_t k=0;k<n;++k)s+=L[r*n+k]*L[c*n+k]; double e=fabs(s-O[i]); ma=std::max(ma,e); mr=std::max(mr,e/(fabs(O[i])+1e-10)); }
    printf("Max absolute error: %.10e\nMax relative error: %.10e\n",ma,mr); return mr<=1e-6;
}
void printUsage(const char*p){printf("Usage: %s [options]\nOptions:\n  -n <num> Matrix size (default: 512)\n  -v Validation\n  -r Print results\n  -h Help\n",p);}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv); int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    size_t n=512; bool val=false,pr=false; for(int i=1;i<argc;++i){if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]); else if(!strcmp(argv[i],"-v"))val=true; else if(!strcmp(argv[i],"-r"))pr=true; else if(!strcmp(argv[i],"-h")){if(!rank)printUsage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)printf("Unknown option: %s\n",argv[i]);MPI_Finalize();return 1;}}
    if(!rank){printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,val?"enabled":"disabled");}
    std::vector<double>A(n*n),O; generatePositiveDefiniteMatrix(A,n); if(val)O=A;
    if(!rank)printf("Computing Cholesky decomposition using MPI ranks=%d, OpenMP threads=%d, CUDA\n",size,omp_get_max_threads());
    MPI_Barrier(MPI_COMM_WORLD); auto st=std::chrono::high_resolution_clock::now(); bool ok=choleskyDecomposition(A,n,rank); int good=ok; MPI_Allreduce(MPI_IN_PLACE,&good,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD); MPI_Bcast(A.data(),(int)(n*n),MPI_DOUBLE,0,MPI_COMM_WORLD); auto en=std::chrono::high_resolution_clock::now();
    if(!good){MPI_Finalize();return 1;} if(!rank){double sec=std::chrono::duration<double>(en-st).count(); printf("Computation time: %ld ms\nPerformance: %.3f GFLOPS\n",(long)(sec*1000),n*n*n/3.0/sec/1e9); if(pr)print_results(A,"CholeskyL"); if(val){printf("Validating result...\n"); bool v=validateCholesky(A,O,n); printf("Validation: %s\n",v?"PASSED":"FAILED"); MPI_Finalize(); return v?0:1;}}
    MPI_Finalize(); return 0;
}
