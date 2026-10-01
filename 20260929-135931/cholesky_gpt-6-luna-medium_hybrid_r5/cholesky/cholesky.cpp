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

__global__ void updateRows(double* a, const double* pivot, size_t n, size_t k,
                           int rank, int ranks) {
    size_t i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && static_cast<int>(i % ranks) == rank) {
        double s = 0.0;
        for (size_t j = 0; j < k; ++j) s += a[i*n+j] * pivot[j];
        a[i*n+k] = (a[i*n+k] - s) / pivot[k];
    }
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int ranks) {
    int deviceCount=0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "No CUDA device is available on MPI rank %d\n", rank);
        return false;
    }
    cudaSetDevice(rank % deviceCount);
    double *deviceA = nullptr, *devicePivot = nullptr;
    cudaError_t err = cudaMalloc(&deviceA, A.size()*sizeof(double));
    if (err != cudaSuccess) { fprintf(stderr, "CUDA allocation failed: %s\n", cudaGetErrorString(err)); return false; }
    cudaMalloc(&devicePivot, n*sizeof(double));
    cudaMemcpy(deviceA, A.data(), A.size()*sizeof(double), cudaMemcpyHostToDevice);
    std::vector<double> pivot(n);
    bool ok = true;
    for (size_t k=0; k<n; ++k) {
        int owner = static_cast<int>(k % ranks);
        if (rank == owner) {
            double s=0.0;
            for (size_t j=0;j<k;++j) s += A[k*n+j]*A[k*n+j];
            double d=A[k*n+k]-s;
            if (d <= 0.0) ok=false;
            else {
                A[k*n+k]=std::sqrt(d);
                for (size_t j=0;j<k;++j) pivot[j]=A[k*n+j];
                pivot[k]=A[k*n+k];
            }
        }
        MPI_Bcast(&ok, 1, MPI_C_BOOL, owner, MPI_COMM_WORLD);
        if (!ok) { if (rank==0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", k); break; }
        MPI_Bcast(&A[k*n+k], 1, MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank==owner) for(size_t j=0;j<k;++j) pivot[j]=A[k*n+j];
        MPI_Bcast(pivot.data(), static_cast<int>(k+1), MPI_DOUBLE, owner, MPI_COMM_WORLD);
        if (rank==owner) for(size_t j=0;j<k;++j) A[k*n+j]=pivot[j];
        cudaMemcpy(devicePivot, pivot.data(), (k+1)*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(deviceA, A.data(), A.size()*sizeof(double), cudaMemcpyHostToDevice);
        size_t count = n-k-1;
        if(count) updateRows<<<(count+255)/256,256>>>(deviceA, devicePivot, n, k, rank, ranks);
        cudaDeviceSynchronize();
        cudaMemcpy(A.data(), deviceA, A.size()*sizeof(double), cudaMemcpyDeviceToHost);
        // Assemble disjoint row updates from all ranks for the next pivot.
        std::vector<double> local(A.size(),0.0);
        for(size_t i=0;i<n;++i) if(static_cast<int>(i%ranks)==rank)
            std::copy(A.begin()+i*n,A.begin()+(i+1)*n,local.begin()+i*n);
        MPI_Allreduce(local.data(), A.data(), static_cast<int>(A.size()), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    cudaFree(devicePivot); cudaFree(deviceA);
    return ok;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n); unsigned int seed=42;
    for(size_t i=0;i<n*n;++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<static_cast<long long>(n);++i) for(size_t j=0;j<n;++j) {
        double s=0.0; for(size_t k=0;k<n;++k) s+=B[static_cast<size_t>(i)*n+k]*B[j*n+k];
        A[static_cast<size_t>(i)*n+j]=s;
    }
    #pragma omp parallel for
    for(long long i=0;i<static_cast<long long>(n);++i) A[static_cast<size_t>(i)*n+i]+=n;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& orig,size_t n) {
    double maxError=0, relError=0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for(long long x=0;x<static_cast<long long>(n*n);++x) {
        size_t i=static_cast<size_t>(x)/n,j=static_cast<size_t>(x)%n; double s=0;
        for(size_t k=0;k<n;++k) s+=L[i*n+k]*L[j*n+k];
        double e=fabs(s-orig[static_cast<size_t>(x)]);
        maxError=std::max(maxError,e); relError=std::max(relError,e/(fabs(orig[static_cast<size_t>(x)])+1e-10));
    }
    printf("Max absolute error: %.10e\n",maxError); printf("Max relative error: %.10e\n",relError);
    if(relError>1e-6) { printf("Validation failed: relative error too large\n"); return false; } return true;
}

void printUsage(const char* p) { printf("Usage: %s [options]\n  -n <num> Matrix size (default: 512)\n  -v Validate\n  -r Print results\n  -h Show help\n",p); }
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t n=512; bool validate=false, printResults=false;
    for(int i=1;i<argc;++i) {
        if(strcmp(argv[i],"-n")==0&&i+1<argc) n=static_cast<size_t>(atoi(argv[++i]));
        else if(strcmp(argv[i],"-v")==0) validate=true; else if(strcmp(argv[i],"-r")==0) printResults=true;
        else if(strcmp(argv[i],"-h")==0) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if(rank==0) { printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\nGenerating positive definite matrix...\n",n,n,validate?"enabled":"disabled"); }
    std::vector<double> A(n*n), original;
    generatePositiveDefiniteMatrix(A,n); if(validate) original=A;
    if(rank==0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    bool success=choleskyDecomposition(A,n,rank,ranks); MPI_Barrier(MPI_COMM_WORLD);
    auto end=std::chrono::high_resolution_clock::now();
    if(rank==0&&!success) printf("Cholesky decomposition failed\n");
    if(!success) { MPI_Finalize(); return 1; }
    if(rank==0) {
        auto ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
        printf("Computation time: %ld ms\n",ms); double secs=ms/1000.0;
        printf("Performance: %.3f GFLOPS\n",secs>0?(double)n*n*n/3.0/secs/1e9:0.0);
        if(printResults) print_results(A,"CholeskyL");
        if(validate) { printf("Validating result...\n"); bool valid=validateCholesky(A,original,n); printf("Validation: %s\n",valid?"PASSED":"FAILED"); MPI_Finalize(); return valid?0:1; }
    }
    MPI_Finalize(); return 0;
}
