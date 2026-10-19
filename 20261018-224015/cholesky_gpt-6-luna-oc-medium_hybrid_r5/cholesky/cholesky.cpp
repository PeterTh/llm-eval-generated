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

// Each MPI rank owns a replicated matrix; rows are block-distributed among
// ranks, while CUDA updates the assigned rows and OpenMP parallelizes setup
// and validation. The pivot is broadcast after each dependency step.
__global__ void update_rows(double* a, size_t n, size_t pivot, size_t first, size_t stride,
                            const double* pivot_row, double diagonal, int rank, int ranks) {
    size_t local = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = pivot + 1 + local;
    if (row >= n || (row / stride) % ranks != rank) return;
    double sum = 0.0;
    for (size_t k = 0; k < pivot; ++k) sum += a[row * n + k] * pivot_row[k];
    a[row * n + pivot] = (a[row * n + pivot] - sum) / diagonal;
    (void)first;
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, int rank, int ranks) {
    double *deviceA = nullptr, *devicePivot = nullptr;
    if (cudaMalloc(&deviceA, A.size() * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&devicePivot, n * sizeof(double)) != cudaSuccess) return false;
    cudaMemcpy(deviceA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
    std::vector<double> pivotRow(n);
    bool success = true;
    for (size_t j = 0; j < n; ++j) {
        // The pivot owner computes its diagonal; all ranks receive the pivot row.
        if (rank == static_cast<int>((j / 1) % ranks)) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) sum += A[j*n+k] * A[j*n+k];
            double value = A[j*n+j] - sum;
            if (!(value > 0.0)) success = false;
            else A[j*n+j] = std::sqrt(value);
            #pragma omp parallel for schedule(static)
            for (long long k = 0; k < static_cast<long long>(j); ++k) pivotRow[k] = A[j*n+k];
            pivotRow[j] = success ? A[j*n+j] : 0.0;
        }
        int ok = success ? 1 : 0;
        MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
        success = ok != 0;
        if (!success) {
            if (rank == 0) std::printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            break;
        }
        MPI_Bcast(pivotRow.data(), static_cast<int>(n), MPI_DOUBLE,
                  static_cast<int>(j % ranks), MPI_COMM_WORLD);
        // Replicas need the same pivot row and lower-triangular prefix.
        for (size_t k = 0; k < j; ++k) A[j*n+k] = pivotRow[k];
        A[j*n+j] = pivotRow[j];
        cudaMemcpy(deviceA, A.data(), A.size()*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(devicePivot, pivotRow.data(), n*sizeof(double), cudaMemcpyHostToDevice);
        size_t rows = n - j - 1;
        if (rows) {
            update_rows<<<static_cast<unsigned>((rows+255)/256), 256>>>(deviceA,n,j,0,1,devicePivot,pivotRow[j],rank,ranks);
            if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) success = false;
        }
        cudaMemcpy(A.data(), deviceA, A.size()*sizeof(double), cudaMemcpyDeviceToHost);
        // Exchange the newly computed column entries so every rank starts the
        // next pivot with a consistent replicated factor.
        std::vector<double> column(n, 0.0);
        for (size_t r = j+1; r < n; ++r) {
            if (static_cast<int>(r % ranks) == rank) column[r] = A[r*n+j];
        }
        MPI_Allreduce(MPI_IN_PLACE, column.data(), static_cast<int>(n), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t r = j+1; r < n; ++r) A[r*n+j] = column[r];
    }
    if (success) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            for (size_t j = static_cast<size_t>(i) + 1; j < n; ++j) A[static_cast<size_t>(i)*n+j] = 0.0;
    }
    cudaFree(devicePivot);
    cudaFree(deviceA);
    return success;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n) {
    std::vector<double> B(n*n);
    unsigned int seed = 42;
    for (size_t i=0;i<n*n;++i) B[i]=(rand_r(&seed)/(double)RAND_MAX)-0.5;
    #pragma omp parallel for schedule(static)
    for (long long ii=0;ii<static_cast<long long>(n);++ii) {
        size_t i=static_cast<size_t>(ii);
        for(size_t j=0;j<n;++j) { double sum=0; for(size_t k=0;k<n;++k) sum+=B[i*n+k]*B[j*n+k]; A[i*n+j]=sum; }
    }
    #pragma omp parallel for schedule(static)
    for(long long i=0;i<static_cast<long long>(n);++i) A[i*n+i]+=n;
}

bool validateCholesky(const std::vector<double>& L,const std::vector<double>& original,size_t n) {
    double maxError=0, relError=0;
    #pragma omp parallel for reduction(max:maxError,relError) schedule(static)
    for(long long ii=0;ii<static_cast<long long>(n);++ii) for(size_t j=0;j<n;++j) {
        double sum=0; for(size_t k=0;k<n;++k) sum+=L[static_cast<size_t>(ii)*n+k]*L[j*n+k];
        double error=std::fabs(sum-original[static_cast<size_t>(ii)*n+j]);
        maxError=std::max(maxError,error); relError=std::max(relError,error/(std::fabs(original[static_cast<size_t>(ii)*n+j])+1e-10));
    }
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n",maxError,relError);
    if(relError>1e-6){std::printf("Validation failed: relative error too large\n");return false;} return true;
}

void printUsage(const char* name){std::printf("Usage: %s [-n num] [-v] [-r] [-h]\n",name);}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    size_t n=512; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){if(!std::strcmp(argv[i],"-n")&&i+1<argc)n=std::strtoull(argv[++i],nullptr,10);else if(!std::strcmp(argv[i],"-v"))validate=true;else if(!std::strcmp(argv[i],"-r"))printResults=true;else if(!std::strcmp(argv[i],"-h")){if(rank==0)printUsage(argv[0]);MPI_Finalize();return 0;}else{if(rank==0){std::printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);}MPI_Finalize();return 1;}}
    int deviceCount=0; cudaGetDeviceCount(&deviceCount); if(deviceCount>0) cudaSetDevice(rank%deviceCount);
    if(rank==0){std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n",n,n,validate?"enabled":"disabled");}
    std::vector<double>A(n*n),original;
    if(rank==0)std::printf("Generating positive definite matrix...\n"); generatePositiveDefiniteMatrix(A,n);
    if(validate)original=A;
    if(rank==0)std::printf("Computing Cholesky decomposition...\n"); MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
    bool success=choleskyDecomposition(A,n,rank,ranks); MPI_Barrier(MPI_COMM_WORLD); auto end=std::chrono::high_resolution_clock::now();
    if(!success){if(rank==0)std::printf("Cholesky decomposition failed\n");MPI_Finalize();return 1;}
    if(rank==0){double seconds=std::chrono::duration<double>(end-start).count();std::printf("Computation time: %ld ms\n",static_cast<long>(seconds*1000));std::printf("Performance: %.3f GFLOPS\n",seconds>0?(double)n*n*n/3.0/seconds/1e9:0);if(printResults)print_results(A,"CholeskyL");}
    int valid=1; if(validate&&rank==0){std::printf("Validating result...\n");valid=validateCholesky(A,original,n)?1:0;if(valid)std::printf("Validation: PASSED\n");else std::printf("Validation: FAILED\n");}
    MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return valid?0:1;
}
