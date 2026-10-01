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

__global__ void updateRows(double* a, size_t n, size_t k, size_t first, size_t stride) {
    size_t i = k + 1 + first + (blockIdx.x * blockDim.x + threadIdx.x) * stride;
    if (i >= n) return;
    for (size_t j=k+1;j<=i;++j) a[i*n+j] -= a[i*n+k]*a[j*n+k];
}

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int deviceCount=0; cudaGetDeviceCount(&deviceCount);
    const bool gpu = deviceCount > 0;
    if (gpu) cudaSetDevice(rank % deviceCount);
    double *dA=nullptr;
    if (gpu && cudaMalloc(&dA, n*n*sizeof(double)) == cudaSuccess)
        cudaMemcpy(dA,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
    else dA=nullptr;

    // Right-looking rank-1 updates. Every rank owns a cyclic subset of rows;
    // the pivot column is reduced/broadcast before the next step.
    std::vector<double> col(n,0.0);
    bool ok=true;
    for (size_t k=0;k<n;++k) {
        if(rank==0) {
            double d=A[k*n+k];
            if (!(d>0.0)) ok=false;
            else A[k*n+k]=sqrt(d);
        }
        MPI_Bcast(&ok,1,MPI_C_BOOL,0,MPI_COMM_WORLD);
        if(!ok) { if(dA) cudaFree(dA); return false; }
        MPI_Bcast(&A[k*n+k],1,MPI_DOUBLE,0,MPI_COMM_WORLD);
        if (rank==0) {
            for(size_t i=k+1;i<n;++i) A[i*n+k]/=A[k*n+k];
        }
        for(size_t i=k+1;i<n;++i) col[i]=A[i*n+k];
        MPI_Bcast(col.data(),(int)n,MPI_DOUBLE,0,MPI_COMM_WORLD);
        for(size_t i=k+1;i<n;++i) A[i*n+k]=col[i];
        // Distribute rows across ranks and update locally with OpenMP. Keep the
        // full matrix replicated so the original result interface is retained.
        std::vector<double> delta(n*n,0.0);
        if (dA) {
            const size_t count = rank < n-k-1 ? (n-k-1-(size_t)rank+(size_t)ranks-1)/(size_t)ranks : 0;
            cudaMemcpy(dA,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
            if(count) updateRows<<<(count+255)/256,256>>>(dA,n,k,(size_t)rank,(size_t)ranks);
            cudaDeviceSynchronize();
            std::vector<double> updated(n*n);
            cudaMemcpy(updated.data(),dA,n*n*sizeof(double),cudaMemcpyDeviceToHost);
            for(size_t i=k+1+rank;i<n;i+=ranks)
                for(size_t j=k+1;j<=i;++j) delta[i*n+j]=updated[i*n+j]-A[i*n+j];
        } else {
            #pragma omp parallel for schedule(static)
            for(long long ii=(long long)k+1+rank; ii<(long long)n; ii+=ranks) {
                const size_t i=(size_t)ii;
                for(size_t j=k+1;j<=i;++j) delta[i*n+j]= -A[i*n+k]*A[j*n+k];
            }
        }
        MPI_Allreduce(MPI_IN_PLACE,delta.data(),(int)(n*n),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static)
        for(long long q=0;q<(long long)(n*n);++q) A[(size_t)q]+=delta[(size_t)q];
    }
    for(size_t i=0;i<n;++i) for(size_t j=i+1;j<n;++j) A[i*n+j]=0.0;
    if(dA) cudaFree(dA);
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if(rank==0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if(rank==0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    
    if(rank==0) printf("Cholesky Decomposition Benchmark\n");
    if(rank==0) { printf("Matrix size: %zu x %zu\n", n, n); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if(rank==0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    if(rank==0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if(rank==0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }
    
    if(rank==0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if(rank==0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank==0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        if(rank==0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (rank==0 && valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else if(rank==0) {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    MPI_Finalize();
    return 0;
}
