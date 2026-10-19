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

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

__global__ void choleskyStep(double* a, size_t n, size_t k, int rank, int ranks, int phase, int* failed) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (phase == 0) {
        if (index == 0 && rank == (int)(k % ranks)) {
            const double v = a[k*n+k];
            if (v <= 0.0) { *failed = 1; return; }
            a[k*n+k] = sqrt(v);
        }
    } else if (phase == 1) {
        if (index < n && index > k && (int)(index % ranks) == rank)
            a[index*n+k] /= a[k*n+k];
    } else if (index < n*n) {
        const size_t i = index / n, j = index % n;
        if (i > k && j > k && (int)(i % ranks) == rank)
            a[i*n+j] -= a[i*n+k] * a[j*n+k];
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int ranks) {
    // A is stored in row-major order
    double* dA = nullptr;
    int* dFailed = nullptr;
    cudaError_t ce = cudaMalloc(&dA, n*n*sizeof(double));
    if (ce != cudaSuccess || cudaMalloc(&dFailed, sizeof(int)) != cudaSuccess) return false;
    cudaMemcpy(dA, A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice);
    int failed = 0;
    cudaMemcpy(dFailed, &failed, sizeof(int), cudaMemcpyHostToDevice);
    const int threads = 256;
    for (size_t i = 0; i < n; ++i) {
        choleskyStep<<<1, threads>>>(dA,n,i,rank,ranks,0,dFailed);
        cudaDeviceSynchronize();
        cudaMemcpy(A.data(),dA,n*n*sizeof(double),cudaMemcpyDeviceToHost);
        MPI_Bcast(A.data()+i*n,(int)n,MPI_DOUBLE,(int)(i%ranks),MPI_COMM_WORLD);
        cudaMemcpy(dA,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
        choleskyStep<<<(n+threads-1)/threads,threads>>>(dA,n,i,rank,ranks,1,dFailed);
        std::vector<double> col(n,0.0);
        // Gather the distributed pivot column before the trailing update.
        std::vector<double> local(n,0.0);
        cudaMemcpy(A.data(), dA, n*n*sizeof(double), cudaMemcpyDeviceToHost);
        for (size_t r=i; r<n; ++r) if ((int)(r%ranks)==rank) local[r]=A[r*n+i];
        MPI_Allreduce(local.data(), col.data(), (int)n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        for (size_t r=i+1;r<n;++r) A[r*n+i]=col[r];
        cudaMemcpy(dA,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
        choleskyStep<<<(n*n+threads-1)/threads,threads>>>(dA,n,i,rank,ranks,2,dFailed);
        cudaDeviceSynchronize();
        cudaMemcpy(&failed,dFailed,sizeof(int),cudaMemcpyDeviceToHost);
        MPI_Allreduce(MPI_IN_PLACE,&failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if (failed) { cudaFree(dA); cudaFree(dFailed); return false; }
        cudaMemcpy(A.data(),dA,n*n*sizeof(double),cudaMemcpyDeviceToHost);
        MPI_Bcast(A.data()+i*n,(int)n,MPI_DOUBLE,(int)(i%ranks),MPI_COMM_WORLD);
        cudaMemcpy(dA,A.data(),n*n*sizeof(double),cudaMemcpyHostToDevice);
        #pragma omp parallel for
        for (long long j=(long long)i+1;j<(long long)n;++j) A[i*n+j]=0.0;
    }
    
    cudaFree(dA); cudaFree(dFailed);
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
    MPI_Init(&argc, &argv);
    int rank=0, ranks=1;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&ranks);
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (rank==0) {
      printf("Cholesky Decomposition Benchmark\n");
      printf("Matrix size: %zu x %zu\n", n, n);
      printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank==0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    if (rank==0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    if(rank!=0) { MPI_Finalize(); return 0; }
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize(); return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize(); return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
