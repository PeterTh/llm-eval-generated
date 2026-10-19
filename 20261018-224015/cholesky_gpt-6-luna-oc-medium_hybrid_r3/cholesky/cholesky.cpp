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

__global__ void updateTrailing(double* a, const double* pivotRow, size_t n,
                               size_t pivot, size_t firstRow, size_t lastRow) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t count = (lastRow - firstRow) * (n - pivot - 1);
    if (idx >= count) return;
    const size_t row = firstRow + idx / (n - pivot - 1);
    const size_t col = pivot + 1 + idx % (n - pivot - 1);
    if (row > pivot && col <= row)
        a[row * n + col] -= a[row * n + pivot] * pivotRow[col];
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n, int rank, int ranks) {
    double *deviceA = nullptr, *devicePivot = nullptr;
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return false;
    cudaSetDevice(rank % devices);
    if (cudaMalloc(&deviceA, n * n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&devicePivot, n * sizeof(double)) != cudaSuccess) return false;
    cudaMemcpy(deviceA, A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice);
    std::vector<double> pivotRow(n);
    bool success = true;
    for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        #pragma omp parallel for reduction(+:sum) schedule(static)
        for (size_t k = 0; k < j; ++k) sum += A[j*n+k] * A[j*n+k];
        const double d = A[j*n+j] - sum;
        if (d <= 0.0) success = false;
        else {
            A[j*n+j] = std::sqrt(d);
            #pragma omp parallel for schedule(static)
            for (size_t row = j + 1; row < n; ++row) {
                double s = 0.0;
                for (size_t k = 0; k < j; ++k) s += A[row*n+k] * A[j*n+k];
                A[row*n+j] = (A[row*n+j] - s) / A[j*n+j];
            }
        }
        if (!success) {
            if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            break;
        }
        pivotRow[j] = A[j*n+j];
        for (size_t row=j+1; row<n; ++row) pivotRow[row] = A[row*n+j];
        MPI_Bcast(pivotRow.data(), (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        for (size_t row=j; row<n; ++row) A[row*n+j] = pivotRow[row];
        cudaMemcpy(deviceA, A.data(), n*n*sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(devicePivot, pivotRow.data(), n*sizeof(double), cudaMemcpyHostToDevice);
        const size_t first = j+1;
        const size_t last = n;
        if (first < last && j+1 < n) {
            const size_t work = (last-first)*(n-j-1);
            updateTrailing<<<(work+255)/256,256>>>(deviceA, devicePivot, n, j, first, last);
            cudaDeviceSynchronize();
            cudaMemcpy(A.data()+first*n, deviceA+first*n, (last-first)*n*sizeof(double), cudaMemcpyDeviceToHost);
        }
        MPI_Bcast(A.data()+first*n, (int)((last-first)*n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    cudaFree(deviceA); cudaFree(devicePivot);
    for (size_t i=0;i<n;++i) for(size_t k=i+1;k<n;++k) A[i*n+k]=0.0;
    return success;
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
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
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
            MPI_Finalize(); return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            MPI_Finalize(); return 1;
        }
    }
    
    if (rank == 0) printf("Cholesky Decomposition Benchmark\n");
    if (rank == 0) { printf("Matrix size: %zu x %zu\n", n, n); printf("Validation: %s\n", validate ? "enabled" : "disabled"); }
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, rank, ranks);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize(); return 1;
    }
    
    if (rank == 0) printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate && rank == 0) {
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
    
    MPI_Finalize(); return 0;
}
