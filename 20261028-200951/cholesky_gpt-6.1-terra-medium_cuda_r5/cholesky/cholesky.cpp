#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA libraries operate on column-major matrices. The generated matrix is
// symmetric, so its row-major storage can be used directly by these routines.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <climits>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define BLAS_CHECK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d (%s:%d)\n", int(e), __FILE__, __LINE__); exit(1); } } while (0)
#define SOLVER_CHECK(call) do { cusolverStatus_t e = (call); if (e != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER error %d (%s:%d)\n", int(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void addDiagonal(double* a, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) a[i * n + i] += double(n);
}

__global__ void clearUpper(double* a, size_t n) {
    size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = n * n;
    if (idx < count && idx % n > idx / n) a[idx] = 0.0;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    if (!n) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double &x : B) x = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    double *dB, *dA;
    const size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));
    cublasHandle_t blas;
    BLAS_CHECK(cublasCreate(&blas));
    const double one = 1.0, zero = 0.0;
    // B is row-major, so column-major B^T * B gives the desired result.
    BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, int(n), int(n), int(n),
                           &one, dB, int(n), dB, int(n), &zero, dA, int(n)));
    addDiagonal<<<(n + 255) / 256, 256>>>(dA, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    BLAS_CHECK(cublasDestroy(blas));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (!n) return true;
    const size_t bytes = n * n * sizeof(double);
    double *dA, *work;
    int *dInfo, info, lwork;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dInfo, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    cusolverDnHandle_t solver;
    SOLVER_CHECK(cusolverDnCreate(&solver));
    // Upper column-major factor is the lower row-major factor.
    SOLVER_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, int(n), dA, int(n), &lwork));
    CUDA_CHECK(cudaMalloc(&work, size_t(lwork) * sizeof(double)));
    SOLVER_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, int(n), dA, int(n), work, lwork, dInfo));
    CUDA_CHECK(cudaMemcpy(&info, dInfo, sizeof(info), cudaMemcpyDeviceToHost));
    if (info == 0) {
        clearUpper<<<(n * n + 255) / 256, 256>>>(dA, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    } else if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
    } else {
        fprintf(stderr, "cuSOLVER potrf: invalid argument %d\n", -info);
    }
    CUDA_CHECK(cudaFree(work));
    CUDA_CHECK(cudaFree(dInfo));
    CUDA_CHECK(cudaFree(dA));
    SOLVER_CHECK(cusolverDnDestroy(solver));
    return info == 0;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    if (n) {
        double *dL, *dR;
        const size_t bytes = n * n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&dL, bytes));
        CUDA_CHECK(cudaMalloc(&dR, bytes));
        CUDA_CHECK(cudaMemcpy(dL, L.data(), bytes, cudaMemcpyHostToDevice));
        cublasHandle_t blas;
        BLAS_CHECK(cublasCreate(&blas));
        const double one = 1.0, zero = 0.0;
        BLAS_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, int(n), int(n), int(n),
                               &one, dL, int(n), dL, int(n), &zero, dR, int(n)));
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), dR, bytes, cudaMemcpyDeviceToHost));
        BLAS_CHECK(cublasDestroy(blas));
        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dR));
    }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
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
    
    if (n > INT_MAX || (n && n > SIZE_MAX / n / sizeof(double))) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
