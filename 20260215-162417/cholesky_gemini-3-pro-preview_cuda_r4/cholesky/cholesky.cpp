#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return false; \
        } \
    } while (0)

// cuSOLVER error checking macro
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, \
                    status); \
            return false; \
        } \
    } while (0)

// Cholesky decomposition using CUDA cuSOLVER
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
bool choleskyDecomposition(cusolverDnHandle_t cusolverH, double* d_A, int* d_info, double* d_work, int lwork, const size_t n) {
    // Compute Cholesky factorization
    CUSOLVER_CHECK(cusolverDnDpotrf(
        cusolverH,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,
        n,
        d_work,
        lwork,
        d_info));

    // Check info
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        if (info < 0) {
            fprintf(stderr, "Error: %d-th parameter is wrong\n", -info);
        } else {
            fprintf(stderr, "Error: The leading minor of order %d is not positive definite\n", info);
        }
        return false;
    }
    
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

    // Initialize CUDA and allocate memory
    cusolverDnHandle_t cusolverH = NULL;
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    double *d_A = NULL;
    int *d_info = NULL;
    int lwork = 0;
    double *d_work = NULL;

    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(double) * n * n));
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));

    // Copy A to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice));

    // Query working space of potrf
    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        cusolverH,
        CUBLAS_FILL_MODE_LOWER,
        n,
        d_A,
        n,
        &lwork));

    CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(double) * lwork));
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    
    // Synchronize to ensure all init is done
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(cusolverH, d_A, d_info, d_work, lwork, n);
    
    // Synchronize to ensure computation is done
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, sizeof(double) * n * n, cudaMemcpyDeviceToHost));
    
    // Transpose result to match row-major lower triangular
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            // i > j
            // Move from upper (j, i) to lower (i, j)
            A[i * n + j] = A[j * n + i];
            A[j * n + i] = 0.0;
        }
    }

    // Cleanup
    if (d_A) cudaFree(d_A);
    if (d_info) cudaFree(d_info);
    if (d_work) cudaFree(d_work);
    if (cusolverH) cusolverDnDestroy(cusolverH);
    
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
