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

// Simple Cholesky decomposition using CUDA (cuSOLVER)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Helper for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Helper for cuSOLVER error checking
#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = call; \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            fprintf(stderr, "CUSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, \
                    status); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Kernel to zero out upper triangular part
__global__ void zeroUpperTriangle(double* A, size_t n) {
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        if (col > row) { // Upper triangle (row < col)
             // Row-major: A[row * n + col]
             A[row * n + col] = 0.0;
        }
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Initialize cuSolver
    static cusolverDnHandle_t cusolverH = NULL;
    if (cusolverH == NULL) {
        CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));
    }

    double *d_A = NULL;
    int *d_info = NULL;
    int lwork = 0;
    double *d_work = NULL;
    int h_info = 0;

    // Allocate GPU memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeof(double) * n * n));
    CUDA_CHECK(cudaMalloc((void**)&d_info, sizeof(int)));

    // Copy A to GPU
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), sizeof(double) * n * n, cudaMemcpyHostToDevice));

    // Query working space
    // Use CUBLAS_FILL_MODE_LOWER.
    // Explanation:
    // Input is row-major symmetric matrix A.
    // A_row_major[i*n + j] stores A_ij.
    // Interpreted as column-major A_col_major, index k = i*n + j corresponds to element at row=j, col=i.
    // So A_col_major is actually A^T.
    // Since A is symmetric, A = A^T. So A_col_major is also A.
    // We want to compute Cholesky decomposition L * L^T = A.
    // cusolverDnDpotrf with CUBLAS_FILL_MODE_LOWER computes L such that L * L^T = A.
    // It reads/writes the lower triangular part of the matrix.
    // The "lower triangular part" of A_col_major corresponds to row >= col, i.e., j >= i.
    // In our row-major view, j >= i corresponds to the UPPER triangular part.
    // So if we use CUBLAS_FILL_MODE_LOWER on the column-major interpretation, we are operating on the UPPER triangle of the row-major data.
    // But we want L, which is lower triangular in the row-major result.
    // Wait. The result L is lower triangular.
    // If we get L in "lower triangular part" of column-major matrix (j >= i),
    // that means we get L stored in the upper triangle of our row-major array.
    // That's L^T in row-major!
    // So we would get L^T where we expect L.
    //
    // Let's reconsider.
    // We want L in lower triangle of row-major A (i >= j).
    // In column-major interpretation (row=j, col=i), i >= j means col >= row.
    // This is the UPPER triangular part of the column-major matrix.
    // So we should ask cusolver to compute the UPPER triangular factor U such that U^T * U = A.
    // Wait, potrf with UPPER computes U.
    // U is upper triangular in column-major (row <= col => j <= i).
    // This corresponds to lower triangular in row-major (i >= j).
    // So if we use CUBLAS_FILL_MODE_UPPER, we get U in the "upper" part of column-major A.
    // This "upper" part corresponds to the "lower" part of our row-major A.
    // So we get a matrix that is lower triangular in row-major.
    // Is this matrix L?
    // U^T * U = A.
    // Let's call our result M. M is lower triangular in row-major.
    // M corresponds to U in column-major.
    // So M_row_major[i][j] = U_col_major[j][i].
    // Since U is stored in column-major, U_col_major[j][i] is the element at row j, col i.
    // In row-major, this element is at index j + i*n? No.
    // Wait.
    // Let's just use the fact that A_row_major = A_col_major^T.
    // We want L such that L * L^T = A.
    // In column major land, we have A' = A^T = A.
    // We want L'.
    // If we use LOWER, we get L' such that L' * L'^T = A'.
    // L' is lower triangular.
    // We receive L' in the buffer.
    // In row-major land, this buffer is (L')^T.
    // Since L' is lower, (L')^T is upper.
    // So we would get an upper triangular matrix.
    // But we want a lower triangular matrix L.
    // So we want (L')^T to be L? No.
    // We want the result in row-major to be L.
    // So we want the result in column-major to be L^T (upper triangular).
    // So we want a matrix U' (upper triangular) such that (U')^T * U' = A' (or something similar).
    // Actually, Cholesky is unique. A = L * L^T = U^T * U.
    // If we get U' (upper) from cusolver, then U'^T * U' = A'.
    // In row-major, this U' is (U')^T = L_out.
    // So L_out is lower triangular.
    // And L_out * L_out^T = (U')^T * ((U')^T)^T = U'^T * U' = A' = A.
    // So YES! We need to use CUBLAS_FILL_MODE_UPPER.
    // This will give us U' in column-major (upper), which is L in row-major (lower).
    // And it satisfies L * L^T = A.

    CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(
        cusolverH,
        CUBLAS_FILL_MODE_UPPER, 
        n,
        d_A,
        n,
        &lwork));

    CUDA_CHECK(cudaMalloc((void**)&d_work, sizeof(double) * lwork));

    // Compute Cholesky
    CUSOLVER_CHECK(cusolverDnDpotrf(
        cusolverH,
        CUBLAS_FILL_MODE_UPPER,
        n,
        d_A,
        n,
        d_work,
        lwork,
        d_info));

    CUDA_CHECK(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (h_info != 0) {
        // ... error handling ...
    }

    // Zero out the upper triangular part (row-major).
    // We computed U' (upper in col-major), which is L (lower in row-major).
    // So the valid data is in the lower triangle of row-major A.
    // The upper triangle of row-major A (which is lower of col-major A) contains garbage/original data?
    // cusolverDnDpotrf with UPPER accesses only the upper triangle of the matrix (in col-major).
    // So the lower triangle (col-major) is untouched.
    // Lower triangle (col-major) is Upper triangle (row-major).
    // So the upper triangle of our row-major A contains original data.
    // We need to zero it out.

    dim3 block(16, 16);
    dim3 grid((n + 15) / 16, (n + 15) / 16);
    zeroUpperTriangle<<<grid, block>>>(d_A, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel before copy back? Not strictly needed if stream 0

    // Copy result back
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, sizeof(double) * n * n, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));
    // CUSOLVER_CHECK(cusolverDnDestroy(cusolverH)); // Static handle

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
    
    // Warmup CUDA
    // printf("Warming up CUDA...\n");
    {
        std::vector<double> dummy(16 * 16);
        generatePositiveDefiniteMatrix(dummy, 16);
        choleskyDecomposition(dummy, 16);
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
