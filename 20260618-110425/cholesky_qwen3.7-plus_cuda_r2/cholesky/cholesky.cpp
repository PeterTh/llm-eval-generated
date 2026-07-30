#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// CPU-based Cholesky decomposition for diagonal block
void choleskyDiagBlockCPU(std::vector<double>& L11, int bs) {
    for (int i = 0; i < bs; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            if (i == j) {
                for (int k = 0; k < j; ++k) {
                    double v = L11[j * bs + k];
                    sum += v * v;
                }
                double val = L11[j * bs + j] - sum;
                if (val <= 0.0) {
                    L11[0] = -1.0; // Signal error
                    return;
                }
                L11[j * bs + j] = sqrt(val);
            } else {
                for (int k = 0; k < j; ++k) {
                    sum += L11[i * bs + k] * L11[j * bs + k];
                }
                L11[i * bs + j] = (L11[i * bs + j] - sum) / L11[j * bs + j];
            }
        }
    }
}

// GPU-parallelized blocked Cholesky decomposition using cuBLAS
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// A is stored in row-major order

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int ni = static_cast<int>(n);

    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to create cuBLAS handle\n");
        return false;
    }

    double* d_A = nullptr;
    if (cudaMalloc(&d_A, n * n * sizeof(double)) != cudaSuccess) {
        printf("Failed to allocate GPU memory\n");
        cublasDestroy(handle);
        return false;
    }

    // Copy A to GPU
    cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Blocked Cholesky decomposition
    // For large matrices, use smaller blocks to keep diagonal factorization fast on CPU
    const int blockSize = (n <= 1024) ? 128 : 256;

    for (int k = 0; k < ni; k += blockSize) {
        int bs = std::min(blockSize, ni - k);
        int m = ni - k - bs;

        // Step 1: Cholesky factorization of diagonal block A[k:k+bs, k:k+bs]
        // Extract diagonal block to contiguous storage using 2D memcpy
        double* d_block = nullptr;
        cudaMalloc(&d_block, bs * bs * sizeof(double));
        cudaMemcpy2D(d_block, bs * sizeof(double),
                     d_A + k * ni + k, ni * sizeof(double),
                     bs * sizeof(double), bs,
                     cudaMemcpyDeviceToDevice);

        // Copy to CPU for factorization
        std::vector<double> L11(bs * bs);
        cudaMemcpy(L11.data(), d_block, bs * bs * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_block);

        // Perform Cholesky on diagonal block (CPU)
        choleskyDiagBlockCPU(L11, bs);

        // Check for error
        if (L11[0] == -1.0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", k);
            cudaFree(d_A);
            cublasDestroy(handle);
            return false;
        }

        // Write back diagonal block
        cudaMalloc(&d_block, bs * bs * sizeof(double));
        cudaMemcpy(d_block, L11.data(), bs * bs * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy2D(d_A + k * ni + k, ni * sizeof(double),
                     d_block, bs * sizeof(double),
                     bs * sizeof(double), bs,
                     cudaMemcpyDeviceToDevice);
        cudaFree(d_block);

        if (m <= 0) continue;

        // Step 2: Triangular solve for sub-diagonal block
        // A21 = A[k+bs:n, k:k+bs] (m x bs, row-major)
        // Solve: A21 = A21 * L11^{-T}
        //
        // cuBLAS column-major view:
        //   d_A21 stores A21^T (bs x m)
        //   d_L11 stores L11^T (bs x bs, upper triangular)
        // We need: A21_new^T = L11^{-1} * A21_old^T
        // => solve L11 * X = B where B = A21^T
        // cuBLAS sees L11^T (upper), so use transA=T to access L11

        // Allocate contiguous L11 storage
        double* d_L11 = nullptr;
        cudaMalloc(&d_L11, bs * bs * sizeof(double));
        cudaMemcpy(d_L11, L11.data(), bs * bs * sizeof(double), cudaMemcpyHostToDevice);

        // Solve in-place on d_A21
        // cublasDtrsm: side=LEFT, uplo=UPPER (cuBLAS view of L11), transA=T (to get L11), diag=NONUNIT
        // Solves: L11 * X = alpha * B, B is bs x m in cuBLAS (d_A21 has leading dim ni)
        const double alpha_val = 1.0;
        cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT,
                    bs, m, &alpha_val, d_L11, bs,
                    d_A + (k + bs) * ni + k, ni);

        cudaFree(d_L11);

        // Step 3: Symmetric rank-k update of trailing submatrix
        // A22 = A[k+bs:n, k+bs:n] (m x m, row-major, symmetric)
        // A21 = A[k+bs:n, k:k+bs] (m x bs, row-major, already updated)
        // Update: A22 = A22 - A21 * A21^T
        //
        // cuBLAS column-major view:
        //   d_A22 stores A22^T = A22 (symmetric, m x m)
        //   d_A21 stores A21^T (bs x m)
        // We want: A22_cm = A22_cm - A21_cm^T * A21_cm
        // => C = alpha * op(A) * op(B) + beta * C
        //    transa=T, transb=N, A = A21_cm (bs x m), B = A21_cm (bs x m)
        //    C_cm = alpha * A21_cm^T * A21_cm + beta * C_cm = alpha * A21 * A21^T + beta * A22

        const double alpha_gemm = -1.0;
        const double beta_gemm = 1.0;
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                    m, m, bs,
                    &alpha_gemm,
                    d_A + (k + bs) * ni + k, ni,   // A21 in row-major, cuBLAS sees A21^T (bs x m), lda=ni
                    d_A + (k + bs) * ni + k, ni,   // same
                    &beta_gemm,
                    d_A + (k + bs) * ni + (k + bs), ni); // A22 in row-major, ldc=ni
    }

    // Copy result back to host
    cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // Zero out upper triangular part
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    cudaFree(d_A);
    cublasDestroy(handle);

    return true;
}

// Generate a symmetric positive definite matrix (GPU-accelerated B*B^T via cuBLAS)
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    const int ni = static_cast<int>(n);

    // Generate random matrix B on CPU (sequential rand_r for reproducibility)
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    double* d_B = nullptr;
    double* d_A = nullptr;
    cudaMalloc(&d_B, n * n * sizeof(double));
    cudaMalloc(&d_A, n * n * sizeof(double));

    // Copy B to GPU
    cudaMemcpy(d_B, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Compute A = B * B^T using cublasDsyrk
    // B stored row-major -> cuBLAS sees B_cm = B^T (column-major)
    // CUBLAS_OP_T: C = alpha * B_cm^T * B_cm = alpha * B * B^T
    // CUBLAS_FILL_MODE_UPPER: fills column-major upper triangle = row-major lower triangle
    const double alpha = 1.0;
    const double beta = 0.0;
    cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                ni, ni, &alpha, d_B, ni, &beta, d_A, ni);

    // Copy result back to host
    cudaMemcpy(A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost);

    // dsyrk fills column-major upper = row-major lower; symmetrize
    for (size_t i = 1; i < n; ++i) {
        for (size_t j = 0; j < i; ++j) {
            A[j * n + i] = A[i * n + j];
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }

    cudaFree(d_B);
    cudaFree(d_A);
    cublasDestroy(handle);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    const int ni = static_cast<int>(n);

    cublasHandle_t handle;
    cublasCreate(&handle);

    double* d_L = nullptr;
    double* d_recon = nullptr;
    cudaMalloc(&d_L, n * n * sizeof(double));
    cudaMalloc(&d_recon, n * n * sizeof(double));

    // Copy L to GPU
    cudaMemcpy(d_L, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Compute reconstructed = L * L^T using cublasDgemm
    // L stored row-major -> cuBLAS sees L_cm = L^T (column-major)
    // C = alpha * op(A) * op(B) + beta * C
    // transa=T, transb=N: C = L_cm^T * L_cm = L * L^T
    const double alpha_val = 1.0;
    const double beta_val = 0.0;
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                ni, ni, ni,
                &alpha_val, d_L, ni, d_L, ni,
                &beta_val, d_recon, ni);

    // Copy result back (dgemm fills entire matrix; result is symmetric so col-major = row-major)
    std::vector<double> reconstructed(n * n);
    cudaMemcpy(reconstructed.data(), d_recon, n * n * sizeof(double), cudaMemcpyDeviceToHost);

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

    bool valid = true;
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        valid = false;
    }

    cudaFree(d_L);
    cudaFree(d_recon);
    cublasDestroy(handle);
    return valid;
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
    printf("Accelerator: CUDA (cuBLAS)\n");

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

    cudaDeviceSynchronize();
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
