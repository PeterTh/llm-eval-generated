#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); }} while(0)

// GPU kernel: compute column k entries for rows i=k+1..n-1
__global__ void compute_column_kernel(double* A, size_t n, size_t k, double diag) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x + (k + 1);
    if (i >= n) return;
    double sum = 0.0;
    size_t base_i = i * n;
    size_t base_k = k * n;
    for (size_t t = 0; t < k; ++t) {
        sum += A[base_i + t] * A[base_k + t];
    }
    A[base_i + k] = (A[base_i + k] - sum) / diag;
}

// GPU kernel: rank-1 update of trailing submatrix for indices i,j >= k+1
__global__ void rank1_update_kernel(double* A, size_t n, size_t k) {
    size_t m = n - (k + 1);
    size_t gid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = m * m;
    if (gid >= total) return;
    size_t ii = gid / m; // 0..m-1
    size_t jj = gid % m;
    size_t i = ii + (k + 1);
    size_t j = jj + (k + 1);
    if (j <= i) {
        A[i * n + j] -= A[i * n + k] * A[j * n + k];
    }
}

// GPU-accelerated Cholesky decomposition (unblocked, with per-step GPU kernels)
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Allocate device memory and copy
    size_t bytes = n * n * sizeof(double);
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));

    // Temporary host buffer for reading row k
    std::vector<double> row_k(n);

    const int block = 256;
    for (size_t k = 0; k < n; ++k) {
        // Copy the first k elements of row k and the diagonal element to host
        if (k > 0) {
            CUDA_CHECK(cudaMemcpy(row_k.data(), dA + k * n, k * sizeof(double), cudaMemcpyDeviceToHost));
        }
        // Get diagonal element
        double diag_host = 0.0;
        CUDA_CHECK(cudaMemcpy(&diag_host, dA + k * n + k, sizeof(double), cudaMemcpyDeviceToHost));

        // Compute sum of squares for diagonal
        double sum = 0.0;
        for (size_t t = 0; t < k; ++t) sum += row_k[t] * row_k[t];

        double val = diag_host - sum;
        if (val <= 0.0) {
            fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %zu\n", k);
            cudaFree(dA);
            return false;
        }
        double diag = sqrt(val);
        // Write diagonal back to device
        CUDA_CHECK(cudaMemcpy(dA + k * n + k, &diag, sizeof(double), cudaMemcpyHostToDevice));

        // Compute column entries for rows i = k+1 .. n-1 in parallel
        int rows = (int)(n - (k + 1));
        if (rows > 0) {
            int g = (rows + block - 1) / block;
            compute_column_kernel<<<g, block>>>(dA, n, k, diag);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Rank-1 update of trailing submatrix
            size_t m = n - (k + 1);
            size_t total = m * m;
            int g2 = (int)((total + block - 1) / block);
            rank1_update_kernel<<<g2, block>>>(dA, n, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Zero upper triangular elements of row k on device (for consistency)
        // We'll copy row and zero on host then write back
        CUDA_CHECK(cudaMemcpy(row_k.data(), dA + k * n, n * sizeof(double), cudaMemcpyDeviceToHost));
        for (size_t j = k + 1; j < n; ++j) row_k[j] = 0.0;
        CUDA_CHECK(cudaMemcpy(dA + k * n, row_k.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    }

    // Copy back result
    CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dA));
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
