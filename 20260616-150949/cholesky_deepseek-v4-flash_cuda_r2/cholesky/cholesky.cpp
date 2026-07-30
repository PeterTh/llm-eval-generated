#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA
#include <cuda_runtime.h>

// ---------------------------------------------------------------------------
// CUDA kernels for right-looking Cholesky decomposition
//
// The right-looking variant processes one column per iteration:
//   1.  A[j][j] = sqrt(A[j][j])
//   2.  A[i][j] = A[i][j] / A[j][j]     for i > j
//   3.  A[i][k] -= A[i][j] * A[k][j]    for i > j, k > j   (rank-1 SYRK)
// ---------------------------------------------------------------------------

// Step 1 – diagonal element
__global__ void diagKernel(double* A, int n, int j) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        A[j * n + j] = sqrt(A[j * n + j]);
}

// Step 2 – scale column j and copy to contiguous buffer d_col
//   d_col[i] = A[i][j] / A[j][j]   for i > j
__global__ void scaleColKernel(double* A, double* d_col, int n, int j) {
    int i = blockIdx.x * blockDim.x + threadIdx.x + j + 1;
    if (i >= n) return;
    double val = A[i * n + j] / A[j * n + j];
    A[i * n + j] = val;
    d_col[i]    = val;
}

// Step 3 – tiled rank-1 SYRK update
//   A[gi][gk] -= d_col[gi] * d_col[gk]    gi, gk in (j, n)
//
// Both column reads are from the contiguous d_col buffer (coalesced).
// The write to A is also coalesced by mapping threadIdx.x to the column
// dimension (so consecutive threads write consecutive doubles).
template<int TX, int TY>
__global__ void syrkKernel(double* A, const double* d_col, int n, int j) {
    int gi = blockIdx.y * TY + threadIdx.y + j + 1;
    int gk = blockIdx.x * TX + threadIdx.x + j + 1;
    if (gi >= n || gk >= n) return;

    double lij = d_col[gi];
    double lkj = d_col[gk];
    A[gi * n + gk] -= lij * lkj;
}

// ---------------------------------------------------------------------------
// CUDA-accelerated Cholesky decomposition (right-looking)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
// ---------------------------------------------------------------------------

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int n_int = static_cast<int>(n);
    double*   d_A;
    double*   d_col;   // contiguous column buffer

    cudaMalloc(&d_A,   static_cast<size_t>(n) * n * sizeof(double));
    cudaMalloc(&d_col, static_cast<size_t>(n) * sizeof(double));
    cudaMemcpy(d_A, A.data(),
               static_cast<size_t>(n) * n * sizeof(double),
               cudaMemcpyHostToDevice);

    constexpr int BLOCK_1D = 256;
    constexpr int TX       = 32;       // tile column dimension (warp-aligned)
    constexpr int TY       = 8;        // tile row    dimension

    // Right-looking Cholesky – three kernels per column.
    for (int j = 0; j < n_int; ++j) {
        const int trailing = n_int - j - 1;
        diagKernel<<<1, 1>>>(d_A, n_int, j);
        if (trailing > 0) {
            const int blocks1d = (trailing + BLOCK_1D - 1) / BLOCK_1D;
            scaleColKernel<<<blocks1d, BLOCK_1D>>>(d_A, d_col, n_int, j);
            const dim3 grid((trailing + TX - 1) / TX,
                            (trailing + TY - 1) / TY);
            syrkKernel<TX, TY><<<grid, dim3(TX, TY)>>>(d_A, d_col, n_int, j);
        }
    }

    // Copy result back and clean up
    cudaMemcpy(A.data(), d_A,
               static_cast<size_t>(n) * n * sizeof(double),
               cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < n; ++i)
        for (size_t j = i + 1; j < n; ++j)
            A[i * n + j] = 0.0;
    cudaFree(d_col);
    cudaFree(d_A);
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
    
    // Eagerly initialise CUDA context so it doesn't pollute the timed section
    cudaFree(0);

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
