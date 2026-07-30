#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// Block size for blocked right-looking Cholesky on GPU
// ---------------------------------------------------------------------------
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

// ---------------------------------------------------------------------------
// CUDA error checking helper (returns false from enclosing function on error)
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _err = call;                                            \
        if (_err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,          \
                    __LINE__, cudaGetErrorString(_err));                     \
            return false;                                                   \
        }                                                                   \
    } while (0)

// Helper: sequential in-place Cholesky for the first panelWidth columns of
// a panelRows × panelWidth sub-matrix stored densely (row-major).
// Returns false if the matrix is not positive definite.
static bool factorPanelCPU(double* panel, int panelRows, int panelWidth) {
    for (int k = 0; k < panelWidth; ++k) {
        // Diagonal element at (k, k)
        double sum = 0.0;
        for (int t = 0; t < k; ++t)
            sum += panel[k * panelWidth + t] * panel[k * panelWidth + t];
        double val = panel[k * panelWidth + k] - sum;
        if (val <= 0.0) return false;
        panel[k * panelWidth + k] = sqrt(val);

        // Off-diagonal elements in column k (rows k+1 .. panelRows-1)
        const double diagVal = panel[k * panelWidth + k];
        for (int i = k + 1; i < panelRows; ++i) {
            sum = 0.0;
            for (int t = 0; t < k; ++t)
                sum += panel[i * panelWidth + t] * panel[k * panelWidth + t];
            panel[i * panelWidth + k] =
                (panel[i * panelWidth + k] - sum) / diagVal;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// Kernel: right-looking symmetric rank-B trailing matrix update
//
//   A22[ i ][ j ] -= Σ_k  L21[ i ][ k ] · L21[ j ][ k ]
//
//   with  A22 = A[panelEnd:n][panelEnd:n]  (only lower triangle updated)
//         L21 = A[panelEnd:n][panelStart:panelEnd]
// ---------------------------------------------------------------------------
__global__ void trailingUpdateKernel(double* __restrict__ A, int n,
                                      int panelStart, int panelWidth) {
    // Global indices inside the trailing matrix (lower triangular part)
    int j = blockIdx.x * blockDim.x + threadIdx.x + panelStart + panelWidth;
    int i = blockIdx.y * blockDim.y + threadIdx.y + panelStart + panelWidth;

    if (i >= n || j >= n || i < j) return;

    double sum = 0.0;
    for (int k = 0; k < panelWidth; ++k) {
        sum += A[i * n + panelStart + k] * A[j * n + panelStart + k];
    }

    A[i * n + j] -= sum;
}

// ---------------------------------------------------------------------------
// Kernel: zero out the strictly upper triangular part
// ---------------------------------------------------------------------------
__global__ void zeroUpperKernel(double* __restrict__ A, int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// CUDA-accelerated Cholesky decomposition (blocked right-looking algorithm)
//
// Decomposes a positive-definite matrix A into L·L^T where L is lower
// triangular.  A is overwritten with L and the result is returned in the
// same row-major vector.
//
// Strategy:
//   The panel (up to BLOCK_SIZE columns) is factored on the CPU using the
//   original sequential algorithm.  The trailing sub-matrix is updated on
//   the GPU with a parallel rank-B symmetric update kernel.  This minimises
//   the number of kernel launches and avoids frequent device synchronisations.
// ---------------------------------------------------------------------------
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    const size_t matrixBytes = static_cast<size_t>(n) * n * sizeof(double);
    const int nn = static_cast<int>(n);
    const int panelWidth = BLOCK_SIZE;

    double* d_A = nullptr;

    // --- Device memory allocation and full matrix copy to GPU -------------
    CUDA_CHECK(cudaMalloc(&d_A, matrixBytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), matrixBytes,
                          cudaMemcpyHostToDevice));

    // --- Host-side panel workspace (dense, row-major) ---------------------
    // The panel is at most (n) rows and B columns.
    std::vector<double> panel(static_cast<size_t>(nn) * panelWidth);

    // --- Blocked right-looking Cholesky -----------------------------------
    for (int panelStart = 0; panelStart < nn; panelStart += panelWidth) {
        const int currentPanelWidth = min(panelWidth, nn - panelStart);
        const int panelEnd = panelStart + currentPanelWidth;
        const int panelRows = nn - panelStart;   // rows in the panel
        const int trailingSize = nn - panelEnd;

        // ---- Copy the current panel from GPU → CPU ----------------------
        // The panel is A[panelStart:nn][panelStart:panelStart+B].
        // On the device it is stored with stride nn; on the host densely.
        {
            const double* src =
                d_A + static_cast<size_t>(panelStart) * nn + panelStart;
            CUDA_CHECK(cudaMemcpy2D(
                panel.data(),                     // dst
                static_cast<size_t>(currentPanelWidth) * sizeof(double), // dst pitch
                src,                              // src
                static_cast<size_t>(nn) * sizeof(double),  // src pitch
                static_cast<size_t>(currentPanelWidth) * sizeof(double), // width
                panelRows,                        // height
                cudaMemcpyDeviceToHost));
        }

        // ---- Factor the panel on the CPU --------------------------------
        if (!factorPanelCPU(panel.data(), panelRows, currentPanelWidth)) {
            fprintf(stderr,
                    "Error: Matrix not positive definite at panel %d\n",
                    panelStart);
            cudaFree(d_A);
            return false;
        }

        // ---- Copy the factored panel back: CPU → GPU --------------------
        {
            double* dst =
                d_A + static_cast<size_t>(panelStart) * nn + panelStart;
            CUDA_CHECK(cudaMemcpy2D(
                dst,                              // dst
                static_cast<size_t>(nn) * sizeof(double),  // dst pitch
                panel.data(),                     // src
                static_cast<size_t>(currentPanelWidth) * sizeof(double), // src pitch
                static_cast<size_t>(currentPanelWidth) * sizeof(double), // width
                panelRows,                        // height
                cudaMemcpyHostToDevice));
        }

        // ---- GPU trailing sub-matrix update (symmetric rank-B) ----------
        if (trailingSize > 0) {
            constexpr int TDIM = 16;
            dim3 tdim(TDIM, TDIM);
            dim3 gdim((trailingSize + TDIM - 1) / TDIM,
                      (trailingSize + TDIM - 1) / TDIM);

            trailingUpdateKernel<<<gdim, tdim>>>(
                d_A, nn, panelStart, currentPanelWidth);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // --- GPU: zero out the strictly upper triangular part -----------------
    {
        constexpr int TDIM = 16;
        dim3 tdim(TDIM, TDIM);
        dim3 gdim((nn + TDIM - 1) / TDIM, (nn + TDIM - 1) / TDIM);
        zeroUpperKernel<<<gdim, tdim>>>(d_A, nn);
        CUDA_CHECK(cudaGetLastError());
    }

    // --- Copy result back to host -----------------------------------------
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, matrixBytes,
                          cudaMemcpyDeviceToHost));

    // --- Clean-up ---------------------------------------------------------
    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Warm-up: force CUDA driver initialisation outside the timed region
    int devCount = 0;
    cudaGetDeviceCount(&devCount);
    cudaFree(0);

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
