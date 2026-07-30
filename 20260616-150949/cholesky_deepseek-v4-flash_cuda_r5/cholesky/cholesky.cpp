#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA error checking macro
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: compute diagonal element A[j][j] for column j
// Launched with 1 block, 1 thread.
// ---------------------------------------------------------------------------
__global__ void cholesky_diag_kernel(double* A, int n, int j, int* error) {
    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
        sum += A[j * n + k] * A[j * n + k];
    }
    const double val = A[j * n + j] - sum;
    if (val <= 0.0) {
        *error = 1;
    }
    A[j * n + j] = sqrt(val);
}

// ---------------------------------------------------------------------------
// Kernel: compute off-diagonal elements A[i][j] for i > j in column j.
// Each thread handles one row.  Shared memory caches the pivot row
// (elements 0 .. j-1) to minimise global-memory reads.
// ---------------------------------------------------------------------------
__global__ void cholesky_off_diag_kernel(double* A, int n, int j) {
    extern __shared__ double pivot[];

    const int tid = threadIdx.x;
    const int bsz = blockDim.x;

    // Cooperative load of the pivot row into shared memory
    for (int k = tid; k < j; k += bsz) {
        pivot[k] = A[j * n + k];
    }
    __syncthreads();

    // Each thread handles exactly one row
    const int i = blockIdx.x * bsz + tid + j + 1;
    if (i >= n) return;

    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
        sum += A[i * n + k] * pivot[k];
    }
    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    A[j * n + i] = 0.0;   // zero upper-triangular counterpart
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// CUDA-accelerated Cholesky decomposition (right-looking, column-by-column).
// Decomposes positive-definite matrix A into L * L^T (A is overwritten).
//
// NOTE: d_A and d_error must be pre-allocated on the device.  Allocation
// is intentionally kept outside this function so that the benchmark timer
// in main() only measures the actual compute + necessary transfers.
// ---------------------------------------------------------------------------
bool choleskyDecomposition(double* d_A, double* h_A, const size_t n) {
    const int nn = static_cast<int>(n);
    const size_t bytes = nn * nn * sizeof(double);

    // Copy matrix to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    int* d_error = nullptr;
    CUDA_CHECK(cudaMalloc(&d_error, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_error, 0, sizeof(int)));

    // Process one column at a time
    for (int j = 0; j < nn; ++j) {
        // --- diagonal element ---
        cholesky_diag_kernel<<<1, 1>>>(d_A, nn, j, d_error);

        // --- off-diagonal elements for rows i > j ---
        const int remaining = nn - j - 1;
        if (remaining > 0) {
            const int threads = 256;
            const int blocks  = (remaining + threads - 1) / threads;
            const size_t smem = static_cast<size_t>(j) * sizeof(double);
            cholesky_off_diag_kernel<<<blocks, threads, smem>>>(d_A, nn, j);
        }
    }

    // Check for positive-definiteness error
    int h_error = 0;
    CUDA_CHECK(cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_error));

    if (h_error) {
        printf("Error: Matrix is not positive definite\n");
        // Copy whatever we have back for diagnostics
        CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));
        return false;
    }

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_A, d_A, bytes, cudaMemcpyDeviceToHost));
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
    
    // --- CUDA setup (outside timed section) ---
    // Warmup: trigger lazy CUDA context creation so it doesn't pollute timing
    {
        double* dummy = nullptr;
        cudaMalloc(&dummy, sizeof(double));
        cudaFree(dummy);
    }

    const size_t bytes = n * n * sizeof(double);
    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));

    // Perform Cholesky decomposition (timed section)
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(d_A, A.data(), n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaFree(d_A));
    
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
