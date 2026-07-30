#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA-accelerated Cholesky decomposition (left-looking algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular.

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _err = (call);                                                \
        if (_err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(_err));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// ---------------------------------------------------------------------------
// Kernel: process one column j of the matrix.
//   1. Diagonal:  shared-memory reduction for A[j][j] = sqrt(A[j][j] - sum)
//   2. Off-diagonal:  A[i][j] = (A[i][j] - sum) / A[j][j]  (i = j+1 … n-1)
// ---------------------------------------------------------------------------
__global__ void choleskyColumnKernel(double* A, int n, int j) {
    extern __shared__ double shared[];
    int tid = threadIdx.x;
    int nthreads = blockDim.x;

    // ----------  diagonal element  ----------
    double partial = 0.0;
    for (int k = tid; k < j; k += nthreads) {
        double v = A[j * n + k];
        partial += v * v;
    }
    shared[tid] = partial;
    __syncthreads();

    for (int s = nthreads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        A[j * n + j] = sqrt(A[j * n + j] - shared[0]);
    }
    __syncthreads();

    // ----------  off-diagonal elements  ----------
    for (int i = j + 1 + tid; i < n; i += nthreads) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) {
            sum += A[i * n + k] * A[j * n + k];
        }
        A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
    }
}

// ---------------------------------------------------------------------------
// GPU-accelerated Cholesky decomposition (left-looking)
// ---------------------------------------------------------------------------
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    int n_int = static_cast<int>(n);

    double* d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    constexpr int blockSize = 256;
    size_t shmemBytes = blockSize * sizeof(double);

    for (int j = 0; j < n_int; ++j) {
        choleskyColumnKernel<<<1, blockSize, shmemBytes>>>(d_A, n_int, j);
    }

    CUDA_CHECK(cudaMemcpy(A.data(), d_A, n * n * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));

    // Zero the upper-triangular part on CPU (O(n²) vs O(n³) decomposition)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
        }
    }

    // Positive-definiteness check
    for (size_t j = 0; j < n; ++j) {
        if (A[j * n + j] <= 0.0 || std::isnan(A[j * n + j])) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", j);
            return false;
        }
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
    
    // Warm-up: initialise CUDA context and JIT-compile kernels so that
    // the timed section measures only actual computation.
    printf("Initializing CUDA...\n");
    {
        double* dummy = nullptr;
        cudaMalloc(&dummy, sizeof(double));
        choleskyColumnKernel<<<1, 32, 32 * sizeof(double)>>>(dummy, 1, 0);
        cudaFree(dummy);
        cudaDeviceSynchronize();
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
