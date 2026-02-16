#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        return false; \
    } \
} while(0)

// GPU kernels: compute diagonal element for column k (row-major storage)
__global__ void kernel_compute_diag(double* A, int n, int k) {
    // Single thread computes diagonal (cheap compared to overall work)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double sum = 0.0;
        for (int t = 0; t < k; ++t) {
            double v = A[k * n + t];
            sum += v * v;
        }
        A[k * n + k] = A[k * n + k] - sum;
        // Do not take sqrt here to allow host to check positivity and report
        // Replace with sqrt after check on host
    }
}

// Compute off-diagonal elements A[i, k] for i = k+1..n-1
__global__ void kernel_compute_offdiag(double* A, int n, int k) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = k + 1 + idx;
    if (i >= n) return;

    double sum = 0.0;
    for (int t = 0; t < k; ++t) {
        sum += A[i * n + t] * A[k * n + t];
    }
    // Use the updated diagonal A[k,k] (already sqrt'ed on host)
    A[i * n + k] = (A[i * n + k] - sum) / A[k * n + k];
}

// Zero the upper triangular part
__global__ void kernel_zero_upper(double* A, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx / n;
    int j = idx % n;
    if (i < n && j < n && j > i) {
        A[i * n + j] = 0.0;
    }
}

// GPU-accelerated Cholesky decomposition. Returns true on success.
bool choleskyDecomposition(std::vector<double>& A_host, const size_t n) {
    const size_t N = n * n;
    double* dA = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&dA, sizeof(double) * N));
    CUDA_CHECK(cudaMemcpy(dA, A_host.data(), sizeof(double) * N, cudaMemcpyHostToDevice));

    const int threads = 256;

    for (int k = 0; k < (int)n; ++k) {
        // Compute diagonal (device writes A[k,k] = original - sum)
        kernel_compute_diag<<<1, 32>>>(dA, (int)n, k);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy diagonal element to host to check positivity and take sqrt
        double diagVal;
        CUDA_CHECK(cudaMemcpy(&diagVal, dA + k * n + k, sizeof(double), cudaMemcpyDeviceToHost));

        if (!(diagVal > 0.0)) {
            fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %d (diag=%e)\n", k, diagVal);
            cudaFree(dA);
            return false;
        }

        diagVal = sqrt(diagVal);
        CUDA_CHECK(cudaMemcpy(dA + k * n + k, &diagVal, sizeof(double), cudaMemcpyHostToDevice));

        // Update column k for rows i = k+1..n-1 in parallel
        int remaining = (int)n - (k + 1);
        if (remaining > 0) {
            int blocks = (remaining + threads - 1) / threads;
            kernel_compute_offdiag<<<blocks, threads>>>(dA, (int)n, k);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }

    // Zero upper triangular part
    int total = (int)n * (int)n;
    int blocks = (total + threads - 1) / threads;
    kernel_zero_upper<<<blocks, threads>>>(dA, (int)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(A_host.data(), dA, sizeof(double) * N, cudaMemcpyDeviceToHost));
    cudaFree(dA);
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
