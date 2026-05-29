#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

// ===== CUDA Kernel =====
// Single-block Cholesky decomposition with grid-stride parallelism.
// All threads in the block cooperate. __syncthreads() ensures proper
// synchronization between diagonal computation and off-diagonal updates.
//
// For each column j:
//   1. Thread 0 computes diagonal A[j][j] = sqrt(A[j][j] - sum(A[j][k]^2))
//   2. __syncthreads() - ensure diagonal is visible to all threads
//   3. All threads compute off-diagonal rows i > j in parallel (grid-stride)
//   4. All threads zero upper triangle A[j][k] = 0 for k > j (grid-stride)
//   5. __syncthreads() - ensure column j is complete before moving to j+1
__global__ void choleskyKernel(double* A, size_t n) {
    unsigned int tid = threadIdx.x;
    unsigned int stride = blockDim.x;

    for (size_t j = 0; j < n; ++j) {
        // Diagonal element: A[j][j] = sqrt(A[j][j] - sum_{k=0}^{j-1} A[j][k]^2)
        if (tid == 0) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[j * n + k] * A[j * n + k];
            }
            double diag_val = A[j * n + j] - sum;
            if (diag_val > 0.0) {
                A[j * n + j] = sqrt(diag_val);
            } else {
                A[j * n + j] = -1.0; // signal error
            }
        }

        __syncthreads();

        // Off-diagonal: parallel update of rows i > j via grid-stride
        for (size_t i = tid + j + 1; i < n; i += stride) {
            double sum = 0.0;
            for (size_t k = 0; k < j; ++k) {
                sum += A[i * n + k] * A[j * n + k];
            }
            A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
        }

        // Zero upper triangle: A[j][k] = 0 for k > j via grid-stride
        for (size_t k = tid + j + 1; k < n; k += stride) {
            A[j * n + k] = 0.0;
        }

        __syncthreads();
    }
}

// ===== Host Functions =====

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    double *d_A = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));

    // Use maximum threads per block for best parallelism within a single block
    const int block_size = 256;
    choleskyKernel<<<1, block_size>>>(d_A, n);

    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<double> h_A(n * n);
    CUDA_CHECK(cudaMemcpy(h_A.data(), d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < n; ++i) {
        if (h_A[i * n + i] < 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
            CUDA_CHECK(cudaFree(d_A));
            return false;
        }
    }

    A = std::move(h_A);
    CUDA_CHECK(cudaFree(d_A));
    return true;
}

bool generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;

    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += B[i * n + k] * B[j * n + k];
            }
            A[i * n + j] = sum;
        }
    }

    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
    return true;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) {
                sum += L[i * n + k] * L[j * n + k];
            }
            reconstructed[i * n + j] = sum;
        }
    }

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

    std::vector<double> A(n * n);
    std::vector<double> A_orig;

    printf("Generating positive definite matrix...\n");
    if (!generatePositiveDefiniteMatrix(A, n)) {
        printf("Failed to generate positive definite matrix\n");
        return 1;
    }

    if (validate) {
        A_orig = A;
    }

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

    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(A, "CholeskyL");
    }

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
