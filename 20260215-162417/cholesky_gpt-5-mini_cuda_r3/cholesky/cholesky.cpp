#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA error checking
static inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

// GPU kernel to compute dot-products sums[j] = sum_{k=0..j-1} A[i,k]*A[j,k]
// Each block computes one j (blockIdx.x == j), threads reduce across k
__global__ void compute_sums_kernel(const double* __restrict__ A, double* __restrict__ sums, size_t n, size_t i) {
    const unsigned int j = blockIdx.x;
    if (j > i) return;

    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    double partial = 0.0;

    size_t offset_i = i * n;
    size_t offset_j = j * n;

    for (size_t k = tid; k < j; k += blockDim.x) {
        partial += A[offset_i + k] * A[offset_j + k];
    }

    sdata[tid] = partial;
    __syncthreads();

    // parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) sums[j] = sdata[0];
}

// Kernel to update off-diagonal elements for row i: A[i,j] = (A[i,j] - sums[j]) / A[j,j]
__global__ void update_offdiag_kernel(double* A, const double* sums, size_t n, size_t i) {
    size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= i) return;
    size_t idx = i * n + j;
    A[idx] = (A[idx] - sums[j]) / A[j * n + j];
}

// GPU-accelerated Cholesky decomposition (unblocked but offloads inner products and row updates)
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Allocate device memory
    double* d_A = nullptr;
    double* d_sums = nullptr;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc((void**)&d_sums, n * sizeof(double)));

    const int THREADS = 256;
    for (size_t i = 0; i < n; ++i) {
        if (i == 0) {
            // diagonal element at (0,0)
            double a00;
            CUDA_CHECK(cudaMemcpy(&a00, d_A + 0, sizeof(double), cudaMemcpyDeviceToHost));
            if (a00 <= 0.0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)0);
                CUDA_CHECK(cudaFree(d_A));
                CUDA_CHECK(cudaFree(d_sums));
                return false;
            }
            double diag = sqrt(a00);
            CUDA_CHECK(cudaMemcpy(d_A + 0, &diag, sizeof(double), cudaMemcpyHostToDevice));
            continue;
        }

        // compute sums for j=0..i
        int blocks = static_cast<int>(i + 1);
        int threads = THREADS;
        size_t shared_mem = threads * sizeof(double);
        compute_sums_kernel<<<blocks, threads, shared_mem>>>(d_A, d_sums, n, i);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // fetch diagonal sum for j == i
        double sum_i = 0.0;
        CUDA_CHECK(cudaMemcpy(&sum_i, d_sums + i, sizeof(double), cudaMemcpyDeviceToHost));

        // fetch A[i,i]
        double aii = 0.0;
        CUDA_CHECK(cudaMemcpy(&aii, d_A + i * n + i, sizeof(double), cudaMemcpyDeviceToHost));

        double val = aii - sum_i;
        if (val <= 0.0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", i);
            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_sums));
            return false;
        }
        double diag = sqrt(val);
        // write diagonal back to device
        CUDA_CHECK(cudaMemcpy(d_A + i * n + i, &diag, sizeof(double), cudaMemcpyHostToDevice));

        // update off-diagonal elements in parallel on GPU for j=0..i-1
        int update_threads = THREADS;
        int update_blocks = (int)((i + update_threads - 1) / update_threads);
        update_offdiag_kernel<<<update_blocks, update_threads>>>(d_A, d_sums, n, i);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // zero out upper triangular elements for row i on device for correctness
        // (we can skip explicit zeroing on device for performance and zero on host at the end, but keep here for equivalence)
        for (size_t j = i + 1; j < n; ++j) {
            double zero = 0.0;
            CUDA_CHECK(cudaMemcpy(d_A + i * n + j, &zero, sizeof(double), cudaMemcpyHostToDevice));
        }
    }

    // copy result back
    CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_sums));
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
