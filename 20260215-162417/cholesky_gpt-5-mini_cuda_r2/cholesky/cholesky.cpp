#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) do { cudaError_t err = (call); if (err != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); return false; }} while(0)

// Device kernel to compute sum of squares for diagonal element (atomicAdd on double)
__global__ void diag_sum_kernel(const double* A, double* result, size_t n, size_t j) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    double local = 0.0;
    for (size_t k = idx; k < j; k += stride) {
        double v = A[j * n + k];
        local += v * v;
    }
    if (local != 0.0) atomicAdd(result, local);
}

// Device kernel to compute dot products between row i and row j for i = j+1..n-1
// sums is an array of length (n - j - 1) where each entry accumulates the dot for a row i
__global__ void row_dot_kernel(const double* A, double* sums, size_t n, size_t j, size_t rows) {
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = rows * j; // rows * number of k
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int index = gid; index < total; index += stride) {
        unsigned int i_rel = index / j; // 0..rows-1
        unsigned int k = index % j;    // 0..j-1
        size_t i = j + 1 + i_rel;
        double a = A[i * n + k];
        double b = A[j * n + k];
        double prod = a * b;
        atomicAdd(&sums[i_rel], prod);
    }
}

// Kernel to update column j entries: A[i*n + j] = (A[i*n + j] - sums[i_rel]) / diag
__global__ void update_column_kernel(double* A, const double* sums, size_t n, size_t j, double diag, size_t rows) {
    unsigned int i_rel = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_rel < rows) {
        size_t i = j + 1 + i_rel;
        double val = (A[i * n + j] - sums[i_rel]) / diag;
        A[i * n + j] = val;
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Move matrix to device
    double* d_A = nullptr;
    size_t bytes = n * n * sizeof(double);
    if (cudaMalloc(&d_A, bytes) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory\n");
        return false;
    }
    if (cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        fprintf(stderr, "Failed to copy matrix to device\n");
        cudaFree(d_A);
        return false;
    }

    // Temporary buffer for diagonal sums and row sums
    double* d_tmp = nullptr;
    if (cudaMalloc(&d_tmp, sizeof(double) * n) != cudaSuccess) {
        fprintf(stderr, "Failed to allocate tmp buffer\n");
        cudaFree(d_A);
        return false;
    }

    // Parameters for kernels
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    for (size_t j = 0; j < n; ++j) {
        // Compute diagonal sum = sum_{k=0..j-1} A[j*n + k]^2
        // set d_tmp[0] = 0
        if (cudaMemset(d_tmp, 0, sizeof(double)) != cudaSuccess) {
            fprintf(stderr, "cudaMemset failed\n");
            cudaFree(d_tmp); cudaFree(d_A); return false;
        }
        int b = (int)((j + threads - 1) / threads);
        if (b < 1) b = 1;
        diag_sum_kernel<<<b, threads>>>(d_A, d_tmp, n, j);
        if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "diag_sum_kernel launch failed\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }

        double diag_sum_host = 0.0;
        if (cudaMemcpy(&diag_sum_host, d_tmp, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr, "Failed to copy diag sum\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }

        // Read diagonal value
        double diag_val_host = 0.0;
        if (cudaMemcpy(&diag_val_host, d_A + j * n + j, sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr, "Failed to copy diag val\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }

        double val = diag_val_host - diag_sum_host;
        if (val <= 0.0) {
            fprintf(stderr, "Error: Matrix is not positive definite at diagonal %zu\n", j);
            cudaFree(d_tmp); cudaFree(d_A);
            return false;
        }
        double diag = sqrt(val);
        // write diag back to device
        if (cudaMemcpy(d_A + j * n + j, &diag, sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess) { fprintf(stderr, "Failed to write diag\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }

        // Compute off-diagonal elements for rows i = j+1..n-1
        size_t rows = (j + 1 < n) ? (n - j - 1) : 0;
        if (rows > 0 && j > 0) {
            // zero sums array of length rows (stored at d_tmp[0..rows-1])
            if (cudaMemset(d_tmp, 0, sizeof(double) * rows) != cudaSuccess) { fprintf(stderr, "cudaMemset sums failed\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }
            // launch kernel to compute dot products
            // total entries = rows * j
            size_t total = rows * j;
            int blocks2 = (int)((total + threads - 1) / threads);
            if (blocks2 < 1) blocks2 = 1;
            row_dot_kernel<<<blocks2, threads>>>(d_A, d_tmp, n, j, rows);
            if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "row_dot_kernel launch failed\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }

            // update column values
            int blocks3 = (int)((rows + threads - 1) / threads);
            update_column_kernel<<<blocks3, threads>>>(d_A, d_tmp, n, j, diag, rows);
            if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "update_column_kernel launch failed\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }
        } else if (rows > 0) {
            // j == 0 mode: sums are zero, just divide by diag
            int blocks3 = (int)((rows + threads - 1) / threads);
            update_column_kernel<<<blocks3, threads>>>(d_A, d_tmp, n, j, diag, rows);
            if (cudaGetLastError() != cudaSuccess) { fprintf(stderr, "update_column_kernel launch failed\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }
        }

        // ensure kernels finished before next iteration
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }
    }

    // Zero out upper triangular part on device
    // Launch kernel to set A[i*n + j] = 0 for j > i
    // We'll do it on host for simplicity by copying back and zeroing
    if (cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) { fprintf(stderr, "Failed to copy result back to host\n"); cudaFree(d_tmp); cudaFree(d_A); return false; }

    cudaFree(d_tmp);
    cudaFree(d_A);

    // Zero upper triangular to match original semantics
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            A[i * n + j] = 0.0;
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
