#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            return false; \
        } \
    } while(0)

#define CUSOLVER_CHECK(call) \
    do { \
        cusolverStatus_t status = (call); \
        if (status != CUSOLVER_STATUS_SUCCESS) { \
            printf("cuSOLVER error at %s:%d: %d\n", __FILE__, __LINE__, (int)status); \
            return false; \
        } \
    } while(0)

// CUDA kernel: convert cuSOLVER column-major lower-triangular Cholesky result
// to row-major lower-triangular format (L in row-major + zero upper triangle)
__global__ void convert_colmajor_lower_to_rowmajor(const double* __restrict__ src,
                                                    double* __restrict__ dst,
                                                    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;

    int i = idx / n;  // row in row-major
    int j = idx % n;  // col in row-major

    if (i >= j) {
        // L(i,j) in column-major is at src[j * n + i]
        dst[idx] = src[j * n + i];
    } else {
        dst[idx] = 0.0;
    }
}

// CUDA Cholesky decomposition using cuSOLVER (GPU-accelerated)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

// Cached handle and workspace for performance
static cusolverDnHandle_t g_handle = nullptr;
static double* g_workspace = nullptr;
static int g_workspace_size = 0;
static double* g_d_A = nullptr;
static double* g_d_L = nullptr;
static int* g_d_info = nullptr;
static size_t g_allocated_size = 0;

static void initialize_cuda_resources(size_t n) {
    if (g_handle == nullptr) {
        cusolverDnCreate(&g_handle);
    }
    
    size_t required_size = n * n;
    if (required_size > g_allocated_size) {
        // Free old allocations if they exist
        if (g_d_A) cudaFree(g_d_A);
        if (g_d_L) cudaFree(g_d_L);
        if (g_d_info) cudaFree(g_d_info);
        if (g_workspace) cudaFree(g_workspace);
        
        // Allocate new larger buffers
        cudaMalloc(&g_d_A, required_size * sizeof(double));
        cudaMalloc(&g_d_L, required_size * sizeof(double));
        cudaMalloc(&g_d_info, sizeof(int));
        g_allocated_size = required_size;
        g_workspace_size = 0; // Force workspace recalculation
    }
    
    // Get workspace size if needed
    if (g_workspace_size == 0) {
        cusolverDnDpotrf_bufferSize(g_handle, CUBLAS_FILL_MODE_LOWER, (int)n, g_d_A, (int)n, &g_workspace_size);
        cudaMalloc(&g_workspace, g_workspace_size * sizeof(double));
    }
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    initialize_cuda_resources(n);

    // Copy matrix to device
    cudaMemcpy(g_d_A, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice);

    // Compute Cholesky: A = L * L^T using cuSOLVER potrf (column-major, lower fill)
    cusolverDnDpotrf(g_handle, CUBLAS_FILL_MODE_LOWER, (int)n, g_d_A, (int)n, g_workspace, g_workspace_size, g_d_info);

    // Check info from device
    int h_info = 0;
    cudaMemcpy(&h_info, g_d_info, sizeof(int), cudaMemcpyDeviceToHost);

    if (h_info != 0) {
        if (h_info > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", h_info - 1);
        } else {
            printf("Error: cuSOLVER potrf parameter error at parameter %d\n", -h_info);
        }
        return false;
    }

    // Convert from column-major lower-triangular to row-major lower-triangular
    int total = n * n;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    convert_colmajor_lower_to_rowmajor<<<blocks, threads>>>(g_d_A, g_d_L, (int)n);
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(A.data(), g_d_L, n * n * sizeof(double), cudaMemcpyDeviceToHost);

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
    
    // Initialize CUDA resources before timing
    initialize_cuda_resources(n);
    
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
