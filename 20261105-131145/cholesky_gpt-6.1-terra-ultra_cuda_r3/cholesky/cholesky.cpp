#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// CUDA library failures are fatal: this benchmark always runs on the GPU.
#define CUDA_CHECK(call) do { \
    cudaError_t status_ = (call); \
    if (status_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)
#define LIB_CHECK(call) do { \
    auto status_ = (call); \
    if (status_ != 0) { \
        fprintf(stderr, "CUDA library error at %s:%d: %d\n", __FILE__, __LINE__, static_cast<int>(status_)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void addDiagonal(double* matrix, size_t n) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) matrix[i * n + i] += static_cast<double>(n);
}

__global__ void clearUpper(double* matrix, size_t n) {
    size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) matrix[row * n + col] = 0.0;
}

// Row-major X is column-major X^T. Compute X X^T using double precision.
void gramMatrix(const std::vector<double>& X, std::vector<double>& result, size_t n,
                bool add_diagonal) {
    if (n == 0) return;
    const int dim = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double *deviceX, *deviceResult;
    CUDA_CHECK(cudaMalloc(&deviceX, bytes));
    CUDA_CHECK(cudaMalloc(&deviceResult, bytes));
    CUDA_CHECK(cudaMemcpy(deviceX, X.data(), bytes, cudaMemcpyHostToDevice));
    cublasHandle_t blas;
    LIB_CHECK(cublasCreate(&blas));
    const double one = 1.0, zero = 0.0;
    LIB_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                          &one, deviceX, dim, deviceX, dim, &zero, deviceResult, dim));
    if (add_diagonal) {
        addDiagonal<<<(n + 255) / 256, 256>>>(deviceResult, n);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(result.data(), deviceResult, bytes, cudaMemcpyDeviceToHost));
    LIB_CHECK(cublasDestroy(blas));
    CUDA_CHECK(cudaFree(deviceResult));
    CUDA_CHECK(cudaFree(deviceX));
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    const int dim = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double *deviceA, *workspace;
    int *deviceInfo;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceInfo, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice));
    cusolverDnHandle_t solver;
    LIB_CHECK(cusolverDnCreate(&solver));
    int workspaceSize = 0;
    // Column-major upper is row-major lower.
    LIB_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, dim,
                                          deviceA, dim, &workspaceSize));
    CUDA_CHECK(cudaMalloc(&workspace, static_cast<size_t>(workspaceSize) * sizeof(double)));
    LIB_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, dim, deviceA, dim,
                              workspace, workspaceSize, deviceInfo));
    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, deviceInfo, sizeof(info), cudaMemcpyDeviceToHost));
    if (info == 0) {
        const dim3 block(32, 8);
        const dim3 grid((n + block.x - 1) / block.x, (n + block.y - 1) / block.y);
        clearUpper<<<grid, block>>>(deviceA, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(workspace));
    LIB_CHECK(cusolverDnDestroy(solver));
    CUDA_CHECK(cudaFree(deviceInfo));
    CUDA_CHECK(cudaFree(deviceA));
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    if (info < 0) {
        fprintf(stderr, "Cholesky CUDA solver rejected argument %d\n", -info);
        return false;
    }
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    gramMatrix(B, A, n, true);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    gramMatrix(L, reconstructed, n, false);
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
    
    if (n > static_cast<size_t>(INT_MAX) || (n != 0 && n > SIZE_MAX / sizeof(double) / n)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
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
