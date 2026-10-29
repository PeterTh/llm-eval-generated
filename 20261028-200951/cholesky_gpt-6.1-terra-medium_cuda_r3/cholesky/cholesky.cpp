#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA stores matrices in column-major order. Symmetric input is identical in
// either layout; an upper-triangular column-major factor is lower-triangular
// when viewed by the rest of this program in row-major order.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <climits>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define BLAS_CHECK(call) do { if ((call) != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error at %s:%d\n", __FILE__, __LINE__); exit(1); } } while (0)
#define SOLVER_CHECK(call) do { if ((call) != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER error at %s:%d\n", __FILE__, __LINE__); exit(1); } } while (0)

__global__ void clearUpper(double* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < n && j < n && j > i) a[i * n + j] = 0.0;
}

struct GPUContext {
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    GPUContext() { BLAS_CHECK(cublasCreate(&blas)); SOLVER_CHECK(cusolverDnCreate(&solver)); }
    ~GPUContext() { cusolverDnDestroy(solver); cublasDestroy(blas); }
};

bool choleskyDecomposition(std::vector<double>& A, const size_t n, GPUContext& gpu) {
    if (n == 0) return true;
    double *d_a, *workspace;
    int *d_info, info, workspaceSize;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_a, A.data(), bytes, cudaMemcpyHostToDevice));
    SOLVER_CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_UPPER,
                                              (int)n, d_a, (int)n, &workspaceSize));
    CUDA_CHECK(cudaMalloc(&workspace, (size_t)workspaceSize * sizeof(double)));
    SOLVER_CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_UPPER, (int)n,
                                  d_a, (int)n, workspace, workspaceSize, d_info));
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info == 0) {
        dim3 block(16, 16), grid((n + 15) / 16, (n + 15) / 16);
        clearUpper<<<grid, block>>>(d_a, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    } else if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
    } else {
        fprintf(stderr, "cuSOLVER Cholesky failed: invalid argument %d\n", -info);
    }
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_a));
    return info == 0;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, GPUContext& gpu) {
    if (n == 0) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    double *d_b, *d_a;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMemcpy(d_b, B.data(), bytes, cudaMemcpyHostToDevice));
    const double alpha = 1.0, beta = 0.0;
    // B is row-major, so its column-major view is B^T.
    BLAS_CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)n, (int)n,
                           &alpha, d_b, (int)n, d_b, (int)n, &beta, d_a, (int)n));
    CUDA_CHECK(cudaMemcpy(A.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n, GPUContext& gpu) {
    std::vector<double> reconstructed(n * n);
    if (n != 0) {
        double *d_l, *d_result;
        size_t bytes = n * n * sizeof(double);
        CUDA_CHECK(cudaMalloc(&d_l, bytes));
        CUDA_CHECK(cudaMalloc(&d_result, bytes));
        CUDA_CHECK(cudaMemcpy(d_l, L.data(), bytes, cudaMemcpyHostToDevice));
        const double alpha = 1.0, beta = 0.0;
        BLAS_CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)n, (int)n,
                               &alpha, d_l, (int)n, d_l, (int)n, &beta, d_result, (int)n));
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), d_result, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_result));
        CUDA_CHECK(cudaFree(d_l));
    }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (n > INT_MAX || (n != 0 && n > SIZE_MAX / n / sizeof(double))) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    GPUContext gpu;

    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, gpu);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, gpu);
    
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
        bool valid = validateCholesky(A, A_orig, n, gpu);
        
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
