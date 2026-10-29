#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA's dense linear algebra routines use column-major matrices.  A row-major
// lower factor is the same memory as a column-major upper factor.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <climits>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define BLAS_CHECK(call) do { cublasStatus_t e = (call); if (e != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d (%s:%d)\n", int(e), __FILE__, __LINE__); exit(1); } } while (0)
#define SOLVER_CHECK(call) do { cusolverStatus_t e = (call); if (e != CUSOLVER_STATUS_SUCCESS) { \
    fprintf(stderr, "cuSOLVER error %d (%s:%d)\n", int(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void addDiagonal(double* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[size_t(i) * n + i] += n;
}

__global__ void clearUpper(double* a, int n, size_t count) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count && i % n > i / n) a[i] = 0.0;
}

struct GpuContext {
    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    GpuContext() { BLAS_CHECK(cublasCreate(&blas)); SOLVER_CHECK(cusolverDnCreate(&solver)); }
    ~GpuContext() { cusolverDnDestroy(solver); cublasDestroy(blas); }
};

void generatePositiveDefiniteMatrix(std::vector<double>& A, size_t n, GpuContext& gpu) {
    if (!n) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& b : B) b = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    double *db, *da;
    size_t bytes = n * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&db, bytes));
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMemcpy(db, B.data(), bytes, cudaMemcpyHostToDevice));
    const double one = 1.0, zero = 0.0;
    // B in column-major is the transpose of the original row-major B.
    BLAS_CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                           &one, db, n, db, n, &zero, da, n));
    addDiagonal<<<(n + 255) / 256, 256>>>(da, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), da, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(da));
    CUDA_CHECK(cudaFree(db));
}

bool choleskyDecomposition(std::vector<double>& A, size_t n, GpuContext& gpu) {
    if (!n) return true;
    size_t bytes = n * n * sizeof(double);
    double *da, *work;
    int *dinfo, info, lwork;
    CUDA_CHECK(cudaMalloc(&da, bytes));
    CUDA_CHECK(cudaMalloc(&dinfo, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(da, A.data(), bytes, cudaMemcpyHostToDevice));
    SOLVER_CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_UPPER, n, da, n, &lwork));
    CUDA_CHECK(cudaMalloc(&work, size_t(lwork) * sizeof(double)));
    SOLVER_CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_UPPER, n, da, n, work, lwork, dinfo));
    CUDA_CHECK(cudaMemcpy(&info, dinfo, sizeof(info), cudaMemcpyDeviceToHost));
    if (info == 0) {
        size_t count = n * n;
        clearUpper<<<(count + 255) / 256, 256>>>(da, n, count);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), da, bytes, cudaMemcpyDeviceToHost));
    } else if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
    } else {
        fprintf(stderr, "cuSOLVER potrf: invalid argument %d\n", -info);
    }
    CUDA_CHECK(cudaFree(work));
    CUDA_CHECK(cudaFree(dinfo));
    CUDA_CHECK(cudaFree(da));
    return info == 0;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& original,
                      size_t n, GpuContext& gpu) {
    std::vector<double> reconstructed(n * n);
    if (n) {
        size_t bytes = n * n * sizeof(double);
        double *dl, *dr;
        CUDA_CHECK(cudaMalloc(&dl, bytes));
        CUDA_CHECK(cudaMalloc(&dr, bytes));
        CUDA_CHECK(cudaMemcpy(dl, L.data(), bytes, cudaMemcpyHostToDevice));
        const double one = 1.0, zero = 0.0;
        BLAS_CHECK(cublasDgemm(gpu.blas, CUBLAS_OP_T, CUBLAS_OP_N, n, n, n,
                               &one, dl, n, dl, n, &zero, dr, n));
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), dr, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dr));
        CUDA_CHECK(cudaFree(dl));
    }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        double error = fabs(reconstructed[i] - original[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(original[i]) + 1e-10));
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
    
    if (n > INT_MAX || (n && n > SIZE_MAX / sizeof(double) / n)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    GpuContext gpu;
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
