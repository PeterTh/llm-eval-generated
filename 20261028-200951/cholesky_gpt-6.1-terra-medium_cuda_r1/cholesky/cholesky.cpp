#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA libraries use column-major matrices. A row-major symmetric matrix has
// the same representation, and an upper Cholesky factor appears lower triangular.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <limits>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while (0)
#define LIB_CHECK(call) do { auto e = (call); if (e != 0) { \
    fprintf(stderr, "CUDA library error %d (%s:%d)\n", int(e), __FILE__, __LINE__); exit(1); } } while (0)

__global__ void addDiagonal(double* a, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i * n + i] += n;
}

__global__ void clearUpper(double* a, size_t n) {
    size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < n * n && index % n > index / n) a[index] = 0.0;
}

static cublasHandle_t blas;
static cusolverDnHandle_t solver;

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double *dA, *workspace;
    int *info, hostInfo, workspaceSize;
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
    LIB_CHECK(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER, int(n), dA, int(n), &workspaceSize));
    CUDA_CHECK(cudaMalloc(&workspace, size_t(workspaceSize) * sizeof(double)));
    LIB_CHECK(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, int(n), dA, int(n), workspace, workspaceSize, info));
    CUDA_CHECK(cudaMemcpy(&hostInfo, info, sizeof(int), cudaMemcpyDeviceToHost));
    if (hostInfo == 0) {
        clearUpper<<<(n * n + 255) / 256, 256>>>(dA, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    } else if (hostInfo > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", hostInfo - 1);
    } else {
        fprintf(stderr, "Cholesky solver: invalid argument %d\n", -hostInfo);
    }
    CUDA_CHECK(cudaFree(workspace));
    CUDA_CHECK(cudaFree(info));
    CUDA_CHECK(cudaFree(dA));
    return hostInfo == 0;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    if (n == 0) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) value = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    double *dB, *dA;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dA, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    const double alpha = 1.0, beta = 0.0;
    LIB_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, int(n), int(n), int(n),
                          &alpha, dB, int(n), dB, int(n), &beta, dA, int(n)));
    addDiagonal<<<(n + 255) / 256, 256>>>(dA, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    if (n) {
        double *dL, *dResult;
        CUDA_CHECK(cudaMalloc(&dL, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dResult, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(dL, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        const double alpha = 1.0, beta = 0.0;
        LIB_CHECK(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, int(n), int(n), int(n),
                              &alpha, dL, int(n), dL, int(n), &beta, dResult, int(n)));
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), dResult, n * n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dL));
        CUDA_CHECK(cudaFree(dResult));
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
    
    if (n > size_t(std::numeric_limits<int>::max()) ||
        (n && n > std::numeric_limits<size_t>::max() / sizeof(double) / n)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    LIB_CHECK(cublasCreate(&blas));
    LIB_CHECK(cusolverDnCreate(&solver));

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
