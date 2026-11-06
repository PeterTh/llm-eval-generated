#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// All matrix arithmetic runs on the GPU. The CUDA libraries are distributed
// with the CUDA toolkit, and use double precision throughout.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

static void checkBlas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuBLAS error %d\n", operation, static_cast<int>(status));
        exit(EXIT_FAILURE);
    }
}

static void checkSolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "%s: cuSOLVER error %d\n", operation, static_cast<int>(status));
        exit(EXIT_FAILURE);
    }
}

static int matrixDimension(size_t n) {
    if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (n != 0 && n > std::numeric_limits<size_t>::max() / n / sizeof(double))) {
        fprintf(stderr, "Matrix size is too large for CUDA\n");
        exit(EXIT_FAILURE);
    }
    return static_cast<int>(n);
}

__global__ void addDiagonal(double* matrix, int n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < static_cast<size_t>(n)) matrix[i * n + i] += n;
}

__global__ void clearUpper(double* matrix, int n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = static_cast<size_t>(n) * n;
    if (idx < count && idx % n > idx / n) matrix[idx] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    const int dim = matrixDimension(n);
    if (dim == 0) return true;
    const size_t bytes = n * n * sizeof(double);
    double* deviceA = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    cusolverDnHandle_t solver;
    checkCuda(cudaMalloc(&deviceA, bytes), "Allocating device matrix");
    checkCuda(cudaMalloc(&deviceInfo, sizeof(int)), "Allocating solver status");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "Copying matrix to GPU");
    checkSolver(cusolverDnCreate(&solver), "Creating cuSOLVER handle");

    // A row-major lower factor is the column-major upper factor of the same
    // symmetric matrix. This avoids transposing the matrix in either direction.
    int workSize = 0;
    checkSolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                            dim, deviceA, dim, &workSize),
                "Querying Cholesky workspace");
    checkCuda(cudaMalloc(&workspace, static_cast<size_t>(workSize) * sizeof(double)),
              "Allocating Cholesky workspace");
    checkSolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, dim, deviceA,
                                 dim, workspace, workSize, deviceInfo),
                "Computing Cholesky decomposition");
    int info = 0;
    checkCuda(cudaMemcpy(&info, deviceInfo, sizeof(info), cudaMemcpyDeviceToHost),
              "Reading Cholesky status");
    if (info == 0) {
        constexpr unsigned threads = 256;
        const size_t blocks = (n * n + threads - 1) / threads;
        clearUpper<<<static_cast<unsigned>(blocks), threads>>>(deviceA, dim);
        checkCuda(cudaGetLastError(), "Clearing upper triangle");
        checkCuda(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                  "Copying Cholesky factor to host");
    } else if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
    } else {
        fprintf(stderr, "Cholesky solver received invalid argument %d\n", -info);
    }
    checkCuda(cudaFree(workspace), "Freeing Cholesky workspace");
    checkCuda(cudaFree(deviceInfo), "Freeing solver status");
    checkCuda(cudaFree(deviceA), "Freeing device matrix");
    checkSolver(cusolverDnDestroy(solver), "Destroying cuSOLVER handle");
    return info == 0;
}

// Compute M * M^T for a row-major matrix using the column-major cuBLAS API.
static void gramMatrix(const std::vector<double>& M, std::vector<double>& result,
                       size_t n, bool addIdentity) {
    const int dim = matrixDimension(n);
    if (dim == 0) return;
    const size_t bytes = n * n * sizeof(double);
    double *deviceM = nullptr, *deviceResult = nullptr;
    cublasHandle_t blas;
    checkCuda(cudaMalloc(&deviceM, bytes), "Allocating GPU input");
    checkCuda(cudaMalloc(&deviceResult, bytes), "Allocating GPU result");
    checkCuda(cudaMemcpy(deviceM, M.data(), bytes, cudaMemcpyHostToDevice), "Copying input to GPU");
    checkBlas(cublasCreate(&blas), "Creating cuBLAS handle");
    const double alpha = 1.0, beta = 0.0;
    checkBlas(cublasDgemm(blas, CUBLAS_OP_T, CUBLAS_OP_N, dim, dim, dim,
                          &alpha, deviceM, dim, deviceM, dim, &beta,
                          deviceResult, dim), "Computing matrix product");
    if (addIdentity) {
        constexpr unsigned threads = 256;
        addDiagonal<<<(static_cast<unsigned>(dim) + threads - 1) / threads, threads>>>(deviceResult, dim);
        checkCuda(cudaGetLastError(), "Adding matrix diagonal");
    }
    checkCuda(cudaMemcpy(result.data(), deviceResult, bytes, cudaMemcpyDeviceToHost),
              "Copying matrix product to host");
    checkBlas(cublasDestroy(blas), "Destroying cuBLAS handle");
    checkCuda(cudaFree(deviceM), "Freeing GPU input");
    checkCuda(cudaFree(deviceResult), "Freeing GPU result");
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    matrixDimension(n);
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
