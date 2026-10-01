#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// A row-major matrix is its transpose when viewed as column-major by CUDA
// libraries. Symmetric A is unchanged, while its lower triangle is the upper
// triangle in cuSOLVER's column-major view.
__global__ void addDiagonal(double* A, size_t n) {
    const size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) A[i * n + i] += static_cast<double>(n);
}

__global__ void zeroUpper(double* A, size_t n) {
    const size_t index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index < n * n && index / n < index % n) A[index] = 0.0;
}

__global__ void mirrorUpper(double* A, size_t n) {
    const size_t index = (static_cast<size_t>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index < n * n) {
        const size_t row = index / n, col = index % n;
        if (row < col) A[index] = A[col * n + row];
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

void checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "%s failed (cuBLAS status %d)\n", operation, static_cast<int>(status));
        std::exit(1);
    }
}

void checkCusolver(cusolverStatus_t status, const char* operation) {
    if (status != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "%s failed (cuSOLVER status %d)\n", operation, static_cast<int>(status));
        std::exit(1);
    }
}

bool choleskyDecomposition(double* dA, int n, cusolverDnHandle_t solver) {
    if (n == 0) return true;

    int workspaceSize = 0;
    checkCusolver(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                               n, dA, n, &workspaceSize), "Cholesky workspace query");
    double* workspace = nullptr;
    int* dInfo = nullptr;
    checkCuda(cudaMalloc(&workspace, static_cast<size_t>(workspaceSize) * sizeof(double)), "Allocate Cholesky workspace");
    checkCuda(cudaMalloc(&dInfo, sizeof(int)), "Allocate Cholesky status");
    checkCusolver(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, n, dA, n,
                                    workspace, workspaceSize, dInfo), "GPU Cholesky decomposition");
    int info = 0;
    checkCuda(cudaMemcpy(&info, dInfo, sizeof(int), cudaMemcpyDeviceToHost), "Read Cholesky status");
    checkCuda(cudaFree(workspace), "Free Cholesky workspace");
    checkCuda(cudaFree(dInfo), "Free Cholesky status");
    if (info > 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        return false;
    }
    if (info < 0) {
        fprintf(stderr, "GPU Cholesky received an invalid argument at position %d\n", -info);
        return false;
    }
    const size_t elements = static_cast<size_t>(n) * n;
    zeroUpper<<<(elements + 255) / 256, 256>>>(dA, n);
    checkCuda(cudaGetLastError(), "Clear upper triangle");
    checkCuda(cudaDeviceSynchronize(), "Finish Cholesky decomposition");
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(double* dA, const size_t n, cublasHandle_t blas) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    if (n == 0) return;
    double* dB = nullptr;
    checkCuda(cudaMalloc(&dB, n * n * sizeof(double)), "Allocate random matrix");
    checkCuda(cudaMemcpy(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice), "Upload random matrix");
    const double alpha = 1.0, beta = 0.0;
    // Column-major B is row-major B^T. Compute the lower triangle of
    // row-major B * B^T using the upper column-major triangle.
    checkCublas(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                            static_cast<int>(n), static_cast<int>(n), &alpha,
                            dB, static_cast<int>(n), &beta, dA, static_cast<int>(n)),
                "Generate positive definite matrix");
    const size_t elements = n * n;
    mirrorUpper<<<(elements + 255) / 256, 256>>>(dA, n);
    checkCuda(cudaGetLastError(), "Mirror generated matrix");
    addDiagonal<<<(n + 255) / 256, 256>>>(dA, n);
    checkCuda(cudaGetLastError(), "Add diagonal dominance");
    checkCuda(cudaFree(dB), "Free random matrix");
}

bool validateCholesky(const double* dL, const std::vector<double>& A_orig,
                      const size_t n, cublasHandle_t blas) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    if (n != 0) {
        double* dReconstructed = nullptr;
        checkCuda(cudaMalloc(&dReconstructed, n * n * sizeof(double)), "Allocate reconstructed matrix");
        const double alpha = 1.0, beta = 0.0;
        checkCublas(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                static_cast<int>(n), static_cast<int>(n), &alpha,
                                dL, static_cast<int>(n), &beta,
                                dReconstructed, static_cast<int>(n)), "Reconstruct matrix");
        const size_t elements = n * n;
        mirrorUpper<<<(elements + 255) / 256, 256>>>(dReconstructed, n);
        checkCuda(cudaGetLastError(), "Mirror reconstructed matrix");
        checkCuda(cudaMemcpy(reconstructed.data(), dReconstructed, n * n * sizeof(double),
                             cudaMemcpyDeviceToHost), "Download reconstructed matrix");
        checkCuda(cudaFree(dReconstructed), "Free reconstructed matrix");
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
    
    if (n > INT_MAX || (n != 0 && n > SIZE_MAX / n / sizeof(double))) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    cublasHandle_t blas;
    cusolverDnHandle_t solver;
    checkCublas(cublasCreate(&blas), "Create cuBLAS handle");
    checkCusolver(cusolverDnCreate(&solver), "Create cuSOLVER handle");
    double* dA = nullptr;
    if (n != 0) checkCuda(cudaMalloc(&dA, n * n * sizeof(double)), "Allocate matrix");

    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(dA, n, blas);
    
    if (validate) {
        A_orig.resize(n * n);
        if (n != 0) checkCuda(cudaMemcpy(A_orig.data(), dA, n * n * sizeof(double),
                                        cudaMemcpyDeviceToHost), "Save original matrix");
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(dA, static_cast<int>(n), solver);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        if (dA) checkCuda(cudaFree(dA), "Free matrix");
        checkCusolver(cusolverDnDestroy(solver), "Destroy cuSOLVER handle");
        checkCublas(cublasDestroy(blas), "Destroy cuBLAS handle");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> A(n * n);
        if (n != 0) checkCuda(cudaMemcpy(A.data(), dA, n * n * sizeof(double),
                                        cudaMemcpyDeviceToHost), "Download Cholesky factor");
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(dA, A_orig, n, blas);

        if (dA) checkCuda(cudaFree(dA), "Free matrix");
        checkCusolver(cusolverDnDestroy(solver), "Destroy cuSOLVER handle");
        checkCublas(cublasDestroy(blas), "Destroy cuBLAS handle");
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    if (dA) checkCuda(cudaFree(dA), "Free matrix");
    checkCusolver(cusolverDnDestroy(solver), "Destroy cuSOLVER handle");
    checkCublas(cublasDestroy(blas), "Destroy cuBLAS handle");
    return 0;
}
