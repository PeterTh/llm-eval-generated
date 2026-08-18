#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// Simple Cholesky decomposition (sequential, unblocked algorithm)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

static bool cudaOk(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

static bool solverOk(cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    fprintf(stderr, "cuSOLVER error in %s: status %d\n", operation, static_cast<int>(status));
    return false;
}

__global__ void clearRowMajorUpper(double* matrix, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) matrix[static_cast<size_t>(row) * n + col] = 0.0;
}

// cuSOLVER is column-major. Factoring its upper triangle maps exactly to the
// lower triangle of the symmetric row-major matrix used by this program.
bool choleskyDecomposition(std::vector<double>& A, const size_t n, float& elapsedMs) {
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Invalid matrix size: %zu\n", n);
        return false;
    }
    const int order = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double* deviceA = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    cusolverDnHandle_t solver = nullptr;
    cudaEvent_t begin = nullptr, end = nullptr;
    bool ok = cudaOk(cudaMalloc(&deviceA, bytes), "matrix allocation") &&
              cudaOk(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "matrix upload") &&
              solverOk(cusolverDnCreate(&solver), "handle creation");
    int workspaceSize = 0;
    if (ok) ok = solverOk(cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                                      order, deviceA, order, &workspaceSize),
                          "workspace query");
    if (ok) ok = cudaOk(cudaMalloc(&workspace, static_cast<size_t>(workspaceSize) * sizeof(double)),
                        "workspace allocation") &&
                 cudaOk(cudaMalloc(&deviceInfo, sizeof(int)), "status allocation") &&
                 cudaOk(cudaEventCreate(&begin), "start event") &&
                 cudaOk(cudaEventCreate(&end), "stop event");
    if (ok) ok = cudaOk(cudaEventRecord(begin), "start timing") &&
                 solverOk(cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, order, deviceA,
                                           order, workspace, workspaceSize, deviceInfo),
                          "factorization");
    const dim3 threads(32, 8);
    const dim3 blocks((order + threads.x - 1) / threads.x,
                      (order + threads.y - 1) / threads.y);
    if (ok) {
        clearRowMajorUpper<<<blocks, threads>>>(deviceA, order);
        ok = cudaOk(cudaGetLastError(), "upper-triangle cleanup") &&
             cudaOk(cudaEventRecord(end), "stop timing") &&
             cudaOk(cudaEventSynchronize(end), "factorization synchronization") &&
             cudaOk(cudaEventElapsedTime(&elapsedMs, begin, end), "elapsed time");
    }
    int info = -1;
    if (ok) ok = cudaOk(cudaMemcpy(&info, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost), "status download") &&
                 cudaOk(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "result download");
    if (ok && info != 0) {
        if (info > 0) fprintf(stderr, "Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        else fprintf(stderr, "Error: cuSOLVER received invalid argument %d\n", -info);
        ok = false;
    }
    if (end) cudaEventDestroy(end);
    if (begin) cudaEventDestroy(begin);
    if (deviceInfo) cudaFree(deviceInfo);
    if (workspace) cudaFree(workspace);
    if (solver) cusolverDnDestroy(solver);
    if (deviceA) cudaFree(deviceA);
    return ok;
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
    float elapsedMs = 0.0f;
    bool success = choleskyDecomposition(A, n, elapsedMs);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (elapsedMs / 1000.0) / 1e9;
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
