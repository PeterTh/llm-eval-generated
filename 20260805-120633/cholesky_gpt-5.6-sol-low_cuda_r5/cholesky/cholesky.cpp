#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// The matrix is symmetric, so its row-major representation is also a valid
// column-major representation.  Factoring the column-major upper triangle
// therefore produces the requested row-major lower triangle in-place.

static bool cudaCheck(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

static bool solverCheck(cusolverStatus_t status, const char* operation) {
    if (status == CUSOLVER_STATUS_SUCCESS) return true;
    fprintf(stderr, "cuSOLVER error in %s: status %d\n", operation,
            static_cast<int>(status));
    return false;
}

static bool initializeCuda() {
    // Keep one-time driver and cuSOLVER initialization out of the measured
    // factorization, just as an already-running accelerator benchmark should.
    if (!cudaCheck(cudaFree(nullptr), "CUDA initialization")) return false;
    cusolverDnHandle_t handle = nullptr;
    if (!solverCheck(cusolverDnCreate(&handle), "cuSOLVER initialization")) return false;
    cusolverDnDestroy(handle);
    return true;
}

__global__ void zeroUpperTriangle(double* matrix, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) matrix[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Matrix is too large for cuSOLVER\n");
        return false;
    }

    double* deviceA = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    cusolverDnHandle_t solver = nullptr;
    const size_t bytes = n * n * sizeof(double);
    bool ok = cudaCheck(cudaMalloc(&deviceA, bytes), "cudaMalloc(matrix)") &&
              cudaCheck(cudaMalloc(&deviceInfo, sizeof(int)), "cudaMalloc(info)") &&
              solverCheck(cusolverDnCreate(&solver), "cusolverDnCreate") &&
              cudaCheck(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
                        "matrix upload");

    int workspaceSize = 0;
    if (ok) ok = solverCheck(cusolverDnDpotrf_bufferSize(
        solver, CUBLAS_FILL_MODE_UPPER, static_cast<int>(n), deviceA,
        static_cast<int>(n), &workspaceSize), "cusolverDnDpotrf_bufferSize");
    if (ok) ok = cudaCheck(cudaMalloc(&workspace,
        static_cast<size_t>(workspaceSize) * sizeof(double)), "cudaMalloc(workspace)");
    if (ok) ok = solverCheck(cusolverDnDpotrf(
        solver, CUBLAS_FILL_MODE_UPPER, static_cast<int>(n), deviceA,
        static_cast<int>(n), workspace, workspaceSize, deviceInfo),
        "cusolverDnDpotrf");

    int info = -1;
    if (ok) ok = cudaCheck(cudaMemcpy(&info, deviceInfo, sizeof(int),
                                      cudaMemcpyDeviceToHost), "factorization status");
    if (ok && info == 0) {
        const dim3 block(32, 8);
        const dim3 grid((n + block.x - 1) / block.x,
                        (n + block.y - 1) / block.y);
        zeroUpperTriangle<<<grid, block>>>(deviceA, n);
        ok = cudaCheck(cudaGetLastError(), "zeroUpperTriangle launch") &&
             cudaCheck(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                       "matrix download");
    } else if (ok) {
        if (info > 0)
            printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
        else
            fprintf(stderr, "cuSOLVER received an invalid argument at position %d\n", -info);
        ok = false;
    }

    if (workspace) cudaFree(workspace);
    if (deviceInfo) cudaFree(deviceInfo);
    if (deviceA) cudaFree(deviceA);
    if (solver) cusolverDnDestroy(solver);
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
    if (!initializeCuda()) return 1;
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
