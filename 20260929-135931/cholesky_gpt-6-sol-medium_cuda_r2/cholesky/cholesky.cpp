#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

// Factor A on the GPU. A is row-major and symmetric, so cuSOLVER's
// column-major upper factor is the row-major lower factor we need.

bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           cusolverDnHandle_t solver) {
    if (n == 0) return true;
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("Error: Matrix dimension exceeds cuSOLVER's integer limit\n");
        return false;
    }

    double* deviceA = nullptr;
    double* workspace = nullptr;
    int* deviceInfo = nullptr;
    bool success = false;

    do {
        cudaError_t cudaStatus = cudaMalloc(reinterpret_cast<void**>(&deviceA), A.size() * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            printf("Error: cudaMalloc matrix failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }
        cudaStatus = cudaMalloc(reinterpret_cast<void**>(&deviceInfo), sizeof(int));
        if (cudaStatus != cudaSuccess) {
            printf("Error: cudaMalloc status failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }
        cudaStatus = cudaMemcpy(deviceA, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            printf("Error: Copy to GPU failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }

        const int dimension = static_cast<int>(n);
        int workspaceSize = 0;
        cusolverStatus_t solverStatus = cusolverDnDpotrf_bufferSize(solver, CUBLAS_FILL_MODE_UPPER,
                                                                   dimension, deviceA, dimension, &workspaceSize);
        if (solverStatus != CUSOLVER_STATUS_SUCCESS) {
            printf("Error: cuSOLVER workspace query failed (%d)\n", static_cast<int>(solverStatus));
            break;
        }
        cudaStatus = cudaMalloc(reinterpret_cast<void**>(&workspace),
                                static_cast<size_t>(workspaceSize) * sizeof(double));
        if (cudaStatus != cudaSuccess) {
            printf("Error: cudaMalloc workspace failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }
        solverStatus = cusolverDnDpotrf(solver, CUBLAS_FILL_MODE_UPPER, dimension,
                                       deviceA, dimension, workspace, workspaceSize, deviceInfo);
        if (solverStatus != CUSOLVER_STATUS_SUCCESS) {
            printf("Error: cuSOLVER factorization failed (%d)\n", static_cast<int>(solverStatus));
            break;
        }
        int info = 0;
        cudaStatus = cudaMemcpy(&info, deviceInfo, sizeof(int), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("Error: Copy of cuSOLVER status failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }
        if (info > 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", info - 1);
            break;
        }
        if (info < 0) {
            printf("Error: cuSOLVER rejected argument %d\n", -info);
            break;
        }
        cudaStatus = cudaMemcpy(A.data(), deviceA, A.size() * sizeof(double), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            printf("Error: Copy from GPU failed: %s\n", cudaGetErrorString(cudaStatus));
            break;
        }
        for (size_t i = 0; i < n; ++i) {
            std::fill(A.begin() + i * n + i + 1, A.begin() + (i + 1) * n, 0.0);
        }
        success = true;
    } while (false);

    cudaFree(workspace);
    cudaFree(deviceInfo);
    cudaFree(deviceA);
    return success;
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
    cusolverDnHandle_t solver = nullptr;
    cusolverStatus_t solverStatus = cusolverDnCreate(&solver);
    if (solverStatus != CUSOLVER_STATUS_SUCCESS) {
        printf("Error: cusolverDnCreate failed (%d)\n", static_cast<int>(solverStatus));
        return 1;
    }
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, solver);
    
    auto end = std::chrono::high_resolution_clock::now();
    cusolverDnDestroy(solver);
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
