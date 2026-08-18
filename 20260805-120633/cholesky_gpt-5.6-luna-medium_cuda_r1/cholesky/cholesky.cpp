#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

__global__ void clearUpperTriangle(double* matrix, const int n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = static_cast<size_t>(n) * n;
    if (index < total && index % n > index / n) {
        matrix[index] = 0.0;
    }
}

__global__ void factorColumnDiagonal(double* matrix, const int n, const int column,
                                     int* positiveDefinite) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const size_t row = static_cast<size_t>(column) * n;
        double sum = 0.0;
        for (int k = 0; k < column; ++k) {
            const double value = matrix[row + k];
            sum += value * value;
        }
        const double value = matrix[row + column] - sum;
        if (value <= 0.0) {
            *positiveDefinite = 0;
        } else {
            matrix[row + column] = sqrt(value);
        }
    }
}

__global__ void factorColumnBelow(double* matrix, const int n, const int column) {
    const int rowIndex = column + 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (rowIndex >= n) {
        return;
    }
    const size_t row = static_cast<size_t>(rowIndex) * n;
    const size_t pivotRow = static_cast<size_t>(column) * n;
    double sum = 0.0;
    for (int k = 0; k < column; ++k) {
        sum += matrix[row + k] * matrix[pivotRow + k];
    }
    matrix[row + column] = (matrix[row + column] - sum) / matrix[pivotRow + column];
}

bool cudaOk(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

} // namespace

// GPU-parallel column-oriented Cholesky decomposition. Every column's lower
// triangular update is distributed across CUDA threads while the sequential
// column dependency is preserved exactly.
// A is stored in row-major order and overwritten with the lower factor.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        return false;
    }

    const int dimension = static_cast<int>(n);
    double* deviceMatrix = nullptr;
    int* devicePositiveDefinite = nullptr;
    const size_t bytes = n * n * sizeof(double);
    bool success = false;
    bool executionError = false;

    if (!cudaOk(cudaMalloc(&deviceMatrix, bytes), "matrix allocation") ||
        !cudaOk(cudaMalloc(&devicePositiveDefinite, sizeof(int)), "status allocation") ||
        !cudaOk(cudaMemcpy(deviceMatrix, A.data(), bytes, cudaMemcpyHostToDevice),
                "matrix upload") ||
        !cudaOk(cudaMemset(devicePositiveDefinite, 1, sizeof(int)), "status initialization")) {
        cudaFree(deviceMatrix);
        cudaFree(devicePositiveDefinite);
        return false;
    }

    for (int column = 0; column < dimension; ++column) {
        factorColumnDiagonal<<<1, 1>>>(deviceMatrix, dimension, column,
                                       devicePositiveDefinite);
        if (!cudaOk(cudaGetLastError(), "diagonal factorization launch") ||
            !cudaOk(cudaDeviceSynchronize(), "diagonal factorization")) {
            executionError = true;
            break;
        }

        int positiveDefinite = 1;
        if (!cudaOk(cudaMemcpy(&positiveDefinite, devicePositiveDefinite, sizeof(int),
                               cudaMemcpyDeviceToHost), "status download") ||
            positiveDefinite == 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", column);
            executionError = positiveDefinite != 0;
            break;
        }

        const int rowsBelow = dimension - column - 1;
        if (rowsBelow > 0) {
            const int blocks = (rowsBelow + 255) / 256;
            factorColumnBelow<<<blocks, 256>>>(deviceMatrix, dimension, column);
            if (!cudaOk(cudaGetLastError(), "panel solve launch") ||
                !cudaOk(cudaDeviceSynchronize(), "panel solve")) {
                executionError = true;
                break;
            }
        }
        success = true;
    }

    if (success && !executionError) {
        const size_t elements = n * n;
        clearUpperTriangle<<<(elements + 255) / 256, 256>>>(deviceMatrix, dimension);
        success = cudaOk(cudaGetLastError(), "upper triangle cleanup launch") &&
                  cudaOk(cudaDeviceSynchronize(), "upper triangle cleanup") &&
                  cudaOk(cudaMemcpy(A.data(), deviceMatrix, bytes, cudaMemcpyDeviceToHost),
                         "matrix download");
    }

    cudaFree(deviceMatrix);
    cudaFree(devicePositiveDefinite);
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
