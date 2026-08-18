#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int FactorThreads = 128;
constexpr int UpdateTile = 16;
constexpr int BlockWidth = 32;

__global__ void factorDiagonalBlock(double* matrix, const int n, const int begin,
                                    const int width, int* failure) {
    const int tid = threadIdx.x;
    for (int column = 0; column < width; ++column) {
        if (tid == 0) {
            double sum = 0.0;
            const int row = begin + column;
            for (int k = 0; k < column; ++k) {
                const double value = matrix[row * n + begin + k];
                sum += value * value;
            }
            const double diagonal = matrix[row * n + row] - sum;
            if (!(diagonal > 0.0)) {
                atomicCAS(failure, -1, row);
                matrix[row * n + row] = 0.0;
            } else {
                matrix[row * n + row] = sqrt(diagonal);
            }
        }
        __syncthreads();

        for (int rowInBlock = column + 1 + tid; rowInBlock < width;
             rowInBlock += blockDim.x) {
            const int row = begin + rowInBlock;
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += matrix[row * n + begin + k] *
                       matrix[(begin + column) * n + begin + k];
            }
            matrix[row * n + begin + column] =
                (matrix[row * n + begin + column] - sum) /
                matrix[(begin + column) * n + begin + column];
        }
        __syncthreads();
    }
}

// Each thread owns one row. Columns are processed in order, preserving the
// dependency within a row while exposing all rows in the panel in parallel.
__global__ void triangularSolve(double* matrix, const int n, const int begin,
                                const int width, const int rows) {
    const int row = begin + width + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= begin + width + rows) return;

    for (int column = 0; column < width; ++column) {
        double sum = 0.0;
        for (int k = 0; k < column; ++k) {
            sum += matrix[row * n + begin + k] *
                   matrix[(begin + column) * n + begin + k];
        }
        matrix[row * n + begin + column] =
            (matrix[row * n + begin + column] - sum) /
            matrix[(begin + column) * n + begin + column];
    }
}

__global__ void trailingUpdate(double* matrix, const int n, const int begin,
                               const int width) {
    __shared__ double left[UpdateTile][UpdateTile];
    __shared__ double right[UpdateTile][UpdateTile];

    const int row = begin + width + blockIdx.y * UpdateTile + threadIdx.y;
    const int column = begin + width + blockIdx.x * UpdateTile + threadIdx.x;
    const bool active = row < n && column < n && row >= column;

    double sum = 0.0;
    for (int tile = 0; tile < width; tile += UpdateTile) {
        const int leftColumn = begin + tile + threadIdx.x;
        const int rightColumn = begin + tile + threadIdx.y;
        left[threadIdx.y][threadIdx.x] =
            (row < n && leftColumn < begin + width) ? matrix[row * n + leftColumn] : 0.0;
        right[threadIdx.y][threadIdx.x] =
            (column < n && rightColumn < begin + width) ? matrix[column * n + rightColumn] : 0.0;
        __syncthreads();
        for (int k = 0; k < UpdateTile && tile + k < width; ++k)
            sum += left[threadIdx.y][k] * right[k][threadIdx.x];
        __syncthreads();
    }
    if (active) matrix[row * n + column] -= sum;
}

bool cudaCheck(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA error during %s: %s\n", operation,
            cudaGetErrorString(status));
    return false;
}

} // namespace

// Blocked right-looking Cholesky: POTRF, TRSM, and SYRK-like updates all run
// on the GPU. The host only launches dependent panels and copies the result.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    if (n > static_cast<size_t>(INT_MAX)) {
        fprintf(stderr, "Matrix is too large for the CUDA implementation\n");
        return false;
    }

    double* deviceMatrix = nullptr;
    int* deviceFailure = nullptr;
    const size_t bytes = n * n * sizeof(double);
    if (!cudaCheck(cudaMalloc(&deviceMatrix, bytes), "matrix allocation") ||
        !cudaCheck(cudaMalloc(&deviceFailure, sizeof(int)), "status allocation")) {
        if (deviceMatrix) cudaFree(deviceMatrix);
        if (deviceFailure) cudaFree(deviceFailure);
        return false;
    }
    int failure = -1;
    bool success = cudaCheck(cudaMemcpy(deviceMatrix, A.data(), bytes,
                                        cudaMemcpyHostToDevice), "host-to-device copy") &&
                   cudaCheck(cudaMemcpy(deviceFailure, &failure, sizeof(int),
                                        cudaMemcpyHostToDevice), "status initialization");

    const int matrixSize = static_cast<int>(n);
    for (int begin = 0; success && begin < matrixSize; begin += BlockWidth) {
        const int width = std::min(BlockWidth, matrixSize - begin);
        factorDiagonalBlock<<<1, FactorThreads>>>(deviceMatrix, matrixSize, begin,
                                                   width, deviceFailure);
        success = cudaCheck(cudaGetLastError(), "diagonal factorization launch") &&
                  cudaCheck(cudaDeviceSynchronize(), "diagonal factorization");

        const int rows = matrixSize - begin - width;
        if (success && rows > 0) {
            triangularSolve<<<(rows + 255) / 256, 256>>>(
                deviceMatrix, matrixSize, begin, width, rows);
            success = cudaCheck(cudaGetLastError(), "triangular solve launch") &&
                      cudaCheck(cudaDeviceSynchronize(), "triangular solve");
            if (success) {
                dim3 block(UpdateTile, UpdateTile);
                dim3 grid((rows + UpdateTile - 1) / UpdateTile,
                          (rows + UpdateTile - 1) / UpdateTile);
                trailingUpdate<<<grid, block>>>(deviceMatrix, matrixSize, begin,
                                                 width);
                success = cudaCheck(cudaGetLastError(), "trailing update launch") &&
                          cudaCheck(cudaDeviceSynchronize(), "trailing update");
            }
        }
    }

    if (success)
        success = cudaCheck(cudaMemcpy(&failure, deviceFailure, sizeof(int),
                                       cudaMemcpyDeviceToHost), "status copy");
    if (success && failure >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failure);
        success = false;
    }
    if (success)
        success = cudaCheck(cudaMemcpy(A.data(), deviceMatrix, bytes,
                                       cudaMemcpyDeviceToHost), "device-to-host copy");
    if (success) {
        for (size_t row = 0; row < n; ++row)
            std::fill(A.begin() + row * n + row + 1,
                      A.begin() + (row + 1) * n, 0.0);
    }
    cudaFree(deviceFailure);
    cudaFree(deviceMatrix);
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
