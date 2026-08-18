#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// A blocked right-looking Cholesky factorization.  The panel and rank-k
// update dominate runtime and are dispatched to cuBLAS; only the tiny diagonal
// blocks are factored by this CUDA kernel.
constexpr int kBlockSize = 64;

__global__ void factorDiagonalBlock(double* matrix, int n, int first, int width,
                                    int* failure) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (int column = 0; column < width; ++column) {
        double diagonal = matrix[(first + column) * n + first + column];
        for (int k = 0; k < column; ++k) {
            const double value = matrix[(first + column) * n + first + k];
            diagonal -= value * value;
        }

        if (!(diagonal > 0.0) || !isfinite(diagonal)) {
            *failure = first + column + 1;
            return;
        }

        const double pivot = sqrt(diagonal);
        matrix[(first + column) * n + first + column] = pivot;
        for (int row = column + 1; row < width; ++row) {
            double value = matrix[(first + row) * n + first + column];
            for (int k = 0; k < column; ++k) {
                value -= matrix[(first + row) * n + first + k] *
                         matrix[(first + column) * n + first + k];
            }
            matrix[(first + row) * n + first + column] = value / pivot;
        }
    }
}

// Each thread owns one panel row, avoiding the synchronization and layout
// conversions that make a small triangular solve inefficient on the GPU.
__global__ void solvePanelRows(double* matrix, int n, int first, int width,
                               int trailing) {
    const int rowOffset = blockIdx.x * blockDim.x + threadIdx.x;
    if (rowOffset >= trailing) {
        return;
    }

    const int row = first + width + rowOffset;
    for (int column = 0; column < width; ++column) {
        double value = matrix[row * n + first + column];
        for (int k = 0; k < column; ++k) {
            value -= matrix[row * n + first + k] *
                     matrix[(first + column) * n + first + k];
        }
        matrix[row * n + first + column] =
            value / matrix[(first + column) * n + first + column];
    }
}

bool cudaOk(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool cublasOk(cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) {
        return true;
    }
    printf("cuBLAS error during %s: %d\n", operation, static_cast<int>(status));
    return false;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX)) {
        printf("Error: Matrix size is too large for CUDA\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double* deviceMatrix = nullptr;
    int* deviceFailure = nullptr;
    cublasHandle_t handle = nullptr;
    bool success = false;
    const double one = 1.0;
    const double minusOne = -1.0;

    if (!cudaOk(cudaMalloc(&deviceMatrix, bytes), "matrix allocation") ||
        !cudaOk(cudaMalloc(&deviceFailure, sizeof(int)), "status allocation") ||
        !cudaOk(cudaMemcpy(deviceMatrix, A.data(), bytes, cudaMemcpyHostToDevice),
                "matrix upload") ||
        !cudaOk(cudaMemset(deviceFailure, 0, sizeof(int)), "status initialization") ||
        !cublasOk(cublasCreate(&handle), "handle creation") ||
        !cublasOk(cublasSetStream(handle, 0), "stream selection")) {
        goto cleanup;
    }

    for (int first = 0; first < dimension; first += kBlockSize) {
        const int width = std::min(kBlockSize, dimension - first);
        factorDiagonalBlock<<<1, 1>>>(deviceMatrix, dimension, first, width, deviceFailure);
        if (!cudaOk(cudaGetLastError(), "diagonal block factorization")) {
            goto cleanup;
        }

        const int trailing = dimension - first - width;
        if (trailing == 0) {
            continue;
        }

        double* panel = deviceMatrix + static_cast<size_t>(first + width) * dimension + first;
        solvePanelRows<<<(trailing + 255) / 256, 256>>>(deviceMatrix, dimension,
                                                         first, width, trailing);
        if (!cudaOk(cudaGetLastError(), "panel solve")) {
            goto cleanup;
        }

        // Row-major storage is the transpose of cuBLAS's column-major view,
        // so CUBLAS_FILL_MODE_UPPER updates the lower half used by later
        // factorization steps.  The other half is cleared after download.
        double* trailingMatrix = deviceMatrix +
            static_cast<size_t>(first + width) * dimension + first + width;
        if (!cublasOk(cublasDsyrk(handle, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                  trailing, width, &minusOne, panel, dimension,
                                  &one, trailingMatrix, dimension),
                      "trailing rank-k update")) {
            goto cleanup;
        }
    }

    {
        if (!cudaOk(cudaDeviceSynchronize(), "factorization synchronization")) {
            goto cleanup;
        }
        int failedAt = 0;
        if (!cudaOk(cudaMemcpy(&failedAt, deviceFailure, sizeof(int), cudaMemcpyDeviceToHost),
                    "diagonal status download")) {
            goto cleanup;
        }
        if (failedAt != 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failedAt - 1);
            goto cleanup;
        }
        if (!cudaOk(cudaMemcpy(A.data(), deviceMatrix, bytes, cudaMemcpyDeviceToHost),
                    "result download")) {
            goto cleanup;
        }
        // Preserve the original routine's exact lower-triangular output even
        // on CUDA implementations whose library stream is nonblocking with
        // respect to the host's default stream.
        for (int row = 0; row < dimension; ++row) {
            std::fill(A.begin() + static_cast<size_t>(row) * dimension + row + 1,
                      A.begin() + static_cast<size_t>(row + 1) * dimension, 0.0);
        }
    }
    success = true;

cleanup:
    if (handle != nullptr) {
        cublasDestroy(handle);
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
        if (error > maxError) {
            maxError = error;
        }
        
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

    // Initialize the CUDA runtime and cuBLAS before timing, just as CPU
    // benchmark setup is excluded from the measured decomposition.
    cublasHandle_t warmupHandle = nullptr;
    if (!cudaOk(cudaFree(nullptr), "runtime initialization") ||
        !cublasOk(cublasCreate(&warmupHandle), "cuBLAS initialization")) {
        return 1;
    }
    cublasDestroy(warmupHandle);

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
