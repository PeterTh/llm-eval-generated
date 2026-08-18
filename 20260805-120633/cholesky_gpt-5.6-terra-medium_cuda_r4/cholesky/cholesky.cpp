#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

// A blocked, right-looking Cholesky factorization.  Keeping the active panel
// small makes its unavoidable dependencies cheap; the O(n^3) trailing update
// is performed by thousands of independent CUDA threads.
constexpr int kPanelWidth = 64;

__global__ void factorDiagonalPanel(double* a, const size_t n, const int first,
                                    const int width, int* positiveDefinite) {
    const int localRow = threadIdx.x;
    for (int j = 0; j < width; ++j) {
        if (localRow == 0) {
            double sum = 0.0;
            const size_t row = static_cast<size_t>(first + j) * n;
            for (int p = 0; p < j; ++p) {
                const double value = a[row + first + p];
                sum += value * value;
            }
            const double value = a[row + first + j] - sum;
            if (value <= 0.0) {
                atomicExch(positiveDefinite, 0);
                a[row + first + j] = 1.0;
            } else {
                a[row + first + j] = sqrt(value);
            }
        }
        __syncthreads();

        const int i = localRow + j + 1;
        if (i < width) {
            const size_t row = static_cast<size_t>(first + i) * n;
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                sum += a[row + first + p] * a[static_cast<size_t>(first + j) * n + first + p];
            }
            a[row + first + j] = (a[row + first + j] - sum) /
                                 a[static_cast<size_t>(first + j) * n + first + j];
        }
        __syncthreads();
    }
}

// One thread factors a row of the panel, avoiding inter-block dependencies.
__global__ void solvePanel(double* a, const size_t n, const int first,
                           const int width, const int rowsBelow) {
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= rowsBelow) return;
    const size_t row = static_cast<size_t>(first + width + offset) * n;
    for (int j = 0; j < width; ++j) {
        double sum = 0.0;
        for (int p = 0; p < j; ++p) {
            sum += a[row + first + p] * a[static_cast<size_t>(first + j) * n + first + p];
        }
        a[row + first + j] = (a[row + first + j] - sum) /
                             a[static_cast<size_t>(first + j) * n + first + j];
    }
}

__global__ void zeroUpperTriangle(double* a, const size_t n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t elements = n * n;
    if (index < elements && index % n > index / n) a[index] = 0.0;
}

bool cudaCheck(const cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    return false;
}

bool cublasCheck(const cublasStatus_t status, const char* operation) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    printf("cuBLAS error during %s: status %d\n", operation, static_cast<int>(status));
    return false;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    double* deviceA = nullptr;
    int* devicePositiveDefinite = nullptr;
    cublasHandle_t cublas = nullptr;
    int positiveDefinite = 1;
    const size_t bytes = n * n * sizeof(double);
    if (!cudaCheck(cudaMalloc(&deviceA, bytes), "matrix allocation") ||
        !cudaCheck(cudaMalloc(&devicePositiveDefinite, sizeof(int)), "status allocation")) {
        cudaFree(deviceA);
        cudaFree(devicePositiveDefinite);
        return false;
    }
    bool ok = cublasCheck(cublasCreate(&cublas), "handle creation") &&
              cudaCheck(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "matrix upload") &&
              cudaCheck(cudaMemcpy(devicePositiveDefinite, &positiveDefinite, sizeof(int), cudaMemcpyHostToDevice), "status upload");
    for (int first = 0; ok && first < static_cast<int>(n); first += kPanelWidth) {
        const int width = std::min(kPanelWidth, static_cast<int>(n) - first);
        const int rowsBelow = static_cast<int>(n) - first - width;
        factorDiagonalPanel<<<1, kPanelWidth>>>(deviceA, n, first, width, devicePositiveDefinite);
        ok = cudaCheck(cudaGetLastError(), "diagonal panel factorization");
        if (!ok) break;
        if (rowsBelow != 0) {
            solvePanel<<<(rowsBelow + 255) / 256, 256>>>(deviceA, n, first, width, rowsBelow);
            // Row-major L21 is viewed by cuBLAS as L21^T.  op(A)=T and
            // op(B)=N therefore form L21 * L21^T directly in the trailing
            // matrix.  cuBLAS supplies the high-throughput O(n^3) update.
            const double minusOne = -1.0;
            const double one = 1.0;
            ok = cudaCheck(cudaGetLastError(), "panel solve") &&
                 cublasCheck(cublasDgemm(cublas, CUBLAS_OP_T, CUBLAS_OP_N,
                                         rowsBelow, rowsBelow, width, &minusOne,
                                         deviceA + static_cast<size_t>(first + width) * n + first,
                                         static_cast<int>(n),
                                         deviceA + static_cast<size_t>(first + width) * n + first,
                                         static_cast<int>(n), &one,
                                         deviceA + static_cast<size_t>(first + width) * n + first + width,
                                         static_cast<int>(n)), "trailing matrix update");
        }
    }
    // A single status transfer also synchronizes all queued panel/update work;
    // doing this once avoids a host/device round trip for every panel.
    if (ok) {
        ok = cudaCheck(cudaMemcpy(&positiveDefinite, devicePositiveDefinite, sizeof(int),
                                  cudaMemcpyDeviceToHost), "status download");
    }
    if (!positiveDefinite) {
        printf("Error: Matrix is not positive definite\n");
    }
    if (ok && positiveDefinite) {
        const size_t threads = 256;
        zeroUpperTriangle<<<(n * n + threads - 1) / threads, threads>>>(deviceA, n);
        ok = cudaCheck(cudaGetLastError(), "upper triangle cleanup") &&
             cudaCheck(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "matrix download");
    }
    cudaFree(deviceA);
    cudaFree(devicePositiveDefinite);
    if (cublas != nullptr) cublasDestroy(cublas);
    return ok && positiveDefinite;
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
