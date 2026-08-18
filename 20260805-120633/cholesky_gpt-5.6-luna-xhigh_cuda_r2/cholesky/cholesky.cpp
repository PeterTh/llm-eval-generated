#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kCholeskyBlockSize = 128;
constexpr int kFactorThreads = 256;

bool checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        printf("CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        return false;
    }
    return true;
}

bool checkCublas(cublasStatus_t status, const char* operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS error in %s (status %d)\n", operation, static_cast<int>(status));
        return false;
    }
    return true;
}

// Factor one diagonal tile of an upper Cholesky factor in place. The tile is
// small enough that the serial dependency between its columns is cheap, while
// each dot product and each row update is distributed over a whole block.
// The device matrix is column-major; its raw storage is also the row-major
// storage used by the host, which lets the final upper-to-lower conversion be
// done by clearing only the unused half of the same allocation.
__global__ void factorUpperTile(double* matrix, const int n, const int offset,
                                const int tileSize, int* info) {
    __shared__ double reduction[kFactorThreads];

    const int tid = static_cast<int>(threadIdx.x);

    for (int j = 0; j < tileSize; ++j) {
        double partial = 0.0;
        for (int p = tid; p < j; p += kFactorThreads) {
            const size_t row = static_cast<size_t>(offset + p);
            const size_t column = static_cast<size_t>(offset + j);
            const size_t index = row + column * static_cast<size_t>(n);
            partial += matrix[index] * matrix[index];
        }

        reduction[tid] = partial;
        __syncthreads();
        for (int stride = kFactorThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                reduction[tid] += reduction[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            const size_t diagonal = static_cast<size_t>(offset + j);
            const size_t index = diagonal + diagonal * static_cast<size_t>(n);
            const double value = matrix[index] - reduction[0];
            if (value <= 0.0) {
                // Preserve the first failing pivot while allowing later
                // blocks to drain without host synchronization.
                if (*info < 0) {
                    *info = offset + j;
                }
            } else if (*info < 0) {
                matrix[index] = sqrt(value);
            }
        }
        __syncthreads();

        // Once U[j,j] is known, all remaining entries in this row are
        // independent. They are the work needed to form the tile's panel.
        if (*info < 0) {
            const int firstColumn = j + 1 + tid;
            for (int localColumn = firstColumn; localColumn < tileSize;
                 localColumn += kFactorThreads) {
                double sum = 0.0;
                for (int p = 0; p < j; ++p) {
                    const size_t row = static_cast<size_t>(offset + p);
                    const size_t leftColumn = static_cast<size_t>(offset + j);
                    const size_t rightColumn =
                        static_cast<size_t>(offset + localColumn);
                    sum += matrix[row + leftColumn * static_cast<size_t>(n)] *
                           matrix[row + rightColumn * static_cast<size_t>(n)];
                }

                const size_t row = static_cast<size_t>(offset + j);
                const size_t column = static_cast<size_t>(offset + localColumn);
                const size_t index = row + column * static_cast<size_t>(n);
                const size_t diagonal = row + row * static_cast<size_t>(n);
                matrix[index] = (matrix[index] - sum) / matrix[diagonal];
            }
        }
        __syncthreads();
    }
}

// The numerical factor is stored as an upper triangle in column-major
// interpretation. Clearing the raw upper half makes the same bytes appear as
// the required lower triangle when copied into the host's row-major vector.
__global__ void clearRawUpperHalf(double* matrix, const size_t elements,
                                  const int n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (index >= elements) {
        return;
    }

    const int row = static_cast<int>(index / static_cast<size_t>(n));
    const int column = static_cast<int>(index % static_cast<size_t>(n));
    if (row < column) {
        matrix[index] = 0.0;
    }
}

} // namespace

// Blocked upper Cholesky on the GPU. If U is computed in column-major
// interpretation, its raw bytes are U^T in the host's row-major
// interpretation, i.e. precisely the lower-triangular result expected by the
// original routine.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        printf("CUDA error: matrix dimension is too large\n");
        return false;
    }

    const int matrixSize = static_cast<int>(n);
    const size_t elements = n * n;
    const size_t bytes = elements * sizeof(double);

    double* deviceMatrix = nullptr;
    int* deviceInfo = nullptr;
    cudaStream_t stream = nullptr;
    cublasHandle_t blas = nullptr;
    bool success = false;
    const double one = 1.0;
    const double minusOne = -1.0;
    int info = -1;

    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceMatrix), bytes),
                   "cudaMalloc(matrix)")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceInfo), sizeof(int)),
                   "cudaMalloc(info)")) {
        goto cleanup;
    }
    if (!checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                   "cudaStreamCreateWithFlags")) {
        goto cleanup;
    }
    if (!checkCublas(cublasCreate(&blas), "cublasCreate")) {
        goto cleanup;
    }
    if (!checkCublas(cublasSetStream(blas, stream), "cublasSetStream")) {
        goto cleanup;
    }
    if (!checkCuda(cudaMemcpyAsync(deviceMatrix, A.data(), bytes,
                                   cudaMemcpyHostToDevice, stream),
                   "cudaMemcpyAsync(host-to-device)")) {
        goto cleanup;
    }

    if (!checkCuda(cudaMemsetAsync(deviceInfo, 0xff, sizeof(int), stream),
                   "cudaMemsetAsync(info)")) {
        goto cleanup;
    }

    for (int offset = 0; offset < matrixSize; offset += kCholeskyBlockSize) {
        const int tileSize = std::min(kCholeskyBlockSize, matrixSize - offset);

        factorUpperTile<<<1, kFactorThreads, 0, stream>>>(
            deviceMatrix, matrixSize, offset, tileSize, deviceInfo);
        if (!checkCuda(cudaGetLastError(), "factorUpperTile launch")) {
            goto cleanup;
        }

        const int trailing = matrixSize - offset - tileSize;
        if (trailing == 0) {
            continue;
        }

        // Solve U_kk^T U_kj = A_kj for the rectangular panel U_kj.
        double* diagonalTile =
            deviceMatrix + static_cast<size_t>(offset) +
            static_cast<size_t>(offset) * static_cast<size_t>(matrixSize);
        double* panel =
            deviceMatrix + static_cast<size_t>(offset) +
            static_cast<size_t>(offset + tileSize) *
                static_cast<size_t>(matrixSize);
        if (!checkCublas(cublasDtrsm(blas, CUBLAS_SIDE_LEFT,
                                     CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     CUBLAS_DIAG_NON_UNIT, tileSize, trailing,
                                     &one, diagonalTile, matrixSize, panel,
                                     matrixSize),
                        "cublasDtrsm")) {
            goto cleanup;
        }

        // A_jj <- A_jj - U_kj^T U_kj. Only the upper triangle is needed by
        // subsequent iterations, so SYRK halves the update's memory traffic.
        double* trailingMatrix =
            deviceMatrix + static_cast<size_t>(offset + tileSize) +
            static_cast<size_t>(offset + tileSize) *
                static_cast<size_t>(matrixSize);
        if (!checkCublas(cublasDsyrk(blas, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
                                     trailing, tileSize, &minusOne, panel,
                                     matrixSize, &one, trailingMatrix,
                                     matrixSize),
                        "cublasDsyrk")) {
            goto cleanup;
        }
    }

    if (!checkCuda(cudaMemcpyAsync(&info, deviceInfo, sizeof(int),
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(info)")) {
        goto cleanup;
    }
    if (!checkCuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(factorization)")) {
        goto cleanup;
    }
    if (info >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               info);
        goto cleanup;
    }

    {
        const size_t threads = 256;
        const size_t blocks = (elements + threads - 1) / threads;
        clearRawUpperHalf<<<static_cast<unsigned int>(blocks),
                            static_cast<unsigned int>(threads), 0, stream>>>(
            deviceMatrix, elements, matrixSize);
        if (!checkCuda(cudaGetLastError(), "clearRawUpperHalf launch")) {
            goto cleanup;
        }
    }

    if (!checkCuda(cudaMemcpyAsync(A.data(), deviceMatrix, bytes,
                                   cudaMemcpyDeviceToHost, stream),
                   "cudaMemcpyAsync(device-to-host)")) {
        goto cleanup;
    }
    if (!checkCuda(cudaStreamSynchronize(stream),
                   "cudaStreamSynchronize(copy)")) {
        goto cleanup;
    }

    success = true;

cleanup:
    if (blas != nullptr) {
        cublasDestroy(blas);
    }
    if (stream != nullptr) {
        cudaStreamDestroy(stream);
    }
    if (deviceInfo != nullptr) {
        cudaFree(deviceInfo);
    }
    if (deviceMatrix != nullptr) {
        cudaFree(deviceMatrix);
    }
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
