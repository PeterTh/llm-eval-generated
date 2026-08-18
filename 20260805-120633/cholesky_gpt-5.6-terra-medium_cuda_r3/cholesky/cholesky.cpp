#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// A blocked, right-looking Cholesky factorization.  The panel and the trailing
// update remain on the GPU for the whole factorization, avoiding a host/device
// synchronization for the O(n^3) work.
constexpr int kTileSize = 32;
constexpr int kUpdateTileSize = 16;

__global__ void factorDiagonalTile(double* A, size_t n, size_t offset,
                                   size_t width, int* notPositiveDefinite) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }

    for (size_t col = 0; col < width; ++col) {
        const size_t diagonal = offset + col;
        double sum = 0.0;
        for (size_t k = 0; k < col; ++k) {
            const double value = A[diagonal * n + offset + k];
            sum += value * value;
        }
        const double value = A[diagonal * n + diagonal] - sum;
        if (value <= 0.0) {
            *notPositiveDefinite = 1;
            return;
        }
        A[diagonal * n + diagonal] = sqrt(value);

        for (size_t row = col + 1; row < width; ++row) {
            const size_t matrixRow = offset + row;
            double dot = 0.0;
            for (size_t k = 0; k < col; ++k) {
                dot += A[matrixRow * n + offset + k] *
                       A[diagonal * n + offset + k];
            }
            A[matrixRow * n + diagonal] =
                (A[matrixRow * n + diagonal] - dot) / A[diagonal * n + diagonal];
        }
    }
}

// Each thread solves one row of a below-diagonal tile.  Rows are independent
// once the diagonal tile has been factored.
__global__ void solvePanel(double* A, size_t n, size_t offset, size_t width) {
    const size_t tile = blockIdx.x;
    const size_t row = offset + width + tile * kTileSize + threadIdx.x;
    if (row >= n) {
        return;
    }

    for (size_t col = 0; col < width; ++col) {
        double sum = 0.0;
        for (size_t k = 0; k < col; ++k) {
            sum += A[row * n + offset + k] * A[(offset + col) * n + offset + k];
        }
        A[row * n + offset + col] =
            (A[row * n + offset + col] - sum) / A[(offset + col) * n + offset + col];
    }
}

// C_ij -= L_ik * L_jk^T for one pair of trailing tiles.  The panel tiles are
// staged in shared memory so each global load contributes to many FMAs.
__global__ void updateTrailingMatrix(double* A, size_t n, size_t offset,
                                     size_t width) {
    const size_t tileRow = blockIdx.y;
    const size_t tileCol = blockIdx.x;
    if (tileRow < tileCol) {
        return;
    }

    const size_t base = offset + width;
    const size_t row = base + tileRow * kUpdateTileSize + threadIdx.y;
    const size_t col = base + tileCol * kUpdateTileSize + threadIdx.x;
    __shared__ double left[kUpdateTileSize][kUpdateTileSize];
    __shared__ double right[kUpdateTileSize][kUpdateTileSize];

    double sum = 0.0;
    for (size_t panel = 0; panel < width; panel += kUpdateTileSize) {
        const size_t panelWidth = min(static_cast<size_t>(kUpdateTileSize), width - panel);
        if (row < n && threadIdx.x < panelWidth) {
            left[threadIdx.y][threadIdx.x] = A[row * n + offset + panel + threadIdx.x];
        } else {
            left[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (col < n && threadIdx.y < panelWidth) {
            right[threadIdx.y][threadIdx.x] = A[col * n + offset + panel + threadIdx.y];
        } else {
            right[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (size_t k = 0; k < panelWidth; ++k) {
            sum += left[threadIdx.y][k] * right[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < n && col < n) {
        A[row * n + col] -= sum;
    }
}

__global__ void zeroUpperTriangle(double* A, size_t n) {
    const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n * n) {
        return;
    }
    const size_t row = index / n;
    const size_t col = index - row * n;
    if (row < n && col > row) {
        A[index] = 0.0;
    }
}

bool cudaSucceeded(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    const size_t bytes = n * n * sizeof(double);
    if (!cudaSucceeded(cudaMalloc(&deviceA, bytes), "matrix allocation") ||
        !cudaSucceeded(cudaMalloc(&deviceFailure, sizeof(int)), "status allocation")) {
        cudaFree(deviceA);
        cudaFree(deviceFailure);
        return false;
    }
    if (!cudaSucceeded(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
                      "matrix upload") ||
        !cudaSucceeded(cudaMemset(deviceFailure, 0, sizeof(int)), "status initialization")) {
        cudaFree(deviceA);
        cudaFree(deviceFailure);
        return false;
    }

    for (size_t offset = 0; offset < n; offset += kTileSize) {
        const size_t width = std::min(static_cast<size_t>(kTileSize), n - offset);
        factorDiagonalTile<<<1, 1>>>(deviceA, n, offset, width, deviceFailure);
        int failed = 0;
        if (!cudaSucceeded(cudaGetLastError(), "diagonal factorization launch") ||
            !cudaSucceeded(cudaMemcpy(&failed, deviceFailure, sizeof(int), cudaMemcpyDeviceToHost),
                          "diagonal factorization status")) {
            cudaFree(deviceA);
            cudaFree(deviceFailure);
            return false;
        }
        if (failed != 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n", offset);
            cudaFree(deviceA);
            cudaFree(deviceFailure);
            return false;
        }

        const size_t rowsRemaining = n - offset - width;
        if (rowsRemaining != 0) {
            const size_t panelTiles = (rowsRemaining + kTileSize - 1) / kTileSize;
            solvePanel<<<static_cast<unsigned int>(panelTiles), kTileSize>>>(deviceA, n, offset, width);

            const size_t updateTiles = (rowsRemaining + kUpdateTileSize - 1) / kUpdateTileSize;
            const dim3 updateGrid(static_cast<unsigned int>(updateTiles),
                                  static_cast<unsigned int>(updateTiles));
            const dim3 updateBlock(kUpdateTileSize, kUpdateTileSize);
            updateTrailingMatrix<<<updateGrid, updateBlock>>>(deviceA, n, offset, width);
            if (!cudaSucceeded(cudaGetLastError(), "panel update launch")) {
                cudaFree(deviceA);
                cudaFree(deviceFailure);
                return false;
            }
        }
    }

    constexpr int cleanupThreads = 256;
    const unsigned int cleanupBlocks = static_cast<unsigned int>(
        (n * n + cleanupThreads - 1) / cleanupThreads);
    zeroUpperTriangle<<<cleanupBlocks, cleanupThreads>>>(deviceA, n);
    const bool copied = cudaSucceeded(cudaGetLastError(), "upper-triangle cleanup launch") &&
                        cudaSucceeded(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                                      "result download");
    cudaFree(deviceA);
    cudaFree(deviceFailure);
    if (!copied) {
        return false;
    }
    return true;
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

    // Establish the CUDA context before the measured region.  This one-time
    // driver setup is not part of the factorization itself.
    if (!cudaSucceeded(cudaFree(nullptr), "CUDA initialization")) {
        return 1;
    }
    
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
