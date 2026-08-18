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

constexpr int kTileSize = 16;

// Factor one diagonal tile.  The panel itself is small, so keeping it in
// shared memory avoids repeatedly reading it from global memory.  The
// surrounding panel operations are parallel; only the inherently sequential
// work inside this diagonal tile is serialized.
__global__ void factorDiagonalTile(double* A, const int n, const int tile,
                                   int* notPositiveDefinite) {
    __shared__ double diagonal[kTileSize][kTileSize];
    const int tx = static_cast<int>(threadIdx.x);
    const int base = tile * kTileSize;
    const int extent = min(kTileSize, n - base);

    if (tx < kTileSize) {
        for (int col = tx; col < kTileSize; col += blockDim.x) {
            for (int row = 0; row < kTileSize; ++row) {
                diagonal[row][col] = (row < extent && col < extent)
                    ? A[(base + row) * n + base + col] : 0.0;
            }
        }
    }
    __syncthreads();

    if (tx == 0) {
        for (int j = 0; j < extent; ++j) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) {
                sum += diagonal[j][k] * diagonal[j][k];
            }
            const double value = diagonal[j][j] - sum;
            if (value <= 0.0) {
                *notPositiveDefinite = 1;
            }
            diagonal[j][j] = sqrt(value);
            for (int row = j + 1; row < extent; ++row) {
                double offDiagonalSum = 0.0;
                for (int k = 0; k < j; ++k) {
                    offDiagonalSum += diagonal[row][k] * diagonal[j][k];
                }
                diagonal[row][j] = (diagonal[row][j] - offDiagonalSum) /
                                   diagonal[j][j];
            }
        }
    }
    __syncthreads();

    for (int row = tx; row < extent; row += blockDim.x) {
        for (int col = 0; col <= row; ++col) {
            A[(base + row) * n + base + col] = diagonal[row][col];
        }
    }
}

// Solve all rows in one tile of the current panel.  Each thread computes one
// matrix element, and its short inner loop is the forward substitution for
// that element.
__global__ void solvePanelTile(double* A, const int n, const int panel) {
    const int rowTile = panel + 1 + static_cast<int>(blockIdx.x);
    const int row = rowTile * kTileSize + static_cast<int>(threadIdx.y);
    const int col = panel * kTileSize + static_cast<int>(threadIdx.x);
    const int panelEnd = min(n, (panel + 1) * kTileSize);
    if (row >= n || col >= panelEnd) {
        return;
    }

    double value = A[row * n + col];
    const int colInTile = col - panel * kTileSize;
    for (int k = 0; k < colInTile; ++k) {
        value -= A[row * n + panel * kTileSize + k] *
                 A[col * n + panel * kTileSize + k];
    }
    A[row * n + col] = value / A[col * n + col];
}

// Rank-k update of a pair of lower-triangular tiles.  The two source tiles
// are staged in shared memory, giving coalesced global loads and reuse across
// the entire output tile.
__global__ void updateTrailingTile(double* A, const int n, const int panel) {
    const int rowTile = panel + 1 + static_cast<int>(blockIdx.y);
    const int colTile = panel + 1 + static_cast<int>(blockIdx.x);
    const int localRow = static_cast<int>(threadIdx.y);
    const int localCol = static_cast<int>(threadIdx.x);
    const int row = rowTile * kTileSize + localRow;
    const int col = colTile * kTileSize + localCol;
    const int panelBase = panel * kTileSize;

    if (row >= n || col >= n || (rowTile == colTile && row < col)) {
        return;
    }

    double value = A[row * n + col];
    const int panelEnd = min(n, panelBase + kTileSize);
    for (int k = panelBase; k < panelEnd; ++k) {
        value -= A[row * n + k] * A[col * n + k];
    }
    A[row * n + col] = value;
}

__global__ void zeroUpperTriangle(double* A, const int n) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n && col < n && col > row) {
        A[row * n + col] = 0.0;
    }
}

__global__ void factorColumnDiagonal(double* A, const int n, const int j,
                                     int* notPositiveDefinite) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k) {
            sum += A[j * n + k] * A[j * n + k];
        }
        const double value = A[j * n + j] - sum;
        if (value <= 0.0) {
            *notPositiveDefinite = 1;
        } else {
            A[j * n + j] = sqrt(value);
        }
    }
}

__global__ void factorColumnOffDiagonal(double* A, const int n, const int j) {
    const int i = j + 1 + static_cast<int>(blockIdx.x * blockDim.x +
                                             threadIdx.x);
    if (i >= n) {
        return;
    }
    double sum = 0.0;
    for (int k = 0; k < j; ++k) {
        sum += A[i * n + k] * A[j * n + k];
    }
    A[i * n + j] = (A[i * n + j] - sum) / A[j * n + j];
}

} // namespace

// GPU blocked Cholesky decomposition.  A is kept in row-major order and is
// overwritten in-place with L, matching the original benchmark's semantics.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    if (n > static_cast<size_t>(INT_MAX) || A.size() < n * n) {
        printf("Error: Matrix is too large for CUDA\n");
        return false;
    }

    const int matrixSize = static_cast<int>(n);
    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    if (cudaMalloc(&deviceA, n * n * sizeof(double)) != cudaSuccess ||
        cudaMalloc(&deviceFailure, sizeof(int)) != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceFailure);
        printf("Error: CUDA allocation failed\n");
        return false;
    }
    if (cudaMemcpy(deviceA, A.data(), n * n * sizeof(double),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(deviceFailure, 0, sizeof(int)) != cudaSuccess) {
        cudaFree(deviceA);
        cudaFree(deviceFailure);
        printf("Error: CUDA host-to-device transfer failed\n");
        return false;
    }

    for (int j = 0; j < matrixSize; ++j) {
        factorColumnDiagonal<<<1, 1>>>(deviceA, matrixSize, j,
                                       deviceFailure);
        const int rowsBelow = matrixSize - j - 1;
        if (rowsBelow > 0) {
            factorColumnOffDiagonal<<<(rowsBelow + 255) / 256, 256>>>(
                deviceA, matrixSize, j);
        }
    }

    zeroUpperTriangle<<<dim3((matrixSize + 15) / 16,
                             (matrixSize + 15) / 16), dim3(16, 16)>>>(
        deviceA, matrixSize);

    const cudaError_t synchronizationStatus = cudaDeviceSynchronize();
    int failed = 0;
    const cudaError_t copyStatus = cudaMemcpy(&failed, deviceFailure,
                                               sizeof(int),
                                               cudaMemcpyDeviceToHost);
    const cudaError_t resultStatus = cudaMemcpy(A.data(), deviceA,
                                                 n * n * sizeof(double),
                                                 cudaMemcpyDeviceToHost);
    cudaFree(deviceFailure);
    cudaFree(deviceA);

    if (synchronizationStatus != cudaSuccess || copyStatus != cudaSuccess ||
        resultStatus != cudaSuccess) {
        printf("Error: CUDA kernel execution failed: %s\n",
               cudaGetErrorString(synchronizationStatus));
        return false;
    }
    if (failed != 0) {
        printf("Error: Matrix is not positive definite\n");
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
