#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kTileSize = 32;
constexpr int kUpdateBlockSize = kTileSize / 2;

// Factor one diagonal tile. A thread owns one row of the tile, so each column
// of the panel is computed in parallel over all rows below its diagonal.
__global__ void factorDiagonalTile(double* matrix, const size_t n,
                                   const size_t tileBase, const int tileWidth,
                                   int* failurePivot) {
    __shared__ double tile[kTileSize][kTileSize];

    const int row = static_cast<int>(threadIdx.x);
    if (row < tileWidth) {
        for (int column = 0; column < tileWidth; ++column) {
            tile[row][column] = matrix[(tileBase + row) * n + tileBase + column];
        }
    }
    __syncthreads();

    for (int column = 0; column < tileWidth; ++column) {
        // The pivot must be available before any row below it performs its
        // division. A second barrier after those divisions advances the
        // panel to the next column safely.
        if (row == column) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += tile[row][k] * tile[column][k];
            }

            const double value = tile[column][column] - sum;
            if (value <= 0.0) {
                // Keep the rest of the kernel well-defined while the host
                // reports the same first non-positive pivot as the scalar
                // implementation.
                atomicExch(failurePivot, static_cast<int>(tileBase + column));
                tile[column][column] = 1.0;
            } else {
                tile[column][column] = sqrt(value);
            }
        }
        __syncthreads();

        if (row > column && row < tileWidth) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += tile[row][k] * tile[column][k];
            }
            tile[row][column] =
                (tile[row][column] - sum) / tile[column][column];
        }
        __syncthreads();
    }

    if (row < tileWidth) {
        for (int column = 0; column < tileWidth; ++column) {
            matrix[(tileBase + row) * n + tileBase + column] =
                (column <= row) ? tile[row][column] : 0.0;
        }
    }
}

// Solve X * L^T = B for every row in one tile below the current diagonal.
// Each thread handles one complete row; rows belonging to different tiles
// are independent and are launched as separate blocks.
__global__ void solvePanelTile(double* matrix, const size_t n,
                               const size_t diagonalBase,
                               const size_t trailingBase,
                               const int tileWidth) {
    __shared__ double diagonal[kTileSize][kTileSize];

    const int row = static_cast<int>(threadIdx.x);
    if (row < tileWidth) {
        for (int column = 0; column < tileWidth; ++column) {
            diagonal[row][column] =
                matrix[(diagonalBase + row) * n + diagonalBase + column];
        }
    }
    __syncthreads();

    const size_t rowBase = trailingBase +
                           static_cast<size_t>(blockIdx.x) * kTileSize;
    const int rowHeight = static_cast<int>(min(
        static_cast<size_t>(kTileSize), n - rowBase));
    if (row < rowHeight) {
        const size_t matrixRow = rowBase + row;
        for (int column = 0; column < tileWidth; ++column) {
            double value = matrix[matrixRow * n + diagonalBase + column];
            for (int k = 0; k < column; ++k) {
                value -= matrix[matrixRow * n + diagonalBase + k] *
                         diagonal[column][k];
            }
            matrix[matrixRow * n + diagonalBase + column] =
                value / diagonal[column][column];
        }
    }
}

// Update one lower-triangular tile of the trailing matrix:
// A_ij -= A_ik * A_jk^T.
// A 16x16 block computes a 32x32 tile, with each thread producing a 2x2
// output. This gives the update high arithmetic intensity without requiring
// a 1024-thread block.
__global__ void updateTrailingTile(double* matrix, const size_t n,
                                   const size_t trailingBase,
                                   const int panelWidth,
                                   const int trailingTileCount) {
    const int tileRow = static_cast<int>(blockIdx.y);
    const int tileColumn = static_cast<int>(blockIdx.x);
    if (tileRow < tileColumn || tileRow >= trailingTileCount ||
        tileColumn >= trailingTileCount) {
        return;
    }

    const size_t rowBase = trailingBase +
                           static_cast<size_t>(tileRow) * kTileSize;
    const size_t columnBase = trailingBase +
                              static_cast<size_t>(tileColumn) * kTileSize;
    const int rowHeight = static_cast<int>(min(
        static_cast<size_t>(kTileSize), n - rowBase));
    const int columnWidth = static_cast<int>(min(
        static_cast<size_t>(kTileSize), n - columnBase));

    __shared__ double left[kTileSize][kTileSize];
    __shared__ double right[kTileSize][kTileSize];

    const int localRow0 = 2 * static_cast<int>(threadIdx.y);
    const int localRow1 = localRow0 + 1;
    const int localColumn0 = 2 * static_cast<int>(threadIdx.x);
    const int localColumn1 = localColumn0 + 1;

    // The second index of both shared tiles is the panel dimension. The
    // loads are coalesced for each matrix row and also cover edge tiles.
    if (localRow0 < rowHeight) {
        left[localRow0][localColumn0] =
            (localColumn0 < panelWidth)
                ? matrix[(rowBase + localRow0) * n + trailingBase - panelWidth +
                         localColumn0]
                : 0.0;
        left[localRow0][localColumn1] =
            (localColumn1 < panelWidth)
                ? matrix[(rowBase + localRow0) * n + trailingBase - panelWidth +
                         localColumn1]
                : 0.0;
    }
    if (localRow1 < rowHeight) {
        left[localRow1][localColumn0] =
            (localColumn0 < panelWidth)
                ? matrix[(rowBase + localRow1) * n + trailingBase - panelWidth +
                         localColumn0]
                : 0.0;
        left[localRow1][localColumn1] =
            (localColumn1 < panelWidth)
                ? matrix[(rowBase + localRow1) * n + trailingBase - panelWidth +
                         localColumn1]
                : 0.0;
    }
    if (localRow0 < columnWidth) {
        right[localRow0][localColumn0] =
            (localColumn0 < panelWidth)
                ? matrix[(columnBase + localRow0) * n + trailingBase - panelWidth +
                         localColumn0]
                : 0.0;
        right[localRow0][localColumn1] =
            (localColumn1 < panelWidth)
                ? matrix[(columnBase + localRow0) * n + trailingBase - panelWidth +
                         localColumn1]
                : 0.0;
    }
    if (localRow1 < columnWidth) {
        right[localRow1][localColumn0] =
            (localColumn0 < panelWidth)
                ? matrix[(columnBase + localRow1) * n + trailingBase - panelWidth +
                         localColumn0]
                : 0.0;
        right[localRow1][localColumn1] =
            (localColumn1 < panelWidth)
                ? matrix[(columnBase + localRow1) * n + trailingBase - panelWidth +
                         localColumn1]
                : 0.0;
    }
    __syncthreads();

    const bool row0Valid = localRow0 < rowHeight;
    const bool row1Valid = localRow1 < rowHeight;
    const bool column0Valid = localColumn0 < columnWidth;
    const bool column1Valid = localColumn1 < columnWidth;

    double sum00 = 0.0;
    double sum01 = 0.0;
    double sum10 = 0.0;
    double sum11 = 0.0;
    for (int k = 0; k < panelWidth; ++k) {
        if (row0Valid && column0Valid) {
            sum00 += left[localRow0][k] * right[localColumn0][k];
        }
        if (row0Valid && column1Valid) {
            sum01 += left[localRow0][k] * right[localColumn1][k];
        }
        if (row1Valid && column0Valid) {
            sum10 += left[localRow1][k] * right[localColumn0][k];
        }
        if (row1Valid && column1Valid) {
            sum11 += left[localRow1][k] * right[localColumn1][k];
        }
    }

    if (row0Valid && column0Valid &&
        (tileRow != tileColumn || localRow0 >= localColumn0)) {
        matrix[(rowBase + localRow0) * n + columnBase + localColumn0] -= sum00;
    }
    if (row0Valid && column1Valid &&
        (tileRow != tileColumn || localRow0 >= localColumn1)) {
        matrix[(rowBase + localRow0) * n + columnBase + localColumn1] -= sum01;
    }
    if (row1Valid && column0Valid &&
        (tileRow != tileColumn || localRow1 >= localColumn0)) {
        matrix[(rowBase + localRow1) * n + columnBase + localColumn0] -= sum10;
    }
    if (row1Valid && column1Valid &&
        (tileRow != tileColumn || localRow1 >= localColumn1)) {
        matrix[(rowBase + localRow1) * n + columnBase + localColumn1] -= sum11;
    }
}

__global__ void zeroUpperTriangle(double* matrix, const size_t n) {
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y +
                       threadIdx.y;
    const size_t column = static_cast<size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    if (row < n && column < n && column > row) {
        matrix[row * n + column] = 0.0;
    }
}

bool reportCudaError(const cudaError_t error, const char* operation) {
    if (error == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation,
            cudaGetErrorString(error));
    return false;
}

}  // namespace

// Blocked, right-looking Cholesky decomposition on the GPU. The matrix is
// overwritten in row-major order with L, where A = L * L^T.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    const size_t elementCount = n * n;
    const size_t bytes = elementCount * sizeof(double);
    double* deviceMatrix = nullptr;
    int* deviceFailurePivot = nullptr;

    auto cleanup = [&]() {
        if (deviceFailurePivot != nullptr) {
            cudaFree(deviceFailurePivot);
        }
        if (deviceMatrix != nullptr) {
            cudaFree(deviceMatrix);
        }
    };

    if (!reportCudaError(cudaMalloc(&deviceMatrix, bytes),
                         "allocating device matrix") ||
        !reportCudaError(cudaMalloc(&deviceFailurePivot, sizeof(int)),
                         "allocating pivot status")) {
        cleanup();
        return false;
    }
    if (!reportCudaError(cudaMemcpy(deviceMatrix, A.data(), bytes,
                                    cudaMemcpyHostToDevice),
                         "copying matrix to device") ||
        !reportCudaError(cudaMemset(deviceFailurePivot, 0xff, sizeof(int)),
                         "initializing pivot status")) {
        cleanup();
        return false;
    }

    const size_t tileCount = (n + kTileSize - 1) / kTileSize;
    for (size_t tile = 0; tile < tileCount; ++tile) {
        const size_t diagonalBase = tile * kTileSize;
        const int diagonalWidth = static_cast<int>(std::min(
            static_cast<size_t>(kTileSize), n - diagonalBase));

        factorDiagonalTile<<<1, kTileSize>>>(
            deviceMatrix, n, diagonalBase, diagonalWidth,
            deviceFailurePivot);

        int failurePivot = -1;
        if (!reportCudaError(cudaMemcpy(&failurePivot, deviceFailurePivot,
                                        sizeof(failurePivot),
                                        cudaMemcpyDeviceToHost),
                             "factoring diagonal tile")) {
            cleanup();
            return false;
        }
        if (failurePivot >= 0) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n",
                   failurePivot);
            cleanup();
            return false;
        }

        const size_t trailingBase = diagonalBase + diagonalWidth;
        if (trailingBase < n) {
            const size_t remaining = n - trailingBase;
            const size_t trailingTileCount =
                (remaining + kTileSize - 1) / kTileSize;
            const int trailingTiles = static_cast<int>(trailingTileCount);

            solvePanelTile<<<trailingTiles, kTileSize>>>(
                deviceMatrix, n, diagonalBase, trailingBase, diagonalWidth);

            updateTrailingTile<<<dim3(trailingTiles, trailingTiles),
                                 dim3(kUpdateBlockSize, kUpdateBlockSize)>>>(
                deviceMatrix, n, trailingBase, diagonalWidth, trailingTiles);
        }
    }

    const dim3 zeroBlock(16, 16);
    const dim3 zeroGrid(static_cast<unsigned int>((n + 15) / 16),
                        static_cast<unsigned int>((n + 15) / 16));
    zeroUpperTriangle<<<zeroGrid, zeroBlock>>>(deviceMatrix, n);

    if (!reportCudaError(cudaMemcpy(A.data(), deviceMatrix, bytes,
                                    cudaMemcpyDeviceToHost),
                         "copying result to host")) {
        cleanup();
        return false;
    }

    cleanup();
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
    if (!reportCudaError(cudaFree(0), "initializing CUDA device")) {
        return 1;
    }
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
