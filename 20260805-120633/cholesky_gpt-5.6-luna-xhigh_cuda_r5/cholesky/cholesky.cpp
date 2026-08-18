#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int kTileSize = 32;
constexpr int kUpdateBlockX = 32;
constexpr int kUpdateBlockY = 8;
constexpr size_t kNoFailure = std::numeric_limits<size_t>::max();

// The factorization is blocked so that the expensive trailing update is a
// collection of independent tiled rank-k updates.  A 32x32 tile gives enough
// independent tiles to occupy the whole GPU even for the benchmark's default
// matrix size, while keeping the shared-memory footprint small enough for
// multiple resident blocks.

__global__ void factorDiagonalTile(double* matrix,
                                   const size_t n,
                                   const size_t base,
                                   const int width,
                                   size_t* failure) {
    __shared__ double tile[kTileSize * kTileSize];
    __shared__ int failed;

    const int tid = static_cast<int>(threadIdx.x);
    const size_t tileElements = static_cast<size_t>(width) * width;

    for (size_t index = tid; index < tileElements; index += blockDim.x) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        tile[index] = matrix[(base + row) * n + base + column];
    }
    __syncthreads();

    for (int column = 0; column < width; ++column) {
        if (tid == 0) {
            failed = (*failure != kNoFailure) ? 1 : 0;
        }
        __syncthreads();
        if (failed != 0) {
            break;
        }

        if (tid == column) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                const double value = tile[column * width + k];
                sum += value * value;
            }

            const double diagonal = tile[column * width + column] - sum;
            if (diagonal <= 0.0) {
                // Only one thread handles each diagonal, so this write is
                // race-free.  The first failing diagonal is the one reported.
                *failure = base + static_cast<size_t>(column);
            } else {
                tile[column * width + column] = sqrt(diagonal);
            }
        }

        if (tid > column && tid < width) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += tile[tid * width + k] * tile[column * width + k];
            }
            tile[tid * width + column] =
                (tile[tid * width + column] - sum) /
                tile[column * width + column];
        }
        __syncthreads();
        if (tid == 0) {
            failed = (*failure != kNoFailure) ? 1 : 0;
        }
        __syncthreads();
        if (failed != 0) {
            break;
        }
    }

    // Store the complete tile and explicitly clear its upper triangle.  The
    // original routine returns a full row-major matrix with zeros above L.
    for (size_t index = tid; index < tileElements; index += blockDim.x) {
        const size_t row = index / width;
        const size_t column = index - row * width;
        matrix[(base + row) * n + base + column] =
            (row >= column) ? tile[index] : 0.0;
    }
}

// Solve B * L^T = B in-place for every tile below the current diagonal tile.
// Each thread owns one matrix row, and rows are independent once the diagonal
// tile has been factored.
__global__ void solvePanel(double* matrix,
                           const size_t n,
                           const size_t rowBase,
                           const size_t columnBase,
                           const int rowCount,
                           const int diagonalWidth,
                           const size_t* failure) {
    if (*failure != kNoFailure) {
        return;
    }

    const int rowOffset = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (rowOffset >= rowCount) {
        return;
    }

    const size_t row = rowBase + static_cast<size_t>(rowOffset);
    for (int column = 0; column < diagonalWidth; ++column) {
        double value = matrix[row * n + columnBase + column];
        for (int k = 0; k < column; ++k) {
            value -= matrix[row * n + columnBase + k] *
                     matrix[(columnBase + column) * n + columnBase + k];
        }
        matrix[row * n + columnBase + column] =
            value / matrix[(columnBase + column) * n + columnBase + column];
    }
}

// Update all lower-triangular tiles in the trailing matrix:
// A(i,j) -= A(i,k) * A(j,k)^T.
// The second input tile is transposed while loading, making the inner loop
// read contiguous shared-memory locations across each warp.
__global__ void updateTrailing(double* matrix,
                               const size_t n,
                               const size_t panelBase,
                               const int panelWidth,
                               const size_t firstTrailingTile,
                               const size_t tileCount,
                               const size_t* failure) {
    if (*failure != kNoFailure) {
        return;
    }

    const size_t columnTile = firstTrailingTile + blockIdx.x;
    const size_t rowTile = firstTrailingTile + blockIdx.y;
    if (rowTile >= tileCount || columnTile > rowTile) {
        return;
    }

    const size_t rowBase = rowTile * kTileSize;
    const size_t columnBase = columnTile * kTileSize;
    const int rowCount = static_cast<int>(min(
        static_cast<size_t>(kTileSize), n - rowBase));
    const int columnCount = static_cast<int>(min(
        static_cast<size_t>(kTileSize), n - columnBase));

    extern __shared__ double shared[];
    double* left = shared;
    double* rightTransposed = shared + kTileSize * kTileSize;

    const int linearThread = static_cast<int>(
        threadIdx.y * blockDim.x + threadIdx.x);
    constexpr int threadsPerBlock = kUpdateBlockX * kUpdateBlockY;
    constexpr size_t tileElements =
        static_cast<size_t>(kTileSize) * kTileSize;

    for (size_t index = linearThread; index < tileElements;
         index += threadsPerBlock) {
        const int tileRow = static_cast<int>(index / kTileSize);
        const int panelColumn = static_cast<int>(index % kTileSize);

        left[index] = (tileRow < rowCount && panelColumn < panelWidth)
            ? matrix[(rowBase + tileRow) * n + panelBase + panelColumn]
            : 0.0;

        // Store the right input transposed: [panelColumn][tileRow].
        rightTransposed[panelColumn * kTileSize + tileRow] =
            (tileRow < columnCount && panelColumn < panelWidth)
            ? matrix[(columnBase + tileRow) * n + panelBase + panelColumn]
            : 0.0;
    }
    __syncthreads();

    const int column = static_cast<int>(threadIdx.x);
    double accum[kUpdateBlockY / 2];
    bool active[kUpdateBlockY / 2];

    #pragma unroll
    for (int r = 0; r < kUpdateBlockY / 2; ++r) {
        const int row = static_cast<int>(threadIdx.y) + r * kUpdateBlockY;
        active[r] = row < rowCount && column < columnCount &&
                    (rowTile != columnTile || row >= column);
        accum[r] = active[r]
            ? matrix[(rowBase + row) * n + columnBase + column]
            : 0.0;
    }

    #pragma unroll
    for (int k = 0; k < kTileSize; ++k) {
        if (k >= panelWidth) {
            break;
        }
        const double right = rightTransposed[k * kTileSize + column];
        #pragma unroll
        for (int r = 0; r < kUpdateBlockY / 2; ++r) {
            const int row = static_cast<int>(threadIdx.y) + r * kUpdateBlockY;
            if (active[r]) {
                accum[r] -= left[row * kTileSize + k] * right;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < kUpdateBlockY / 2; ++r) {
        const int row = static_cast<int>(threadIdx.y) + r * kUpdateBlockY;
        if (active[r]) {
            matrix[(rowBase + row) * n + columnBase + column] = accum[r];
        }
    }
}

__global__ void zeroUpperTriangle(double* matrix, const size_t n) {
    const size_t elements = n * n;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
         index < elements; index += stride) {
        const size_t row = index / n;
        const size_t column = index - row * n;
        if (column > row) {
            matrix[index] = 0.0;
        }
    }
}

bool cudaSucceeded(const cudaError_t error, const char* operation) {
    if (error == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation,
            cudaGetErrorString(error));
    return false;
}

} // namespace

// Decomposes positive definite matrix A into L * L^T where L is lower
// triangular.  All factorization work is performed unconditionally on the
// CUDA device; there is deliberately no sequential CPU fallback.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    double* deviceMatrix = nullptr;
    size_t* deviceFailure = nullptr;
    const size_t bytes = n * n * sizeof(double);

    if (!cudaSucceeded(cudaMalloc(&deviceMatrix, bytes), "cudaMalloc(matrix)")) {
        return false;
    }
    if (!cudaSucceeded(cudaMalloc(&deviceFailure, sizeof(size_t)),
                       "cudaMalloc(failure)")) {
        cudaFree(deviceMatrix);
        return false;
    }

    auto cleanup = [&]() {
        cudaFree(deviceFailure);
        cudaFree(deviceMatrix);
    };

    if (!cudaSucceeded(cudaMemcpy(deviceMatrix, A.data(), bytes,
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpyHostToDevice")) {
        cleanup();
        return false;
    }
    if (!cudaSucceeded(cudaMemset(deviceFailure, 0xff, sizeof(size_t)),
                       "cudaMemset(failure)")) {
        cleanup();
        return false;
    }

    const size_t tileCount = (n + kTileSize - 1) / kTileSize;
    const size_t updateBytes = 2 * static_cast<size_t>(kTileSize) *
                               kTileSize * sizeof(double);
    if (!cudaSucceeded(cudaFuncSetAttribute(
            updateTrailing, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(updateBytes)),
                       "cudaFuncSetAttribute(shared memory)")) {
        cleanup();
        return false;
    }

    for (size_t tile = 0; tile < tileCount; ++tile) {
        const size_t base = tile * kTileSize;
        const int width = static_cast<int>(min(
            static_cast<size_t>(kTileSize), n - base));

        factorDiagonalTile<<<1, kTileSize>>>(
            deviceMatrix, n, base, width, deviceFailure);
        if (!cudaSucceeded(cudaGetLastError(), "factorDiagonalTile launch")) {
            cleanup();
            return false;
        }

        const size_t firstRow = base + kTileSize;
        if (firstRow < n) {
            const size_t rowTiles = (n - firstRow + kTileSize - 1) /
                                    kTileSize;
            solvePanel<<<static_cast<unsigned int>(rowTiles), kTileSize>>>(
                deviceMatrix, n, firstRow, base,
                static_cast<int>(n - firstRow),
                width, deviceFailure);
            if (!cudaSucceeded(cudaGetLastError(), "solvePanel launch")) {
                cleanup();
                return false;
            }

            const dim3 grid(static_cast<unsigned int>(rowTiles),
                            static_cast<unsigned int>(rowTiles), 1);
            updateTrailing<<<grid,
                             dim3(kUpdateBlockX, kUpdateBlockY),
                             updateBytes>>>(
                deviceMatrix, n, base, width, tile + 1, tileCount,
                deviceFailure);
            if (!cudaSucceeded(cudaGetLastError(),
                               "updateTrailing launch")) {
                cleanup();
                return false;
            }
        }
    }

    const size_t zeroBlocks = std::min<size_t>(
        65535, (n * n + 255) / 256);
    zeroUpperTriangle<<<static_cast<unsigned int>(zeroBlocks), 256>>>(
        deviceMatrix, n);
    if (!cudaSucceeded(cudaGetLastError(), "zeroUpperTriangle launch")) {
        cleanup();
        return false;
    }

    size_t failure = kNoFailure;
    if (!cudaSucceeded(cudaMemcpy(&failure, deviceFailure, sizeof(size_t),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy(failure)")) {
        cleanup();
        return false;
    }
    if (failure != kNoFailure) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               failure);
        cleanup();
        return false;
    }

    if (!cudaSucceeded(cudaMemcpy(A.data(), deviceMatrix, bytes,
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpyDeviceToHost")) {
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
    
    // Initialize the CUDA context before starting the measured section.  CUDA
    // context creation is a one-time runtime cost, not factorization work.
    if (!cudaSucceeded(cudaFree(nullptr), "CUDA initialization")) {
        return 1;
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
