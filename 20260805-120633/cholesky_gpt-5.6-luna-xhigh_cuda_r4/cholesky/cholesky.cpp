#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// A 32-column panel gives the diagonal and triangular-solve kernels enough
// parallelism while keeping their shared-memory footprints small.  The
// trailing update uses 16 x 16 output tiles, which leaves room for two
// double-precision shared-memory tiles and sustains good occupancy.
constexpr int kPanelSize = 32;
constexpr int kUpdateTileSize = 16;

bool checkCuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) {
        return true;
    }

    printf("CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    return false;
}

// Factor one diagonal panel.  The panel is transposed in shared memory:
// tile[k][i] contains the matrix element at row i, column k.  This layout
// makes the dot products coalesced across the rows handled by the warp.
__global__ void factorDiagonalKernel(double* __restrict__ matrix,
                                     const size_t n,
                                     const size_t offset,
                                     const int width,
                                     int* failurePivot) {
    __shared__ double tile[kPanelSize][kPanelSize];

    const int row = static_cast<int>(threadIdx.x);

    for (int index = row; index < width * width; index += kPanelSize) {
        const int localRow = index / width;
        const int localColumn = index % width;
        tile[localColumn][localRow] =
            matrix[(offset + static_cast<size_t>(localRow)) * n +
                   offset + static_cast<size_t>(localColumn)];
    }
    __syncthreads();

    for (int column = 0; column < width; ++column) {
        if (row == column) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += tile[k][row] * tile[k][column];
            }

            const double value = tile[column][column] - sum;
            if (value <= 0.0) {
                atomicCAS(failurePivot, -1, static_cast<int>(offset + column));
                tile[column][row] = 0.0;
            } else {
                tile[column][row] = sqrt(value);
            }
        }
        __syncthreads();

        // Do not generate dependent values after a non-positive pivot.  The
        // host observes failurePivot after this kernel and reports the same
        // failure condition as the original implementation.
        if (row > column && *failurePivot < 0) {
            double sum = 0.0;
            for (int k = 0; k < column; ++k) {
                sum += tile[k][row] * tile[k][column];
            }
            tile[column][row] =
                (tile[column][row] - sum) / tile[column][column];
        }
        __syncthreads();
    }

    for (int index = row; index < width * width; index += kPanelSize) {
        const int localRow = index / width;
        const int localColumn = index % width;
        matrix[(offset + static_cast<size_t>(localRow)) * n +
               offset + static_cast<size_t>(localColumn)] =
            tile[localColumn][localRow];
    }
}

// Solve all block rows below the current diagonal block:
//     A(i, k) = A(i, k) * inv(L(k, k)^T)
// Each CUDA thread owns one matrix row, and columns of that row are solved
// sequentially because they are a triangular dependency.  Rows in all block
// rows execute concurrently.
__global__ void triangularSolveKernel(double* __restrict__ matrix,
                                      const size_t n,
                                      const size_t offset,
                                      const int width) {
    __shared__ double diagonal[kPanelSize][kPanelSize];

    const int threadRow = static_cast<int>(threadIdx.x);
    for (int index = threadRow; index < width * width; index += kPanelSize) {
        const int localRow = index / width;
        const int localColumn = index % width;
        diagonal[localColumn][localRow] =
            matrix[(offset + static_cast<size_t>(localRow)) * n +
                   offset + static_cast<size_t>(localColumn)];
    }
    __syncthreads();

    const size_t row = offset + static_cast<size_t>(width) +
                       static_cast<size_t>(blockIdx.x) * kPanelSize +
                       static_cast<size_t>(threadRow);
    if (row >= n) {
        return;
    }

    // This fixed-size array is register-resident for the normal 32-column
    // panel and avoids repeatedly reading the same row from device memory.
    double values[kPanelSize];
    for (int column = 0; column < width; ++column) {
        values[column] =
            matrix[row * n + offset + static_cast<size_t>(column)];
    }

    for (int column = 0; column < width; ++column) {
        double value = values[column];
        for (int k = 0; k < column; ++k) {
            value -= values[k] * diagonal[k][column];
        }
        values[column] = value / diagonal[column][column];
    }

    for (int column = 0; column < width; ++column) {
        matrix[row * n + offset + static_cast<size_t>(column)] =
            values[column];
    }
}

// Update one lower-triangular output tile of the trailing matrix:
//     A(i, j) -= L(i, k) * L(j, k)^T
// The grid covers a square, but upper-triangular tiles return immediately.
// The padded right-hand shared tile prevents 64-bit shared-memory bank
// conflicts when neighboring threads read different output rows.
__global__ void trailingUpdateKernel(double* __restrict__ matrix,
                                     const size_t n,
                                     const size_t panelOffset,
                                     const int width) {
    if (blockIdx.y < blockIdx.x) {
        return;
    }

    __shared__ double left[kUpdateTileSize][kPanelSize + 1];
    __shared__ double right[kUpdateTileSize][kPanelSize + 1];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const size_t trailingOffset = panelOffset + static_cast<size_t>(width);
    const size_t row = trailingOffset + static_cast<size_t>(blockIdx.y) *
                                             kUpdateTileSize +
                       static_cast<size_t>(ty);
    const size_t column = trailingOffset + static_cast<size_t>(blockIdx.x) *
                                             kUpdateTileSize +
                          static_cast<size_t>(tx);
    const size_t rightRow = trailingOffset + static_cast<size_t>(blockIdx.x) *
                                                  kUpdateTileSize +
                            static_cast<size_t>(ty);

    for (int k = tx; k < width; k += kUpdateTileSize) {
        left[ty][k] = row < n
            ? matrix[row * n + panelOffset + static_cast<size_t>(k)]
            : 0.0;
        right[ty][k] = rightRow < n
            ? matrix[rightRow * n + panelOffset + static_cast<size_t>(k)]
            : 0.0;
    }
    __syncthreads();

    if (row < n && column < n && row >= column) {
        double sum = 0.0;
        for (int k = 0; k < width; ++k) {
            sum += left[ty][k] * right[tx][k];
        }
        matrix[row * n + column] -= sum;
    }
}

__global__ void zeroUpperTriangleKernel(double* __restrict__ matrix,
                                        const size_t n) {
    const size_t row = static_cast<size_t>(blockIdx.y) * kUpdateTileSize +
                       static_cast<size_t>(threadIdx.y);
    const size_t column = static_cast<size_t>(blockIdx.x) * kUpdateTileSize +
                          static_cast<size_t>(threadIdx.x);
    if (row < n && column < n && column > row) {
        matrix[row * n + column] = 0.0;
    }
}

// Blocked right-looking CUDA Cholesky factorization.  The only host-side
// synchronization inside the factorization is the dependency between panel
// iterations; all rows, tiles, dot products, and trailing updates are GPU
// work.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }

    double* deviceMatrix = nullptr;
    int* failurePivot = nullptr;
    const size_t bytes = n * n * sizeof(double);

    auto cleanup = [&]() {
        if (failurePivot != nullptr) {
            cudaFree(failurePivot);
        }
        if (deviceMatrix != nullptr) {
            cudaFree(deviceMatrix);
        }
    };

    if (!checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceMatrix), bytes),
                   "allocating device matrix") ||
        !checkCuda(cudaMalloc(reinterpret_cast<void**>(&failurePivot),
                              sizeof(int)),
                   "allocating CUDA status") ||
        !checkCuda(cudaMemcpy(deviceMatrix, A.data(), bytes,
                              cudaMemcpyHostToDevice),
                   "copying matrix to device")) {
        cleanup();
        return false;
    }

    for (size_t offset = 0; offset < n; offset += kPanelSize) {
        const int width = static_cast<int>(
            std::min(static_cast<size_t>(kPanelSize), n - offset));

        if (!checkCuda(cudaMemset(failurePivot, 0xff, sizeof(int)),
                       "resetting CUDA status")) {
            cleanup();
            return false;
        }

        factorDiagonalKernel<<<1, kPanelSize>>>(
            deviceMatrix, n, offset, width, failurePivot);
        if (!checkCuda(cudaGetLastError(), "launching diagonal factorization")) {
            cleanup();
            return false;
        }

        int pivot = -1;
        if (!checkCuda(cudaMemcpy(&pivot, failurePivot, sizeof(int),
                                  cudaMemcpyDeviceToHost),
                       "checking diagonal factorization")) {
            cleanup();
            return false;
        }
        if (pivot >= 0) {
            printf("Error: Matrix is not positive definite at diagonal element %zu\n",
                   static_cast<size_t>(pivot));
            cleanup();
            return false;
        }

        const size_t trailing = offset + static_cast<size_t>(width);
        if (trailing < n) {
            const size_t trailingRows = n - trailing;
            const unsigned int panelBlocks = static_cast<unsigned int>(
                (trailingRows + kPanelSize - 1) / kPanelSize);

            triangularSolveKernel<<<panelBlocks, kPanelSize>>>(
                deviceMatrix, n, offset, width);
            if (!checkCuda(cudaGetLastError(), "launching triangular solve")) {
                cleanup();
                return false;
            }

            const unsigned int updateBlocks = static_cast<unsigned int>(
                (trailingRows + kUpdateTileSize - 1) / kUpdateTileSize);
            trailingUpdateKernel<<<dim3(updateBlocks, updateBlocks),
                                   dim3(kUpdateTileSize, kUpdateTileSize)>>>(
                deviceMatrix, n, offset, width);
            if (!checkCuda(cudaGetLastError(), "launching trailing update")) {
                cleanup();
                return false;
            }
        }
    }

    const unsigned int zeroBlocks = static_cast<unsigned int>(
        (n + kUpdateTileSize - 1) / kUpdateTileSize);
    zeroUpperTriangleKernel<<<dim3(zeroBlocks, zeroBlocks),
                              dim3(kUpdateTileSize, kUpdateTileSize)>>>(
        deviceMatrix, n);
    if (!checkCuda(cudaGetLastError(), "launching upper-triangle cleanup") ||
        !checkCuda(cudaMemcpy(A.data(), deviceMatrix, bytes,
                              cudaMemcpyDeviceToHost),
                   "copying matrix to host")) {
        cleanup();
        return false;
    }

    cleanup();
    return true;
}

}  // namespace

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
    // Initialize the CUDA context before starting the benchmark timer so the
    // reported duration measures the decomposition and transfers, not driver
    // startup on the first CUDA call.
    if (!checkCuda(cudaFree(nullptr), "initializing CUDA")) {
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
    const double seconds = duration.count() / 1000.0;
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
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
