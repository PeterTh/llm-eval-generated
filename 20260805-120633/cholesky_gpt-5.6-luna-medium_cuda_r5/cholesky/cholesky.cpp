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

constexpr int TILE = 32;

inline bool cudaOk(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    return false;
}

// Factor one diagonal tile. The tile is small enough that keeping it in shared
// memory makes the panel boundary cheap; the trailing update is the dominant
// part of the factorization and is fully tiled and parallel.
__global__ void factorTile(double* __restrict__ a, const size_t n, const size_t k,
                           const int width, int* __restrict__ failure) {
    extern __shared__ double tile[];
    const int tid = static_cast<int>(threadIdx.x);
    const int elements = width * width;
    for (int p = tid; p < elements; p += blockDim.x) {
        const int r = p / width;
        const int c = p % width;
        tile[p] = a[(k + static_cast<size_t>(r)) * n + k + static_cast<size_t>(c)];
    }
    __syncthreads();

    if (tid == 0) {
        for (int i = 0; i < width; ++i) {
            double sum = 0.0;
            for (int p = 0; p < i; ++p) {
                const double v = tile[i * width + p];
                sum += v * v;
            }
            const double diagonal = tile[i * width + i] - sum;
            if (!(diagonal > 0.0)) {
                *failure = 1;
                return;
            }
            tile[i * width + i] = sqrt(diagonal);
            for (int r = i + 1; r < width; ++r) {
                double value = tile[r * width + i];
                for (int p = 0; p < i; ++p) {
                    value -= tile[r * width + p] * tile[i * width + p];
                }
                tile[r * width + i] = value / tile[i * width + i];
            }
        }
        for (int r = 0; r < width; ++r) {
            for (int c = r + 1; c < width; ++c) {
                tile[r * width + c] = 0.0;
            }
        }
        for (int p = 0; p < elements; ++p) {
            const int r = p / width;
            const int c = p % width;
            a[(k + static_cast<size_t>(r)) * n + k + static_cast<size_t>(c)] = tile[p];
        }
    }
}

// Solve X * L^T = A_panel. One block owns several rows and advances through
// the columns of each row in lockstep, avoiding a separate kernel per column.
__global__ void solvePanel(double* __restrict__ a, const size_t n, const size_t k,
                           const int panelWidth, const size_t rowCount) {
    __shared__ double rows[8 * TILE];
    const int col = static_cast<int>(threadIdx.x);
    const int row = static_cast<int>(threadIdx.y);
    const size_t globalRow = k + TILE + static_cast<size_t>(blockIdx.x) * 8 + row;
    const bool active = row < 8 && globalRow < k + TILE + rowCount && col < panelWidth;

    if (active) {
        rows[row * TILE + col] = a[globalRow * n + k + static_cast<size_t>(col)];
    }
    __syncthreads();

    for (int c = 0; c < panelWidth; ++c) {
        if (active && col == c) {
            double value = rows[row * TILE + c];
            for (int p = 0; p < c; ++p) {
                value -= rows[row * TILE + p] * a[(k + static_cast<size_t>(c)) * n +
                                                   k + static_cast<size_t>(p)];
            }
            rows[row * TILE + c] = value /
                a[(k + static_cast<size_t>(c)) * n + k + static_cast<size_t>(c)];
        }
        __syncthreads();
    }

    if (active) {
        a[globalRow * n + k + static_cast<size_t>(col)] = rows[row * TILE + col];
    }
}

// C = C - A_panel * B_panel^T for one lower-triangular tile of the trailing
// matrix. Shared-memory tiles are reused across all dot products in a block.
__global__ void updateTrailing(double* __restrict__ a, const size_t n, const size_t k,
                               const int tileRow, const int tileColumn,
                               const int leftWidth, const int rightWidth,
                               const int panelWidth, const bool diagonalTile) {
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const size_t row = k + TILE + static_cast<size_t>(tileRow) * TILE +
                       static_cast<size_t>(blockIdx.y) * 16 + ty;
    const size_t col = k + TILE + static_cast<size_t>(tileColumn) * TILE +
                       static_cast<size_t>(blockIdx.x) * 16 + tx;
    const bool active = row < k + TILE + static_cast<size_t>(tileRow) * TILE + leftWidth &&
                        col < k + TILE + static_cast<size_t>(tileColumn) * TILE + rightWidth;

    double sum = 0.0;
    if (active && (!diagonalTile || row >= col)) {
        for (int p = 0; p < panelWidth; ++p) {
            sum += a[row * n + k + static_cast<size_t>(p)] *
                   a[col * n + k + static_cast<size_t>(p)];
        }
    }
    if (active && (!diagonalTile || row >= col)) {
        a[row * n + col] -= sum;
    }
}

__global__ void zeroUpper(double* __restrict__ a, const size_t n) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = n * n;
    if (index < total) {
        const size_t row = index / n;
        const size_t column = index % n;
        if (column > row) {
            a[index] = 0.0;
        }
    }
}

} // namespace

// GPU tiled Cholesky decomposition. A is retained in row-major order and is
// overwritten with its lower-triangular factor, matching the original API.

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) {
        return true;
    }
    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    int failure = 0;
    const size_t bytes = n * n * sizeof(double);
    if (!cudaOk(cudaMalloc(&deviceA, bytes), "matrix allocation") ||
        !cudaOk(cudaMalloc(&deviceFailure, sizeof(int)), "status allocation")) {
        cudaFree(deviceA);
        cudaFree(deviceFailure);
        return false;
    }
    bool success = cudaOk(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
                          "matrix upload") &&
                   cudaOk(cudaMemset(deviceFailure, 0, sizeof(int)), "status reset");

    const size_t tileCount = (n + TILE - 1) / TILE;
    for (size_t tileIndex = 0; success && tileIndex < tileCount; ++tileIndex) {
        const size_t k = tileIndex * TILE;
        const int panelWidth = static_cast<int>(std::min<size_t>(TILE, n - k));
        factorTile<<<1, 256, TILE * TILE * sizeof(double)>>>(
            deviceA, n, k, panelWidth, deviceFailure);
        success = cudaOk(cudaGetLastError(), "diagonal tile factorization") &&
                  cudaOk(cudaDeviceSynchronize(), "diagonal tile synchronization") &&
                  cudaOk(cudaMemcpy(&failure, deviceFailure, sizeof(int), cudaMemcpyDeviceToHost),
                        "status download");
        if (!success || failure != 0) {
            if (failure != 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k);
            }
            break;
        }

        const size_t trailing = n - k - panelWidth;
        if (trailing == 0) {
            continue;
        }
        const dim3 solveBlock(TILE, 8);
        const dim3 solveGrid(static_cast<unsigned>((trailing + 7) / 8));
        solvePanel<<<solveGrid, solveBlock>>>(deviceA, n, k, panelWidth, trailing);
        success = cudaOk(cudaGetLastError(), "panel solve") &&
                  cudaOk(cudaDeviceSynchronize(), "panel solve synchronization");
        if (!success) {
            break;
        }

        const int trailingTiles = static_cast<int>((trailing + TILE - 1) / TILE);
        for (int ti = 0; success && ti < trailingTiles; ++ti) {
            const int leftWidth = std::min(TILE, static_cast<int>(trailing) - ti * TILE);
            for (int tj = 0; tj <= ti; ++tj) {
                const int rightWidth = std::min(TILE, static_cast<int>(trailing) - tj * TILE);
                const dim3 grid(1, 1);
                updateTrailing<<<grid, dim3(TILE, TILE)>>>(
                    deviceA, n, k, ti, tj,
                    leftWidth, rightWidth, panelWidth, ti == tj);
                success = cudaOk(cudaGetLastError(), "trailing update") &&
                          cudaOk(cudaDeviceSynchronize(), "trailing update synchronization");
                if (!success) {
                    break;
                }
            }
        }
    }

    if (success) {
        zeroUpper<<<static_cast<unsigned>((n * n + 255) / 256), 256>>>(deviceA, n);
        success = cudaOk(cudaGetLastError(), "upper triangle cleanup") &&
                  cudaOk(cudaDeviceSynchronize(), "upper triangle cleanup synchronization");
    }
    if (success) {
        success = cudaOk(cudaMemcpy(A.data(), deviceA, bytes, cudaMemcpyDeviceToHost),
                         "matrix download");
    }
    cudaFree(deviceA);
    cudaFree(deviceFailure);
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
