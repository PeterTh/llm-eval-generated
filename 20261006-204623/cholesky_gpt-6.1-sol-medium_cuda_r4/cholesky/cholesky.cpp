#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {
constexpr int Tile = 32;

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
}

// Own device allocations so errors also release all previously allocated memory.
template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Each thread accumulates a 2x2 register tile. Only lower-triangular blocks
// execute; Gram matrices are mirrored to preserve exact symmetry.
template <bool Update>
__global__ void productKernel(const double* input, double* output, size_t n,
                              size_t offset, size_t first, size_t count,
                              bool addDiagonal) {
    if (blockIdx.x > blockIdx.y) return;
    __shared__ double left[Tile][Tile + 1];
    __shared__ double right[Tile][Tile + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t rowBase = offset + size_t(blockIdx.y) * Tile;
    const size_t colBase = offset + size_t(blockIdx.x) * Tile;
    double sums[2][2] = {};
    for (size_t base = 0; base < count; base += Tile) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r = y + i * 16, k = x + j * 16;
                const size_t column = first + base + k;
                left[r][k] = rowBase + r < n && base + k < count
                    ? input[(rowBase + r) * n + column] : 0.0;
                right[r][k] = colBase + r < n && base + k < count
                    ? input[(colBase + r) * n + column] : 0.0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Tile; ++k) {
            const double a0 = left[y][k], a1 = left[y + 16][k];
            const double b0 = right[x][k], b1 = right[x + 16][k];
            sums[0][0] += a0 * b0;
            sums[0][1] += a0 * b1;
            sums[1][0] += a1 * b0;
            sums[1][1] += a1 * b1;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const size_t row = rowBase + y + i * 16;
            const size_t col = colBase + x + j * 16;
            if (row < n && col < n && row >= col) {
                if constexpr (Update) {
                    output[row * n + col] -= sums[i][j];
                } else {
                    const double value = sums[i][j] + (addDiagonal && row == col ? double(n) : 0.0);
                    output[row * n + col] = value;
                    if (row != col) output[col * n + row] = value;
                }
            }
        }
    }
}

// The dependency chain is confined to a small shared-memory diagonal tile.
// Columns below each pivot are computed concurrently.
__global__ void diagonalKernel(double* matrix, size_t n, size_t offset,
                               int width, unsigned long long* failure) {
    __shared__ double tile[Tile][Tile + 1];
    __shared__ int failed;
    const int row = threadIdx.x;
    if (row == 0) failed = *failure != 0;
    for (int col = 0; col < width; ++col)
        if (row < width) tile[row][col] = matrix[(offset + row) * n + offset + col];
    __syncthreads();
    if (failed) return;
    for (int col = 0; col < width; ++col) {
        if (row >= col && row < width) {
            double value = tile[row][col];
            for (int k = 0; k < col; ++k) value -= tile[row][k] * tile[col][k];
            if (row == col) {
                if (value <= 0.0) {
                    *failure = offset + col + 1;
                    failed = 1;
                } else {
                    tile[row][col] = sqrt(value);
                }
            } else {
                tile[row][col] = value;
            }
        }
        __syncthreads();
        if (failed) return;
        if (row > col && row < width) tile[row][col] /= tile[col][col];
        __syncthreads();
    }
    for (int col = 0; col <= row && col < width; ++col)
        if (row < width) matrix[(offset + row) * n + offset + col] = tile[row][col];
}

// Solve all rows of the panel independently against the diagonal factor.
__global__ void panelKernel(double* matrix, size_t n, size_t offset,
                            int width, const unsigned long long* failure) {
    if (*failure) return;
    __shared__ double diagonal[Tile][Tile + 1];
    for (int index = threadIdx.x; index < width * width; index += blockDim.x) {
        const int row = index / width, col = index % width;
        diagonal[row][col] = matrix[(offset + row) * n + offset + col];
    }
    __syncthreads();
    const int lane = threadIdx.x % 32;
    const size_t row = offset + width + size_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= n) return; // A whole warp exits together.
    double value = lane < width ? matrix[row * n + offset + lane] : 0.0;
    for (int col = 0; col < width; ++col) {
        if (lane == col) value /= diagonal[col][col];
        const double pivot = __shfl_sync(0xffffffffu, value, col);
        if (lane > col && lane < width) value -= pivot * diagonal[lane][col];
    }
    if (lane < width) matrix[row * n + offset + lane] = value;
}

__global__ void clearUpperKernel(double* matrix, size_t n) {
    const size_t count = n * n;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(gridDim.x) * blockDim.x)
        if (index % n > index / n) matrix[index] = 0.0;
}

void gramMatrix(const std::vector<double>& input, std::vector<double>& output,
                size_t n, bool addDiagonal) {
    if (!n) return;
    DeviceBuffer<double> source(n * n), result(n * n);
    checkCuda(cudaMemcpy(source.data, input.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned tiles = static_cast<unsigned>((n + Tile - 1) / Tile);
    productKernel<false><<<dim3(tiles, tiles), dim3(16, 16)>>>(
        source.data, result.data, n, 0, 0, n, addDiagonal);
    checkCuda(cudaGetLastError());
    checkCuda(cudaMemcpy(output.data(), result.data, n * n * sizeof(double), cudaMemcpyDeviceToHost));
}
} // namespace

// Right-looking blocked Cholesky: factor a diagonal tile, solve its panel,
// then update the trailing matrix with a parallel rank-Tile product.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    // Initialize CUDA even for the empty matrix; there is no CPU fallback.
    checkCuda(cudaFree(nullptr));
    if (!n) return true;
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<unsigned long long> failure(1);
    checkCuda(cudaMemcpy(matrix.data, A.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    checkCuda(cudaMemset(failure.data, 0, sizeof(unsigned long long)));
    for (size_t offset = 0; offset < n; offset += Tile) {
        const int width = static_cast<int>(std::min(size_t(Tile), n - offset));
        diagonalKernel<<<1, Tile>>>(matrix.data, n, offset, width, failure.data);
        const size_t remaining = n - offset - width;
        if (remaining) {
            panelKernel<<<static_cast<unsigned>((remaining + 3) / 4), 128>>>(
                matrix.data, n, offset, width, failure.data);
            const unsigned tiles = static_cast<unsigned>((remaining + Tile - 1) / Tile);
            productKernel<true><<<dim3(tiles, tiles), dim3(16, 16)>>>(
                matrix.data, matrix.data, n, offset + width, offset, width, false);
        }
    }
    const unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), (n * n + 255) / 256));
    clearUpperKernel<<<blocks, 256>>>(matrix.data, n);
    checkCuda(cudaGetLastError());
    unsigned long long failed = 0;
    checkCuda(cudaMemcpy(&failed, failure.data, sizeof(failed), cudaMemcpyDeviceToHost));
    if (failed) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n", size_t(failed - 1));
        return false;
    }
    checkCuda(cudaMemcpy(A.data(), matrix.data, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // Preserve the original rand_r sequence exactly; parallelize the cubic work.
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    gramMatrix(B, A, n, true);
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    gramMatrix(L, reconstructed, n, false);
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

int main(int argc, char** argv) try {
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
    
    if (n > std::numeric_limits<size_t>::max() / sizeof(double) / (n ? n : 1) ||
        (n + Tile - 1) / Tile > 65535) {
        fprintf(stderr, "Matrix size is too large\n");
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
    const double seconds = std::chrono::duration<double>(end - start).count();
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
 catch (const std::exception& error) {
    fprintf(stderr, "%s\n", error.what());
    return 1;
}
