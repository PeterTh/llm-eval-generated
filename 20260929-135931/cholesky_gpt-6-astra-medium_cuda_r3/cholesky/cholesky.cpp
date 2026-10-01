#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// CUDA is required: errors are reported rather than falling back to the CPU.
#include <cuda_runtime.h>
#include <limits>

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int tile = 32;

// Compute lower tiles of X X^T, or update the trailing matrix with a panel.
// Each thread accumulates four outputs; adjacent threads load adjacent columns.
// The two operands are staged in padded shared memory for reuse across a tile.
template <bool update, bool addDiagonal>
__global__ void gramKernel(const double* X, double* A, size_t n,
                           size_t offset, size_t first, size_t count) {
    if (blockIdx.y < blockIdx.x) return;
    __shared__ double left[tile][tile + 1];
    __shared__ double right[tile][tile + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t row = offset + size_t(blockIdx.y) * tile;
    const size_t col = offset + size_t(blockIdx.x) * tile;
    double sums[2][2] = {};
    for (size_t base = 0; base < count; base += tile) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r = y + i * 16, k = x + j * 16;
                left[r][k] = row + r < n && base + k < count
                    ? X[(row + r) * n + first + base + k] : 0.0;
                right[r][k] = col + r < n && base + k < count
                    ? X[(col + r) * n + first + base + k] : 0.0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < tile; ++k) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                    sums[i][j] += left[y + i * 16][k] * right[x + j * 16][k];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const size_t r = row + y + i * 16, c = col + x + j * 16;
            if (r < n && c < n && r >= c) {
                if (update) A[r * n + c] -= sums[i][j];
                else {
                    const double value = sums[i][j] + (addDiagonal && r == c ? double(n) : 0.0);
                    A[r * n + c] = value;
                    if (r != c) A[c * n + r] = value;
                }
            }
        }
    }
}

// The small diagonal block has sequential pivots, with parallel rank-one updates.
__global__ void diagonalKernel(double* A, size_t n, size_t offset, int width, int* info) {
    __shared__ double block[tile][tile + 1];
    if (*info) return;
    const int t = threadIdx.x;
    for (int index = t; index < tile * tile; index += blockDim.x) {
        const int r = index / tile, c = index % tile;
        block[r][c] = r < width && c <= r ? A[(offset + r) * n + offset + c] : 0.0;
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (t == 0) {
            if (!(block[k][k] > 0.0)) *info = int(offset) + k + 1;
            else block[k][k] = sqrt(block[k][k]);
        }
        __syncthreads();
        if (*info) return;
        if (t > k && t < width) block[t][k] /= block[k][k];
        __syncthreads();
        for (int index = t; index < tile * tile; index += blockDim.x) {
            const int r = index / tile, c = index % tile;
            if (r >= c && c > k && r < width)
                block[r][c] -= block[r][k] * block[c][k];
        }
        __syncthreads();
    }
    for (int index = t; index < tile * tile; index += blockDim.x) {
        const int r = index / tile, c = index % tile;
        if (r < width && c <= r) A[(offset + r) * n + offset + c] = block[r][c];
    }
}

// Solve all panel rows independently against the factored diagonal block.
__global__ void panelKernel(double* A, size_t n, size_t offset, int width, const int* info) {
    if (*info) return;
    __shared__ double diagonal[tile][tile + 1];
    __shared__ double panel[tile][129];
    const int t = threadIdx.x;
    const size_t firstRow = offset + width + size_t(blockIdx.x) * 128;
    for (int index = t; index < tile * tile; index += 128) {
        const int r = index / tile, c = index % tile;
        diagonal[r][c] = r < width && c <= r ? A[(offset + r) * n + offset + c] : 0.0;
    }
    for (int index = t; index < 128 * tile; index += 128) {
        const int r = index / tile, c = index % tile;
        panel[c][r] = firstRow + r < n && c < width ? A[(firstRow + r) * n + offset + c] : 0.0;
    }
    __syncthreads();
    if (firstRow + t < n) {
        for (int j = 0; j < width; ++j) {
            double value = panel[j][t];
            for (int k = 0; k < j; ++k) value -= panel[k][t] * diagonal[j][k];
            panel[j][t] = value / diagonal[j][j];
        }
    }
    __syncthreads();
    for (int index = t; index < 128 * tile; index += 128) {
        const int r = index / tile, c = index % tile;
        if (firstRow + r < n && c < width) A[(firstRow + r) * n + offset + c] = panel[c][r];
    }
}

__global__ void zeroUpperKernel(double* A, size_t n) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < n * n; index += size_t(gridDim.x) * blockDim.x)
        if (index % n > index / n) A[index] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<int> info(1);
    cudaCheck(cudaMemcpy(matrix.data, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(info.data, 0, sizeof(int)));
    for (size_t offset = 0; offset < n; offset += tile) {
        const int width = int(std::min(size_t(tile), n - offset));
        diagonalKernel<<<1, 128>>>(matrix.data, n, offset, width, info.data);
        const size_t remaining = n - offset - width;
        if (remaining) {
            panelKernel<<<(remaining + 127) / 128, 128>>>(matrix.data, n, offset, width, info.data);
            const unsigned blocks = unsigned((remaining + tile - 1) / tile);
            gramKernel<true, false><<<dim3(blocks, blocks), dim3(16, 16)>>>(
                matrix.data, matrix.data, n, offset + width, offset, width);
        }
    }
    cudaCheck(cudaGetLastError());
    int hostInfo = 0;
    cudaCheck(cudaMemcpy(&hostInfo, info.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (hostInfo) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", hostInfo - 1);
        return false;
    }
    zeroUpperKernel<<<unsigned(std::min(size_t(65535), (n * n + 255) / 256)), 256>>>(matrix.data, n);
    cudaCheck(cudaGetLastError());
    // This synchronous copy includes completion of all GPU work in the timed interval.
    cudaCheck(cudaMemcpy(A.data(), matrix.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
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
    
    DeviceBuffer<double> input(n * n), output(n * n);
    cudaCheck(cudaMemcpy(input.data, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned blocks = unsigned((n + tile - 1) / tile);
    gramKernel<false, true><<<dim3(blocks, blocks), dim3(16, 16)>>>(input.data, output.data, n, 0, 0, n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(A.data(), output.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    DeviceBuffer<double> input(n * n), output(n * n);
    cudaCheck(cudaMemcpy(input.data, L.data(), L.size() * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned blocks = unsigned((n + tile - 1) / tile);
    gramKernel<false, false><<<dim3(blocks, blocks), dim3(16, 16)>>>(input.data, output.data, n, 0, 0, n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(reconstructed.data(), output.data, reconstructed.size() * sizeof(double), cudaMemcpyDeviceToHost));

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
    
    if (n == 0 || n > size_t(std::numeric_limits<int>::max()) ||
        n > std::numeric_limits<size_t>::max() / n / sizeof(double) ||
        (n + tile - 1) / tile > 65535) {
        fprintf(stderr, "Invalid matrix size\n");
        return 1;
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
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
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
