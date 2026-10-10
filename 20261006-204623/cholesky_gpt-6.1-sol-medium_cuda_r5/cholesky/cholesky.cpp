#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// All cubic work runs on CUDA. Matrices retain the original row-major layout.
#include <cuda_runtime.h>
#include <limits>

namespace {
constexpr int tile = 32;

void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(&data, count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// One block factors a diagonal tile. Only the pivot dependencies are serial;
// each column's dot products and divisions execute in parallel.
__global__ void factorDiagonal(double* a, size_t n, size_t base, int width,
                               unsigned long long* failure) {
    if (*failure != ~0ULL) return;
    __shared__ double d[tile][tile + 1];
    const int row = threadIdx.x;
    for (int col = 0; col < width; ++col)
        d[row][col] = row < width ? a[(base + row) * n + base + col] : 0.0;
    __syncthreads();
    for (int col = 0; col < width; ++col) {
        double value = 0.0;
        if (row >= col && row < width) {
            double sum = 0.0;
            for (int k = 0; k < col; ++k) sum += d[row][k] * d[col][k];
            value = d[row][col] - sum;
        }
        if (row == col) {
            if (!(value > 0.0)) *failure = base + col;
            else d[col][col] = sqrt(value);
        }
        __syncthreads();
        if (*failure != ~0ULL) return; // uniform after the block barrier
        if (row > col && row < width) d[row][col] = value / d[col][col];
        __syncthreads();
    }
    if (row < width)
        for (int col = 0; col <= row; ++col)
            a[(base + row) * n + base + col] = d[row][col];
}

// Solve L_panel * L_diagonal^T = A_panel, with a row per thread.
// Cooperative tile loads/stores make global memory accesses contiguous.
__global__ void solvePanel(double* a, size_t n, size_t base, int width,
                           unsigned long long* failure) {
    if (*failure != ~0ULL) return;
    __shared__ double d[tile][tile + 1];
    __shared__ double p[tile][tile + 1];
    const int tid = threadIdx.x;
    const size_t firstRow = base + width + size_t(blockIdx.x) * tile;
    for (int i = tid; i < tile * tile; i += blockDim.x) {
        const int row = i / tile, col = i % tile;
        d[row][col] = row < width && col <= row
            ? a[(base + row) * n + base + col] : 0.0;
        p[row][col] = firstRow + row < n && col < width
            ? a[(firstRow + row) * n + base + col] : 0.0;
    }
    __syncthreads();
    if (tid < tile && firstRow + tid < n) {
        for (int col = 0; col < width; ++col) {
            double sum = 0.0;
            for (int k = 0; k < col; ++k) sum += p[tid][k] * d[col][k];
            p[tid][col] = (p[tid][col] - sum) / d[col][col];
        }
    }
    __syncthreads();
    for (int i = tid; i < tile * tile; i += blockDim.x) {
        const int row = i / tile, col = i % tile;
        if (firstRow + row < n && col < width)
            a[(firstRow + row) * n + base + col] = p[row][col];
    }
}

// Register-blocked symmetric product. Each thread computes four outputs and
// reuses shared-memory operands. Update only the lower triangle for Cholesky;
// generation/reconstruction mirror it to form a full symmetric matrix.
template <bool Update>
__global__ void symmetricProduct(const double* input, double* output, size_t n,
                                 size_t first, size_t kFirst, size_t kEnd,
                                 double diagonalAdd,
                                 const unsigned long long* failure) {
    if (blockIdx.y < blockIdx.x) return;
    if (Update && *failure != ~0ULL) return;
    __shared__ double left[tile][tile + 1];
    __shared__ double right[tile][tile + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const int tid = y * 16 + x;
    const size_t rowBase = first + size_t(blockIdx.y) * tile;
    const size_t colBase = first + size_t(blockIdx.x) * tile;
    double sum[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    for (size_t kb = kFirst; kb < kEnd; kb += tile) {
        for (int i = tid; i < tile * tile; i += 256) {
            const int row = i / tile, k = i % tile;
            left[row][k] = rowBase + row < n && kb + k < kEnd
                ? input[(rowBase + row) * n + kb + k] : 0.0;
            right[row][k] = colBase + row < n && kb + k < kEnd
                ? input[(colBase + row) * n + kb + k] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < tile; ++k) {
            const double a0 = left[y][k], a1 = left[y + 16][k];
            const double b0 = right[x][k], b1 = right[x + 16][k];
            sum[0][0] += a0 * b0;
            sum[0][1] += a0 * b1;
            sum[1][0] += a1 * b0;
            sum[1][1] += a1 * b1;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            const size_t row = rowBase + y + 16 * i;
            const size_t col = colBase + x + 16 * j;
            if (row < n && col < n && row >= col) {
                if (Update) output[row * n + col] -= sum[i][j];
                else {
                    const double value = sum[i][j] + (row == col ? diagonalAdd : 0.0);
                    output[row * n + col] = value;
                    if (row != col) output[col * n + row] = value;
                }
            }
        }
    }
}

__global__ void zeroUpper(double* a, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n * n; i += size_t(gridDim.x) * blockDim.x)
        if (i % n > i / n) a[i] = 0.0;
}

void productOnGPU(const std::vector<double>& input, std::vector<double>& output,
                  size_t n, double diagonalAdd) {
    if (!n) return;
    const size_t bytes = n * n * sizeof(double);
    DeviceBuffer<double> dInput(n * n), dOutput(n * n);
    cudaCheck(cudaMemcpy(dInput.data, input.data(), bytes, cudaMemcpyHostToDevice));
    const unsigned blocks = static_cast<unsigned>((n + tile - 1) / tile);
    symmetricProduct<false><<<dim3(blocks, blocks), dim3(16, 16)>>>(
        dInput.data, dOutput.data, n, 0, 0, n, diagonalAdd, nullptr);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(output.data(), dOutput.data, bytes, cudaMemcpyDeviceToHost));
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (!n) return true;
    DeviceBuffer<double> dA(n * n);
    DeviceBuffer<unsigned long long> failure(1);
    const size_t bytes = n * n * sizeof(double);
    cudaCheck(cudaMemcpy(dA.data, A.data(), bytes, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(failure.data, 0xff, sizeof(unsigned long long)));
    // A single stream orders dependent phases without host synchronization
    // between tiles. The trailing update spans all remaining GPU blocks.
    for (size_t base = 0; base < n; base += tile) {
        const int width = static_cast<int>(std::min(size_t(tile), n - base));
        factorDiagonal<<<1, tile>>>(dA.data, n, base, width, failure.data);
        const size_t first = base + width;
        if (first < n) {
            const unsigned blocks = static_cast<unsigned>((n - first + tile - 1) / tile);
            solvePanel<<<blocks, 128>>>(dA.data, n, base, width, failure.data);
            symmetricProduct<true><<<dim3(blocks, blocks), dim3(16, 16)>>>(
                dA.data, dA.data, n, first, base, first, 0.0, failure.data);
        }
    }
    const unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), (n * n + 255) / 256));
    zeroUpper<<<blocks, 256>>>(dA.data, n);
    cudaCheck(cudaGetLastError());
    unsigned long long failed;
    cudaCheck(cudaMemcpy(&failed, failure.data, sizeof(failed), cudaMemcpyDeviceToHost));
    if (failed != ~0ULL) {
        printf("Error: Matrix is not positive definite at diagonal element %llu\n", failed);
        return false;
    }
    cudaCheck(cudaMemcpy(A.data(), dA.data, bytes, cudaMemcpyDeviceToHost));
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    productOnGPU(B, A, n, static_cast<double>(n));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    productOnGPU(L, reconstructed, n, 0.0);
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        if (!std::isfinite(error)) {
            maxError = relError = std::numeric_limits<double>::infinity();
            break;
        }
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
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
