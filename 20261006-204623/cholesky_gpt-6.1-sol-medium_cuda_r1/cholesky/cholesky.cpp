#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// All cubic work runs on CUDA. The input and result remain row-major doubles.
#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <limits>

namespace {
constexpr int Tile = 32;

void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T>
class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

// Each thread computes a 2x2 register tile. Loads are coalesced, and shared
// memory reuses both operands across the entire 32x32 output tile.
// Update subtracts a panel product from the lower trailing matrix; otherwise
// this computes a symmetric Gram matrix (optionally adding n to its diagonal).
template <bool Update, bool AddDiagonal>
__global__ void gramKernel(const double* input, double* output, size_t n,
                           size_t start, size_t firstK, size_t countK) {
    if (blockIdx.y < blockIdx.x) return;
    __shared__ double left[Tile][Tile + 1];
    __shared__ double right[Tile][Tile + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t rowBase = start + size_t(blockIdx.y) * Tile;
    const size_t colBase = start + size_t(blockIdx.x) * Tile;
    double sums[2][2] = {};
    for (size_t base = 0; base < countK; base += Tile) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int r = y + i * 16, k = x + j * 16;
                const size_t col = firstK + base + k;
                left[r][k] = rowBase + r < n && base + k < countK
                    ? input[(rowBase + r) * n + col] : 0.0;
                right[r][k] = colBase + r < n && base + k < countK
                    ? input[(colBase + r) * n + col] : 0.0;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Tile; ++k) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    sums[i][j] = fma(left[y + i * 16][k],
                                     right[x + j * 16][k], sums[i][j]);
                }
            }
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
                    const double value = sums[i][j] + (AddDiagonal && row == col ? double(n) : 0.0);
                    output[row * n + col] = value;
                    if (row != col) output[col * n + row] = value;
                }
            }
        }
    }
}

// Factor a diagonal tile entirely in shared memory. Only the pivots are
// sequential; the remaining entries of each column are solved in parallel.
__global__ void diagonalKernel(double* A, size_t n, size_t offset, int width, int* failure) {
    if (*failure >= 0) return;
    __shared__ double tile[Tile][Tile + 1];
    __shared__ int failed;
    const int t = threadIdx.x;
    if (t == 0) failed = 0;
    for (int index = t; index < Tile * Tile; index += blockDim.x) {
        const int r = index / Tile, c = index % Tile;
        tile[r][c] = r < width && c <= r
            ? A[(offset + r) * n + offset + c] : 0.0;
    }
    __syncthreads();
    for (int col = 0; col < width; ++col) {
        if (t == col) {
            double value = tile[col][col];
            for (int k = 0; k < col; ++k)
                value = fma(-tile[col][k], tile[col][k], value);
            if (!(value > 0.0)) {
                *failure = int(offset) + col;
                failed = 1;
            } else {
                tile[col][col] = sqrt(value);
            }
        }
        __syncthreads();
        if (failed) return; // Uniform across this block.
        if (t > col && t < width) {
            double value = tile[t][col];
            for (int k = 0; k < col; ++k)
                value = fma(-tile[t][k], tile[col][k], value);
            tile[t][col] = value / tile[col][col];
        }
        __syncthreads();
    }
    for (int index = t; index < Tile * Tile; index += blockDim.x) {
        const int r = index / Tile, c = index % Tile;
        if (r < width && c <= r)
            A[(offset + r) * n + offset + c] = tile[r][c];
    }
}

// Independent rows solve A21 = L21 * L11^T. The shared panel provides
// coalesced global loads/stores even though each row has serial dependencies.
__global__ void panelKernel(double* A, size_t n, size_t offset, int width, const int* failure) {
    if (*failure >= 0) return;
    __shared__ double diagonal[Tile][Tile + 1];
    __shared__ double panel[Tile][Tile + 1];
    const int t = threadIdx.x;
    const size_t rowBase = offset + width + size_t(blockIdx.x) * Tile;
    for (int index = t; index < Tile * Tile; index += blockDim.x) {
        const int r = index / Tile, c = index % Tile;
        diagonal[r][c] = r < width && c <= r
            ? A[(offset + r) * n + offset + c] : 0.0;
        panel[r][c] = rowBase + r < n && c < width
            ? A[(rowBase + r) * n + offset + c] : 0.0;
    }
    __syncthreads();
    if (t < Tile && rowBase + t < n) {
        for (int col = 0; col < width; ++col) {
            double value = panel[t][col];
            for (int k = 0; k < col; ++k)
                value = fma(-panel[t][k], diagonal[col][k], value);
            panel[t][col] = value / diagonal[col][col];
        }
    }
    __syncthreads();
    for (int index = t; index < Tile * Tile; index += blockDim.x) {
        const int r = index / Tile, c = index % Tile;
        if (rowBase + r < n && c < width)
            A[(rowBase + r) * n + offset + c] = panel[r][c];
    }
}

__global__ void zeroUpperKernel(double* A, size_t n) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < n * n; index += size_t(blockDim.x) * gridDim.x) {
        if (index % n > index / n) A[index] = 0.0;
    }
}

__global__ void errorKernel(const double* reconstructed, const double* original,
                            size_t count, double* errors) {
    __shared__ double absolute[256], relative[256];
    double maxAbs = 0.0, maxRel = 0.0;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x) {
        const double error = fabs(reconstructed[index] - original[index]);
        const double rel = error / (fabs(original[index]) + 1e-10);
        maxAbs = fmax(maxAbs, isfinite(error) ? error : CUDART_INF);
        maxRel = fmax(maxRel, isfinite(rel) ? rel : CUDART_INF);
    }
    absolute[threadIdx.x] = maxAbs;
    relative[threadIdx.x] = maxRel;
    __syncthreads();
    for (int step = blockDim.x / 2; step; step /= 2) {
        if (threadIdx.x < step) {
            absolute[threadIdx.x] = fmax(absolute[threadIdx.x], absolute[threadIdx.x + step]);
            relative[threadIdx.x] = fmax(relative[threadIdx.x], relative[threadIdx.x + step]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        // Nonnegative IEEE doubles have the same order as their unsigned bits.
        atomicMax(reinterpret_cast<unsigned long long*>(errors),
                  static_cast<unsigned long long>(__double_as_longlong(absolute[0])));
        atomicMax(reinterpret_cast<unsigned long long*>(errors + 1),
                  static_cast<unsigned long long>(__double_as_longlong(relative[0])));
    }
}

unsigned linearBlocks(size_t count) {
    return unsigned(std::min(size_t(65535), (count + 255) / 256));
}
} // namespace

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<int> failure(1);
    cudaCheck(cudaMemcpy(matrix.data, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(failure.data, 0xff, sizeof(int)));
    // Same-stream kernel order enforces panel dependencies without any
    // per-pivot host/device synchronization.
    for (size_t offset = 0; offset < n; offset += Tile) {
        const int width = int(std::min(size_t(Tile), n - offset));
        diagonalKernel<<<1, 128>>>(matrix.data, n, offset, width, failure.data);
        const size_t start = offset + width;
        if (start < n) {
            const unsigned tiles = unsigned((n - start + Tile - 1) / Tile);
            panelKernel<<<tiles, 128>>>(matrix.data, n, offset, width, failure.data);
            gramKernel<true, false><<<dim3(tiles, tiles), dim3(16, 16)>>>(
                matrix.data, matrix.data, n, start, offset, width);
        }
    }
    zeroUpperKernel<<<linearBlocks(n * n), 256>>>(matrix.data, n);
    cudaCheck(cudaGetLastError());
    int failed;
    cudaCheck(cudaMemcpy(&failed, failure.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (failed >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
        return false;
    }
    cudaCheck(cudaMemcpy(A.data(), matrix.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    if (n == 0) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    // Keep the original rand_r sequence so input semantics stay deterministic.
    for (double& value : B) value = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
    DeviceBuffer<double> input(n * n), output(n * n);
    cudaCheck(cudaMemcpy(input.data, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned tiles = unsigned((n + Tile - 1) / Tile);
    gramKernel<false, true><<<dim3(tiles, tiles), dim3(16, 16)>>>(input.data, output.data, n, 0, 0, n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(A.data(), output.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    double errors[2] = {};
    if (n) {
        DeviceBuffer<double> input(n * n), original(n * n), reconstructed(n * n), maxima(2);
        cudaCheck(cudaMemcpy(input.data, L.data(), L.size() * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(original.data, A_orig.data(), A_orig.size() * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemset(maxima.data, 0, sizeof(errors)));
        const unsigned tiles = unsigned((n + Tile - 1) / Tile);
        gramKernel<false, false><<<dim3(tiles, tiles), dim3(16, 16)>>>(
            input.data, reconstructed.data, n, 0, 0, n);
        errorKernel<<<linearBlocks(n * n), 256>>>(reconstructed.data, original.data, n * n, maxima.data);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(errors, maxima.data, sizeof(errors), cudaMemcpyDeviceToHost));
    }
    printf("Max absolute error: %.10e\n", errors[0]);
    printf("Max relative error: %.10e\n", errors[1]);
    if (errors[1] > 1e-6) {
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
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            // Bounds cover matrix allocation arithmetic and CUDA grid limits.
            if (value[0] == '-' || end == value || *end != '\0' ||
                parsed > size_t(65535) * Tile ||
                parsed > std::numeric_limits<size_t>::max() / sizeof(double) / std::max(1ULL, parsed)) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = size_t(parsed);
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
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
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
