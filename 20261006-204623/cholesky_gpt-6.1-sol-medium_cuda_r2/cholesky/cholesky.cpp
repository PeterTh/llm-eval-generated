#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// All matrix arithmetic runs on CUDA; host-side random generation preserves
// the original rand_r sequence. No CPU factorization fallback is provided.
#include <cuda_runtime.h>
#include <limits>

static void cudaCheck(cudaError_t result) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

template <typename T> class DeviceBuffer {
public:
    T* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

constexpr int TILE = 32;

// Each thread accumulates a 4x4 register tile. Shared memory loads are
// coalesced across rows; padding avoids bank conflicts in transposed access.
// Update computes the lower trailing triangle; otherwise compute X * X^T.
template <bool Update>
__global__ void gramKernel(const double* X, double* A, size_t n,
                           size_t offset, size_t begin, size_t end, bool addDiagonal) {
    const size_t rowBase = offset + size_t(blockIdx.y) * TILE;
    const size_t colBase = offset + size_t(blockIdx.x) * TILE;
    if (Update && blockIdx.x > blockIdx.y) return;
    __shared__ double left[TILE][TILE + 1];
    __shared__ double right[TILE][TILE + 1];
    const int tid = threadIdx.y * 8 + threadIdx.x;
    double acc[4][4] = {};
    for (size_t base = begin; base < end; base += TILE) {
        for (int t = tid; t < TILE * TILE; t += 64) {
            const int r = t / TILE, k = t % TILE;
            left[r][k] = rowBase + r < n && base + k < end
                ? X[(rowBase + r) * n + base + k] : 0.0;
            right[r][k] = colBase + r < n && base + k < end
                ? X[(colBase + r) * n + base + k] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const double x = left[threadIdx.y + i * 8][k];
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    acc[i][j] += x * right[threadIdx.x + j * 8][k];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t row = rowBase + threadIdx.y + i * 8;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t col = colBase + threadIdx.x + j * 8;
            if (row < n && col < n && (!Update || row >= col)) {
                if (Update) A[row * n + col] -= acc[i][j];
                else A[row * n + col] = acc[i][j] + (addDiagonal && row == col ? double(n) : 0.0);
            }
        }
    }
}

// Factor a diagonal block entirely in shared memory. The column dependency
// is local to this block, while rows within each column run in parallel.
__global__ void diagonalKernel(double* A, size_t n, size_t offset, int width, int* failure) {
    __shared__ double tile[TILE][TILE + 1];
    __shared__ int failed;
    const int row = threadIdx.x;
    if (row == 0) failed = 0;
    for (int col = 0; col < width; ++col)
        tile[row][col] = row < width && row >= col ? A[(offset + row) * n + offset + col] : 0.0;
    __syncthreads();
    if (*failure >= 0) return;
    for (int col = 0; col < width; ++col) {
        if (row == col) {
            const double value = tile[col][col];
            if (value <= 0.0) {
                *failure = int(offset + col);
                failed = 1;
            } else tile[col][col] = sqrt(value);
        }
        __syncthreads();
        if (failed) return;
        if (row > col && row < width) tile[row][col] /= tile[col][col];
        __syncthreads();
        if (row > col && row < width)
            for (int j = col + 1; j <= row; ++j)
                tile[row][j] -= tile[row][col] * tile[j][col];
        __syncthreads();
    }
    if (row < width)
        for (int col = 0; col <= row; ++col)
            A[(offset + row) * n + offset + col] = tile[row][col];
}

// One warp solves each trailing row. Broadcasts implement forward
// substitution without global synchronization or uncoalesced column loads.
__global__ void panelKernel(double* A, size_t n, size_t offset, int width, const int* failure) {
    if (*failure >= 0) return;
    __shared__ double diagonal[TILE][TILE + 1];
    const int lane = threadIdx.x;
    for (int row = threadIdx.y; row < width; row += 4)
        diagonal[row][lane] = lane <= row && lane < width
            ? A[(offset + row) * n + offset + lane] : 0.0;
    __syncthreads();
    const size_t row = offset + width + size_t(blockIdx.x) * 4 + threadIdx.y;
    double value = row < n && lane < width ? A[row * n + offset + lane] : 0.0;
    for (int col = 0; col < width; ++col) {
        if (lane == col) value /= diagonal[col][col];
        const double solved = __shfl_sync(0xffffffff, value, col);
        if (lane > col && lane < width) value -= solved * diagonal[lane][col];
    }
    if (row < n && lane < width) A[row * n + offset + lane] = value;
}

__global__ void clearUpperKernel(double* A, size_t n) {
    const size_t col = size_t(blockIdx.x) * 32 + threadIdx.x;
    const size_t row = size_t(blockIdx.y) * 8 + threadIdx.y;
    if (row < n && col < n && col > row) A[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (!n) return true;
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<int> failure(1);
    int failed = -1;
    cudaCheck(cudaMemcpy(matrix.data, A.data(), A.size() * sizeof(double), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(failure.data, &failed, sizeof(int), cudaMemcpyHostToDevice));
    for (size_t offset = 0; offset < n; offset += TILE) {
        const int width = int(std::min(size_t(TILE), n - offset));
        diagonalKernel<<<1, TILE>>>(matrix.data, n, offset, width, failure.data);
        const size_t trailing = n - offset - width;
        if (trailing) {
            panelKernel<<<(trailing + 3) / 4, dim3(32, 4)>>>(matrix.data, n, offset, width, failure.data);
            const unsigned tiles = unsigned((trailing + TILE - 1) / TILE);
            gramKernel<true><<<dim3(tiles, tiles), dim3(8, 8)>>>(
                matrix.data, matrix.data, n, offset + width, offset, offset + width, false);
        }
    }
    clearUpperKernel<<<dim3((n + 31) / 32, (n + 7) / 8), dim3(32, 8)>>>(matrix.data, n);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(&failed, failure.data, sizeof(int), cudaMemcpyDeviceToHost));
    if (failed >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
        return false;
    }
    cudaCheck(cudaMemcpy(A.data(), matrix.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    if (!n) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B) value = (rand_r(&seed) / double(RAND_MAX)) - 0.5;
    DeviceBuffer<double> input(n * n), matrix(n * n);
    cudaCheck(cudaMemcpy(input.data, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice));
    const unsigned tiles = unsigned((n + TILE - 1) / TILE);
    gramKernel<false><<<dim3(tiles, tiles), dim3(8, 8)>>>(input.data, matrix.data, n, 0, 0, n, true);
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaMemcpy(A.data(), matrix.data, A.size() * sizeof(double), cudaMemcpyDeviceToHost));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    if (n) {
        DeviceBuffer<double> input(n * n), matrix(n * n);
        cudaCheck(cudaMemcpy(input.data, L.data(), L.size() * sizeof(double), cudaMemcpyHostToDevice));
        const unsigned tiles = unsigned((n + TILE - 1) / TILE);
        gramKernel<false><<<dim3(tiles, tiles), dim3(8, 8)>>>(input.data, matrix.data, n, 0, 0, n, false);
        cudaCheck(cudaGetLastError());
        cudaCheck(cudaMemcpy(reconstructed.data(), matrix.data, L.size() * sizeof(double), cudaMemcpyDeviceToHost));
    }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
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
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end ||
                parsed > size_t(std::numeric_limits<int>::max()) ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
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
    double gflops = ops / seconds / 1e9;
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
