#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// All matrices use row-major storage. CUDA errors are fatal so that a failed
// device operation cannot be mistaken for a valid decomposition.
#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(1); \
    } \
} while (0)

constexpr int TILE = 16;
constexpr int PANEL = 32;

// Calculate B * B^T in tiles. For validation, compare each entry with the
// original and reduce the two maxima on the GPU instead of copying a matrix.
__global__ void gramMatrix(const double* B, double* A, const double* original,
                           unsigned long long* maxima, size_t n, bool addDiagonal) {
    __shared__ double x[TILE][TILE], y[TILE][TILE];
    __shared__ double errors[TILE * TILE], relatives[TILE * TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t row = size_t(blockIdx.y) * TILE + ty;
    const size_t col = size_t(blockIdx.x) * TILE + tx;
    double sum = 0.0;
    for (size_t base = 0; base < n; base += TILE) {
        x[ty][tx] = row < n && base + tx < n ? B[row * n + base + tx] : 0.0;
        const size_t other = size_t(blockIdx.x) * TILE + ty;
        y[ty][tx] = other < n && base + tx < n ? B[other * n + base + tx] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += x[ty][k] * y[tx][k];
        __syncthreads();
    }
    if (row < n && col < n) {
        if (addDiagonal && row == col) sum += double(n);
        if (A) A[row * n + col] = sum;
    }
    if (original) {
        const int id = ty * TILE + tx;
        double error = 0.0, relative = 0.0;
        if (row < n && col < n) {
            error = fabs(sum - original[row * n + col]);
            relative = error / (fabs(original[row * n + col]) + 1e-10);
            // Match std::max's treatment of NaNs in the original reduction.
            if (isnan(error)) error = 0.0;
            if (isnan(relative)) relative = 0.0;
        }
        errors[id] = error;
        relatives[id] = relative;
        __syncthreads();
        for (int stride = TILE * TILE / 2; stride; stride /= 2) {
            if (id < stride) {
                errors[id] = fmax(errors[id], errors[id + stride]);
                relatives[id] = fmax(relatives[id], relatives[id + stride]);
            }
            __syncthreads();
        }
        if (id == 0) {
            atomicMax(maxima, __double_as_longlong(errors[0]));
            atomicMax(maxima + 1, __double_as_longlong(relatives[0]));
        }
    }
}

// Factor a diagonal panel in shared memory. Each column is formed in parallel;
// the columns themselves have the usual Cholesky dependency.
__global__ void factorDiagonal(double* A, size_t n, size_t base, int width, int* failure) {
    __shared__ double tile[PANEL][PANEL];
    __shared__ int failed;
    const int t = threadIdx.x;
    if (t == 0) failed = *failure >= 0;
    for (int i = t; i < PANEL * PANEL; i += PANEL) {
        const int r = i / PANEL, c = i % PANEL;
        tile[r][c] = r < width && c < width ? A[(base + r) * n + base + c] : 0.0;
    }
    __syncthreads();
    for (int j = 0; j < width; ++j) {
        if (t == j && !failed) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += tile[j][k] * tile[j][k];
            const double val = tile[j][j] - sum;
            if (val <= 0.0) {
                *failure = int(base + j);
                failed = 1;
            } else tile[j][j] = sqrt(val);
        }
        __syncthreads();
        if (t > j && t < width && !failed) {
            double sum = 0.0;
            for (int k = 0; k < j; ++k) sum += tile[t][k] * tile[j][k];
            tile[t][j] = (tile[t][j] - sum) / tile[j][j];
        }
        __syncthreads();
    }
    if (!failed && t < width)
        for (int j = 0; j <= t; ++j) A[(base + t) * n + base + j] = tile[t][j];
}

// Solve the block column under the factored panel, independently for each row.
__global__ void solvePanel(double* A, size_t n, size_t base, int width, const int* failure) {
    const size_t row = base + width + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n || *failure >= 0) return;
    for (int j = 0; j < width; ++j) {
        double sum = 0.0;
        for (int k = 0; k < j; ++k)
            sum += A[row * n + base + k] * A[(base + j) * n + base + k];
        A[row * n + base + j] = (A[row * n + base + j] - sum) / A[(base + j) * n + base + j];
    }
}

// Symmetric rank-k update of the remaining lower triangle.
__global__ void updateTrailing(double* A, size_t n, size_t base, int width, const int* failure) {
    if (*failure >= 0 || blockIdx.x > blockIdx.y) return;
    __shared__ double x[TILE][TILE], y[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t row = base + width + size_t(blockIdx.y) * TILE + ty;
    const size_t col = base + width + size_t(blockIdx.x) * TILE + tx;
    double sum = 0.0;
    for (int off = 0; off < width; off += TILE) {
        x[ty][tx] = row < n && off + tx < width ? A[row * n + base + off + tx] : 0.0;
        const size_t other = base + width + size_t(blockIdx.x) * TILE + ty;
        y[ty][tx] = other < n && off + tx < width ? A[other * n + base + off + tx] : 0.0;
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) sum += x[ty][k] * y[tx][k];
        __syncthreads();
    }
    if (row < n && col < n && col <= row) A[row * n + col] -= sum;
}

__global__ void clearUpper(double* A, size_t n) {
    size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t row = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) A[row * n + col] = 0.0;
}

bool choleskyDecomposition(double* A, size_t n) {
    if (!n) return true;
    int* failure;
    CUDA_CHECK(cudaMalloc(&failure, sizeof(int)));
    CUDA_CHECK(cudaMemset(failure, 0xff, sizeof(int)));
    const dim3 threads(TILE, TILE);
    for (size_t base = 0; base < n; base += PANEL) {
        int width = int(std::min(size_t(PANEL), n - base));
        factorDiagonal<<<1, PANEL>>>(A, n, base, width, failure);
        const size_t remaining = n - base - width;
        if (remaining) {
            solvePanel<<<(remaining + 127) / 128, 128>>>(A, n, base, width, failure);
            const unsigned tiles = (remaining + TILE - 1) / TILE;
            updateTrailing<<<dim3(tiles, tiles), threads>>>(A, n, base, width, failure);
        }
    }
    clearUpper<<<dim3((n + TILE - 1) / TILE, (n + TILE - 1) / TILE), threads>>>(A, n);
    CUDA_CHECK(cudaGetLastError());
    int failed;
    CUDA_CHECK(cudaMemcpy(&failed, failure, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(failure));
    if (failed >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failed);
        return false;
    }
    return true;
}

// Preserve the original random sequence while performing the cubic work on GPU.
void generatePositiveDefiniteMatrix(double* A, size_t n) {
    if (!n) return;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    double* deviceB;
    CUDA_CHECK(cudaMalloc(&deviceB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(deviceB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    dim3 threads(TILE, TILE), blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
    gramMatrix<<<blocks, threads>>>(deviceB, A, nullptr, nullptr, n, true);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(deviceB));
}

bool validateCholesky(const double* L, const double* original, size_t n) {
    double values[2] = {0.0, 0.0};
    if (n) {
        unsigned long long* maxima;
        CUDA_CHECK(cudaMalloc(&maxima, 2 * sizeof(unsigned long long)));
        CUDA_CHECK(cudaMemset(maxima, 0, 2 * sizeof(unsigned long long)));
        dim3 threads(TILE, TILE), blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        gramMatrix<<<blocks, threads>>>(L, nullptr, original, maxima, n, false);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(values, maxima, sizeof(values), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(maxima));
    }
    printf("Max absolute error: %.10e\n", values[0]);
    printf("Max relative error: %.10e\n", values[1]);
    if (values[1] > 1e-6) {
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
    std::vector<double> A;
    double* deviceA = nullptr;
    double* deviceOriginal = nullptr;
    if (n) CUDA_CHECK(cudaMalloc(&deviceA, n * n * sizeof(double)));
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(deviceA, n);
    
    if (validate) {
        if (n) {
            CUDA_CHECK(cudaMalloc(&deviceOriginal, n * n * sizeof(double)));
            CUDA_CHECK(cudaMemcpy(deviceOriginal, deviceA, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
        }
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(deviceA, n);
    
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
        A.resize(n * n);
        if (n) CUDA_CHECK(cudaMemcpy(A.data(), deviceA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(deviceA, deviceOriginal, n);
        
        if (deviceOriginal) CUDA_CHECK(cudaFree(deviceOriginal));
        if (deviceA) CUDA_CHECK(cudaFree(deviceA));
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    if (deviceA) CUDA_CHECK(cudaFree(deviceA));
    return 0;
}
