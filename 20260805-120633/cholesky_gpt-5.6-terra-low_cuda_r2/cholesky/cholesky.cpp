#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {
constexpr int kBlockSize = 32;

bool cudaCheck(cudaError_t error, const char* expression, const char* file, int line) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, expression,
                 cudaGetErrorString(error));
    return false;
}
#define CUDA_CHECK(expr) cudaCheck((expr), #expr, __FILE__, __LINE__)

// A single block factors one diagonal panel.  The panel is deliberately kept
// small: dependencies remain local while the O(n^3) update is fully parallel.
__global__ void factorDiagonalPanel(double* a, size_t n, size_t begin, int width, int* failure) {
    __shared__ double panel[kBlockSize][kBlockSize];
    const int tid = threadIdx.x;
    if (tid < width) {
        for (int col = 0; col <= tid; ++col)
            panel[tid][col] = a[(begin + tid) * n + begin + col];
    }
    __syncthreads();

    for (int col = 0; col < width; ++col) {
        if (tid == 0) {
            double diagonal = panel[col][col];
            for (int k = 0; k < col; ++k) diagonal -= panel[col][k] * panel[col][k];
            if (diagonal <= 0.0 || !isfinite(diagonal)) {
                atomicExch(failure, 1);
                panel[col][col] = 1.0;
            } else {
                panel[col][col] = sqrt(diagonal);
            }
        }
        __syncthreads();
        if (tid > col && tid < width) {
            double value = panel[tid][col];
            for (int k = 0; k < col; ++k) value -= panel[tid][k] * panel[col][k];
            panel[tid][col] = value / panel[col][col];
        }
        __syncthreads();
    }
    if (tid < width) {
        for (int col = 0; col <= tid; ++col)
            a[(begin + tid) * n + begin + col] = panel[tid][col];
    }
}

// Each thread solves an entire row of the block column, preserving the short
// dependency chain within a row and exposing all remaining rows concurrently.
__global__ void solvePanelRows(double* a, size_t n, size_t begin, int width) {
    const size_t row = begin + width + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int col = 0; col < width; ++col) {
        double value = a[row * n + begin + col];
        for (int k = 0; k < col; ++k)
            value -= a[row * n + begin + k] * a[(begin + col) * n + begin + k];
        a[row * n + begin + col] = value / a[(begin + col) * n + begin + col];
    }
}

// Tiled rank-k update of the lower trailing matrix.  This is the performance
// critical BLAS-3 phase and has no inter-block dependencies.
__global__ void updateTrailing(double* a, size_t n, size_t begin, int width, size_t trailing) {
    __shared__ double left[kBlockSize][kBlockSize];
    __shared__ double right[kBlockSize][kBlockSize];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t col = begin + width + static_cast<size_t>(blockIdx.x) * kBlockSize + x;
    const size_t row = begin + width + static_cast<size_t>(blockIdx.y) * kBlockSize + y;

    if (row < n && x < width) left[y][x] = a[row * n + begin + x];
    if (col < n && y < width) right[x][y] = a[col * n + begin + y];
    __syncthreads();

    if (row < n && col < n && row >= col) {
        double sum = 0.0;
        #pragma unroll
        for (int k = 0; k < kBlockSize; ++k) {
            if (k < width) sum += left[y][k] * right[x][k];
        }
        a[row * n + col] -= sum;
    }
}

__global__ void zeroUpper(double* a, size_t n) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col > row) a[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& a, size_t n) {
    if (n == 0) return true;
    if (n > std::numeric_limits<size_t>::max() / n || n * n > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix is too large\n");
        return false;
    }
    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    const size_t bytes = n * n * sizeof(double);
    if (!CUDA_CHECK(cudaMalloc(&deviceA, bytes)) || !CUDA_CHECK(cudaMalloc(&deviceFailure, sizeof(int)))) {
        cudaFree(deviceA); cudaFree(deviceFailure); return false;
    }
    bool ok = CUDA_CHECK(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice)) &&
              CUDA_CHECK(cudaMemset(deviceFailure, 0, sizeof(int)));
    for (size_t begin = 0; ok && begin < n; begin += kBlockSize) {
        const int width = static_cast<int>(std::min(static_cast<size_t>(kBlockSize), n - begin));
        factorDiagonalPanel<<<1, kBlockSize>>>(deviceA, n, begin, width, deviceFailure);
        ok = CUDA_CHECK(cudaGetLastError());
        const size_t remaining = n - begin - width;
        if (ok && remaining) {
            solvePanelRows<<<static_cast<unsigned>((remaining + 255) / 256), 256>>>(deviceA, n, begin, width);
            ok = CUDA_CHECK(cudaGetLastError());
            if (ok) {
                dim3 threads(kBlockSize, kBlockSize);
                dim3 blocks(static_cast<unsigned>((remaining + kBlockSize - 1) / kBlockSize),
                            static_cast<unsigned>((remaining + kBlockSize - 1) / kBlockSize));
                updateTrailing<<<blocks, threads>>>(deviceA, n, begin, width, remaining);
                ok = CUDA_CHECK(cudaGetLastError());
            }
        }
    }
    if (ok) {
        dim3 threads(32, 8);
        dim3 blocks(static_cast<unsigned>((n + threads.x - 1) / threads.x),
                    static_cast<unsigned>((n + threads.y - 1) / threads.y));
        zeroUpper<<<blocks, threads>>>(deviceA, n);
        ok = CUDA_CHECK(cudaGetLastError()) && CUDA_CHECK(cudaDeviceSynchronize());
    }
    int failure = 0;
    if (ok) ok = CUDA_CHECK(cudaMemcpy(&failure, deviceFailure, sizeof(failure), cudaMemcpyDeviceToHost));
    if (ok && failure) {
        std::printf("Error: Matrix is not positive definite\n");
        ok = false;
    }
    if (ok) ok = CUDA_CHECK(cudaMemcpy(a.data(), deviceA, bytes, cudaMemcpyDeviceToHost));
    cudaFree(deviceFailure);
    cudaFree(deviceA);
    return ok;
}
} // namespace

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
            double sum = 0.0;
            for (size_t k = 0; k < n; ++k) sum += B[i * n + k] * B[j * n + k];
            A[i * n + j] = sum;
        }
    for (size_t i = 0; i < n; ++i) A[i * n + i] += n;
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    std::vector<double> reconstructed(n * n);
    for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < n; ++k) sum += L[i * n + k] * L[j * n + k];
        reconstructed[i * n + j] = sum;
    }
    double maxError = 0.0, relError = 0.0;
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (fabs(A_orig[i]) + 1e-10));
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) { printf("Validation failed: relative error too large\n"); return false; }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
    std::vector<double> A(n * n), A_orig;
    printf("Generating positive definite matrix...\n"); generatePositiveDefiniteMatrix(A, n);
    if (validate) A_orig = A;
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDecomposition(A, n);
    auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (!success) { printf("Cholesky decomposition failed\n"); return 1; }
    printf("Computation time: %ld ms\n", duration.count());
    const double ops = (double)n * n * n / 3.0;
    printf("Performance: %.3f GFLOPS\n", ops / (duration.count() / 1000.0) / 1e9);
    if (printResults) print_results(A, "CholeskyL");
    if (validate) { printf("Validating result...\n"); const bool valid = validateCholesky(A, A_orig, n); printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); return valid ? 0 : 1; }
    return 0;
}
