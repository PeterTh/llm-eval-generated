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

constexpr int PANEL_SIZE = 32;
constexpr int TILE_SIZE = 16;

bool cudaOk(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(error));
    return false;
}

// A panel is deliberately kept small.  This exposes the large triangular solve
// and trailing-matrix update to the whole GPU while retaining cache locality.
__global__ void factorPanel(double* a, int n, int first, int width, int* failure) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int j = 0; j < width; ++j) {
        const int col = first + j;
        double diagonal = a[col * n + col];
        for (int k = 0; k < j; ++k) {
            const double v = a[col * n + first + k];
            diagonal -= v * v;
        }
        if (diagonal <= 0.0) {
            *failure = col + 1;
            return;
        }
        const double pivot = sqrt(diagonal);
        a[col * n + col] = pivot;
        for (int row = j + 1; row < width; ++row) {
            const int i = first + row;
            double value = a[i * n + col];
            for (int k = 0; k < j; ++k)
                value -= a[i * n + first + k] * a[col * n + first + k];
            a[i * n + col] = value / pivot;
        }
    }
}

// One independent forward substitution per row below the diagonal panel.
__global__ void solvePanel(double* a, int n, int first, int width) {
    const int row = first + width + blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int j = 0; j < width; ++j) {
        const int col = first + j;
        double value = a[row * n + col];
        for (int k = 0; k < j; ++k)
            value -= a[row * n + first + k] * a[col * n + first + k];
        a[row * n + col] = value / a[col * n + col];
    }
}

// Parallel rank-k update of the lower trailing triangle.  Each thread owns an
// output element, avoiding atomics and making the update bandwidth-friendly.
__global__ void updateTrailing(double* a, int n, int first, int width) {
    const int col = first + width + blockIdx.x * blockDim.x + threadIdx.x;
    const int row = first + width + blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ double rowPanel[TILE_SIZE][PANEL_SIZE];
    __shared__ double colPanel[TILE_SIZE][PANEL_SIZE];
    // Cooperatively cache the two panel strips.  Each cached value is reused by
    // all 16 threads in its row or column of the output tile.
    if (row < n)
        for (int k = threadIdx.x; k < width; k += blockDim.x)
            rowPanel[threadIdx.y][k] = a[row * n + first + k];
    if (col < n)
        for (int k = threadIdx.y; k < width; k += blockDim.y)
            colPanel[threadIdx.x][k] = a[col * n + first + k];
    __syncthreads();
    if (row >= n || col >= n || row < col) return;
    double value = a[row * n + col];
#pragma unroll
    for (int k = 0; k < PANEL_SIZE; ++k) {
        if (k < width)
            value -= rowPanel[threadIdx.y][k] * colPanel[threadIdx.x][k];
    }
    a[row * n + col] = value;
}

__global__ void zeroUpperTriangle(double* a, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) a[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& a, size_t n) {
    if (n == 0) return true;
    if (n > static_cast<size_t>(INT_MAX)) {
        std::fprintf(stderr, "Matrix dimension is too large for CUDA kernels\n");
        return false;
    }
    const int dimension = static_cast<int>(n);
    const size_t bytes = n * n * sizeof(double);
    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    bool ok = cudaOk(cudaMalloc(&deviceA, bytes), "matrix allocation") &&
              cudaOk(cudaMalloc(&deviceFailure, sizeof(int)), "status allocation") &&
              cudaOk(cudaMemcpy(deviceA, a.data(), bytes, cudaMemcpyHostToDevice), "matrix upload") &&
              cudaOk(cudaMemset(deviceFailure, 0, sizeof(int)), "status initialization");

    for (int first = 0; ok && first < dimension; first += PANEL_SIZE) {
        const int width = std::min(PANEL_SIZE, dimension - first);
        factorPanel<<<1, 1>>>(deviceA, dimension, first, width, deviceFailure);
        ok = cudaOk(cudaGetLastError(), "diagonal panel factorization") &&
             cudaOk(cudaDeviceSynchronize(), "diagonal panel synchronization");
        if (first + width < dimension && ok) {
            const int rows = dimension - first - width;
            solvePanel<<<(rows + 255) / 256, 256>>>(deviceA, dimension, first, width);
            ok = cudaOk(cudaGetLastError(), "panel triangular solve");
        }
        if (first + width < dimension && ok) {
            const int trailing = dimension - first - width;
            const dim3 block(TILE_SIZE, TILE_SIZE);
            const dim3 grid((trailing + TILE_SIZE - 1) / TILE_SIZE,
                            (trailing + TILE_SIZE - 1) / TILE_SIZE);
            updateTrailing<<<grid, block>>>(deviceA, dimension, first, width);
            ok = cudaOk(cudaGetLastError(), "trailing rank-k update");
        }
    }

    int failure = 0;
    if (ok) ok = cudaOk(cudaDeviceSynchronize(), "factorization synchronization") &&
                    cudaOk(cudaMemcpy(&failure, deviceFailure, sizeof(int), cudaMemcpyDeviceToHost), "status download");
    if (ok && failure != 0) {
        std::printf("Error: Matrix is not positive definite at diagonal element %d\n", failure - 1);
        ok = false;
    }
    if (ok) {
        const dim3 block(TILE_SIZE, TILE_SIZE);
        const dim3 grid((dimension + TILE_SIZE - 1) / TILE_SIZE, (dimension + TILE_SIZE - 1) / TILE_SIZE);
        zeroUpperTriangle<<<grid, block>>>(deviceA, dimension);
        ok = cudaOk(cudaGetLastError(), "upper-triangle cleanup") &&
             cudaOk(cudaMemcpy(a.data(), deviceA, bytes, cudaMemcpyDeviceToHost), "result download");
    }
    if (deviceFailure) cudaFree(deviceFailure);
    if (deviceA) cudaFree(deviceA);
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
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j) {
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
    std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", maxError, relError);
    if (relError > 1e-6) { std::printf("Validation failed: relative error too large\n"); return false; }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Matrix size (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", progName);
}

int main(int argc, char** argv) {
    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
    std::vector<double> A(n * n), A_orig;
    std::printf("Generating positive definite matrix...\n"); generatePositiveDefiniteMatrix(A, n);
    if (validate) A_orig = A;
    // Create the CUDA context before the benchmark interval.  Context startup is
    // a one-time runtime cost, not part of the decomposition itself.
    if (!cudaOk(cudaFree(nullptr), "CUDA initialization")) return 1;
    std::printf("Computing Cholesky decomposition...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDecomposition(A, n);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (!success) { std::printf("Cholesky decomposition failed\n"); return 1; }
    std::printf("Computation time: %ld ms\n", duration.count());
    const double ops = (double)n * n * n / 3.0;
    const double seconds = duration.count() / 1000.0;
    std::printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
    if (printResults) print_results(A, "CholeskyL");
    if (validate) { std::printf("Validating result...\n"); const bool valid = validateCholesky(A, A_orig, n); std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED"); return valid ? 0 : 1; }
    return 0;
}
