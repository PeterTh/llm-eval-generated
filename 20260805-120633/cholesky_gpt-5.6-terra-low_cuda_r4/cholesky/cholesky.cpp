#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kBlockSize = 64;

bool cudaOk(cudaError_t status, const char* what) {
    if (status == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(status));
    return false;
}

bool cublasOk(cublasStatus_t status, const char* what) {
    if (status == CUBLAS_STATUS_SUCCESS) return true;
    std::fprintf(stderr, "cuBLAS error in %s: status %d\n", what, static_cast<int>(status));
    return false;
}

// Each invocation factors one diagonal tile.  A single thread is intentional:
// the tile is tiny and this avoids global synchronization between its columns.
__global__ void factorDiagonalTile(double* a, int n, int offset, int width, int* failure) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int j = 0; j < width; ++j) {
        const int col = offset + j;
        double diagonal = a[static_cast<size_t>(col) * n + col];
        for (int k = 0; k < j; ++k) {
            const double value = a[static_cast<size_t>(col) * n + offset + k];
            diagonal -= value * value;
        }
        if (!(diagonal > 0.0) || !isfinite(diagonal)) {
            *failure = col + 1;
            return;
        }
        a[static_cast<size_t>(col) * n + col] = sqrt(diagonal);
        for (int row = j + 1; row < width; ++row) {
            const int globalRow = offset + row;
            double value = a[static_cast<size_t>(globalRow) * n + col];
            for (int k = 0; k < j; ++k) {
                value -= a[static_cast<size_t>(globalRow) * n + offset + k] *
                         a[static_cast<size_t>(col) * n + offset + k];
            }
            a[static_cast<size_t>(globalRow) * n + col] =
                value / a[static_cast<size_t>(col) * n + col];
        }
        for (int upper = j + 1; upper < width; ++upper)
            a[static_cast<size_t>(col) * n + offset + upper] = 0.0;
    }
}

// Solves every row of L21 independently: L21 * L11^T = A21.
__global__ void solvePanel(double* a, int n, int offset, int width, int rows) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const int globalRow = offset + width + row;
    for (int j = 0; j < width; ++j) {
        double value = a[static_cast<size_t>(globalRow) * n + offset + j];
        for (int k = 0; k < j; ++k)
            value -= a[static_cast<size_t>(globalRow) * n + offset + k] *
                     a[static_cast<size_t>(offset + j) * n + offset + k];
        a[static_cast<size_t>(globalRow) * n + offset + j] =
            value / a[static_cast<size_t>(offset + j) * n + offset + j];
    }
}

__global__ void zeroUpperTriangle(double* a, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col > row) a[static_cast<size_t>(row) * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& a, size_t n) {
    if (n == 0) return true;
    if (n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Matrix dimension is too large for CUDA BLAS\n");
        return false;
    }

    const int dimension = static_cast<int>(n);
    // Pad the device leading dimension so every panel begins on a tile boundary.
    // This also gives cuBLAS a regular leading dimension for a final short tile.
    const int leadingDimension = ((dimension + kBlockSize - 1) / kBlockSize) * kBlockSize;
    const size_t bytes = static_cast<size_t>(leadingDimension) * leadingDimension * sizeof(double);
    double* deviceA = nullptr;
    int* deviceFailure = nullptr;
    cublasHandle_t handle = nullptr;
    bool success = cudaOk(cudaMalloc(&deviceA, bytes), "matrix allocation") &&
                   cudaOk(cudaMalloc(&deviceFailure, sizeof(int)), "status allocation") &&
                   cublasOk(cublasCreate(&handle), "handle creation");
    if (!success) {
        if (handle) cublasDestroy(handle);
        cudaFree(deviceFailure);
        cudaFree(deviceA);
        return false;
    }

    success = cudaOk(cudaMemcpy2D(deviceA, static_cast<size_t>(leadingDimension) * sizeof(double),
                                  a.data(), n * sizeof(double), n * sizeof(double), n,
                                  cudaMemcpyHostToDevice), "matrix upload") &&
              cudaOk(cudaMemset(deviceFailure, 0, sizeof(int)), "status initialization");

    for (int offset = 0; success && offset < dimension; offset += kBlockSize) {
        const int width = std::min(kBlockSize, dimension - offset);
        factorDiagonalTile<<<1, 1>>>(deviceA, leadingDimension, offset, width, deviceFailure);
        success = cudaOk(cudaGetLastError(), "diagonal tile launch") &&
                  cudaOk(cudaDeviceSynchronize(), "diagonal tile factorization");

        int failure = 0;
        if (success) success = cudaOk(cudaMemcpy(&failure, deviceFailure, sizeof(failure),
                                                 cudaMemcpyDeviceToHost), "status download");
        if (success && failure != 0) {
            std::printf("Error: Matrix is not positive definite at diagonal element %d\n", failure - 1);
            success = false;
            break;
        }

        const int trailing = dimension - offset - width;
        if (trailing == 0 || !success) continue;
        constexpr int threads = 256;
        solvePanel<<<(trailing + threads - 1) / threads, threads>>>(deviceA, leadingDimension, offset, width, trailing);
        success = cudaOk(cudaGetLastError(), "panel solve launch");
        if (!success) break;

        // Row-major A21 (trailing x width) is column-major A21^T (width x trailing).
        // Therefore C = A21*A21^T is cuBLAS op(A21^T)^T * op(A21^T).
        const double alpha = -1.0;
        const double beta = 1.0;
        double* panel = deviceA + static_cast<size_t>(offset + width) * leadingDimension + offset;
        double* trailingMatrix = deviceA + static_cast<size_t>(offset + width) * leadingDimension + offset + width;
        success = cublasOk(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                                       trailing, trailing, width, &alpha,
                                       panel, leadingDimension, panel, leadingDimension, &beta,
                                       trailingMatrix, leadingDimension), "trailing update");
    }

    if (success) {
        dim3 threads(16, 16);
        dim3 blocks((dimension + threads.x - 1) / threads.x, (dimension + threads.y - 1) / threads.y);
        zeroUpperTriangle<<<blocks, threads>>>(deviceA, leadingDimension);
        success = cudaOk(cudaGetLastError(), "upper-triangle cleanup launch") &&
                  cudaOk(cudaDeviceSynchronize(), "GPU factorization");
    }
    if (success) success = cudaOk(cudaMemcpy2D(a.data(), n * sizeof(double),
                                                deviceA, static_cast<size_t>(leadingDimension) * sizeof(double),
                                                n * sizeof(double), n, cudaMemcpyDeviceToHost), "matrix download");

    cublasDestroy(handle);
    cudaFree(deviceFailure);
    cudaFree(deviceA);
    return success;
}

} // namespace

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) B[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
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
        const double error = std::fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        relError = std::max(relError, error / (std::fabs(A_orig[i]) + 1e-10));
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
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
    std::vector<double> A(n * n), A_orig;
    std::printf("Generating positive definite matrix...\n"); generatePositiveDefiniteMatrix(A, n);
    if (validate) A_orig = A;
    std::printf("Computing Cholesky decomposition...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    const bool success = choleskyDecomposition(A, n);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (!success) { std::printf("Cholesky decomposition failed\n"); return 1; }
    std::printf("Computation time: %ld ms\n", duration.count());
    const double seconds = duration.count() / 1000.0;
    const double gflops = seconds > 0.0 ? (static_cast<double>(n) * n * n / 3.0) / seconds / 1e9 : 0.0;
    std::printf("Performance: %.3f GFLOPS\n", gflops);
    if (printResults) print_results(A, "CholeskyL");
    if (!validate) return 0;
    std::printf("Validating result...\n");
    const bool valid = validateCholesky(A, A_orig, n);
    std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    return valid ? 0 : 1;
}
