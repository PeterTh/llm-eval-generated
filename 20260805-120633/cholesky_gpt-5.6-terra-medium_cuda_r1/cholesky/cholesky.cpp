#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// A right-looking blocked Cholesky decomposition.  The matrix stays on the GPU
// for the complete factorization; only the final lower factor is copied back.
namespace {
constexpr int kBlockSize = 32;

bool cudaOk(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    return false;
}

__global__ void factorDiagonalBlock(double* a, size_t n, size_t offset, int width, int* failed) {
    // A diagonal block is deliberately processed serially.  It is small, and
    // this avoids global synchronization inside its dependency chain.
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 0; i < width; ++i) {
        const size_t row = offset + static_cast<size_t>(i);
        for (int j = 0; j <= i; ++j) {
            double sum = 0.0;
            for (int p = 0; p < j; ++p) {
                const double x = a[row * n + offset + p];
                sum += x * a[(offset + j) * n + offset + p];
            }
            const size_t index = row * n + offset + j;
            if (i == j) {
                const double value = a[index] - sum;
                if (value <= 0.0) {
                    *failed = 1;
                    return;
                }
                a[index] = sqrt(value);
            } else {
                a[index] = (a[index] - sum) / a[(offset + j) * n + offset + j];
            }
        }
        for (int j = i + 1; j < width; ++j) a[row * n + offset + j] = 0.0;
    }
}

__global__ void solvePanel(double* a, size_t n, size_t offset, int diagonalWidth) {
    const size_t row = offset + diagonalWidth +
                       static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= n) return;
    for (int j = 0; j < diagonalWidth; ++j) {
        double sum = 0.0;
        for (int p = 0; p < j; ++p)
            sum += a[row * n + offset + p] * a[(offset + j) * n + offset + p];
        a[row * n + offset + j] =
            (a[row * n + offset + j] - sum) / a[(offset + j) * n + offset + j];
    }
}

__global__ void updateTrailing(double* a, size_t n, size_t offset, int diagonalWidth) {
    // One 16x16 thread block updates a 32x32 output tile.  Each thread owns a
    // 2x2 result, giving coalesced loads and enough arithmetic per thread.
    __shared__ double left[kBlockSize][kBlockSize + 1];
    __shared__ double right[kBlockSize][kBlockSize + 1];
    const size_t base = offset + diagonalWidth;
    const size_t tileRow = base + static_cast<size_t>(blockIdx.y) * kBlockSize;
    const size_t tileCol = base + static_cast<size_t>(blockIdx.x) * kBlockSize;
    if (tileRow < tileCol) return;

    const int r0 = static_cast<int>(threadIdx.y) * 2;
    const int c0 = static_cast<int>(threadIdx.x) * 2;
    const int rows[2] = {r0, r0 + 1};
    const int cols[2] = {c0, c0 + 1};

    for (int rr : rows) {
        for (int cc : cols) {
            const size_t globalRow = tileRow + rr;
            const size_t globalCol = offset + cc;
            left[rr][cc] = (globalRow < n && cc < diagonalWidth) ? a[globalRow * n + globalCol] : 0.0;
            const size_t rightRow = tileCol + rr;
            right[rr][cc] = (rightRow < n && cc < diagonalWidth) ? a[rightRow * n + globalCol] : 0.0;
        }
    }
    __syncthreads();

    for (int rr : rows) {
        const size_t globalRow = tileRow + rr;
        if (globalRow >= n) continue;
        for (int cc : cols) {
            const size_t globalCol = tileCol + cc;
            if (globalCol >= n || globalRow < globalCol) continue;
            double sum = 0.0;
            #pragma unroll
            for (int p = 0; p < kBlockSize; ++p) sum += left[rr][p] * right[cc][p];
            a[globalRow * n + globalCol] -= sum;
        }
    }
}

__global__ void zeroUpperTriangle(double* a, size_t n) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (row < n && col > row) a[row * n + col] = 0.0;
}

bool choleskyDecomposition(std::vector<double>& a, size_t n, float& elapsedMs) {
    if (n == 0) { elapsedMs = 0.0f; return true; }
    // Pad to complete tiles with an identity block.  This keeps every GPU
    // update regular while leaving the requested leading principal factor
    // mathematically unchanged.
    const size_t workN = ((n + kBlockSize - 1) / kBlockSize) * kBlockSize;
    std::vector<double> padded(workN * workN, 0.0);
    for (size_t row = 0; row < n; ++row)
        std::copy_n(a.data() + row * n, n, padded.data() + row * workN);
    for (size_t row = n; row < workN; ++row) padded[row * workN + row] = 1.0;
    double* deviceA = nullptr;
    int* deviceFailed = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    bool success = false;
    int failed = 0;

    if (!cudaOk(cudaMalloc(&deviceA, padded.size() * sizeof(double)), "matrix allocation") ||
        !cudaOk(cudaMalloc(&deviceFailed, sizeof(int)), "status allocation") ||
        !cudaOk(cudaMemcpy(deviceA, padded.data(), padded.size() * sizeof(double), cudaMemcpyHostToDevice), "matrix upload") ||
        !cudaOk(cudaMemset(deviceFailed, 0, sizeof(int)), "status initialization") ||
        !cudaOk(cudaEventCreate(&start), "start event creation") ||
        !cudaOk(cudaEventCreate(&stop), "stop event creation")) goto cleanup;

    cudaEventRecord(start);
    for (size_t offset = 0; offset < workN; offset += kBlockSize) {
        const int width = kBlockSize;
        factorDiagonalBlock<<<1, 1>>>(deviceA, workN, offset, width, deviceFailed);
        if (!cudaOk(cudaGetLastError(), "diagonal block factorization")) goto cleanup;

        const size_t remaining = workN - offset - width;
        if (remaining == 0) continue;
        solvePanel<<<static_cast<unsigned>((remaining + 255) / 256), 256>>>(deviceA, workN, offset, width);
        if (!cudaOk(cudaGetLastError(), "panel solve")) goto cleanup;

        const unsigned tiles = static_cast<unsigned>((remaining + kBlockSize - 1) / kBlockSize);
        updateTrailing<<<dim3(tiles, tiles), dim3(16, 16)>>>(deviceA, workN, offset, width);
        if (!cudaOk(cudaGetLastError(), "trailing update")) goto cleanup;
    }
    zeroUpperTriangle<<<dim3(static_cast<unsigned>((workN + 31) / 32), static_cast<unsigned>((workN + 31) / 32)), dim3(32, 32)>>>(deviceA, workN);
    if (!cudaOk(cudaGetLastError(), "upper triangle cleanup")) goto cleanup;
    cudaEventRecord(stop);
    if (!cudaOk(cudaEventSynchronize(stop), "factorization synchronization") ||
        !cudaOk(cudaEventElapsedTime(&elapsedMs, start, stop), "elapsed-time measurement")) goto cleanup;
    if (!cudaOk(cudaMemcpy(&failed, deviceFailed, sizeof(int), cudaMemcpyDeviceToHost), "status download")) goto cleanup;
    if (failed) {
        std::printf("Error: Matrix is not positive definite\n");
        goto cleanup;
    }
    if (!cudaOk(cudaMemcpy2D(a.data(), n * sizeof(double), deviceA, workN * sizeof(double),
                             n * sizeof(double), n, cudaMemcpyDeviceToHost), "factor download")) goto cleanup;
    success = true;

cleanup:
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (deviceFailed) cudaFree(deviceFailed);
    if (deviceA) cudaFree(deviceA);
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
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    std::printf("Cholesky Decomposition Benchmark\nMatrix size: %zu x %zu\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
    std::vector<double> A(n * n), A_orig;
    std::printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);
    if (validate) A_orig = A;
    std::printf("Computing Cholesky decomposition...\n");
    float elapsedMs = 0.0f;
    const bool success = choleskyDecomposition(A, n, elapsedMs);
    if (!success) { std::printf("Cholesky decomposition failed\n"); return 1; }
    const long durationMs = static_cast<long>(elapsedMs);
    std::printf("Computation time: %ld ms\n", durationMs);
    const double seconds = std::max(static_cast<double>(elapsedMs) / 1000.0, 1e-9);
    std::printf("Performance: %.3f GFLOPS\n", (static_cast<double>(n) * n * n / 3.0) / seconds / 1e9);
    if (printResults) print_results(A, "CholeskyL");
    if (validate) {
        std::printf("Validating result...\n");
        if (!validateCholesky(A, A_orig, n)) { std::printf("Validation: FAILED\n"); return 1; }
        std::printf("Validation: PASSED\n");
    }
    return 0;
}
