#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int kTile = 32;
constexpr int kBlock = 16;

// Generate pseudo-random values for matrix initialization.
constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                   const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) /
           static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

[[noreturn]] void cudaFailure(const char* expression, const cudaError_t error,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file,
                 line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_result = (expression);                       \
        if (cuda_check_result != cudaSuccess) {                                   \
            cudaFailure(#expression, cuda_check_result, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

// Each block produces a 32x32 output tile.  A 16x16 thread block computes four
// outputs per thread, increasing arithmetic intensity and exposing independent
// FP64 operations to the scheduler.  Padding removes shared-memory bank
// conflicts when a warp reads columns.
template <bool GuardEdges>
__global__ __launch_bounds__(kBlock * kBlock, 2)
void matrixMultiplyKernel(const double* __restrict__ A,
                          const double* __restrict__ B,
                          double* __restrict__ C, const int N) {
    __shared__ double aTile[kTile][kTile + 1];
    __shared__ double bTile[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread = ty * kBlock + tx;
    const int outputRow = static_cast<int>(blockIdx.y) * kTile;
    const int outputCol = static_cast<int>(blockIdx.x) * kTile;

    double c00 = 0.0;
    double c01 = 0.0;
    double c10 = 0.0;
    double c11 = 0.0;

    for (int tileStart = 0; tileStart < N; tileStart += kTile) {
#pragma unroll
        for (int load = 0; load < (kTile * kTile) / (kBlock * kBlock);
             ++load) {
            const int element = thread + load * kBlock * kBlock;
            const int row = element / kTile;
            const int col = element % kTile;

            if constexpr (GuardEdges) {
                const int aRow = outputRow + row;
                const int aCol = tileStart + col;
                const int bRow = tileStart + row;
                const int bCol = outputCol + col;
                aTile[row][col] =
                    (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0;
                bTile[row][col] =
                    (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0;
            } else {
                aTile[row][col] = A[(outputRow + row) * N + tileStart + col];
                bTile[row][col] = B[(tileStart + row) * N + outputCol + col];
            }
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTile; ++k) {
            const double a0 = aTile[ty][k];
            const double a1 = aTile[ty + kBlock][k];
            const double b0 = bTile[k][tx];
            const double b1 = bTile[k][tx + kBlock];
            c00 = fma(a0, b0, c00);
            c01 = fma(a0, b1, c01);
            c10 = fma(a1, b0, c10);
            c11 = fma(a1, b1, c11);
        }
        __syncthreads();
    }

    const int row0 = outputRow + ty;
    const int row1 = row0 + kBlock;
    const int col0 = outputCol + tx;
    const int col1 = col0 + kBlock;
    if constexpr (GuardEdges) {
        if (row0 < N && col0 < N) C[row0 * N + col0] = c00;
        if (row0 < N && col1 < N) C[row0 * N + col1] = c01;
        if (row1 < N && col0 < N) C[row1 * N + col0] = c10;
        if (row1 < N && col1 < N) C[row1 * N + col1] = c11;
    } else {
        C[row0 * N + col0] = c00;
        C[row0 * N + col1] = c01;
        C[row1 * N + col0] = c10;
        C[row1 * N + col1] = c11;
    }
}

// The returned time covers the multiplication kernel, matching the original
// benchmark's compute-only interval.  Allocation and PCIe transfers are kept
// outside that interval, but the final synchronization is always checked.
float matrixMultiply(const std::vector<double>& A,
                     const std::vector<double>& B, std::vector<double>& C,
                     const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    CUDA_CHECK(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const dim3 threads(kBlock, kBlock);
    const dim3 blocks((static_cast<unsigned int>(N) + kTile - 1) / kTile,
                      (static_cast<unsigned int>(N) + kTile - 1) / kTile);
    CUDA_CHECK(cudaEventRecord(start));
    if (N % kTile == 0) {
        matrixMultiplyKernel<false><<<blocks, threads>>>(deviceA, deviceB,
                                                         deviceC,
                                                         static_cast<int>(N));
    } else {
        matrixMultiplyKernel<true><<<blocks, threads>>>(deviceA, deviceB,
                                                        deviceC,
                                                        static_cast<int>(N));
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(deviceC));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceA));
    return milliseconds;
}

bool validateResult(const std::vector<double>& A,
                    const std::vector<double>& B,
                    const std::vector<double>& C, const size_t N) {
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError =
                std::abs((actual - expected) / (expected + 1e-10));
            if (relError > 1e-6) {
                std::printf("Validation failed at (%zu, %zu): expected %.10f, "
                            "got %.10f (error: %.10e)\n",
                            i, j, expected, actual, relError);
                return false;
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '\0' || *end != '\0' || parsed == 0 ||
                parsed > static_cast<unsigned long long>(
                             std::numeric_limits<int>::max())) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    std::printf("Matrix Multiplication Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", N, N);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    std::printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    std::printf("Computing matrix multiplication...\n");
    const float milliseconds = matrixMultiply(A, B, C, N);
    std::printf("Computation time: %.3f ms\n", milliseconds);

    const double gflops =
        (2.0 * static_cast<double>(N) * static_cast<double>(N) *
         static_cast<double>(N)) /
        (static_cast<double>(milliseconds) * 1.0e6);
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(C, "MatrixC");
    }

    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(A, B, C, N);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
