#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Tiling: each block computes a BM x BN tile of C, stepping over the k dimension in BK slices.
// Each thread computes a TM x TN micro-tile held in registers.
constexpr unsigned BM = 64;
constexpr unsigned BN = 64;
constexpr unsigned BK = 16;
constexpr unsigned TM = 4;
constexpr unsigned TN = 4;
constexpr unsigned THREADS_X = BN / TN;  // 16
constexpr unsigned THREADS_Y = BM / TM;  // 16

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// C (M x N) = A (M x N) * B (N x N), row-major. A and C hold only this device's rows of the full matrices.
// Every C element is produced by exactly one thread that accumulates k = 0..N-1 in order, so the
// result is identical to the sequential reference (out-of-range tile entries are zero and add nothing).
__global__ void matrixMultiplyKernel(const double* __restrict__ A, const double* __restrict__ B,
                                     double* __restrict__ C, const size_t M, const size_t N) {
    __shared__ double As[BM][BK + 1];
    __shared__ double Bs[BK][BN];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tid = ty * THREADS_X + tx;
    constexpr unsigned nThreads = THREADS_X * THREADS_Y;

    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN] = {};

    for (size_t kBase = 0; kBase < N; kBase += BK) {
        for (unsigned idx = tid; idx < BM * BK; idx += nThreads) {
            const unsigned r = idx / BK;
            const unsigned c = idx % BK;
            const size_t gr = rowBase + r;
            const size_t gk = kBase + c;
            As[r][c] = (gr < M && gk < N) ? A[gr * N + gk] : 0.0;
        }
        for (unsigned idx = tid; idx < BK * BN; idx += nThreads) {
            const unsigned r = idx / BN;
            const unsigned c = idx % BN;
            const size_t gk = kBase + r;
            const size_t gc = colBase + c;
            Bs[r][c] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (unsigned k = 0; k < BK; ++k) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (unsigned i = 0; i < TM; ++i) a[i] = As[ty * TM + i][k];
#pragma unroll
            for (unsigned j = 0; j < TN; ++j) b[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (unsigned i = 0; i < TM; ++i) {
#pragma unroll
                for (unsigned j = 0; j < TN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (unsigned i = 0; i < TM; ++i) {
        const size_t gr = rowBase + ty * TM + i;
        if (gr >= M) continue;
#pragma unroll
        for (unsigned j = 0; j < TN; ++j) {
            const size_t gc = colBase + tx * TN + j;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

// Device-resident state for one GPU: a contiguous block of rows of A and C, plus a full copy of B.
struct GpuPartition {
    int device;
    size_t rowBegin;
    size_t rows;
    double* dA;
    double* dB;
    double* dC;
};

// Splits the rows of C evenly across all visible GPUs and uploads the inputs (not timed).
std::vector<GpuPartition> uploadMatrices(const std::vector<double>& A, const std::vector<double>& B,
                                         const size_t N) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(EXIT_FAILURE);
    }

    std::vector<GpuPartition> parts;
    const size_t base = N / deviceCount;
    const size_t extra = N % deviceCount;
    size_t row = 0;
    for (int d = 0; d < deviceCount; ++d) {
        const size_t rows = base + (static_cast<size_t>(d) < extra ? 1 : 0);
        if (rows == 0) continue;

        GpuPartition p{d, row, rows, nullptr, nullptr, nullptr};
        CUDA_CHECK(cudaSetDevice(d));
        CUDA_CHECK(cudaMalloc(&p.dA, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&p.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&p.dC, rows * N * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(p.dA, A.data() + row * N, rows * N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(p.dB, B.data(), N * N * sizeof(double), cudaMemcpyHostToDevice));
        parts.push_back(p);
        row += rows;
    }
    return parts;
}

// Launches the kernel on every GPU concurrently and waits for all of them to finish.
void matrixMultiply(const std::vector<GpuPartition>& parts, const size_t N) {
    const dim3 block(THREADS_X, THREADS_Y);
    for (const auto& p : parts) {
        CUDA_CHECK(cudaSetDevice(p.device));
        const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                        static_cast<unsigned>((p.rows + BM - 1) / BM));
        matrixMultiplyKernel<<<grid, block>>>(p.dA, p.dB, p.dC, p.rows, N);
        CUDA_CHECK(cudaGetLastError());
    }
    for (const auto& p : parts) {
        CUDA_CHECK(cudaSetDevice(p.device));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

// Copies the computed rows of C back to the host (not timed).
void downloadResult(const std::vector<GpuPartition>& parts, std::vector<double>& C, const size_t N) {
    for (const auto& p : parts) {
        CUDA_CHECK(cudaSetDevice(p.device));
        CUDA_CHECK(cudaMemcpy(C.data() + p.rowBegin * N, p.dC, p.rows * N * sizeof(double),
                              cudaMemcpyDeviceToHost));
    }
}

void releaseDevices(const std::vector<GpuPartition>& parts) {
    for (const auto& p : parts) {
        CUDA_CHECK(cudaSetDevice(p.device));
        CUDA_CHECK(cudaFree(p.dA));
        CUDA_CHECK(cudaFree(p.dB));
        CUDA_CHECK(cudaFree(p.dC));
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few random positions
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
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Upload inputs to all GPUs
    std::vector<GpuPartition> parts = uploadMatrices(A, B, N);

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(parts, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    downloadResult(parts, C, N);
    releaseDevices(parts);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);

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
