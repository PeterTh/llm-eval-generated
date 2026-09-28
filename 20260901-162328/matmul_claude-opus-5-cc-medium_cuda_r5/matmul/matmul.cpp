#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                          \
    do {                                                                                          \
        const cudaError_t err_ = (call);                                                          \
        if (err_ != cudaSuccess) {                                                                \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__,    \
                   cudaGetErrorString(err_));                                                     \
            exit(1);                                                                              \
        }                                                                                         \
    } while (0)

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

// --- CUDA matrix multiplication -------------------------------------------------
//
// Tiled SGEMM-style kernel: each thread block computes a BM x BN tile of C, each
// thread a TM x TN sub-tile held in registers. A and B tiles are staged through
// shared memory (A transposed, so both operands are read with unit stride during
// the inner product). The k-loop walks k in increasing order, matching the
// accumulation order of the sequential reference implementation.

static constexpr int BM = 64; // rows of C per block
static constexpr int BN = 64; // cols of C per block
static constexpr int BK = 16; // k-slice depth per iteration
static constexpr int TM = 4;  // rows of C per thread
static constexpr int TN = 4;  // cols of C per thread
static constexpr int BLOCK_THREADS = (BM / TM) * (BN / TN); // 256

__global__ __launch_bounds__(BLOCK_THREADS) void matrixMultiplyKernel(
    const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
    const int N) {
    __shared__ double As[BK][BM]; // transposed A tile
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.y * (BN / TN) + threadIdx.x;
    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    // Load index mapping: 256 threads x 4 contiguous elements = one 64x16 tile.
    const int aLoadRow = tid / (BK / 4);          // 0..63
    const int aLoadCol = (tid % (BK / 4)) * 4;    // 0, 4, 8, 12
    const int bLoadRow = tid / (BN / 4);          // 0..15
    const int bLoadCol = (tid % (BN / 4)) * 4;    // 0..60

    double acc[TM][TN] = {};

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Stage A tile (transposed) and B tile in shared memory.
        const int aRow = rowBase + aLoadRow;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int aCol = k0 + aLoadCol + i;
            As[aLoadCol + i][aLoadRow] =
                (aRow < N && aCol < N) ? A[static_cast<size_t>(aRow) * N + aCol] : 0.0;
        }

        const int bRow = k0 + bLoadRow;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int bCol = colBase + bLoadCol + i;
            Bs[bLoadRow][bLoadCol + i] =
                (bRow < N && bCol < N) ? B[static_cast<size_t>(bRow) * N + bCol] : 0.0;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            double regA[TM];
            double regB[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = As[k][threadIdx.y * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = Bs[k][threadIdx.x * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = rowBase + threadIdx.y * TM + i;
        if (row >= N) {
            continue;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = colBase + threadIdx.x * TN + j;
            if (col < N) {
                C[static_cast<size_t>(row) * N + col] = acc[i][j];
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BN / TN, BM / TM);
    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN), static_cast<unsigned>((N + BM - 1) / BM));
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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

    // Initialize the CUDA context and force loading of the kernel module up front,
    // so that these one-time costs are not attributed to the measured computation.
    CUDA_CHECK(cudaFree(nullptr));
    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, matrixMultiplyKernel));
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(A, B, C, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

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
