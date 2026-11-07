// CUDA-parallel matrix multiplication benchmark.
// Each C element is accumulated sequentially over k with separately rounded
// multiply and add (sum = sum + round(a*b)); if N is odd the final k uses an
// fma. This reproduces bit-for-bit the rounding of the original serial loop as
// compiled by GCC -O3 -march=native (in-order vectorized reduction + fma tail).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                                       const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ---------------------------------------------------------------------------
// Tiled, register-blocked GEMM kernel (double precision).
// Block tile: BM x BN of C, K-slab of BK. 256 threads, each computes TM x TN.
// C[row0 + i][j] = sum_k A[row0 + i][k] * B[k][j], k ascending, no contraction.
// ---------------------------------------------------------------------------
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int NTHREADS = (BM / TM) * (BN / TN);  // 256

__global__ void __launch_bounds__(NTHREADS)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t N, const size_t rowOffset,
             const size_t numRows) {
    __shared__ double As[BK][BM + 1];  // transposed: As[k][m]
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.x;
    const int tx = tid % (BN / TN);  // 0..15
    const int ty = tid / (BN / TN);  // 0..15

    const size_t blockRow = (size_t)blockIdx.y * BM;  // relative to rowOffset
    const size_t blockCol = (size_t)blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0;

    const size_t Neven = N & ~(size_t)1;

    // Loads: BM*BK = 1024 elements of A and BK*BN = 1024 of B, 4 per thread each.
    for (size_t k0 = 0; k0 < Neven; k0 += BK) {
#pragma unroll
        for (int r = 0; r < (BM * BK) / NTHREADS; ++r) {
            const int idx = tid + r * NTHREADS;
            const int m = idx / BK;
            const int kk = idx % BK;
            const size_t gRow = blockRow + m;
            const size_t gK = k0 + kk;
            As[kk][m] = (gRow < numRows && gK < Neven) ? A[(rowOffset + gRow) * N + gK] : 0.0;
        }
#pragma unroll
        for (int r = 0; r < (BK * BN) / NTHREADS; ++r) {
            const int idx = tid + r * NTHREADS;
            const int kk = idx / BN;
            const int n = idx % BN;
            const size_t gK = k0 + kk;
            const size_t gCol = blockCol + n;
            Bs[kk][n] = (gK < Neven && gCol < N) ? B[gK * N + gCol] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM], b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[kk][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[kk][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j)
                    acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
        }
        __syncthreads();
    }

    // Odd-N tail: the last k is accumulated with a single fma (as in the original).
    if (Neven != N) {
        const size_t kl = N - 1;
        double a[TM], b[TN];
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const size_t gRow = blockRow + ty * TM + i;
            a[i] = (gRow < numRows) ? A[(rowOffset + gRow) * N + kl] : 0.0;
        }
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t gCol = blockCol + tx * TN + j;
            b[j] = (gCol < N) ? B[kl * N + gCol] : 0.0;
        }
#pragma unroll
        for (int i = 0; i < TM; ++i)
#pragma unroll
            for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t gRow = blockRow + ty * TM + i;
        if (gRow >= numRows) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t gCol = blockCol + tx * TN + j;
            if (gCol < N) C[(rowOffset + gRow) * N + gCol] = acc[i][j];
        }
    }
}

struct DeviceCtx {
    int dev = 0;
    size_t rowOffset = 0;
    size_t numRows = 0;
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaStream_t stream = nullptr;
};

// Split rows of C across all visible GPUs (only when there is enough work).
static std::vector<DeviceCtx> setupDevices(const size_t N) {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) {
        fprintf(stderr, "No CUDA devices found\n");
        exit(1);
    }
    constexpr size_t MIN_ROWS_PER_DEV = 1024;
    size_t useDev = std::max<size_t>(1, std::min<size_t>(ndev, N / MIN_ROWS_PER_DEV));

    std::vector<DeviceCtx> ctxs;
    size_t rowsPer = (N + useDev - 1) / useDev;
    for (size_t d = 0; d < useDev; ++d) {
        DeviceCtx c;
        c.dev = (int)d;
        c.rowOffset = d * rowsPer;
        if (c.rowOffset >= N) break;
        c.numRows = std::min(rowsPer, N - c.rowOffset);
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaStreamCreateWithFlags(&c.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaMalloc(&c.dA, c.numRows * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.dB, N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&c.dC, c.numRows * N * sizeof(double)));
        // Warm-up launch so module loading is not counted in the timed region.
        matmulKernel<<<1, NTHREADS, 0, c.stream>>>(c.dA, c.dB, c.dC, 0, 0, 0);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
        ctxs.push_back(c);
    }
    return ctxs;
}

static void teardownDevices(std::vector<DeviceCtx>& ctxs) {
    for (auto& c : ctxs) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaFree(c.dA));
        CUDA_CHECK(cudaFree(c.dB));
        CUDA_CHECK(cudaFree(c.dC));
        CUDA_CHECK(cudaStreamDestroy(c.stream));
    }
}

// Host-side vectors are pinned so that async copies overlap across devices.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N, std::vector<DeviceCtx>& ctxs) {
    if (N == 0) return;
    const dim3 block(NTHREADS);
    for (auto& c : ctxs) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaMemcpyAsync(c.dA, A.data() + c.rowOffset * N,
                                   c.numRows * N * sizeof(double), cudaMemcpyHostToDevice,
                                   c.stream));
        CUDA_CHECK(cudaMemcpyAsync(c.dB, B.data(), N * N * sizeof(double),
                                   cudaMemcpyHostToDevice, c.stream));
        const dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((c.numRows + BM - 1) / BM));
        matmulKernel<<<grid, block, 0, c.stream>>>(c.dA, c.dB, c.dC, N, 0, c.numRows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(C.data() + c.rowOffset * N, c.dC,
                                   c.numRows * N * sizeof(double), cudaMemcpyDeviceToHost,
                                   c.stream));
    }
    for (auto& c : ctxs) {
        CUDA_CHECK(cudaSetDevice(c.dev));
        CUDA_CHECK(cudaStreamSynchronize(c.stream));
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

    // Set up GPU(s): contexts, streams, device buffers; pin host memory.
    std::vector<DeviceCtx> ctxs = setupDevices(N);
    if (N > 0) {
        CUDA_CHECK(cudaHostRegister(A.data(), N * N * sizeof(double), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(B.data(), N * N * sizeof(double), cudaHostRegisterDefault));
        CUDA_CHECK(cudaHostRegister(C.data(), N * N * sizeof(double), cudaHostRegisterDefault));
    }
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N, ctxs);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (N > 0) {
        CUDA_CHECK(cudaHostUnregister(A.data()));
        CUDA_CHECK(cudaHostUnregister(B.data()));
        CUDA_CHECK(cudaHostUnregister(C.data()));
    }
    teardownDevices(ctxs);
    
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
