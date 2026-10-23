#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ---------------------------------------------------------------------------
// CUDA tiled double-precision GEMM
//
// Each thread block computes a BM x BN tile of C.  The K dimension is streamed
// through shared memory in TILE_K sized chunks.  Every thread owns a TM x TN
// register tile of accumulators, which gives a high ratio of arithmetic to
// shared-memory traffic without consuming too many registers.  The shared
// memory layouts are padded to avoid bank conflicts.  A smaller tile is used
// for small matrices so that enough thread blocks exist to fill the GPU.
// ---------------------------------------------------------------------------

namespace {

constexpr int TILE_K = 16;

template <int BM, int BN, int TM, int TN>
__global__ __launch_bounds__((BM / TM) * (BN / TN))
void matmulKernel(const double* __restrict__ A,
                  const double* __restrict__ B,
                  double* __restrict__ C,
                  const int N) {
    constexpr int BLOCK_X = BN / TN;
    constexpr int BLOCK_Y = BM / TM;
    constexpr int THREADS = BLOCK_X * BLOCK_Y;
    constexpr int A_ELEMS = BM * TILE_K;
    constexpr int B_ELEMS = TILE_K * BN;
    constexpr int A_LOADS = A_ELEMS / THREADS;
    constexpr int B_LOADS = B_ELEMS / THREADS;

    static_assert(A_ELEMS % THREADS == 0, "A tile load imbalance");
    static_assert(B_ELEMS % THREADS == 0, "B tile load imbalance");

    __shared__ double As[TILE_K][BM + 1];
    __shared__ double Bs[TILE_K][BN + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * BLOCK_X + tx;

    const int rowBase = blockIdx.y * BM;
    const int colBase = blockIdx.x * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i) {
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            acc[i][j] = 0.0;
        }
    }

    for (int k0 = 0; k0 < N; k0 += TILE_K) {
        // Load and transpose the A tile (coalesced reads along K).
#pragma unroll
        for (int s = 0; s < A_LOADS; ++s) {
            const int idx = tid + s * THREADS;
            const int row = idx / TILE_K;
            const int kk = idx % TILE_K;
            const int gm = rowBase + row;
            const int gk = k0 + kk;
            double v = 0.0;
            if (gm < N && gk < N) {
                v = A[static_cast<size_t>(gm) * N + gk];
            }
            As[kk][row] = v;
        }

        // Load the B tile (coalesced reads along N).
#pragma unroll
        for (int s = 0; s < B_LOADS; ++s) {
            const int idx = tid + s * THREADS;
            const int kk = idx / BN;
            const int col = idx % BN;
            const int gk = k0 + kk;
            const int gn = colBase + col;
            double v = 0.0;
            if (gk < N && gn < N) {
                v = B[static_cast<size_t>(gk) * N + gn];
            }
            Bs[kk][col] = v;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][ty * TM + i];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][tx * TN + j];
            }
#pragma unroll
            for (int i = 0; i < TM; ++i) {
#pragma unroll
                for (int j = 0; j < TN; ++j) {
                    acc[i][j] = fma(a[i], b[j], acc[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // Store the register tile back to global memory.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowBase + ty * TM + i;
        if (gr < N) {
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                const int gc = colBase + tx * TN + j;
                if (gc < N) {
                    C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
                }
            }
        }
    }
}

// For matrices whose 64x64 tiling would not produce enough blocks to fill the
// GPU, a 32x32 tile (four times as many blocks) is faster.
constexpr size_t SMALL_TILE_MAX_N = 1024;

template <int BM, int BN, int TM, int TN>
void launchKernel(double* A, double* B, double* C, const size_t N, const bool singleBlock) {
    const dim3 block(BN / TN, BM / TM, 1);
    const dim3 grid = singleBlock
        ? dim3(1, 1, 1)
        : dim3(static_cast<unsigned>((N + BN - 1) / BN),
               static_cast<unsigned>((N + BM - 1) / BM), 1);
    matmulKernel<BM, BN, TM, TN><<<grid, block>>>(A, B, C, static_cast<int>(N));
}

void launchMatmul(double* A, double* B, double* C, const size_t N, const bool singleBlock) {
    if (N <= SMALL_TILE_MAX_N) {
        launchKernel<32, 32, 2, 2>(A, B, C, N, singleBlock);
    } else {
        launchKernel<64, 64, 4, 4>(A, B, C, N, singleBlock);
    }
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err__));     \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// Device buffers are allocated once and reused so that driver allocation costs
// are not part of the measured computation.
double* g_dA = nullptr;
double* g_dB = nullptr;
double* g_dC = nullptr;
size_t g_capacity = 0;

void ensureDeviceBuffers(const size_t bytes) {
    if (bytes <= g_capacity) {
        return;
    }
    if (g_dA != nullptr) {
        CUDA_CHECK(cudaFree(g_dA));
        CUDA_CHECK(cudaFree(g_dB));
        CUDA_CHECK(cudaFree(g_dC));
        g_dA = nullptr;
        g_dB = nullptr;
        g_dC = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&g_dA, bytes));
    CUDA_CHECK(cudaMalloc(&g_dB, bytes));
    CUDA_CHECK(cudaMalloc(&g_dC, bytes));
    g_capacity = bytes;
}

}  // namespace

// Initialize the CUDA context, allocate the device buffers, warm up the copy
// engine and the kernel before timing begins.  The first transfers in a fresh
// process pay a large one-time driver cost, so they are issued here.
void prepareDevice(const std::vector<double>& A, const std::vector<double>& B,
                   std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }
    const size_t bytes = N * N * sizeof(double);
    ensureDeviceBuffers(bytes);

    CUDA_CHECK(cudaMemcpy(g_dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(g_dB, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(C.data(), g_dC, bytes, cudaMemcpyDeviceToHost));

    launchMatmul(g_dA, g_dB, g_dC, N, true);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }
    const size_t bytes = N * N * sizeof(double);
    ensureDeviceBuffers(bytes);
    double* const dA = g_dA;
    double* const dB = g_dB;
    double* const dC = g_dC;

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    launchMatmul(dA, dB, dC, N, false);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
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
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    prepareDevice(A, B, C, N);
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    const double elapsedMs = std::max(duration.count(), 1e-6);
    
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (elapsedMs / 1000.0) / 1e9;
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
