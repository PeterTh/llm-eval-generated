#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err = (call);                                         \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

namespace {

// Tile configuration: each block computes a TILE x TILE output tile using
// 256 threads (16x16), each thread accumulating a 2x2 register tile.
// K is consumed in chunks of BK elements staged through shared memory.
constexpr int TILE = 32;
constexpr int BK = 32;
constexpr int THREADS = 256;

__global__ __launch_bounds__(THREADS)
void matrixMultiplyKernel(const double* __restrict__ A,
                          const double* __restrict__ B,
                          double* __restrict__ C,
                          const int N) {
    // Padded so that consecutive columns map to consecutive shared banks.
    __shared__ double As[BK][TILE + 1];
    __shared__ double Bs[BK][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    const long long blockRow = static_cast<long long>(blockIdx.y) * TILE;
    const long long blockCol = static_cast<long long>(blockIdx.x) * TILE;

    const int row0 = ty * 2;
    const int col0 = tx * 2;

    double acc00 = 0.0;
    double acc01 = 0.0;
    double acc10 = 0.0;
    double acc11 = 0.0;

    for (int k0 = 0; k0 < N; k0 += BK) {
        // Stage the A tile transposed: As[k][m] = A[blockRow + m][k0 + k].
#pragma unroll
        for (int idx = tid; idx < TILE * BK; idx += THREADS) {
            const int r = idx / BK;
            const int c = idx % BK;
            const long long gr = blockRow + r;
            const int gk = k0 + c;
            As[c][r] = (gr < N && gk < N) ? A[gr * N + gk] : 0.0;
        }

        // Stage the B tile: Bs[k][n] = B[k0 + k][blockCol + n].
#pragma unroll
        for (int idx = tid; idx < BK * TILE; idx += THREADS) {
            const int r = idx / TILE;
            const int c = idx % TILE;
            const int gk = k0 + r;
            const long long gc = blockCol + c;
            Bs[r][c] = (gk < N && gc < N) ? B[static_cast<long long>(gk) * N + gc] : 0.0;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            const double a0 = As[kk][row0];
            const double a1 = As[kk][row0 + 1];
            const double b0 = Bs[kk][col0];
            const double b1 = Bs[kk][col0 + 1];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    const long long gr = blockRow + row0;
    const long long gc = blockCol + col0;

    if (gr < N) {
        const long long rowOffset = gr * N;
        if (gc < N) {
            C[rowOffset + gc] = acc00;
        }
        if (gc + 1 < N) {
            C[rowOffset + gc + 1] = acc01;
        }
    }
    if (gr + 1 < N) {
        const long long rowOffset = (gr + 1) * N;
        if (gc < N) {
            C[rowOffset + gc] = acc10;
        }
        if (gc + 1 < N) {
            C[rowOffset + gc + 1] = acc11;
        }
    }
}

// Launch the tiled matrix multiplication and wait for completion.
void matrixMultiply(const double* A, const double* B, double* C, const size_t N) {
    if (N == 0) {
        return;
    }

    const int n = static_cast<int>(N);
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned int>((N + TILE - 1) / TILE),
                    static_cast<unsigned int>((N + TILE - 1) / TILE));

    matrixMultiplyKernel<<<grid, block>>>(A, B, C, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace

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

    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    const size_t bytes = N * N * sizeof(double);

    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));
    }
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Copy inputs to the device before timing so the measured region only
    // covers the matrix multiplication itself.
    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

        // Warm up the device (lazy module loading, clock ramp) so that the
        // timed region reflects steady-state matrix multiplication only.
        matrixMultiply(dA, dB, dC, N);
    }
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(dA, dB, dC, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    // Sub-millisecond resolution so fast GPU runs do not truncate to zero.
    std::chrono::duration<double, std::milli> duration = end - start;

    if (bytes > 0) {
        CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC));
    }
    
    printf("Computation time: %.3f ms\n", duration.count());
    
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
