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
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,          \
                    __LINE__, cudaGetErrorString(err_));                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i,
                                                       const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialize matrix directly on the GPU
__global__ void initMatrixKernel(double* mat, const size_t N) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        mat[i * N + j] = getPseudoRndValue(N, i, j);
    }
}

void initMatrix(double* d_mat, const size_t N) {
    const dim3 block(32, 8);
    const dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    initMatrixKernel<<<grid, block>>>(d_mat, N);
    CUDA_CHECK(cudaGetLastError());
}

// Tiled matrix multiplication with shared memory and register blocking.
// Each thread block computes a BM x BN tile of C; each thread computes a
// TM x TN sub-tile held in registers.
constexpr int BM = 64;  // C tile rows per block
constexpr int BN = 64;  // C tile cols per block
constexpr int BK = 16;  // K-dimension tile depth
constexpr int TM = 4;   // C rows per thread
constexpr int TN = 4;   // C cols per thread
// Threads per block: (BM/TM) * (BN/TN) = 16 * 16 = 256

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    __shared__ double As[BM][BK];
    __shared__ double Bs[BK][BN + 1];  // +1 to avoid bank conflicts

    const int tx = threadIdx.x;  // 0..15
    const int ty = threadIdx.y;  // 0..15
    const int tid = ty * (BN / TN) + tx;

    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN] = {};
    double aReg[TM];
    double bReg[TN];

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // Cooperative load of A tile (BM x BK = 1024 elems, 256 threads -> 4 each)
        for (int idx = tid; idx < BM * BK; idx += 256) {
            const int r = idx / BK;
            const int c = idx % BK;
            const size_t gr = rowBase + r;
            const size_t gc = k0 + c;
            As[r][c] = (gr < N && gc < N) ? A[gr * N + gc] : 0.0;
        }
        // Cooperative load of B tile (BK x BN = 1024 elems)
        for (int idx = tid; idx < BK * BN; idx += 256) {
            const int r = idx / BN;
            const int c = idx % BN;
            const size_t gr = k0 + r;
            const size_t gc = colBase + c;
            Bs[r][c] = (gr < N && gc < N) ? B[gr * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
#pragma unroll
            for (int m = 0; m < TM; ++m) {
                aReg[m] = As[ty * TM + m][k];
            }
#pragma unroll
            for (int n = 0; n < TN; ++n) {
                bReg[n] = Bs[k][tx * TN + n];
            }
#pragma unroll
            for (int m = 0; m < TM; ++m) {
#pragma unroll
                for (int n = 0; n < TN; ++n) {
                    acc[m][n] += aReg[m] * bReg[n];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int m = 0; m < TM; ++m) {
        const size_t row = rowBase + ty * TM + m;
        if (row >= N) continue;
#pragma unroll
        for (int n = 0; n < TN; ++n) {
            const size_t col = colBase + tx * TN + n;
            if (col < N) {
                C[row * N + col] = acc[m][n];
            }
        }
    }
}

void matrixMultiply(const double* d_A, const double* d_B, double* d_C, const size_t N) {
    const dim3 block(BN / TN, BM / TM);  // 16 x 16 = 256 threads
    const dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    matrixMultiplyKernel<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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

    // Allocate matrices on the GPU
    const size_t bytes = N * N * sizeof(double);
    double* d_A = nullptr;
    double* d_B = nullptr;
    double* d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(d_A, N);
    initMatrix(d_B, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(d_A, d_B, d_C, N);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy result (and inputs, if needed) back to the host
    std::vector<double> C(N * N);
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }

    int exitCode = 0;

    // Validation
    if (validate) {
        printf("Validating result...\n");
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(B.data(), d_B, bytes, cudaMemcpyDeviceToHost));
        bool valid = validateResult(A, B, C, N);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return exitCode;
}
