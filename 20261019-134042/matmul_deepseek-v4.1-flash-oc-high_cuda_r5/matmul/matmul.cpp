#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void checkCuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

// Generate pseudo-random values for matrix initialization on the device.
// The row-wise formulation keeps the modulo arithmetic incremental, avoiding
// an expensive 64-bit division for every element.
__global__ void initMatrixKernel(double* __restrict__ mat, const int N) {
    const unsigned long long mod = static_cast<unsigned long long>(N) * static_cast<unsigned long long>(N);
    const int i = blockIdx.x;
    const unsigned long long step =
        (static_cast<unsigned long long>(i + 1) * 1299709ULL) % mod;
    const unsigned long long base =
        (static_cast<unsigned long long>(i + 1) * step) % mod;
    const unsigned long long stride = (step * blockDim.x) % mod;

    unsigned long long v = (base + (step * threadIdx.x) % mod) % mod;
    for (int j = threadIdx.x; j < N; j += blockDim.x) {
        mat[static_cast<size_t>(i) * N + j] = static_cast<double>(v) / static_cast<double>(mod);
        v += stride;
        if (v >= mod) v -= mod;
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ __launch_bounds__((BM / TM) * (BN / TN))
void matrixMultiplyKernel(const double* __restrict__ A,
                          const double* __restrict__ B,
                          double* __restrict__ C,
                          const int N) {
    constexpr int TX = BN / TN;
    constexpr int TY = BM / TM;
    constexpr int NT = TX * TY;
    constexpr int PAD = 1;

    __shared__ double As[BK][BM + PAD];
    __shared__ double Bs[BK][BN + PAD];

    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TX + tx;

    const int rowStart = by * BM;
    const int colStart = bx * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0;

    for (int k0 = 0; k0 < N; k0 += BK) {
        for (int idx = tid; idx < BM * BK; idx += NT) {
            const int r = idx / BK;
            const int kk = idx - r * BK;
            const int gr = rowStart + r;
            const int gk = k0 + kk;
            As[kk][r] = (gr < N && gk < N) ? A[static_cast<size_t>(gr) * N + gk] : 0.0;
        }
        for (int idx = tid; idx < BK * BN; idx += NT) {
            const int kk = idx / BN;
            const int c = idx - kk * BN;
            const int gk = k0 + kk;
            const int gc = colStart + c;
            Bs[kk][c] = (gk < N && gc < N) ? B[static_cast<size_t>(gk) * N + gc] : 0.0;
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
                    acc[i][j] += a[i] * b[j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gr = rowStart + ty * TM + i;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gc = colStart + tx * TN + j;
            if (gr < N && gc < N)
                C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
static void launchGemm(const double* dA, const double* dB, double* dC, const int N) {
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (N + BM - 1) / BM);
    matrixMultiplyKernel<BM, BN, BK, TM, TN><<<grid, block>>>(dA, dB, dC, N);
}

void matrixMultiply(const double* dA, const double* dB, double* dC, const size_t N) {
    if (N == 0) return;
    const int n = static_cast<int>(N);

    if (n <= 1792) {
        launchGemm<16, 16, 16, 2, 2>(dA, dB, dC, n);
    } else {
        launchGemm<32, 32, 32, 4, 4>(dA, dB, dC, n);
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
    
    const size_t matrixBytes = N * N * sizeof(double);
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    if (N > 0) {
        checkCuda(cudaMalloc(&dA, matrixBytes), "cudaMalloc A");
        checkCuda(cudaMalloc(&dB, matrixBytes), "cudaMalloc B");
        checkCuda(cudaMalloc(&dC, matrixBytes), "cudaMalloc C");
    }

    // Initialize matrices directly on the device
    printf("Initializing matrices...\n");
    if (N > 0) {
        const int initThreads = 256;
        initMatrixKernel<<<static_cast<unsigned int>(N), initThreads>>>(dA, static_cast<int>(N));
        initMatrixKernel<<<static_cast<unsigned int>(N), initThreads>>>(dB, static_cast<int>(N));
        checkCuda(cudaGetLastError(), "initMatrixKernel");
    }

    // Make sure initialization has completed before starting the timed region.
    checkCuda(cudaDeviceSynchronize(), "init synchronization");

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(dA, dB, dC, N);
    checkCuda(cudaGetLastError(), "matrixMultiplyKernel");
    checkCuda(cudaDeviceSynchronize(), "matrixMultiplication synchronization");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const double durationMs = duration.count() / 1000.0;
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (durationMs / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Bring the result back only when it is actually needed.
    std::vector<double> C;
    if (printResults || validate) {
        C.resize(N * N);
        if (N > 0) {
            checkCuda(cudaMemcpy(C.data(), dC, matrixBytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy C");
        }
    }
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    int exitCode = 0;
    if (validate) {
        printf("Validating result...\n");
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        if (N > 0) {
            checkCuda(cudaMemcpy(A.data(), dA, matrixBytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy A");
            checkCuda(cudaMemcpy(B.data(), dB, matrixBytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy B");
        }
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    if (dA) cudaFree(dA);
    if (dB) cudaFree(dB);
    if (dC) cudaFree(dC);
    
    return exitCode;
}
