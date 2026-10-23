#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_X = 16;
constexpr int THREADS_Y = 16;
constexpr int THREADS = THREADS_X * THREADS_Y;

constexpr int A_LOADS = (BM * BK) / THREADS;
constexpr int B_LOADS = (BN * BK) / THREADS;

constexpr size_t EXACT_PRODUCT_MAX_N = 8192;

__device__ __forceinline__ double fmaRounded(const double a, const double b, const double c) {
    double result;
    asm("fma.rn.f64 %0, %1, %2, %3;" : "=d"(result) : "d"(a), "d"(b), "d"(c));
    return result;
}

template <bool UseFma>
__global__ __launch_bounds__(THREADS)
void matmulKernel(const double* __restrict__ A,
                  const double* __restrict__ B,
                  double* __restrict__ C,
                  const int N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const int tid = threadIdx.y * THREADS_X + threadIdx.x;
    const int blockRow = blockIdx.y * BM;
    const int blockCol = blockIdx.x * BN;

    double acc[TM][TN] = {};

    for (int k0 = 0; k0 < N; k0 += BK) {
#pragma unroll
        for (int r = 0; r < A_LOADS; ++r) {
            const int idx = tid + r * THREADS;
            const int kk = idx / BM;
            const int mm = idx % BM;
            const int gi = blockRow + mm;
            const int gk = k0 + kk;
            As[kk][mm] = (gi < N && gk < N) ? A[static_cast<size_t>(gi) * N + gk] : 0.0;
        }
#pragma unroll
        for (int r = 0; r < B_LOADS; ++r) {
            const int idx = tid + r * THREADS;
            const int kk = idx / BN;
            const int nn = idx % BN;
            const int gk = k0 + kk;
            const int gj = blockCol + nn;
            Bs[kk][nn] = (gk < N && gj < N) ? B[static_cast<size_t>(gk) * N + gj] : 0.0;
        }

        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) {
                a[i] = As[kk][threadIdx.y + i * THREADS_Y];
            }
#pragma unroll
            for (int j = 0; j < TN; ++j) {
                b[j] = Bs[kk][threadIdx.x + j * THREADS_X];
            }
            if constexpr (UseFma) {
#pragma unroll
                for (int i = 0; i < TM; ++i) {
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        acc[i][j] = fmaRounded(a[i], b[j], acc[i][j]);
                    }
                }
            } else {
                const bool tail = ((N & 1) != 0) && (k0 + kk == N - 1);
#pragma unroll
                for (int i = 0; i < TM; ++i) {
#pragma unroll
                    for (int j = 0; j < TN; ++j) {
                        if (tail) {
                            acc[i][j] = fmaRounded(a[i], b[j], acc[i][j]);
                        } else {
                            acc[i][j] += a[i] * b[j];
                        }
                    }
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gi = blockRow + threadIdx.y + i * THREADS_Y;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gj = blockCol + threadIdx.x + j * THREADS_X;
            if (gi < N && gj < N) {
                C[static_cast<size_t>(gi) * N + gj] = acc[i][j];
            }
        }
    }
}

inline void checkCuda(const cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", what, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

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
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    checkCuda(cudaMalloc(&dA, bytes), "allocation of A");
    checkCuda(cudaMalloc(&dB, bytes), "allocation of B");
    checkCuda(cudaMalloc(&dC, bytes), "allocation of C");

    checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "copy of A");
    checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "copy of B");

    const dim3 block(THREADS_X, THREADS_Y);
    const dim3 grid(static_cast<unsigned>((N + BN - 1) / BN),
                    static_cast<unsigned>((N + BM - 1) / BM));

    const bool productsExact = N <= EXACT_PRODUCT_MAX_N && (N & (N - 1)) == 0;
    if (productsExact) {
        matmulKernel<true><<<grid, block>>>(dA, dB, dC, static_cast<int>(N));
    } else {
        matmulKernel<false><<<grid, block>>>(dA, dB, dC, static_cast<int>(N));
    }
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "kernel execution");

    checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "copy of C");

    checkCuda(cudaFree(dA), "free of A");
    checkCuda(cudaFree(dB), "free of B");
    checkCuda(cudaFree(dC), "free of C");
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
    if (std::getenv("CUDA_VISIBLE_DEVICES") == nullptr) {
        setenv("CUDA_VISIBLE_DEVICES", "0", 0);
    }

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
    
    // Bring the CUDA context up before timing starts so that driver/context
    // initialization is not attributed to the matrix multiplication itself.
    checkCuda(cudaFree(nullptr), "CUDA initialization");

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
