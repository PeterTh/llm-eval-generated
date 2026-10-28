#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Tiling parameters for the GPU matmul kernel.
// Each block computes a BM x BN tile of C using 16x16 threads, each thread
// computing a TM x TN register tile. The K dimension is staged through shared
// memory in chunks of BK. Within each C element, products are accumulated in
// increasing k order, matching the sequential reference loop.
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 16;
constexpr int TM = 4;
constexpr int TN = 4;
constexpr int THREADS_X = BN / TN; // 16
constexpr int THREADS_Y = BM / TM; // 16

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double As[BM][BK];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS_X + tx;
    const int nThreads = THREADS_X * THREADS_Y;

    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN];
#pragma unroll
    for (int r = 0; r < TM; ++r)
#pragma unroll
        for (int c = 0; c < TN; ++c)
            acc[r][c] = 0.0;

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // Cooperative load of A tile (BM x BK)
        for (int idx = tid; idx < BM * BK; idx += nThreads) {
            const int r = idx / BK;
            const int c = idx % BK;
            const size_t gr = rowBase + r;
            const size_t gk = k0 + c;
            As[r][c] = (gr < N && gk < N) ? A[gr * N + gk] : 0.0;
        }
        // Cooperative load of B tile (BK x BN)
        for (int idx = tid; idx < BK * BN; idx += nThreads) {
            const int r = idx / BN;
            const int c = idx % BN;
            const size_t gk = k0 + r;
            const size_t gc = colBase + c;
            Bs[r][c] = (gk < N && gc < N) ? B[gk * N + gc] : 0.0;
        }
        __syncthreads();

        // Only iterate over valid k entries so no extra terms are added
        const int kmax = (N - k0 < static_cast<size_t>(BK)) ? static_cast<int>(N - k0) : BK;
        for (int kk = 0; kk < kmax; ++kk) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (int r = 0; r < TM; ++r) a[r] = As[ty * TM + r][kk];
#pragma unroll
            for (int c = 0; c < TN; ++c) b[c] = Bs[kk][tx * TN + c];
#pragma unroll
            for (int r = 0; r < TM; ++r)
#pragma unroll
                for (int c = 0; c < TN; ++c)
                    acc[r][c] += a[r] * b[c];
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < TM; ++r) {
        const size_t gr = rowBase + ty * TM + r;
        if (gr >= N) continue;
#pragma unroll
        for (int c = 0; c < TN; ++c) {
            const size_t gc = colBase + tx * TN + c;
            if (gc < N) C[gr * N + gc] = acc[r][c];
        }
    }
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

// Computes C = A * B on the GPU. Inputs are copied to device memory, the
// kernel runs over all output tiles in parallel, and the result is copied back.
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;

    const size_t bytes = N * N * sizeof(double);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(THREADS_X, THREADS_Y);
    const dim3 grid(static_cast<unsigned int>((N + BN - 1) / BN),
                    static_cast<unsigned int>((N + BM - 1) / BM));
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N);
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
    // Initialize the CUDA context up front so one-time driver setup is not timed
    CUDA_CHECK(cudaFree(nullptr));
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
