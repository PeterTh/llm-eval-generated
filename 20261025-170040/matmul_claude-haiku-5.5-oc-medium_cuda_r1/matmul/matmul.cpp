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

// Tiled GPU matrix multiplication.
//
// Each thread block computes a BM x BN tile of C using a 16x16 thread block;
// every thread owns a TM x TN register sub-tile. The K dimension is walked in
// tiles of BK, staged through shared memory. Within each thread the products
// are accumulated in strictly ascending k order, exactly as in the sequential
// reference, so the numerical result is the same as the original algorithm.
constexpr int BM = 64;   // block tile rows
constexpr int BN = 64;   // block tile cols
constexpr int BK = 16;   // K-tile depth
constexpr int TDIM = 16; // threads per block dimension
constexpr int TM = BM / TDIM; // 4 outputs per thread along rows
constexpr int TN = BN / TDIM; // 4 outputs per thread along cols

__global__ void __launch_bounds__(TDIM * TDIM)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t N) {
    // As is stored transposed (k-major) and padded to avoid shared-memory bank conflicts.
    __shared__ double As[BK][BM + 1];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * TDIM + tx;
    constexpr int NTHREADS = TDIM * TDIM;

    const size_t rowBase = static_cast<size_t>(blockIdx.y) * BM;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * BN;

    double acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j)
            acc[i][j] = 0.0;

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        // Load A tile (BM x BK) into As[kk][row]; zero-pad out-of-range entries.
#pragma unroll
        for (int r = 0; r < (BM * BK) / NTHREADS; ++r) {
            const int idx = tid + r * NTHREADS;
            const int row = idx / BK;
            const int kk = idx % BK;
            const size_t gRow = rowBase + row;
            const size_t gK = k0 + kk;
            As[kk][row] = (gRow < N && gK < N) ? A[gRow * N + gK] : 0.0;
        }
        // Load B tile (BK x BN) into Bs[kk][col]; zero-pad out-of-range entries.
#pragma unroll
        for (int r = 0; r < (BK * BN) / NTHREADS; ++r) {
            const int idx = tid + r * NTHREADS;
            const int kk = idx / BN;
            const int col = idx % BN;
            const size_t gK = k0 + kk;
            const size_t gCol = colBase + col;
            Bs[kk][col] = (gK < N && gCol < N) ? B[gK * N + gCol] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            double a[TM];
            double b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i)
                a[i] = As[kk][ty + TDIM * i];
#pragma unroll
            for (int j = 0; j < TN; ++j)
                b[j] = Bs[kk][tx + TDIM * j];
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
        const size_t gRow = rowBase + ty + TDIM * i;
        if (gRow >= N) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t gCol = colBase + tx + TDIM * j;
            if (gCol < N) {
                C[gRow * N + gCol] = acc[i][j];
            }
        }
    }
}

// Multiply A * B on the GPU. Inputs and output are host vectors; device buffers
// are allocated, transferred, computed and released within this call.
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

    const dim3 block(TDIM, TDIM);
    const dim3 grid(static_cast<unsigned int>((N + BN - 1) / BN),
                    static_cast<unsigned int>((N + BM - 1) / BM));
    matmulKernel<<<grid, block>>>(dA, dB, dC, N);
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
    
    // Initialize the CUDA runtime/context up front so that one-time driver
    // setup is not included in the timed computation.
    CUDA_CHECK(cudaFree(nullptr));
    
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
    CUDA_CHECK(cudaDeviceSynchronize());
    
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
