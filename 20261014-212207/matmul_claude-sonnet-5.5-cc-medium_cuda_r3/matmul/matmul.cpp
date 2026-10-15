#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            exit(2);                                                                      \
        }                                                                                 \
    } while (0)

__global__ void initMatrixKernel(double* mat, const size_t N) {
    const size_t total = N * N;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        mat[idx] = getPseudoRndValue(N, idx / N, idx % N);
    }
}

// Tiled FP64 GEMM: 128x128 block tile, BK=8, 16x16 threads, 8x8 outputs per thread.
// Each output accumulates over k in ascending order, as in the serial code.
constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

__global__ void __launch_bounds__(256)
matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
             double* __restrict__ C, const size_t N) {
    __shared__ double As[BK][BM];
    __shared__ double Bs[BK][BN];

    const int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const size_t row0 = (size_t)blockIdx.y * BM;
    const size_t col0 = (size_t)blockIdx.x * BN;

    double acc[TM][TN] = {};

    for (size_t k0 = 0; k0 < N; k0 += BK) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int e = threadIdx.x + l * 256;
            // A tile: 128 rows x 8 cols
            const int ar = e / BK, ac = e % BK;
            const size_t gr = row0 + ar, gc = k0 + ac;
            As[ac][ar] = (gr < N && gc < N) ? A[gr * N + gc] : 0.0;
            // B tile: 8 rows x 128 cols
            const int br = e / BN, bc = e % BN;
            const size_t kr = k0 + br, cc = col0 + bc;
            Bs[br][bc] = (kr < N && cc < N) ? B[kr * N + cc] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            double a[TM], b[TN];
            #pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[k][ty + i * 16];
            #pragma unroll
            for (int j = 0; j < TN; ++j) b[j] = Bs[k][tx + j * 16];
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                #pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fma(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        const size_t r = row0 + ty + i * 16;
        if (r >= N) continue;
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            const size_t c = col0 + tx + j * 16;
            if (c < N) C[r * N + c] = acc[i][j];
        }
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
    
    // Allocate matrices (device)
    const size_t bytes = N * N * sizeof(double);
    std::vector<double> A, B;
    std::vector<double> C(N * N);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Initialize matrices on the GPU
    printf("Initializing matrices...\n");
    initMatrixKernel<<<1024, 256>>>(dA, N);
    initMatrixKernel<<<1024, 256>>>(dB, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    dim3 block(256);
    dim3 grid((unsigned)((N + BN - 1) / BN), (unsigned)((N + BM - 1) / BM));
    matmulKernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    if (validate) {
        A.resize(N * N);
        B.resize(N * N);
        CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(B.data(), dB, bytes, cudaMemcpyDeviceToHost));
    }
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

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
