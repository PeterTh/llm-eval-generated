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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each 16x16 block computes a (16*R)x(16*R) tile. The input tiles are
// reused from shared memory, while every thread accumulates R*R outputs.
template <int R>
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    constexpr int side = 16 * R;
    __shared__ double tileA[side][16];
    __shared__ double tileB[16][side];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t firstRow = static_cast<size_t>(blockIdx.y) * side + ty;
    const size_t firstCol = static_cast<size_t>(blockIdx.x) * side + tx;
    double sums[R][R] = {};

    for (size_t firstK = 0; firstK < N; firstK += 16) {
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const size_t row = firstRow + 16 * r;
            const size_t col = firstCol + 16 * r;
            tileA[ty + 16 * r][tx] =
                row < N && firstK + tx < N ? A[row * N + firstK + tx] : 0.0;
            tileB[ty][tx + 16 * r] =
                firstK + ty < N && col < N ? B[(firstK + ty) * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < 16; ++k) {
            double a[R], b[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                a[r] = tileA[ty + 16 * r][k];
                b[r] = tileB[k][tx + 16 * r];
            }
#pragma unroll
            for (int i = 0; i < R; ++i) {
#pragma unroll
                for (int j = 0; j < R; ++j) {
                    sums[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < R; ++i) {
        const size_t row = firstRow + 16 * i;
        if (row < N) {
#pragma unroll
            for (int j = 0; j < R; ++j) {
                const size_t col = firstCol + 16 * j;
                if (col < N) C[row * N + col] = sums[i][j];
            }
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;

    const size_t bytes = N * N * sizeof(double);
    double *deviceA, *deviceB, *deviceC;
    checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(A)");
    checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(B)");
    checkCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc(C)");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copy A to GPU");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copy B to GPU");

    const dim3 threads(16, 16);
    if (N <= 512) {
        const dim3 blocks((N + 31) / 32, (N + 31) / 32);
        matrixMultiplyKernel<2><<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    } else {
        const dim3 blocks((N + 63) / 64, (N + 63) / 64);
        matrixMultiplyKernel<4><<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    }
    checkCuda(cudaGetLastError(), "launch matrix multiplication kernel");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copy C from GPU");
    checkCuda(cudaFree(deviceA), "cudaFree(A)");
    checkCuda(cudaFree(deviceB), "cudaFree(B)");
    checkCuda(cudaFree(deviceC), "cudaFree(C)");
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

    // Initialize the CUDA context outside the measured multiplication.
    checkCuda(cudaFree(nullptr), "initialize CUDA");
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    const std::chrono::duration<double, std::milli> duration = end - start;
    
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
