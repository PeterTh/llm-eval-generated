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

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

// Each block computes a 64x64 output tile. Its 256 threads each own 4x4
// results, reusing input tiles from shared memory across all 16 outputs.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    constexpr int TILE = 64;
    constexpr int STEP = 16;
    __shared__ double tileA[TILE][STEP + 1];
    __shared__ double tileB[STEP][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE + ty;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE + tx;
    double sum[4][4] = {};

    for (size_t base = 0; base < N; base += STEP) {
        const size_t aCol = base + tx;
        const size_t bRow = base + ty;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
            const size_t row = rowBase + r * STEP;
            tileA[ty + r * STEP][tx] = row < N && aCol < N
                ? A[row * N + aCol] : 0.0;
        }
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t col = colBase + c * STEP;
            tileB[ty][tx + c * STEP] = bRow < N && col < N
                ? B[bRow * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < STEP; ++k) {
            double a[4], b[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) a[r] = tileA[ty + r * STEP][k];
#pragma unroll
            for (int c = 0; c < 4; ++c) b[c] = tileB[k][tx + c * STEP];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
#pragma unroll
                for (int c = 0; c < 4; ++c) sum[r][c] += a[r] * b[c];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t row = rowBase + r * STEP;
        if (row < N) {
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                const size_t col = colBase + c * STEP;
                if (col < N) C[row * N + col] = sum[r][c];
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
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copy B");

    const dim3 threads(16, 16);
    const dim3 blocks((N + 63) / 64, (N + 63) / 64);
    matrixMultiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "matrixMultiplyKernel launch");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copy C");
    checkCuda(cudaFree(deviceA), "cudaFree(A)");
    checkCuda(cudaFree(deviceB), "cudaFree(B)");
    checkCuda(cudaFree(deviceC), "cudaFree(C)");
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check the original positions plus the center and far edges of tiles.
    const size_t checkPoints[] = {0, 1, 2, 3, 4, N / 2, N - 1};
    
    for (size_t pi = 0; pi < 7; ++pi) {
        for (size_t pj = 0; pj < 7; ++pj) {
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
    checkCuda(cudaFree(nullptr), "CUDA initialization");
    
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
