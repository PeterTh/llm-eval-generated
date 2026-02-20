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

constexpr int kTileSize = 32;

inline void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t N) {
    __shared__ double As[kTileSize][kTileSize];
    __shared__ double Bs[kTileSize][kTileSize];

    const size_t row = static_cast<size_t>(blockIdx.y) * kTileSize + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * kTileSize + threadIdx.x;

    double sum = 0.0;
    for (size_t tile = 0; tile < N; tile += kTileSize) {
        const size_t tiledCol = tile + threadIdx.x;
        const size_t tiledRow = tile + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && tiledCol < N) ? A[row * N + tiledCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] =
            (col < N && tiledRow < N) ? B[tiledRow * N + col] : 0.0;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    const size_t elements = N * N;
    const size_t bytes = elements * sizeof(double);
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    checkCuda(cudaMalloc(&dA, bytes), "cudaMalloc A");
    checkCuda(cudaMalloc(&dB, bytes), "cudaMalloc B");
    checkCuda(cudaMalloc(&dC, bytes), "cudaMalloc C");
    checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy B");

    const dim3 block(kTileSize, kTileSize);
    const dim3 grid((N + kTileSize - 1) / kTileSize, (N + kTileSize - 1) / kTileSize);
    matmulKernel<<<grid, block>>>(dA, dB, dC, N);
    checkCuda(cudaGetLastError(), "matmulKernel launch");
    checkCuda(cudaDeviceSynchronize(), "matmulKernel sync");
    checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C");

    checkCuda(cudaFree(dA), "cudaFree A");
    checkCuda(cudaFree(dB), "cudaFree B");
    checkCuda(cudaFree(dC), "cudaFree C");
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
