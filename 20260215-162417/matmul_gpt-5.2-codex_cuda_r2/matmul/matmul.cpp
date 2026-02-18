#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline void checkCuda(const cudaError_t status, const char* msg) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
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

constexpr int kTileSize = 16;

template <int TileSize>
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, const size_t N) {
    __shared__ double As[TileSize][TileSize];
    __shared__ double Bs[TileSize][TileSize];

    const size_t row = blockIdx.y * TileSize + threadIdx.y;
    const size_t col = blockIdx.x * TileSize + threadIdx.x;

    double sum = 0.0;
    const size_t tiles = (N + TileSize - 1) / TileSize;

    for (size_t t = 0; t < tiles; ++t) {
        const size_t tiledCol = t * TileSize + threadIdx.x;
        const size_t tiledRow = t * TileSize + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && tiledCol < N)
            ? A[row * N + tiledCol]
            : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (tiledRow < N && col < N)
            ? B[tiledRow * N + col]
            : 0.0;

        __syncthreads();
#pragma unroll
        for (int k = 0; k < TileSize; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return 0.0;
    }

    const size_t bytes = N * N * sizeof(double);
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;

    checkCuda(cudaMalloc(&dA, bytes), "allocating A");
    checkCuda(cudaMalloc(&dB, bytes), "allocating B");
    checkCuda(cudaMalloc(&dC, bytes), "allocating C");

    checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A");
    checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B");

    checkCuda(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared), "setting cache config");

    const size_t gridSize = (N + kTileSize - 1) / kTileSize;
    const dim3 block(kTileSize, kTileSize);
    const dim3 grid(static_cast<unsigned int>(gridSize), static_cast<unsigned int>(gridSize));

    cudaEvent_t start{};
    cudaEvent_t stop{};
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&stop), "creating stop event");

    checkCuda(cudaEventRecord(start), "recording start event");
    matmulKernel<kTileSize><<<grid, block>>>(dA, dB, dC, N);
    checkCuda(cudaGetLastError(), "launching kernel");
    checkCuda(cudaEventRecord(stop), "recording stop event");
    checkCuda(cudaEventSynchronize(stop), "synchronizing stop event");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "elapsed time");

    checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "copying C");

    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(stop), "destroying stop event");
    checkCuda(cudaFree(dA), "freeing A");
    checkCuda(cudaFree(dB), "freeing B");
    checkCuda(cudaFree(dC), "freeing C");

    return static_cast<double>(elapsedMs);
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
    double elapsedMs = matrixMultiply(A, B, C, N);
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (elapsedMs / 1000.0) / 1e9;
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
