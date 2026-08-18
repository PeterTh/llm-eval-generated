#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

namespace {

constexpr int kTileSize = 32;

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        cudaFailure(operation, error);
    }
}

__global__ void tiledMatrixMultiply(const double* __restrict__ A,
                                    const double* __restrict__ B,
                                    double* __restrict__ C,
                                    const int N) {
    __shared__ double tileA[kTileSize][kTileSize];
    __shared__ double tileB[kTileSize][kTileSize];

    const int row = static_cast<int>(blockIdx.y) * kTileSize + threadIdx.y;
    const int column = static_cast<int>(blockIdx.x) * kTileSize + threadIdx.x;
    double sum = 0.0;

    for (int tile = 0; tile < N; tile += kTileSize) {
        const int aColumn = tile + threadIdx.x;
        const int bRow = tile + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < N && aColumn < N) ? A[row * N + aColumn] : 0.0;
        tileB[threadIdx.y][threadIdx.x] =
            (bRow < N && column < N) ? B[bRow * N + column] : 0.0;
        __syncthreads();

        const int tileWidth = min(kTileSize, N - tile);
        for (int k = 0; k < tileWidth; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && column < N) {
        C[row * N + column] = sum;
    }
}

}  // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    // The benchmark is intentionally GPU-only: all matrix storage used by the
    // kernel is allocated on the selected CUDA device.
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Matrix size must be in the range [1, %d]\n",
                std::numeric_limits<int>::max());
        std::exit(EXIT_FAILURE);
    }

    const int dimension = static_cast<int>(N);
    const size_t bytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;

    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytes), "allocating A");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytes), "allocating B");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), bytes), "allocating C");

    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B");

    const dim3 block(kTileSize, kTileSize);
    const dim3 grid((dimension + kTileSize - 1) / kTileSize,
                    (dimension + kTileSize - 1) / kTileSize);
    tiledMatrixMultiply<<<grid, block>>>(deviceA, deviceB, deviceC, dimension);
    checkCuda(cudaGetLastError(), "launching matrix multiplication");
    checkCuda(cudaDeviceSynchronize(), "executing matrix multiplication");

    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying C");

    cudaFree(deviceA);
    cudaFree(deviceB);
    cudaFree(deviceC);
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
    // Pay one-time CUDA context startup before starting the benchmark clock.
    checkCuda(cudaFree(nullptr), "initializing CUDA");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const double elapsedMilliseconds = duration.count() / 1000.0;
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (elapsedMilliseconds / 1000.0) / 1e9;
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
