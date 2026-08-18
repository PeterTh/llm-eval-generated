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

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(error, operation);
}

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    constexpr unsigned TILE = 32;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + ty;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    double sum = 0.0;

    for (size_t tile = 0; tile < N; tile += TILE) {
        const size_t aCol = tile + tx;
        const size_t bRow = tile + ty;
        tileA[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0;
        tileB[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        __syncthreads();
        const unsigned limit = static_cast<unsigned>((tile + TILE < N) ? TILE : N - tile);
        for (unsigned k = 0; k < limit; ++k) sum += tileA[ty][k] * tileB[k][tx];
        __syncthreads();
    }
    if (row < N && col < N) C[row * N + col] = sum;
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    const size_t bytes = N * N * sizeof(double);
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dA), bytes), "allocating A");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dB), bytes), "allocating B");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dC), bytes), "allocating C");
    cudaCheck(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A");
    cudaCheck(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B");
    const dim3 block(32, 32);
    const dim3 grid((N + 31) / 32, (N + 31) / 32);
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N);
    cudaCheck(cudaGetLastError(), "launching matrix multiplication");
    cudaCheck(cudaDeviceSynchronize(), "executing matrix multiplication");
    cudaCheck(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "copying C");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
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
