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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A 16x16 block computes a 32x32 output tile. Each thread accumulates four
// results, reusing the two A and two B values it reads from shared memory.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, int N) {
    __shared__ double tileA[32][33];
    __shared__ double tileB[32][33];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * 32 + ty;
    const int col = blockIdx.x * 32 + tx;
    double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;

    for (int base = 0; base < N; base += 32) {
        // All threads participate in every barrier, including boundary tiles.
        for (int offset = 0; offset < 1024; offset += 256) {
            const int index = ty * 16 + tx + offset;
            const int localRow = index / 32;
            const int localCol = index % 32;
            const int aRow = blockIdx.y * 32 + localRow;
            const int aCol = base + localCol;
            const int bRow = base + localRow;
            const int bCol = blockIdx.x * 32 + localCol;
            tileA[localRow][localCol] = (aRow < N && aCol < N)
                ? A[static_cast<size_t>(aRow) * N + aCol] : 0.0;
            tileB[localRow][localCol] = (bRow < N && bCol < N)
                ? B[static_cast<size_t>(bRow) * N + bCol] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < 32; ++k) {
            const double a0 = tileA[ty][k];
            const double a1 = tileA[ty + 16][k];
            const double b0 = tileB[k][tx];
            const double b1 = tileB[k][tx + 16];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row < N && col < N) C[static_cast<size_t>(row) * N + col] = c00;
    if (row < N && col + 16 < N) C[static_cast<size_t>(row) * N + col + 16] = c01;
    if (row + 16 < N && col < N) C[static_cast<size_t>(row + 16) * N + col] = c10;
    if (row + 16 < N && col + 16 < N)
        C[static_cast<size_t>(row + 16) * N + col + 16] = c11;
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, int N,
                    double* deviceA, double* deviceB, double* deviceC) {
    const size_t bytes = C.size() * sizeof(double);
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copy A to GPU");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copy B to GPU");

    const dim3 threads(16, 16);
    const dim3 blocks((N + 31) / 32, (N + 31) / 32);
    matrixMultiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launch matrix multiplication");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copy result from GPU");
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
    
    if (N == 0 || N > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        fprintf(stderr, "Invalid matrix size\n");
        return 1;
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    const size_t bytes = N * N * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceA), bytes), "allocate GPU A");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceB), bytes), "allocate GPU B");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceC), bytes), "allocate GPU C");
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, static_cast<int>(N), deviceA, deviceB, deviceC);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    checkCuda(cudaFree(deviceA), "free GPU A");
    checkCuda(cudaFree(deviceB), "free GPU B");
    checkCuda(cudaFree(deviceC), "free GPU C");
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / seconds / 1e9;
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
