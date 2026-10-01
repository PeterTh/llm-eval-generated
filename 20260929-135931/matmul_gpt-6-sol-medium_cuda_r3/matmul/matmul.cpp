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

// Each block computes a 32 x 32 output tile. Its 256 threads each own four
// results, reusing input values from shared memory across the entire tile.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    __shared__ double aTile[32][32];
    __shared__ double bTile[32][32];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const size_t row0 = static_cast<size_t>(blockIdx.y) * 32 + ty;
    const size_t row1 = row0 + 16;
    const size_t col0 = static_cast<size_t>(blockIdx.x) * 32 + tx;
    const size_t col1 = col0 + 16;
    double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;

    for (size_t base = 0; base < N; base += 32) {
        const size_t k0 = base + tx;
        const size_t k1 = k0 + 16;
        const size_t bRow0 = base + ty;
        const size_t bRow1 = bRow0 + 16;
        aTile[ty][tx] = row0 < N && k0 < N ? A[row0 * N + k0] : 0.0;
        aTile[ty][tx + 16] = row0 < N && k1 < N ? A[row0 * N + k1] : 0.0;
        aTile[ty + 16][tx] = row1 < N && k0 < N ? A[row1 * N + k0] : 0.0;
        aTile[ty + 16][tx + 16] = row1 < N && k1 < N ? A[row1 * N + k1] : 0.0;
        bTile[ty][tx] = bRow0 < N && col0 < N ? B[bRow0 * N + col0] : 0.0;
        bTile[ty][tx + 16] = bRow0 < N && col1 < N ? B[bRow0 * N + col1] : 0.0;
        bTile[ty + 16][tx] = bRow1 < N && col0 < N ? B[bRow1 * N + col0] : 0.0;
        bTile[ty + 16][tx + 16] = bRow1 < N && col1 < N ? B[bRow1 * N + col1] : 0.0;
        __syncthreads();

#pragma unroll
        for (unsigned k = 0; k < 32; ++k) {
            const double a0 = aTile[ty][k];
            const double a1 = aTile[ty + 16][k];
            const double b0 = bTile[k][tx];
            const double b1 = bTile[k][tx + 16];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row0 < N && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < N && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < N && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < N && col1 < N) C[row1 * N + col1] = c11;
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dA), bytes), "cudaMalloc(A)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dB), bytes), "cudaMalloc(B)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&dC), bytes), "cudaMalloc(C)");
    checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(A)");
    checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy(B)");

    const dim3 threads(16, 16);
    const dim3 blocks(static_cast<unsigned>((N + 31) / 32),
                      static_cast<unsigned>((N + 31) / 32));
    matrixMultiplyKernel<<<blocks, threads>>>(dA, dB, dC, N);
    checkCuda(cudaGetLastError(), "matrixMultiplyKernel launch");
    checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(C)");
    checkCuda(cudaFree(dA), "cudaFree(A)");
    checkCuda(cudaFree(dB), "cudaFree(B)");
    checkCuda(cudaFree(dC), "cudaFree(C)");
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
    checkCuda(cudaSetDevice(0), "cudaSetDevice");
    
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
