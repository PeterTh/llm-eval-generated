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

// A block covers a 32x32 output tile.  Each of its 256 threads computes a
// 2x2 sub-tile, reducing the number of global loads while keeping the shared
// memory footprint and register pressure bounded.  The two input tiles are
// loaded coalescently and reused for all 32 multiply-adds in the tile.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                      const double* __restrict__ B,
                                      double* __restrict__ C,
                                      const size_t N) {
    constexpr unsigned TILE = 32;
    __shared__ double tileA[TILE][TILE];
    __shared__ double tileB[TILE][TILE];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE + ty * 2;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx * 2;

    double c00 = 0.0;
    double c01 = 0.0;
    double c10 = 0.0;
    double c11 = 0.0;

    for (size_t tile = 0; tile < N; tile += TILE) {
        const size_t aRow0 = row;
        const size_t aRow1 = row + 1;
        const size_t bCol0 = col;
        const size_t bCol1 = col + 1;

        const size_t aIndex0 = aRow0 * N + tile + tx * 2;
        const size_t aIndex1 = aRow1 * N + tile + tx * 2;
        tileA[ty * 2][tx * 2] =
            (aRow0 < N && tile + tx * 2 < N) ? A[aIndex0] : 0.0;
        tileA[ty * 2][tx * 2 + 1] =
            (aRow0 < N && tile + tx * 2 + 1 < N) ? A[aIndex0 + 1] : 0.0;
        tileA[ty * 2 + 1][tx * 2] =
            (aRow1 < N && tile + tx * 2 < N) ? A[aIndex1] : 0.0;
        tileA[ty * 2 + 1][tx * 2 + 1] =
            (aRow1 < N && tile + tx * 2 + 1 < N) ? A[aIndex1 + 1] : 0.0;

        const size_t bIndex0 = (tile + ty * 2) * N + bCol0;
        const size_t bIndex1 = (tile + ty * 2 + 1) * N + bCol0;
        tileB[ty * 2][tx * 2] =
            (tile + ty * 2 < N && bCol0 < N) ? B[bIndex0] : 0.0;
        tileB[ty * 2][tx * 2 + 1] =
            (tile + ty * 2 < N && bCol1 < N) ? B[bIndex0 + 1] : 0.0;
        tileB[ty * 2 + 1][tx * 2] =
            (tile + ty * 2 + 1 < N && bCol0 < N) ? B[bIndex1] : 0.0;
        tileB[ty * 2 + 1][tx * 2 + 1] =
            (tile + ty * 2 + 1 < N && bCol1 < N) ? B[bIndex1 + 1] : 0.0;

        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < TILE; ++k) {
            const double a0 = tileA[ty * 2][k];
            const double a1 = tileA[ty * 2 + 1][k];
            const double b0 = tileB[k][tx * 2];
            const double b1 = tileB[k][tx * 2 + 1];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = c00;
    if (row < N && col + 1 < N) C[row * N + col + 1] = c01;
    if (row + 1 < N && col < N) C[(row + 1) * N + col] = c10;
    if (row + 1 < N && col + 1 < N) C[(row + 1) * N + col + 1] = c11;
}

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "cudaMalloc(A)");
    checkCuda(cudaMalloc(&deviceB, bytes), "cudaMalloc(B)");
    checkCuda(cudaMalloc(&deviceC, bytes), "cudaMalloc(C)");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
              "copying A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice),
              "copying B");

    constexpr unsigned THREADS = 16;
    constexpr unsigned OUTPUT_TILE = THREADS * 2;
    const dim3 block(THREADS, THREADS);
    const dim3 grid(static_cast<unsigned>((N + OUTPUT_TILE - 1) / OUTPUT_TILE),
                    static_cast<unsigned>((N + OUTPUT_TILE - 1) / OUTPUT_TILE));
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost),
              "copying C");

    cudaFree(deviceC);
    cudaFree(deviceB);
    cudaFree(deviceA);
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
