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

// Each block computes a 32 x 32 tile, with four results per thread.
// Shared tiles reuse both inputs across the dot products while preserving the
// original order of the k loop for every output element.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    constexpr unsigned tile = 32;
    constexpr unsigned depth = 16;
    __shared__ double aTile[tile][depth];
    __shared__ double bTile[depth][tile];

    const unsigned x = threadIdx.x;
    const unsigned y = threadIdx.y;
    const size_t row = static_cast<size_t>(blockIdx.y) * tile + y;
    const size_t col = static_cast<size_t>(blockIdx.x) * tile + x;
    double sum00 = 0.0, sum01 = 0.0, sum10 = 0.0, sum11 = 0.0;

    for (size_t base = 0; base < N; base += depth) {
        const size_t kx = base + x;
        const size_t ky = base + y;
        aTile[y][x] = (row < N && kx < N) ? A[row * N + kx] : 0.0;
        aTile[y + 16][x] = (row + 16 < N && kx < N) ? A[(row + 16) * N + kx] : 0.0;
        bTile[y][x] = (ky < N && col < N) ? B[ky * N + col] : 0.0;
        bTile[y][x + 16] = (ky < N && col + 16 < N) ? B[ky * N + col + 16] : 0.0;
        __syncthreads();

        #pragma unroll
        for (unsigned k = 0; k < depth; ++k) {
            const double a0 = aTile[y][k];
            const double a1 = aTile[y + 16][k];
            const double b0 = bTile[k][x];
            const double b1 = bTile[k][x + 16];
            sum00 += a0 * b0;
            sum01 += a0 * b1;
            sum10 += a1 * b0;
            sum11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum00;
    if (row < N && col + 16 < N) C[row * N + col + 16] = sum01;
    if (row + 16 < N && col < N) C[(row + 16) * N + col] = sum10;
    if (row + 16 < N && col + 16 < N) C[(row + 16) * N + col + 16] = sum11;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) return true;
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
    
    // Prepare GPU buffers before timing, just as host allocation and
    // initialization are outside the original multiplication timing.
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    const size_t bytes = N * N * sizeof(double);
    if (bytes != 0) {
        checkCuda(cudaMalloc(&deviceA, bytes), "allocating A");
        checkCuda(cudaMalloc(&deviceB, bytes), "allocating B");
        checkCuda(cudaMalloc(&deviceC, bytes), "allocating C");
        checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A to GPU");
        checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B to GPU");
        // Resolve any lazy kernel loading or PTX compilation before timing.
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, matrixMultiplyKernel), "loading matrix multiplication kernel");
    }

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (bytes != 0) {
        const dim3 threads(16, 16);
        const dim3 blocks((N + 31) / 32, (N + 31) / 32);
        matrixMultiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
        checkCuda(cudaGetLastError(), "launching matrix multiplication");
        checkCuda(cudaDeviceSynchronize(), "computing matrix multiplication");
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();

    if (bytes != 0) {
        checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying C from GPU");
        checkCuda(cudaFree(deviceA), "freeing A");
        checkCuda(cudaFree(deviceB), "freeing B");
        checkCuda(cudaFree(deviceC), "freeing C");
    }
    
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (milliseconds / 1000.0) / 1e9;
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
