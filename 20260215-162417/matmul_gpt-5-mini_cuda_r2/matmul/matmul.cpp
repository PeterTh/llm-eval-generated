#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

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

// CUDA error checking
inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        std::exit(EXIT_FAILURE);
    }
}

// Tiled shared-memory kernel for double-precision matrix multiply
// TILE size tuned for shared memory / occupancy; 16 is safe for many GPUs
#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

extern "C" __global__ void matmul_kernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C, int N) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE_SIZE + ty;
    int col = blockIdx.x * TILE_SIZE + tx;

    double sum = 0.0;
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE_SIZE + tx;
        int bRow = t * TILE_SIZE + ty;

        As[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0;
        Bs[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    // Check a few positions
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
            N = static_cast<size_t>(atoi(argv[++i]));
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
    
    if (N == 0) {
        fprintf(stderr, "Matrix size must be > 0\n");
        return 1;
    }

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate host matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Device pointers
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    size_t bytes = N * N * sizeof(double);

    checkCuda(cudaSetDevice(0), "set device");
    checkCuda(cudaMalloc(&dA, bytes), "alloc dA");
    checkCuda(cudaMalloc(&dB, bytes), "alloc dB");
    checkCuda(cudaMalloc(&dC, bytes), "alloc dC");

    // Copy inputs
    checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "copy A to device");
    checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "copy B to device");

    // Launch kernel
    printf("Computing matrix multiplication on GPU...\n");

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    cudaEvent_t startEvent, stopEvent;
    checkCuda(cudaEventCreate(&startEvent), "create start event");
    checkCuda(cudaEventCreate(&stopEvent), "create stop event");

    checkCuda(cudaEventRecord(startEvent, 0), "record start");

    // Kernel launch; cast to int for N (safe unless extremely large)
    matmul_kernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N));

    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(stopEvent, 0), "record stop");
    checkCuda(cudaEventSynchronize(stopEvent), "sync stop");

    float ms = 0.0f;
    checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent), "elapsed time");

    // Copy result back
    checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "copy C to host");

    // Cleanup device memory and events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);

    long long msll = static_cast<long long>(ms);
    printf("Computation time: %lld ms\n", msll);

    double gflops = (2.0 * static_cast<double>(N) * static_cast<double>(N) * static_cast<double>(N)) / (ms / 1000.0) / 1e9;
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
