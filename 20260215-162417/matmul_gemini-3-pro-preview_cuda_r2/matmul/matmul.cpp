#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <iostream>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// Tiled matrix multiplication kernel
// Block size should be tuned, 32 is a common choice for modern GPUs
constexpr int TILE_SIZE = 32;

__global__ void matrixMultiplyKernel(const double* A, const double* B, double* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    double sum = 0.0;
    
    // Shared memory for tiles
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];
    
    for (int k = 0; k < N; k += TILE_SIZE) {
        // Load tile from A into shared memory
        // Each thread loads one element of the tile
        int tiledRow = row;                 // Row index for A
        int tiledCol = k + threadIdx.x;     // Column index for A
        if (tiledRow < N && tiledCol < N) {
            As[threadIdx.y][threadIdx.x] = A[tiledRow * N + tiledCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load tile from B into shared memory
        tiledRow = k + threadIdx.y;         // Row index for B
        tiledCol = col;                     // Column index for B
        if (tiledRow < N && tiledCol < N) {
            Bs[threadIdx.y][threadIdx.x] = B[tiledRow * N + tiledCol];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        
        __syncthreads();
        
        // Compute partial product for this tile
        for (int n = 0; n < TILE_SIZE; ++n) {
            sum += As[threadIdx.y][n] * Bs[n][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (row < N && col < N) {
        C[row * N + col] = sum;
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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    
    double *d_A, *d_B, *d_C;
    size_t size = N * N * sizeof(double);
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice));
    
    // Setup grid and block dimensions
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    
    // Launch kernel
    matrixMultiplyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    
    // Synchronize to ensure completion before stopping timer (although cudaMemcpy would sync too)
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost));
    
    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
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

    // Pin memory for faster transfer
    cudaHostRegister(A.data(), N * N * sizeof(double), cudaHostRegisterDefault);
    cudaHostRegister(B.data(), N * N * sizeof(double), cudaHostRegisterDefault);
    cudaHostRegister(C.data(), N * N * sizeof(double), cudaHostRegisterDefault);
    
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
    
    // Unpin memory
    cudaHostUnregister(A.data());
    cudaHostUnregister(B.data());
    cudaHostUnregister(C.data());
    
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
