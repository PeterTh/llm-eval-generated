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

// Optimized tiled matrix multiplication kernel using shared memory
// Uses double precision (double) for correctness with the original code
template <int BLOCK_SIZE>
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const size_t N) {
    // Shared memory tile for sub-matrices of A and B
    __shared__ double tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tileB[BLOCK_SIZE][BLOCK_SIZE];

    const int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    double sum = 0.0;

    const int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A into shared memory
        const int aCol = t * BLOCK_SIZE + threadIdx.x;
        if (row < N && aCol < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + aCol];
        } else {
            tileA[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Load tile from B into shared memory
        const int bRow = t * BLOCK_SIZE + threadIdx.y;
        if (col < N && bRow < N) {
            tileB[threadIdx.y][threadIdx.x] = B[bRow * N + col];
        } else {
            tileB[threadIdx.y][threadIdx.x] = 0.0;
        }

        // Synchronize to ensure the tile is loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Synchronize before loading next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CUDA wrapper that selects kernel and launches
void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);

    // Device pointers
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // Allocate device memory
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);

    // Copy input matrices to device
    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

    // Zero out C on device
    cudaMemset(dC, 0, bytes);

    // Choose block size: 32 is optimal for double precision on modern GPUs
    // (better occupancy with double due to wider registers)
    constexpr int BLOCK_SIZE = 32;

    // Grid dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    matmulKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(dA, dB, dC, N);

    // Synchronize and copy result back
    cudaDeviceSynchronize();
    cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

// Simple validation: compute a single element and compare on CPU
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

    // Print GPU info
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("GPU: %s (%d GPU(s) available)\n", prop.name, deviceCount);
    }

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
