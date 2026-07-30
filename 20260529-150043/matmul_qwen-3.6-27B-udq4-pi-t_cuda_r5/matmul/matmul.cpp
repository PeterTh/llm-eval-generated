#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err), #call);       \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

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

// Optimized CUDA matrix multiplication kernel with shared memory tiling
// Uses 32x32 tiles for optimal shared memory utilization and coalesced access
template <unsigned int BLOCK_SIZE>
__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             const size_t N) {
    // Shared memory tile for sub-matrices
    __shared__ double tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double tileB[BLOCK_SIZE][BLOCK_SIZE];

    // Thread indices within the block
    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;

    // Row and column of the output element this thread computes
    const size_t row = blockIdx.y * BLOCK_SIZE + ty;
    const size_t col = blockIdx.x * BLOCK_SIZE + tx;

    double sum = 0.0;

    // Number of tiles to iterate over in the K dimension
    const int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        // Load tile from A: each thread loads one element
        // A is accessed as A[row][tile_col] -> A[row * N + (t * BLOCK_SIZE + tx)]
        const size_t aCol = t * BLOCK_SIZE + tx;
        if (row < N && aCol < N) {
            tileA[ty][tx] = A[row * N + aCol];
        } else {
            tileA[ty][tx] = 0.0;
        }

        // Load tile from B: each thread loads one element
        // B is accessed as B[tile_row][col] -> B[(t * BLOCK_SIZE + ty) * N + col]
        const size_t bRow = t * BLOCK_SIZE + ty;
        if (bRow < N && col < N) {
            tileB[ty][tx] = B[bRow * N + col];
        } else {
            tileB[ty][tx] = 0.0;
        }

        // Synchronize to ensure the entire tile is loaded
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += tileA[ty][k] * tileB[k][tx];
        }

        // Synchronize before loading the next tile
        __syncthreads();
    }

    // Write result to global memory
    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t totalElements = N * N;
    const size_t bytes = totalElements * sizeof(double);

    // Device pointers
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    // Copy input matrices to device
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Clear output matrix on device
    CUDA_CHECK(cudaMemset(dC, 0, bytes));

    // Configure kernel launch parameters
    // Block size: 32x32 = 1024 threads per block
    constexpr unsigned int BLOCK_SIZE = 32;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the optimized tiled kernel
    matmulKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure completion before timing
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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
