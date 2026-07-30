#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// CUDA tiled matrix multiplication kernel
template <const int BLOCK_SIZE>
__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                              double* __restrict__ C, const int N) {
    __shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * BLOCK_SIZE + ty;
    const int col = blockIdx.x * BLOCK_SIZE + tx;

    double sum = 0.0;
    const int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int t = 0; t < numTiles; ++t) {
        const int tOff = t * BLOCK_SIZE;
        // Load tile of A
        if (row < N && tOff + tx < N) {
            As[ty][tx] = A[row * N + tOff + tx];
        } else {
            As[ty][tx] = 0.0;
        }
        // Load tile of B
        if (tOff + ty < N && col < N) {
            Bs[ty][tx] = B[(tOff + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0;
        }
        __syncthreads();

        // Compute partial dot product for this tile
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CUDA tiled matrix multiplication (device pointers)
void matrixMultiplyDevice(double* d_A, double* d_B, double* d_C, const size_t N) {
    constexpr int BLOCK_SIZE = 16;
    const dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE,
                       (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matmulKernel<BLOCK_SIZE><<<gridDim, blockDim>>>(d_A, d_B, d_C, static_cast<int>(N));
}

// Simple validation: compute elements on CPU and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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

    // Allocate host matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);

    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Allocate device memory once (before timing)
    const size_t bytes = N * N * sizeof(double);
    double *d_A, *d_B, *d_C;
    cudaError_t err;
    err = cudaMalloc(&d_A, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_B, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMalloc(&d_C, bytes);
    if (err != cudaSuccess) { printf("cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Copy inputs to device
    err = cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy H2D A failed: %s\n", cudaGetErrorString(err)); return 1; }
    err = cudaMemcpy(d_B, B.data(), bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { printf("cudaMemcpy H2D B failed: %s\n", cudaGetErrorString(err)); return 1; }

    // Perform matrix multiplication on GPU (timed)
    printf("Computing matrix multiplication (CUDA)...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplyDevice(d_A, d_B, d_C, N);

    // Copy result back
    err = cudaMemcpy(C.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { printf("cudaMemcpy D2H C failed: %s\n", cudaGetErrorString(err)); return 1; }

    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto durationUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double durationMs = durationUs.count() / 1000.0;

    printf("Computation time: %.3f ms\n", durationMs);

    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (durationMs / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Cleanup device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

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
