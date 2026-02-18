#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = (call); \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

#define TILE_SIZE 32

__global__ void initMatrixKernel(double* mat, size_t N) {
    size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (idx < N * N) {
        size_t i = idx / N;
        size_t j = idx % N;
        mat[idx] = (((i + 1) * (i + j + 1) * (size_t)1299709) % (N * N)) / static_cast<double>(N * N);
    }
}

__global__ void matmulKernel(const double* __restrict__ A, const double* __restrict__ B,
                             double* __restrict__ C, size_t N) {
    __shared__ double As[TILE_SIZE][TILE_SIZE];
    __shared__ double Bs[TILE_SIZE][TILE_SIZE];

    size_t row = blockIdx.y * TILE_SIZE + threadIdx.y;
    size_t col = blockIdx.x * TILE_SIZE + threadIdx.x;

    double sum = 0.0;
    size_t numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    for (size_t t = 0; t < numTiles; ++t) {
        size_t tCol = t * TILE_SIZE + threadIdx.x;
        size_t tRow = t * TILE_SIZE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && tCol < N) ? A[row * N + tCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] = (tRow < N && col < N) ? B[tRow * N + col] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

// CPU validation: recompute a few elements and compare
bool validateResult(const double* A, const double* B,
                   const double* C, size_t N) {
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

    size_t matBytes = N * N * sizeof(double);

    // Allocate device memory
    double *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, matBytes));
    CUDA_CHECK(cudaMalloc(&d_B, matBytes));
    CUDA_CHECK(cudaMalloc(&d_C, matBytes));

    // Initialize matrices on GPU
    printf("Initializing matrices...\n");
    int initBlock = 256;
    int initGrid = (int)((N * N + initBlock - 1) / initBlock);
    initMatrixKernel<<<initGrid, initBlock>>>(d_A, N);
    initMatrixKernel<<<initGrid, initBlock>>>(d_B, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Perform matrix multiplication on GPU
    printf("Computing matrix multiplication...\n");
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((unsigned int)((N + TILE_SIZE - 1) / TILE_SIZE),
                 (unsigned int)((N + TILE_SIZE - 1) / TILE_SIZE));

    auto start = std::chrono::high_resolution_clock::now();

    matmulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    // Copy result back to host
    std::vector<double> C(N * N);
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, matBytes, cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        CUDA_CHECK(cudaMemcpy(A.data(), d_A, matBytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(B.data(), d_B, matBytes, cudaMemcpyDeviceToHost));

        bool valid = validateResult(A.data(), B.data(), C.data(), N);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            return 1;
        }
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
