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

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int TILE_SIZE = 32;
constexpr int ROWS_PER_THREAD = 4;

// Each 32x8 block computes a 32x32 output tile.  Padding avoids shared-memory
// bank conflicts while four output rows per thread increase data reuse.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE + 1];
    __shared__ double tileB[TILE_SIZE][TILE_SIZE + 1];

    const size_t col = static_cast<size_t>(blockIdx.x) * TILE_SIZE + threadIdx.x;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE_SIZE + threadIdx.y;
    double sums[ROWS_PER_THREAD] = {0.0, 0.0, 0.0, 0.0};

    for (size_t tile = 0; tile < N; tile += TILE_SIZE) {
#pragma unroll
        for (int rowOffset = 0; rowOffset < TILE_SIZE; rowOffset += blockDim.y) {
            const size_t aRow = rowBase + rowOffset;
            const size_t bRow = tile + threadIdx.y + rowOffset;
            tileA[threadIdx.y + rowOffset][threadIdx.x] =
                (aRow < N && tile + threadIdx.x < N) ? A[aRow * N + tile + threadIdx.x] : 0.0;
            tileB[threadIdx.y + rowOffset][threadIdx.x] =
                (bRow < N && col < N) ? B[bRow * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
#pragma unroll
            for (int output = 0; output < ROWS_PER_THREAD; ++output) {
                sums[output] += tileA[threadIdx.y + output * blockDim.y][k] * tileB[k][threadIdx.x];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int output = 0; output < ROWS_PER_THREAD; ++output) {
        const size_t row = rowBase + output * blockDim.y;
        if (row < N && col < N) {
            C[row * N + col] = sums[output];
        }
    }
}

void matrixMultiply(const double* A, const double* B, double* C, const size_t N) {
    const dim3 block(TILE_SIZE, TILE_SIZE / ROWS_PER_THREAD);
    const dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    matrixMultiplyKernel<<<grid, block>>>(A, B, C, N);
    checkCuda(cudaGetLastError(), "kernel launch");
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
    
    const size_t matrixBytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, matrixBytes), "allocating matrix A");
    checkCuda(cudaMalloc(&deviceB, matrixBytes), "allocating matrix B");
    checkCuda(cudaMalloc(&deviceC, matrixBytes), "allocating matrix C");
    checkCuda(cudaMemcpy(deviceA, A.data(), matrixBytes, cudaMemcpyHostToDevice), "copying matrix A");
    checkCuda(cudaMemcpy(deviceB, B.data(), matrixBytes, cudaMemcpyHostToDevice), "copying matrix B");

    // Time only the multiplication, matching the original benchmark's timing scope.
    printf("Computing matrix multiplication...\n");
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    checkCuda(cudaEventCreate(&startEvent), "creating start event");
    checkCuda(cudaEventCreate(&endEvent), "creating end event");
    checkCuda(cudaEventRecord(startEvent), "recording start event");
    matrixMultiply(deviceA, deviceB, deviceC, N);
    checkCuda(cudaEventRecord(endEvent), "recording end event");
    checkCuda(cudaEventSynchronize(endEvent), "synchronizing computation");
    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, endEvent), "reading elapsed time");
    const auto duration = std::chrono::milliseconds(static_cast<long long>(elapsedMilliseconds));
    checkCuda(cudaMemcpy(C.data(), deviceC, matrixBytes, cudaMemcpyDeviceToHost), "copying matrix C");
    checkCuda(cudaEventDestroy(startEvent), "destroying start event");
    checkCuda(cudaEventDestroy(endEvent), "destroying end event");
    checkCuda(cudaFree(deviceA), "freeing matrix A");
    checkCuda(cudaFree(deviceB), "freeing matrix B");
    checkCuda(cudaFree(deviceC), "freeing matrix C");
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    // CUDA events provide sub-millisecond timing, which is important for GPU kernels.
    double gflops = (2.0 * N * N * N) / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e9;
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
