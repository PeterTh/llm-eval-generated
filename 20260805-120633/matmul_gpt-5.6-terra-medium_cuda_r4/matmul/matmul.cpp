#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int TILE_DIM = 32;
constexpr int THREAD_DIM = 16;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block produces a 32x32 output tile.  A 16x16 thread block computes a
// 2x2 result per thread, increasing arithmetic intensity while retaining
// coalesced loads and shared-memory reuse for both input tiles.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double aTile[TILE_DIM][TILE_DIM];
    __shared__ double bTile[TILE_DIM][TILE_DIM];

    const size_t row = static_cast<size_t>(blockIdx.y) * TILE_DIM + threadIdx.y;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE_DIM + threadIdx.x;
    const size_t row2 = row + THREAD_DIM;
    const size_t col2 = col + THREAD_DIM;

    double sum00 = 0.0;
    double sum01 = 0.0;
    double sum10 = 0.0;
    double sum11 = 0.0;

    for (size_t tileStart = 0; tileStart < N; tileStart += TILE_DIM) {
        const size_t aCol = tileStart + threadIdx.x;
        const size_t bRow = tileStart + threadIdx.y;

        aTile[threadIdx.y][threadIdx.x] = row < N && aCol < N ? A[row * N + aCol] : 0.0;
        aTile[threadIdx.y][threadIdx.x + THREAD_DIM] =
            row < N && aCol + THREAD_DIM < N ? A[row * N + aCol + THREAD_DIM] : 0.0;
        aTile[threadIdx.y + THREAD_DIM][threadIdx.x] =
            row2 < N && aCol < N ? A[row2 * N + aCol] : 0.0;
        aTile[threadIdx.y + THREAD_DIM][threadIdx.x + THREAD_DIM] =
            row2 < N && aCol + THREAD_DIM < N ? A[row2 * N + aCol + THREAD_DIM] : 0.0;

        bTile[threadIdx.y][threadIdx.x] = bRow < N && col < N ? B[bRow * N + col] : 0.0;
        bTile[threadIdx.y][threadIdx.x + THREAD_DIM] =
            bRow < N && col2 < N ? B[bRow * N + col2] : 0.0;
        bTile[threadIdx.y + THREAD_DIM][threadIdx.x] =
            bRow + THREAD_DIM < N && col < N ? B[(bRow + THREAD_DIM) * N + col] : 0.0;
        bTile[threadIdx.y + THREAD_DIM][threadIdx.x + THREAD_DIM] =
            bRow + THREAD_DIM < N && col2 < N ? B[(bRow + THREAD_DIM) * N + col2] : 0.0;

        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            sum00 += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
            sum01 += aTile[threadIdx.y][k] * bTile[k][threadIdx.x + THREAD_DIM];
            sum10 += aTile[threadIdx.y + THREAD_DIM][k] * bTile[k][threadIdx.x];
            sum11 += aTile[threadIdx.y + THREAD_DIM][k] * bTile[k][threadIdx.x + THREAD_DIM];
        }

        __syncthreads();
    }

    if (row < N && col < N) C[row * N + col] = sum00;
    if (row < N && col2 < N) C[row * N + col2] = sum01;
    if (row2 < N && col < N) C[row2 * N + col] = sum10;
    if (row2 < N && col2 < N) C[row2 * N + col2] = sum11;
}

} // namespace

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

float matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                     std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cudaEvent_t start{}, end{};

    checkCuda(cudaMalloc(&deviceA, bytes), "allocating matrix A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating matrix B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocating matrix C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying matrix A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying matrix B");
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");

    const dim3 block(THREAD_DIM, THREAD_DIM);
    const dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);
    checkCuda(cudaEventRecord(start), "recording start event");
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "synchronizing matrix multiplication kernel");

    float elapsedMs = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "measuring kernel time");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying matrix C");

    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    checkCuda(cudaFree(deviceA), "freeing matrix A");
    checkCuda(cudaFree(deviceB), "freeing matrix B");
    checkCuda(cudaFree(deviceC), "freeing matrix C");
    return elapsedMs;
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
    const float durationMs = matrixMultiply(A, B, C, N);
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (durationMs / 1000.0) / 1e9;
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
