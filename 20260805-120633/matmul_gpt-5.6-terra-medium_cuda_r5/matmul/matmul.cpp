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
constexpr int THREAD_ROWS = 4;
constexpr int THREAD_Y = TILE_DIM / THREAD_ROWS;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block calculates a 32x32 output tile.  Eight rows of threads produce
// four output rows each, which gives coalesced global accesses while keeping
// the shared-memory tiles resident for all 32 dot products.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double aTile[TILE_DIM][TILE_DIM + 1];
    __shared__ double bTile[TILE_DIM][TILE_DIM + 1];

    const size_t column = static_cast<size_t>(blockIdx.x) * TILE_DIM + threadIdx.x;
    const size_t firstRow = static_cast<size_t>(blockIdx.y) * TILE_DIM + threadIdx.y;
    double sums[THREAD_ROWS] = {0.0, 0.0, 0.0, 0.0};

    for (size_t tile = 0; tile < N; tile += TILE_DIM) {
#pragma unroll
        for (int rowOffset = 0; rowOffset < THREAD_ROWS; ++rowOffset) {
            const size_t tileRow = threadIdx.y + rowOffset * THREAD_Y;
            const size_t row = static_cast<size_t>(blockIdx.y) * TILE_DIM + tileRow;
            const size_t k = tile + threadIdx.x;
            const size_t bRow = tile + tileRow;

            aTile[tileRow][threadIdx.x] = (row < N && k < N) ? A[row * N + k] : 0.0;
            bTile[tileRow][threadIdx.x] = (bRow < N && column < N) ? B[bRow * N + column] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE_DIM; ++k) {
            const double b = bTile[k][threadIdx.x];
#pragma unroll
            for (int rowOffset = 0; rowOffset < THREAD_ROWS; ++rowOffset) {
                sums[rowOffset] += aTile[threadIdx.y + rowOffset * THREAD_Y][k] * b;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int rowOffset = 0; rowOffset < THREAD_ROWS; ++rowOffset) {
        const size_t row = firstRow + rowOffset * THREAD_Y;
        if (row < N && column < N) {
            C[row * N + column] = sums[rowOffset];
        }
    }
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

long matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return 0;
    }

    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocating A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocating C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A to device");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B to device");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&stop), "creating stop event");
    const dim3 block(TILE_DIM, THREAD_Y);
    const dim3 grid((N + TILE_DIM - 1) / TILE_DIM, (N + TILE_DIM - 1) / TILE_DIM);

    checkCuda(cudaEventRecord(start), "recording start event");
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaEventRecord(stop), "recording stop event");
    checkCuda(cudaEventSynchronize(stop), "synchronizing matrix multiplication kernel");

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, stop), "measuring kernel time");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying C to host");

    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(stop), "destroying stop event");
    checkCuda(cudaFree(deviceA), "freeing A");
    checkCuda(cudaFree(deviceB), "freeing B");
    checkCuda(cudaFree(deviceC), "freeing C");
    return static_cast<long>(elapsedMilliseconds);
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
    const long elapsedMilliseconds = matrixMultiply(A, B, C, N);
    
    printf("Computation time: %ld ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (elapsedMilliseconds / 1000.0) / 1e9;
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
