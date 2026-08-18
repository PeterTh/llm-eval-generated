#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

namespace {

constexpr unsigned TILE = 32;
constexpr unsigned BLOCK_ROWS = 8;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ __launch_bounds__(TILE * BLOCK_ROWS)
void matrixMultiplyKernel(const double* __restrict__ A,
                          const double* __restrict__ B,
                          double* __restrict__ C, unsigned N) {
    __shared__ double tileA[TILE][TILE + 1];
    __shared__ double tileB[TILE][TILE + 1];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned column = blockIdx.x * TILE + tx;
    const unsigned rowBase = blockIdx.y * TILE;
    double sums[4] = {0.0, 0.0, 0.0, 0.0};

    for (unsigned tile = 0; tile < N; tile += TILE) {
#pragma unroll
        for (unsigned item = 0; item < 4; ++item) {
            const unsigned localRow = ty + item * BLOCK_ROWS;
            const unsigned row = rowBase + localRow;
            const unsigned k = tile + tx;
            tileA[localRow][tx] = (row < N && k < N) ? A[static_cast<size_t>(row) * N + k] : 0.0;

            const unsigned bRow = tile + localRow;
            tileB[localRow][tx] = (bRow < N && column < N)
                ? B[static_cast<size_t>(bRow) * N + column] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (unsigned k = 0; k < TILE; ++k) {
            const double b = tileB[k][tx];
#pragma unroll
            for (unsigned item = 0; item < 4; ++item) {
                sums[item] = fma(tileA[ty + item * BLOCK_ROWS][k], b, sums[item]);
            }
        }
        __syncthreads();
    }

    if (column < N) {
#pragma unroll
        for (unsigned item = 0; item < 4; ++item) {
            const unsigned row = rowBase + ty + item * BLOCK_ROWS;
            if (row < N) C[static_cast<size_t>(row) * N + column] = sums[item];
        }
    }
}

} // namespace

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    if (N > std::numeric_limits<unsigned>::max()) {
        std::fprintf(stderr, "Matrix dimension is too large for the CUDA kernel\n");
        std::exit(EXIT_FAILURE);
    }

    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocating matrix A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating matrix B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocating matrix C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying matrix A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying matrix B");

    const unsigned dimension = static_cast<unsigned>(N);
    const dim3 block(TILE, BLOCK_ROWS);
    const dim3 grid((dimension + TILE - 1) / TILE,
                    (dimension + TILE - 1) / TILE);
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, dimension);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying matrix C");

    checkCuda(cudaFree(deviceA), "freeing matrix A");
    checkCuda(cudaFree(deviceB), "freeing matrix B");
    checkCuda(cudaFree(deviceC), "freeing matrix C");
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

    // Create the CUDA context before timing so one-time driver startup is not
    // incorrectly reported as matrix multiplication work.
    checkCuda(cudaFree(nullptr), "initializing the CUDA runtime");
    
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
