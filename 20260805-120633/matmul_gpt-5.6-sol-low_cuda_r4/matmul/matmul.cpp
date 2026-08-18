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
constexpr int TILE = 32;
constexpr int BLOCK_ROWS = 8;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    // Padding avoids shared-memory bank conflicts when traversing a tile.
    __shared__ double tileA[TILE][TILE + 1];
    __shared__ double tileB[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t blockRow = static_cast<size_t>(blockIdx.y) * TILE;
    const size_t col = static_cast<size_t>(blockIdx.x) * TILE + tx;
    double sums[TILE / BLOCK_ROWS] = {};

    for (size_t base = 0; base < N; base += TILE) {
#pragma unroll
        for (int r = 0; r < TILE / BLOCK_ROWS; ++r) {
            const int localRow = ty + r * BLOCK_ROWS;
            const size_t row = blockRow + localRow;
            const size_t bRow = base + localRow;
            tileA[localRow][tx] = (row < N && base + tx < N)
                                      ? A[row * N + base + tx] : 0.0;
            tileB[localRow][tx] = (bRow < N && col < N)
                                      ? B[bRow * N + col] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const double b = tileB[k][tx];
#pragma unroll
            for (int r = 0; r < TILE / BLOCK_ROWS; ++r)
                sums[r] = fma(tileA[ty + r * BLOCK_ROWS][k], b, sums[r]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < TILE / BLOCK_ROWS; ++r) {
        const size_t row = blockRow + ty + r * BLOCK_ROWS;
        if (row < N && col < N)
            C[row * N + col] = sums[r];
    }
}

double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;

    checkCuda(cudaMalloc(&deviceA, bytes), "allocating matrix A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating matrix B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocating matrix C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying matrix A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying matrix B");
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&stop), "creating stop event");

    const dim3 threads(TILE, BLOCK_ROWS);
    const dim3 blocks(static_cast<unsigned>((N + TILE - 1) / TILE),
                      static_cast<unsigned>((N + TILE - 1) / TILE));
    checkCuda(cudaEventRecord(start), "recording start event");
    matrixMultiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaEventRecord(stop), "recording stop event");
    checkCuda(cudaEventSynchronize(stop), "running matrix multiplication kernel");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "measuring kernel time");
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying matrix C");
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(stop), "destroying stop event");
    checkCuda(cudaFree(deviceA), "freeing matrix A");
    checkCuda(cudaFree(deviceB), "freeing matrix B");
    checkCuda(cudaFree(deviceC), "freeing matrix C");
    return elapsedMs;
}
} // namespace

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
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '-' || *value == '\0' || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                std::fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
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
    
    if (N > std::numeric_limits<size_t>::max() / N ||
        N * N > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return 1;
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
    const double durationMs = matrixMultiply(A, B, C, N);
    const long duration = static_cast<long>(std::llround(durationMs));
    
    printf("Computation time: %ld ms\n", duration);
    
    // Calculate GFLOPS
    const double n = static_cast<double>(N);
    double gflops = (2.0 * n * n * n) / (durationMs / 1000.0) / 1e9;
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
