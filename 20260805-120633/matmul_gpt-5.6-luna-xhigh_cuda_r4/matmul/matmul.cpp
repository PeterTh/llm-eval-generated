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

constexpr int kTileSize = 32;
constexpr int kThreadsPerDimension = kTileSize / 2;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each 16x16 thread block computes a 32x32 output tile.  A thread owns a
// 2x2 result tile, which increases reuse of values loaded into shared memory
// while keeping global loads and stores coalesced.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                      const double* __restrict__ B,
                                      double* __restrict__ C, const int N) {
    __shared__ double tileA[kTileSize][kTileSize];
    __shared__ double tileB[kTileSize][kTileSize];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int row = static_cast<int>(blockIdx.y) * kTileSize + ty * 2;
    const int col = static_cast<int>(blockIdx.x) * kTileSize + tx * 2;

    double c00 = 0.0;
    double c01 = 0.0;
    double c10 = 0.0;
    double c11 = 0.0;

    for (int tileStart = 0; tileStart < N; tileStart += kTileSize) {
        const int aCol = tileStart + tx * 2;
        const int bRow = tileStart + ty * 2;

        tileA[ty * 2][tx * 2] =
            row < N && aCol < N ? A[static_cast<size_t>(row) * N + aCol] : 0.0;
        tileA[ty * 2][tx * 2 + 1] =
            row < N && aCol + 1 < N
                ? A[static_cast<size_t>(row) * N + aCol + 1]
                : 0.0;
        tileA[ty * 2 + 1][tx * 2] =
            row + 1 < N && aCol < N
                ? A[static_cast<size_t>(row + 1) * N + aCol]
                : 0.0;
        tileA[ty * 2 + 1][tx * 2 + 1] =
            row + 1 < N && aCol + 1 < N
                ? A[static_cast<size_t>(row + 1) * N + aCol + 1]
                : 0.0;

        const int bCol = col;
        tileB[ty * 2][tx * 2] =
            bRow < N && bCol < N
                ? B[static_cast<size_t>(bRow) * N + bCol]
                : 0.0;
        tileB[ty * 2][tx * 2 + 1] =
            bRow < N && bCol + 1 < N
                ? B[static_cast<size_t>(bRow) * N + bCol + 1]
                : 0.0;
        tileB[ty * 2 + 1][tx * 2] =
            bRow + 1 < N && bCol < N
                ? B[static_cast<size_t>(bRow + 1) * N + bCol]
                : 0.0;
        tileB[ty * 2 + 1][tx * 2 + 1] =
            bRow + 1 < N && bCol + 1 < N
                ? B[static_cast<size_t>(bRow + 1) * N + bCol + 1]
                : 0.0;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTileSize && tileStart + k < N; ++k) {
            const double a0 = tileA[ty * 2][k];
            const double a1 = tileA[ty * 2 + 1][k];
            const double b0 = tileB[k][tx * 2];
            const double b1 = tileB[k][tx * 2 + 1];

            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row < N && col < N) {
        C[static_cast<size_t>(row) * N + col] = c00;
    }
    if (row < N && col + 1 < N) {
        C[static_cast<size_t>(row) * N + col + 1] = c01;
    }
    if (row + 1 < N && col < N) {
        C[static_cast<size_t>(row + 1) * N + col] = c10;
    }
    if (row + 1 < N && col + 1 < N) {
        C[static_cast<size_t>(row + 1) * N + col + 1] = c11;
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N,
                    double& elapsedMilliseconds) {
    if (N == 0) {
        elapsedMilliseconds = 0.0;
        return;
    }
    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Matrix size is too large for the CUDA kernel\n");
        std::exit(EXIT_FAILURE);
    }

    const size_t elementCount = N * N;
    const size_t bytes = elementCount * sizeof(double);
    const int n = static_cast<int>(N);

    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocating matrix A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating matrix B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocating matrix C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix B");

    const dim3 block(kThreadsPerDimension, kThreadsPerDimension);
    const unsigned int gridDimension =
        static_cast<unsigned int>((N + kTileSize - 1) / kTileSize);
    const dim3 grid(gridDimension, gridDimension);

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&stop), "creating stop event");
    checkCuda(cudaEventRecord(start), "recording start event");

    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, n);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaEventRecord(stop), "recording stop event");
    checkCuda(cudaEventSynchronize(stop), "waiting for matrix multiplication");
    float elapsed = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsed, start, stop),
              "measuring matrix multiplication");
    elapsedMilliseconds = static_cast<double>(elapsed);

    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost),
              "copying matrix C");

    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(stop), "destroying stop event");
    checkCuda(cudaFree(deviceA), "freeing matrix A");
    checkCuda(cudaFree(deviceB), "freeing matrix B");
    checkCuda(cudaFree(deviceC), "freeing matrix C");
}

}  // namespace

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
    double elapsedMilliseconds = 0.0;
    matrixMultiply(A, B, C, N, elapsedMilliseconds);

    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS
    const double gflops = elapsedMilliseconds > 0.0
                              ? (2.0 * static_cast<double>(N) * N * N) /
                                    (elapsedMilliseconds * 1.0e6)
                              : 0.0;
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
