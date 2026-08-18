#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kTileM = 64;
constexpr int kTileN = 64;
constexpr int kTileK = 16;
constexpr int kThreadsX = 16;
constexpr int kThreadsY = 16;
constexpr int kRowsPerThread = kTileM / kThreadsY;
constexpr int kColsPerThread = kTileN / kThreadsX;
constexpr int kThreadsPerBlock = kThreadsX * kThreadsY;
constexpr int kTileElements = kTileM * kTileK;

static_assert(kTileM % kThreadsY == 0 && kTileN % kThreadsX == 0);
static_assert(kTileElements % kThreadsPerBlock == 0);

void checkCuda(const cudaError_t status, const char* const expression,
               const char* const file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n",
                     file, line, expression, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

// Each 16x16 block calculates a 64x64 output tile.  A thread retains a 4x4
// output fragment in registers while the block stages 64x16 tiles of A and B
// in shared memory.  This provides coalesced global accesses and enough work
// per launch to scale across GPU SMs without changing the row-major ABI.
__global__ __launch_bounds__(kThreadsPerBlock, 4)
void matrixMultiplyKernel(const double* __restrict__ A,
                          const double* __restrict__ B,
                          double* __restrict__ C, const int n) {
    __shared__ double aTile[kTileM][kTileK];
    __shared__ double bTile[kTileK][kTileN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int threadLinear = ty * kThreadsX + tx;
    const int blockRow = static_cast<int>(blockIdx.y) * kTileM;
    const int blockCol = static_cast<int>(blockIdx.x) * kTileN;
    const int row = blockRow + ty * kRowsPerThread;
    const int col = blockCol + tx * kColsPerThread;

    double accum[kRowsPerThread][kColsPerThread] = {};

    for (int tileK = 0; tileK < n; tileK += kTileK) {
        // Both tiles contain kTileElements entries, so every thread performs
        // the same number of coalesced loads.  Zero padding handles arbitrary
        // N without a separate remainder kernel.
#pragma unroll
        for (int load = 0; load < kTileElements / kThreadsPerBlock; ++load) {
            const int index = threadLinear + load * kThreadsPerBlock;
            const int aRow = index / kTileK;
            const int aCol = index % kTileK;
            const int bRow = index / kTileN;
            const int bCol = index % kTileN;
            const int globalARow = blockRow + aRow;
            const int globalBCol = blockCol + bCol;

            aTile[aRow][aCol] = (globalARow < n && tileK + aCol < n)
                ? A[static_cast<size_t>(globalARow) * n + tileK + aCol]
                : 0.0;
            bTile[bRow][bCol] = (tileK + bRow < n && globalBCol < n)
                ? B[static_cast<size_t>(tileK + bRow) * n + globalBCol]
                : 0.0;
        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTileK; ++k) {
            double bValues[kColsPerThread];
#pragma unroll
            for (int j = 0; j < kColsPerThread; ++j) {
                bValues[j] = bTile[k][tx * kColsPerThread + j];
            }

#pragma unroll
            for (int i = 0; i < kRowsPerThread; ++i) {
                const double aValue = aTile[ty * kRowsPerThread + i][k];
#pragma unroll
                for (int j = 0; j < kColsPerThread; ++j) {
                    accum[i][j] += aValue * bValues[j];
                }
            }
        }

        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kRowsPerThread; ++i) {
        if (row + i < n) {
#pragma unroll
            for (int j = 0; j < kColsPerThread; ++j) {
                if (col + j < n) {
                    C[static_cast<size_t>(row + i) * n + col + j] = accum[i][j];
                }
            }
        }
    }
}

// The returned duration measures only the matrix multiplication kernel,
// matching the original benchmark's computation-only timing boundary.
float matrixMultiplyCuda(const std::vector<double>& A, const std::vector<double>& B,
                         std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return 0.0F;
    }
    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Matrix dimension is too large for CUDA indexing: %zu\n", N);
        std::exit(EXIT_FAILURE);
    }

    const size_t bytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMalloc(&deviceC, bytes));
    CUDA_CHECK(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const dim3 block(kThreadsX, kThreadsY);
    const dim3 grid(static_cast<unsigned int>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned int>((N + kTileM - 1) / kTileM));

    CUDA_CHECK(cudaEventRecord(start));
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, static_cast<int>(N));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
    CUDA_CHECK(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaFree(deviceC));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceA));
    return elapsedMilliseconds;
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
    const float elapsedMilliseconds = matrixMultiplyCuda(A, B, C, N);
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate GFLOPS
    const double gflops = elapsedMilliseconds > 0.0F
        ? (2.0 * N * N * N) / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e9
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
