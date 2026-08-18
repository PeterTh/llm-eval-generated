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

namespace {

constexpr int kTileSize = 32;
constexpr int kBlockRows = 8;
constexpr int kRowsPerThread = kTileSize / kBlockRows;

void checkCuda(const cudaError_t error, const char* const operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// Each 32x8 block computes a 32x32 output tile. Every thread computes four
// rows, which keeps the tile dimensions warp-friendly while reducing the
// number of blocks and synchronizations needed for a large matrix.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double tileA[kTileSize][kTileSize];
    __shared__ double tileB[kTileSize][kTileSize];

    const int threadColumn = static_cast<int>(threadIdx.x);
    const int threadRow = static_cast<int>(threadIdx.y);
    const size_t tileRow = static_cast<size_t>(blockIdx.y) * kTileSize;
    const size_t tileColumn = static_cast<size_t>(blockIdx.x) * kTileSize;

    double sums[kRowsPerThread] = {0.0, 0.0, 0.0, 0.0};

    for (size_t tileStart = 0; tileStart < N; tileStart += kTileSize) {
        // The 32x8 block has enough threads to load four rows of each tile
        // per thread. Out-of-range elements are zero for non-multiple sizes.
        #pragma unroll
        for (int rowOffset = 0; rowOffset < kRowsPerThread; ++rowOffset) {
            const int tileLocalRow = threadRow + rowOffset * kBlockRows;
            const size_t globalRow = tileRow + static_cast<size_t>(tileLocalRow);
            const size_t globalColumn = tileColumn + static_cast<size_t>(threadColumn);

            tileA[tileLocalRow][threadColumn] =
                (globalRow < N && tileStart + static_cast<size_t>(threadColumn) < N)
                    ? A[globalRow * N + tileStart + static_cast<size_t>(threadColumn)]
                    : 0.0;
            tileB[tileLocalRow][threadColumn] =
                (tileStart + static_cast<size_t>(tileLocalRow) < N && globalColumn < N)
                    ? B[(tileStart + static_cast<size_t>(tileLocalRow)) * N + globalColumn]
                    : 0.0;
        }

        __syncthreads();

        // Keeping this loop in increasing k order matches the scalar
        // reference implementation's accumulation order.
        #pragma unroll
        for (int k = 0; k < kTileSize; ++k) {
            const double b = tileB[k][threadColumn];
            #pragma unroll
            for (int rowOffset = 0; rowOffset < kRowsPerThread; ++rowOffset) {
                const int tileLocalRow = threadRow + rowOffset * kBlockRows;
                sums[rowOffset] += tileA[tileLocalRow][k] * b;
            }
        }

        __syncthreads();
    }

    const size_t globalColumn = tileColumn + static_cast<size_t>(threadColumn);
    if (globalColumn < N) {
        #pragma unroll
        for (int rowOffset = 0; rowOffset < kRowsPerThread; ++rowOffset) {
            const size_t globalRow = tileRow + static_cast<size_t>(threadRow + rowOffset * kBlockRows);
            if (globalRow < N) {
                C[globalRow * N + globalColumn] = sums[rowOffset];
            }
        }
    }
}

double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return 0.0;
    }

    const size_t elementCount = N * N;
    const size_t allocationSize = elementCount * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceA), allocationSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceB), allocationSize));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceC), allocationSize));

    CUDA_CHECK(cudaMemcpy(deviceA, A.data(), allocationSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceB, B.data(), allocationSize, cudaMemcpyHostToDevice));

    const dim3 block(kTileSize, kBlockRows);
    const dim3 grid(static_cast<unsigned int>((N + kTileSize - 1) / kTileSize),
                    static_cast<unsigned int>((N + kTileSize - 1) / kTileSize));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));

    CUDA_CHECK(cudaMemcpy(C.data(), deviceC, allocationSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    CUDA_CHECK(cudaFree(deviceC));

    return static_cast<double>(elapsedMilliseconds);
}

} // namespace

#undef CUDA_CHECK

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
    const double computationMilliseconds = matrixMultiply(A, B, C, N);

    printf("Computation time: %.3f ms\n", computationMilliseconds);
    
    // Calculate GFLOPS
    double gflops = (computationMilliseconds > 0.0)
                        ? (2.0 * N * N * N) / (computationMilliseconds / 1000.0) / 1e9
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
