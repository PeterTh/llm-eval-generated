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

namespace {

constexpr int kTileM = 8;
constexpr int kTileN = 32;
constexpr int kTileK = 32;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each 32x8 block computes one contiguous 8x32 C tile.  All global loads and
// stores are coalesced, and a B tile is reused by every output row in the block.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    __shared__ double aTile[kTileM][kTileK];
    __shared__ double bTile[kTileK][kTileN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t row = static_cast<size_t>(blockIdx.y) * kTileM + ty;
    const size_t column = static_cast<size_t>(blockIdx.x) * kTileN + tx;

    double sum = 0.0;
    for (size_t tile = 0; tile < N; tile += kTileK) {
        aTile[ty][tx] = row < N && tile + tx < N ? A[row * N + tile + tx] : 0.0;

        // The eight thread rows collectively load all 32 B rows.
#pragma unroll
        for (int load = 0; load < 4; ++load) {
            const int bRow = ty + load * kTileM;
            bTile[bRow][tx] = tile + bRow < N && column < N
                                  ? B[(tile + bRow) * N + column]
                                  : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kTileK; ++k)
            sum += aTile[ty][k] * bTile[k][tx];
        __syncthreads();
    }

    if (row < N && column < N) C[row * N + column] = sum;
}

} // namespace

double matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    cudaEvent_t startEvent, endEvent;

    checkCuda(cudaMalloc(&deviceA, bytes), "allocating A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocating C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B");
    checkCuda(cudaEventCreate(&startEvent), "creating start event");
    checkCuda(cudaEventCreate(&endEvent), "creating end event");

    const dim3 block(kTileN, kTileM);
    const dim3 grid(static_cast<unsigned int>((N + kTileN - 1) / kTileN),
                    static_cast<unsigned int>((N + kTileM - 1) / kTileM));
    checkCuda(cudaEventRecord(startEvent), "recording start event");
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaEventRecord(endEvent), "recording end event");
    checkCuda(cudaEventSynchronize(endEvent), "waiting for matrix multiplication kernel");

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, endEvent),
              "measuring matrix multiplication kernel");

    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying C");
    checkCuda(cudaEventDestroy(startEvent), "destroying start event");
    checkCuda(cudaEventDestroy(endEvent), "destroying end event");
    checkCuda(cudaFree(deviceA), "freeing A");
    checkCuda(cudaFree(deviceB), "freeing B");
    checkCuda(cudaFree(deviceC), "freeing C");
    return elapsedMilliseconds;
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
    const double durationMilliseconds = matrixMultiply(A, B, C, N);
    
    printf("Computation time: %.3f ms\n", durationMilliseconds);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (durationMilliseconds / 1000.0) / 1e9;
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
