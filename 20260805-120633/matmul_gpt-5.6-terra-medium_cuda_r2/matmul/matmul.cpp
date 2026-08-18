#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kTile = 32;
constexpr int kRowsPerThread = 2;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each 32x16 block produces a 32x32 output tile. Two rows per thread keep the
// number of concurrent blocks high while reusing each shared-memory tile.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double aTile[kTile][kTile];
    __shared__ double bTile[kTile][kTile];

    const size_t column = static_cast<size_t>(blockIdx.x) * kTile + threadIdx.x;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * (blockDim.y * kRowsPerThread) + threadIdx.y;
    double sums[kRowsPerThread] = {0.0, 0.0};

    for (size_t tileStart = 0; tileStart < N; tileStart += kTile) {
        // Cooperatively load full 32x32 A and B tiles.  Every thread performs
        // four loads, which is coalesced for both matrices.
        for (int loadRow = threadIdx.y; loadRow < kTile; loadRow += blockDim.y) {
            const size_t aRow = static_cast<size_t>(blockIdx.y) * (blockDim.y * kRowsPerThread) + loadRow;
            const size_t bRow = tileStart + loadRow;
            aTile[loadRow][threadIdx.x] =
                (aRow < N && tileStart + threadIdx.x < N) ? A[aRow * N + tileStart + threadIdx.x] : 0.0;
            bTile[loadRow][threadIdx.x] =
                (bRow < N && column < N) ? B[bRow * N + column] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < kTile; ++k) {
            const double b = bTile[k][threadIdx.x];
            #pragma unroll
            for (int r = 0; r < kRowsPerThread; ++r) {
                sums[r] += aTile[threadIdx.y + r * blockDim.y][k] * b;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < kRowsPerThread; ++r) {
        const size_t row = rowBase + r * blockDim.y;
        if (row < N && column < N) {
            C[row * N + column] = sums[r];
        }
    }
}

}  // namespace

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
    cudaEvent_t start = nullptr, end = nullptr;

    // Establish the CUDA context outside the timed region.
    checkCuda(cudaFree(nullptr), "context initialization");
    checkCuda(cudaMalloc(&deviceA, bytes), "allocation of A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocation of B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocation of C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copy of A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copy of B");
    checkCuda(cudaEventCreate(&start), "creation of start event");
    checkCuda(cudaEventCreate(&end), "creation of end event");

    const dim3 threads(kTile, kTile / kRowsPerThread);
    const dim3 blocks((N + kTile - 1) / kTile,
                      (N + threads.y * kRowsPerThread - 1) / (threads.y * kRowsPerThread));
    checkCuda(cudaEventRecord(start), "recording of start event");
    matrixMultiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaEventRecord(end), "recording of end event");
    checkCuda(cudaEventSynchronize(end), "kernel completion");

    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, end), "timing of kernel");

    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copy of C");
    checkCuda(cudaEventDestroy(start), "destruction of start event");
    checkCuda(cudaEventDestroy(end), "destruction of end event");
    checkCuda(cudaFree(deviceA), "free of A");
    checkCuda(cudaFree(deviceB), "free of B");
    checkCuda(cudaFree(deviceC), "free of C");
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
    const float elapsedMilliseconds = matrixMultiply(A, B, C, N);
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
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
