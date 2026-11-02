#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Each block computes a 32 x 32 output tile, with four results per thread.
// The K dimension is tiled separately so each input is reused from shared memory.
__global__ void matrixMultiply(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t N) {
    constexpr int tile = 32;
    constexpr int depth = 16;
    __shared__ double aTile[tile][depth];
    __shared__ double bTile[depth][tile];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t row0 = static_cast<size_t>(blockIdx.y) * tile + y;
    const size_t row1 = row0 + 16;
    const size_t col0 = static_cast<size_t>(blockIdx.x) * tile + x;
    const size_t col1 = col0 + 16;
    double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;

    for (size_t base = 0; base < N; base += depth) {
        const size_t ak = base + x;
        const size_t bk = base + y;
        aTile[y][x] = row0 < N && ak < N ? A[row0 * N + ak] : 0.0;
        aTile[y + 16][x] = row1 < N && ak < N ? A[row1 * N + ak] : 0.0;
        bTile[y][x] = bk < N && col0 < N ? B[bk * N + col0] : 0.0;
        bTile[y][x + 16] = bk < N && col1 < N ? B[bk * N + col1] : 0.0;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < depth; ++k) {
            const double a0 = aTile[y][k];
            const double a1 = aTile[y + 16][k];
            const double b0 = bTile[k][x];
            const double b1 = bTile[k][x + 16];
            c00 += a0 * b0;
            c01 += a0 * b1;
            c10 += a1 * b0;
            c11 += a1 * b1;
        }
        __syncthreads();
    }
    if (row0 < N && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < N && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < N && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < N && col1 < N) C[row1 * N + col1] = c11;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) return true;
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
    
    // Upload inputs before timing, and time only the GPU multiplication.
    checkCuda(cudaFree(nullptr), "initializing the device");
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    if (bytes != 0) {
        checkCuda(cudaMalloc(&deviceA, bytes), "allocating A");
        checkCuda(cudaMalloc(&deviceB, bytes), "allocating B");
        checkCuda(cudaMalloc(&deviceC, bytes), "allocating C");
        checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "uploading A");
        checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "uploading B");
    }
    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    printf("Computing matrix multiplication...\n");
    checkCuda(cudaEventRecord(start), "recording start event");
    if (N != 0) {
        const dim3 threads(16, 16);
        const dim3 blocks((N + 31) / 32, (N + 31) / 32);
        matrixMultiply<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
        checkCuda(cudaGetLastError(), "launching matrix multiplication");
    }
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "executing matrix multiplication");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, end), "measuring execution time");
    if (bytes != 0) {
        checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "downloading C");
    }
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    checkCuda(cudaFree(deviceA), "freeing A");
    checkCuda(cudaFree(deviceB), "freeing B");
    checkCuda(cudaFree(deviceC), "freeing C");

    printf("Computation time: %ld ms\n", static_cast<long>(milliseconds));
    double gflops = milliseconds > 0.0f ? (2.0 * N * N * N) / (milliseconds * 1.0e6) : 0.0;
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
