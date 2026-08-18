#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int THREADS_X = 32;
constexpr int THREADS_Y = 8;
constexpr int ROWS_PER_THREAD = TILE_M / THREADS_Y;
constexpr int COLS_PER_THREAD = TILE_N / THREADS_X;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     const size_t N) {
    __shared__ double tileA[TILE_M][TILE_K];
    __shared__ double tileB[TILE_K][TILE_N];

    const int thread = threadIdx.y * THREADS_X + threadIdx.x;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE_M +
                           static_cast<size_t>(threadIdx.y) * ROWS_PER_THREAD;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE_N +
                           static_cast<size_t>(threadIdx.x) * COLS_PER_THREAD;

    double accum[ROWS_PER_THREAD][COLS_PER_THREAD] = {};

    const size_t tileCount = (N + TILE_K - 1) / TILE_K;
    for (size_t tile = 0; tile < tileCount; ++tile) {
        // Each thread loads four elements of each shared-memory tile. The
        // 32-thread x dimension makes both global loads coalesced.
        #pragma unroll
        for (int element = thread; element < TILE_M * TILE_K;
             element += THREADS_X * THREADS_Y) {
            const int tileRow = element / TILE_K;
            const int tileCol = element % TILE_K;
            const size_t row = static_cast<size_t>(blockIdx.y) * TILE_M + tileRow;
            const size_t col = tile * TILE_K + tileCol;
            tileA[tileRow][tileCol] =
                (row < N && col < N) ? A[row * N + col] : 0.0;
        }

        #pragma unroll
        for (int element = thread; element < TILE_K * TILE_N;
             element += THREADS_X * THREADS_Y) {
            const int tileRow = element / TILE_N;
            const int tileCol = element % TILE_N;
            const size_t row = tile * TILE_K + tileRow;
            const size_t col = static_cast<size_t>(blockIdx.x) * TILE_N + tileCol;
            tileB[tileRow][tileCol] =
                (row < N && col < N) ? B[row * N + col] : 0.0;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_K; ++k) {
            double a[ROWS_PER_THREAD];
            #pragma unroll
            for (int row = 0; row < ROWS_PER_THREAD; ++row) {
                a[row] = tileA[threadIdx.y * ROWS_PER_THREAD + row][k];
            }

            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                const double b = tileB[k][threadIdx.x * COLS_PER_THREAD + col];
                #pragma unroll
                for (int row = 0; row < ROWS_PER_THREAD; ++row) {
                    accum[row][col] += a[row] * b;
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int row = 0; row < ROWS_PER_THREAD; ++row) {
        const size_t outputRow = rowBase + row;
        if (outputRow < N) {
            #pragma unroll
            for (int col = 0; col < COLS_PER_THREAD; ++col) {
                const size_t outputCol = colBase + col;
                if (outputCol < N) {
                    C[outputRow * N + outputCol] = accum[row][col];
                }
            }
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

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B, 
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const size_t bytes = N * N * sizeof(double);
    double* deviceA = nullptr;
    double* deviceB = nullptr;
    double* deviceC = nullptr;

    checkCuda(cudaMalloc(&deviceA, bytes), "allocating matrix A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocating matrix B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocating matrix C");

    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix A to the GPU");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice),
              "copying matrix B to the GPU");

    const dim3 block(THREADS_X, THREADS_Y);
    const dim3 grid(static_cast<unsigned int>((N + TILE_N - 1) / TILE_N),
                    static_cast<unsigned int>((N + TILE_M - 1) / TILE_M));
    matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication kernel");
    checkCuda(cudaDeviceSynchronize(), "running matrix multiplication kernel");

    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost),
              "copying matrix C to the host");

    checkCuda(cudaFree(deviceC), "freeing matrix C");
    checkCuda(cudaFree(deviceB), "freeing matrix B");
    checkCuda(cudaFree(deviceA), "freeing matrix A");
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

    // Keep one-time CUDA context creation out of the multiplication timing.
    checkCuda(cudaFree(nullptr), "initializing CUDA context");
    
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
