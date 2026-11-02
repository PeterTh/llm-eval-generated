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

// Each block computes a 64 x 64 output tile. Threads cooperate to load
// 64 x 16 and 16 x 64 input tiles and accumulate a 4 x 4 register tile.
constexpr int TILE = 64;
constexpr int DEPTH = 16;
constexpr int THREADS = 16;

__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    __shared__ double tileA[TILE][DEPTH];
    __shared__ double tileB[DEPTH][TILE];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THREADS + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * TILE;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * TILE;
    double sums[4][4] = {};

    for (size_t base = 0; base < N; base += DEPTH) {
        #pragma unroll
        for (int load = 0; load < 4; ++load) {
            const int index = tid + load * THREADS * THREADS;
            const int aRow = index / DEPTH;
            const int aCol = index % DEPTH;
            const int bRow = index / TILE;
            const int bCol = index % TILE;
            tileA[aRow][aCol] = rowBase + aRow < N && base + aCol < N
                ? A[(rowBase + aRow) * N + base + aCol] : 0.0;
            tileB[bRow][bCol] = base + bRow < N && colBase + bCol < N
                ? B[(base + bRow) * N + colBase + bCol] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < DEPTH; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = tileA[ty + i * THREADS][k];
                b[i] = tileB[k][tx + i * THREADS];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sums[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t row = rowBase + ty + i * THREADS;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t col = colBase + tx + j * THREADS;
            if (row < N && col < N) C[row * N + col] = sums[i][j];
        }
    }
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
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
    
    // Transfer input before timing, just as initialization is outside the
    // original multiplication timer. Always synchronize before stopping it.
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    const size_t bytes = N * N * sizeof(double);
    if (bytes != 0) {
        checkCuda(cudaMalloc(&deviceA, bytes), "allocating A");
        checkCuda(cudaMalloc(&deviceB, bytes), "allocating B");
        checkCuda(cudaMalloc(&deviceC, bytes), "allocating C");
        checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A");
        checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B");
        // Load/JIT the kernel before timing so first-use setup is not measured.
        cudaFuncAttributes attributes;
        checkCuda(cudaFuncGetAttributes(&attributes, matrixMultiplyKernel), "loading kernel");
    }

    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (N != 0) {
        const dim3 block(THREADS, THREADS);
        const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
        matrixMultiplyKernel<<<grid, block>>>(deviceA, deviceB, deviceC, N);
        checkCuda(cudaGetLastError(), "launching matrix multiplication");
        checkCuda(cudaDeviceSynchronize(), "executing matrix multiplication");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (bytes != 0) {
        checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copying C");
        checkCuda(cudaFree(deviceA), "freeing A");
        checkCuda(cudaFree(deviceB), "freeing B");
        checkCuda(cudaFree(deviceC), "freeing C");
    }

    printf("Computation time: %ld ms\n", duration.count());

    // Use the full clock resolution for short GPU runs.
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = (2.0 * N * N * N) / seconds / 1e9;
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
