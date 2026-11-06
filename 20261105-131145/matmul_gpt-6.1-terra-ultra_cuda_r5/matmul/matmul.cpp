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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block computes a 64x64 tile.  The 16x16 threads each keep a 4x4
// patch of C in registers and reuse tiles of A and B from shared memory.
template <bool fullTiles>
__global__ void multiplyKernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t N) {
    constexpr int tile = 64;
    constexpr int depth = 16;
    __shared__ double aTile[tile][depth];
    __shared__ double bTile[depth][tile];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * tile;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * tile;
    double sums[4][4] = {};

    for (size_t base = 0; base < N; base += depth) {
        #pragma unroll
        for (int load = 0; load < 4; ++load) {
            const int index = tid + load * 256;
            const int ar = index / depth;
            const int ak = index % depth;
            const int bk = index / tile;
            const int bc = index % tile;
            aTile[ar][ak] = (fullTiles || (rowBase + ar < N && base + ak < N))
                ? A[(rowBase + ar) * N + base + ak] : 0.0;
            bTile[bk][bc] = (fullTiles || (base + bk < N && colBase + bc < N))
                ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < depth; ++k) {
            double av[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                av[i] = aTile[ty + i * 16][k];
                bv[i] = bTile[k][tx + i * 16];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sums[i][j] += av[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t row = rowBase + ty + i * 16;
            const size_t col = colBase + tx + j * 16;
            if (fullTiles || (row < N && col < N)) C[row * N + col] = sums[i][j];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) return;
    const size_t bytes = A.size() * sizeof(double);
    double *dA, *dB, *dC;
    checkCuda(cudaMalloc(&dA, bytes), "allocating A");
    checkCuda(cudaMalloc(&dB, bytes), "allocating B");
    checkCuda(cudaMalloc(&dC, bytes), "allocating C");
    checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "copying A to GPU");
    checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "copying B to GPU");
    const dim3 block(16, 16);
    const dim3 grid((N + 63) / 64, (N + 63) / 64);
    if (N % 64 == 0) multiplyKernel<true><<<grid, block>>>(dA, dB, dC, N);
    else multiplyKernel<false><<<grid, block>>>(dA, dB, dC, N);
    checkCuda(cudaGetLastError(), "launching matrix multiplication");
    checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "copying C from GPU");
    checkCuda(cudaFree(dA), "freeing A");
    checkCuda(cudaFree(dB), "freeing B");
    checkCuda(cudaFree(dC), "freeing C");
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

    // Context creation is one-time setup, like matrix initialization, and is
    // not part of the multiplication measured below.
    checkCuda(cudaFree(nullptr), "initializing CUDA");
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
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
