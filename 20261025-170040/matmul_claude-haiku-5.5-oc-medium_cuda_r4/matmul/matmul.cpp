#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Tile edge along i, j and k for the shared-memory matmul kernel.
constexpr size_t TILE = 32;
// Threads per block edge; each thread computes a 2x2 block of C.
constexpr size_t BLOCK = 16;

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

__global__ void initMatrixKernel(double* __restrict__ mat, const size_t N) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        mat[i * N + j] = getPseudoRndValue(N, i, j);
    }
}

// C = A * B, row-major NxN. Each output element is accumulated over k in
// strictly increasing order (tile by tile, then within tile), matching the
// sequential reference summation order.
__global__ void matrixMultiplyKernel(const double* __restrict__ A, const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    __shared__ double As[TILE][TILE];
    __shared__ double Bs[TILE][TILE];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t tid = ty * BLOCK + tx;
    const size_t row = blockIdx.y * TILE + ty * 2;  // rows row, row+1
    const size_t col = blockIdx.x * TILE + tx * 2;  // cols col, col+1

    double s00 = 0.0, s01 = 0.0, s10 = 0.0, s11 = 0.0;

    for (size_t t0 = 0; t0 < N; t0 += TILE) {
        // Cooperative load of one TILE x TILE slab of A and B (zero-padded at edges).
        for (size_t idx = tid; idx < TILE * TILE; idx += BLOCK * BLOCK) {
            const size_t r = idx / TILE;
            const size_t c = idx % TILE;

            const size_t aRow = blockIdx.y * TILE + r;
            const size_t aCol = t0 + c;
            As[r][c] = (aRow < N && aCol < N) ? A[aRow * N + aCol] : 0.0;

            const size_t bRow = t0 + r;
            const size_t bCol = blockIdx.x * TILE + c;
            Bs[r][c] = (bRow < N && bCol < N) ? B[bRow * N + bCol] : 0.0;
        }
        __syncthreads();

        const size_t kMax = min(TILE, N - t0);
        for (size_t k = 0; k < kMax; ++k) {
            const double a0 = As[ty * 2][k];
            const double a1 = As[ty * 2 + 1][k];
            const double b0 = Bs[k][tx * 2];
            const double b1 = Bs[k][tx * 2 + 1];
            s00 += a0 * b0;
            s01 += a0 * b1;
            s10 += a1 * b0;
            s11 += a1 * b1;
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = s00;
    }
    if (row < N && col + 1 < N) {
        C[row * N + col + 1] = s01;
    }
    if (row + 1 < N && col < N) {
        C[(row + 1) * N + col] = s10;
    }
    if (row + 1 < N && col + 1 < N) {
        C[(row + 1) * N + col + 1] = s11;
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
    
    const size_t elems = N * N;
    const size_t bytes = elems * sizeof(double);

    // Allocate device matrices
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    const dim3 block2d(BLOCK, BLOCK);
    const dim3 initGrid(static_cast<unsigned int>((N + BLOCK - 1) / BLOCK),
                        static_cast<unsigned int>((N + BLOCK - 1) / BLOCK));
    const dim3 mulGrid(static_cast<unsigned int>((N + TILE - 1) / TILE),
                       static_cast<unsigned int>((N + TILE - 1) / TILE));

    // Initialize matrices on the GPU
    printf("Initializing matrices...\n");
    if (N > 0) {
        initMatrixKernel<<<initGrid, block2d>>>(dA, N);
        CUDA_CHECK(cudaGetLastError());
        initMatrixKernel<<<initGrid, block2d>>>(dB, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Perform matrix multiplication on the GPU
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (N > 0) {
        matrixMultiplyKernel<<<mulGrid, block2d>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    int status = 0;

    // Copy results back to host only when they are needed
    std::vector<double> A;
    std::vector<double> B;
    std::vector<double> C;
    if (validate) {
        A.resize(elems);
        B.resize(elems);
        CUDA_CHECK(cudaMemcpy(A.data(), dA, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(B.data(), dB, bytes, cudaMemcpyDeviceToHost));
    }
    if (validate || printResults) {
        C.resize(elems);
        CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));
    }
    
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
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    
    return status;
}
