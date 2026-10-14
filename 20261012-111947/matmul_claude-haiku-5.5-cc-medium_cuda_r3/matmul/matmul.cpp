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
        const cudaError_t err_ = (call);                                        \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// GPU tiling: each thread block computes a kTile x kTile tile of C using a
// kBlockDim x kBlockDim thread grid; each thread computes a kThreadTile x
// kThreadTile sub-block of that tile. A and B are staged through shared memory
// in kBlockK-deep slabs. Each C element is still accumulated in ascending k
// order, so the result matches the sequential reference.
constexpr size_t kThreadTile = 4;
constexpr size_t kBlockDim = 16;
constexpr size_t kTile = kBlockDim * kThreadTile;
constexpr size_t kBlockK = 16;

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

__global__ void matrixMultiplyKernel(const double* __restrict__ A, const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    __shared__ double As[kTile][kBlockK];
    __shared__ double Bs[kBlockK][kTile];

    const size_t tx = threadIdx.x;
    const size_t ty = threadIdx.y;
    const size_t tid = ty * kBlockDim + tx;
    const size_t rowBase = blockIdx.y * kTile;
    const size_t colBase = blockIdx.x * kTile;

    // Thread (tx, ty) owns rows ty + kBlockDim * i and columns tx + kBlockDim * j
    // (interleaved so that a warp reads consecutive shared-memory doubles).
    double acc[kThreadTile][kThreadTile] = {};

    for (size_t kt = 0; kt < N; kt += kBlockK) {
        for (size_t idx = tid; idx < kTile * kBlockK; idx += kBlockDim * kBlockDim) {
            const size_t r = idx / kBlockK;
            const size_t c = idx % kBlockK;
            As[r][c] = (rowBase + r < N && kt + c < N) ? A[(rowBase + r) * N + kt + c] : 0.0;
        }
        for (size_t idx = tid; idx < kBlockK * kTile; idx += kBlockDim * kBlockDim) {
            const size_t r = idx / kTile;
            const size_t c = idx % kTile;
            Bs[r][c] = (kt + r < N && colBase + c < N) ? B[(kt + r) * N + colBase + c] : 0.0;
        }
        __syncthreads();

        const size_t kEnd = (N - kt < kBlockK) ? (N - kt) : kBlockK;
        for (size_t k = 0; k < kEnd; ++k) {
            double a[kThreadTile];
            double b[kThreadTile];
            for (size_t i = 0; i < kThreadTile; ++i) {
                a[i] = As[ty + i * kBlockDim][k];
                b[i] = Bs[k][tx + i * kBlockDim];
            }
            for (size_t i = 0; i < kThreadTile; ++i) {
                for (size_t j = 0; j < kThreadTile; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        __syncthreads();
    }

    for (size_t i = 0; i < kThreadTile; ++i) {
        const size_t row = rowBase + ty + i * kBlockDim;
        if (row >= N) continue;
        for (size_t j = 0; j < kThreadTile; ++j) {
            const size_t col = colBase + tx + j * kBlockDim;
            if (col < N) {
                C[row * N + col] = acc[i][j];
            }
        }
    }
}

// Device-side multiply: A, B, C are device pointers to N x N row-major matrices.
void matrixMultiply(const double* dA, const double* dB, double* dC, const size_t N) {
    if (N == 0) return;
    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid(static_cast<unsigned int>((N + kTile - 1) / kTile),
                    static_cast<unsigned int>((N + kTile - 1) / kTile));
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
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

    // Allocate device memory and upload inputs (not included in the timing)
    const size_t bytes = N * N * sizeof(double);
    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));
    CUDA_CHECK(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice));

    // Perform matrix multiplication on the GPU, then copy the result back to host
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiply(dA, dB, dC, N);
    CUDA_CHECK(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost));

    auto end = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
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
