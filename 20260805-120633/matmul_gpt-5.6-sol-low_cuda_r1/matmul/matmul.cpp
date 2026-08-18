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

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(error_));                               \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (false)

// Each 16x16 block computes a 32x32 tile.  Four results per thread increase
// instruction-level parallelism and halve the number of threads needed while
// retaining fully coalesced global-memory accesses.
__global__ __launch_bounds__(256)
void matrixMultiplyKernel(const double* __restrict__ A,
                          const double* __restrict__ B,
                          double* __restrict__ C, const size_t N) {
    constexpr int TILE = 32;
    constexpr int STEP = 16;
    __shared__ double aTile[TILE][STEP];
    __shared__ double bTile[STEP][TILE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t row0 = static_cast<size_t>(blockIdx.y) * TILE + ty;
    const size_t row1 = row0 + STEP;
    const size_t col0 = static_cast<size_t>(blockIdx.x) * TILE + tx;
    const size_t col1 = col0 + STEP;
    double c00 = 0.0, c01 = 0.0, c10 = 0.0, c11 = 0.0;

    for (size_t base = 0; base < N; base += STEP) {
        const size_t ak = base + tx;
        const size_t bk = base + ty;
        aTile[ty][tx] = (row0 < N && ak < N) ? A[row0 * N + ak] : 0.0;
        aTile[ty + STEP][tx] = (row1 < N && ak < N) ? A[row1 * N + ak] : 0.0;
        bTile[ty][tx] = (bk < N && col0 < N) ? B[bk * N + col0] : 0.0;
        bTile[ty][tx + STEP] = (bk < N && col1 < N) ? B[bk * N + col1] : 0.0;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < STEP; ++k) {
            const double a0 = aTile[ty][k];
            const double a1 = aTile[ty + STEP][k];
            c00 = fma(a0, bTile[k][tx], c00);
            c01 = fma(a0, bTile[k][tx + STEP], c01);
            c10 = fma(a1, bTile[k][tx], c10);
            c11 = fma(a1, bTile[k][tx + STEP], c11);
        }
        __syncthreads();
    }
    if (row0 < N && col0 < N) C[row0 * N + col0] = c00;
    if (row0 < N && col1 < N) C[row0 * N + col1] = c01;
    if (row1 < N && col0 < N) C[row1 * N + col0] = c10;
    if (row1 < N && col1 < N) C[row1 * N + col1] = c11;
}

void uploadMatrices(const std::vector<double>& A, const std::vector<double>& B,
                    double** dA, double** dB, double** dC, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    CUDA_CHECK(cudaMalloc(dA, bytes));
    CUDA_CHECK(cudaMalloc(dB, bytes));
    CUDA_CHECK(cudaMalloc(dC, bytes));
    CUDA_CHECK(cudaMemcpy(*dA, A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(*dB, B.data(), bytes, cudaMemcpyHostToDevice));
}

void matrixMultiply(const double* dA, const double* dB, double* dC,
                    const size_t N) {
    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned>((N + 31) / 32),
                    static_cast<unsigned>((N + 31) / 32));
    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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

    double *dA = nullptr, *dB = nullptr, *dC = nullptr;
    uploadMatrices(A, B, &dA, &dB, &dC, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(dA, dB, dC, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(C.data(), dC, N * N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dA));
    const std::chrono::duration<double, std::milli> duration = end - start;
    
    printf("Computation time: %.3f ms\n", duration.count());
    
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
