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

__global__ __launch_bounds__(256) void matrixMultiplyKernel(
    const double* __restrict__ A,
    const double* __restrict__ B,
    double* __restrict__ C,
    const int N) {

    __shared__ double tileA[32][33];
    __shared__ double tileB[32][33];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int row0 = blockIdx.y * 32 + ty;
    const int row1 = row0 + 16;
    const int col0 = blockIdx.x * 32 + tx;
    const int col1 = col0 + 16;

    double acc00 = 0.0;
    double acc01 = 0.0;
    double acc10 = 0.0;
    double acc11 = 0.0;

    const int numTiles = (N + 31) / 32;

    for (int t = 0; t < numTiles; ++t) {
        const int k0 = t * 32;

        const int aRow0 = row0;
        const int aRow1 = row1;
        const int aCol0 = k0 + tx;
        const int aCol1 = k0 + tx + 16;

        tileA[ty][tx]      = (aRow0 < N && aCol0 < N) ? A[aRow0 * N + aCol0] : 0.0;
        tileA[ty][tx + 16] = (aRow0 < N && aCol1 < N) ? A[aRow0 * N + aCol1] : 0.0;
        tileA[ty + 16][tx]      = (aRow1 < N && aCol0 < N) ? A[aRow1 * N + aCol0] : 0.0;
        tileA[ty + 16][tx + 16] = (aRow1 < N && aCol1 < N) ? A[aRow1 * N + aCol1] : 0.0;

        const int bRow0 = k0 + ty;
        const int bRow1 = k0 + ty + 16;
        const int bCol0 = blockIdx.x * 32 + tx;
        const int bCol1 = bCol0 + 16;

        tileB[ty][tx]      = (bRow0 < N && bCol0 < N) ? B[bRow0 * N + bCol0] : 0.0;
        tileB[ty][tx + 16] = (bRow0 < N && bCol1 < N) ? B[bRow0 * N + bCol1] : 0.0;
        tileB[ty + 16][tx]      = (bRow1 < N && bCol0 < N) ? B[bRow1 * N + bCol0] : 0.0;
        tileB[ty + 16][tx + 16] = (bRow1 < N && bCol1 < N) ? B[bRow1 * N + bCol1] : 0.0;

        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            const double a0 = tileA[ty][kk];
            const double a1 = tileA[ty + 16][kk];
            const double b0 = tileB[kk][tx];
            const double b1 = tileB[kk][tx + 16];

            acc00 += a0 * b0;
            acc01 += a0 * b1;
            acc10 += a1 * b0;
            acc11 += a1 * b1;
        }

        __syncthreads();
    }

    if (row0 < N) {
        if (col0 < N) C[row0 * N + col0] = acc00;
        if (col1 < N) C[row0 * N + col1] = acc01;
    }
    if (row1 < N) {
        if (col0 < N) C[row1 * N + col0] = acc10;
        if (col1 < N) C[row1 * N + col1] = acc11;
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return;
    }

    const size_t bytes = N * N * sizeof(double);

    double* dAll = nullptr;
    if (cudaMalloc(&dAll, 3 * bytes) != cudaSuccess) {
        fprintf(stderr, "CUDA memory allocation failed: %s\n", cudaGetErrorString(cudaGetLastError()));
        return;
    }

    double* dA = dAll;
    double* dB = dAll + N * N;
    double* dC = dAll + 2 * N * N;

    cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice);

    const dim3 block(16, 16);
    const dim3 grid(static_cast<unsigned>((N + 31) / 32),
                    static_cast<unsigned>((N + 31) / 32));

    matrixMultiplyKernel<<<grid, block>>>(dA, dB, dC, static_cast<int>(N));

    const cudaError_t launchStatus = cudaGetLastError();
    if (launchStatus != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch failed: %s\n", cudaGetErrorString(launchStatus));
        cudaFree(dAll);
        return;
    }

    cudaDeviceSynchronize();
    cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dAll);
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
    cudaFree(nullptr);
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
