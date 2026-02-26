#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

constexpr int kBlockSize = 16;

bool checkCuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(err));
        return false;
    }
    return true;
}

__global__ void matmulKernel(const double* __restrict__ A,
                             const double* __restrict__ B,
                             double* __restrict__ C,
                             int N) {
    __shared__ double As[kBlockSize][kBlockSize];
    __shared__ double Bs[kBlockSize][kBlockSize];

    const int row = blockIdx.y * kBlockSize + threadIdx.y;
    const int col = blockIdx.x * kBlockSize + threadIdx.x;
    const int tiles = (N + kBlockSize - 1) / kBlockSize;
    double sum = 0.0;

    for (int t = 0; t < tiles; ++t) {
        const int tiledCol = t * kBlockSize + threadIdx.x;
        const int tiledRow = t * kBlockSize + threadIdx.y;

        As[threadIdx.y][threadIdx.x] =
            (row < N && tiledCol < N) ? A[row * N + tiledCol] : 0.0;
        Bs[threadIdx.y][threadIdx.x] =
            (tiledRow < N && col < N) ? B[tiledRow * N + col] : 0.0;

        __syncthreads();
#pragma unroll
        for (int k = 0; k < kBlockSize; ++k) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

bool matrixMultiplyCUDA(const std::vector<double>& A, const std::vector<double>& B,
                        std::vector<double>& C, const size_t N, double* elapsedMs) {
    if (N == 0) {
        if (elapsedMs) {
            *elapsedMs = 0.0;
        }
        return true;
    }

    if (N > static_cast<size_t>(std::numeric_limits<int>::max())) {
        fprintf(stderr, "Matrix size too large for CUDA kernel.\n");
        return false;
    }

    const size_t bytes = N * N * sizeof(double);
    double* dA = nullptr;
    double* dB = nullptr;
    double* dC = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    if (!checkCuda(cudaMalloc(&dA, bytes), "cudaMalloc A")) {
        return false;
    }
    if (!checkCuda(cudaMalloc(&dB, bytes), "cudaMalloc B")) {
        cudaFree(dA);
        return false;
    }
    if (!checkCuda(cudaMalloc(&dC, bytes), "cudaMalloc C")) {
        cudaFree(dA);
        cudaFree(dB);
        return false;
    }

    auto cleanup = [&]() {
        if (start) {
            cudaEventDestroy(start);
        }
        if (stop) {
            cudaEventDestroy(stop);
        }
        if (dA) {
            cudaFree(dA);
        }
        if (dB) {
            cudaFree(dB);
        }
        if (dC) {
            cudaFree(dC);
        }
    };

    if (!checkCuda(cudaMemcpy(dA, A.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy A H2D") ||
        !checkCuda(cudaMemcpy(dB, B.data(), bytes, cudaMemcpyHostToDevice), "cudaMemcpy B H2D")) {
        cleanup();
        return false;
    }

    if (!checkCuda(cudaEventCreate(&start), "cudaEventCreate start") ||
        !checkCuda(cudaEventCreate(&stop), "cudaEventCreate stop")) {
        cleanup();
        return false;
    }

    const int n = static_cast<int>(N);
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((n + kBlockSize - 1) / kBlockSize,
                    (n + kBlockSize - 1) / kBlockSize);

    if (!checkCuda(cudaEventRecord(start), "cudaEventRecord start")) {
        cleanup();
        return false;
    }
    matmulKernel<<<grid, block>>>(dA, dB, dC, n);
    if (!checkCuda(cudaGetLastError(), "matmulKernel launch")) {
        cleanup();
        return false;
    }
    if (!checkCuda(cudaEventRecord(stop), "cudaEventRecord stop") ||
        !checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop")) {
        cleanup();
        return false;
    }

    float ms = 0.0f;
    if (!checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime")) {
        cleanup();
        return false;
    }
    if (elapsedMs) {
        *elapsedMs = static_cast<double>(ms);
    }

    if (!checkCuda(cudaMemcpy(C.data(), dC, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy C D2H")) {
        cleanup();
        return false;
    }

    cleanup();
    return true;
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
    double elapsedMs = 0.0;
    if (!matrixMultiplyCUDA(A, B, C, N, &elapsedMs)) {
        fprintf(stderr, "CUDA matrix multiplication failed.\n");
        return 1;
    }

    const long elapsedMsRounded = static_cast<long>(std::llround(elapsedMs));
    printf("Computation time: %ld ms\n", elapsedMsRounded);

    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (elapsedMsRounded / 1000.0) / 1e9;
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
