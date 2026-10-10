#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

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

// Every CUDA failure is fatal: this benchmark never falls back to the CPU.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceMatrices {
    double *A = nullptr, *B = nullptr, *C = nullptr;

    DeviceMatrices(const std::vector<double>& hostA,
                   const std::vector<double>& hostB) {
        const size_t bytes = hostA.size() * sizeof(double);
        checkCuda(cudaMalloc(&A, bytes), "Allocate A");
        checkCuda(cudaMalloc(&B, bytes), "Allocate B");
        checkCuda(cudaMalloc(&C, bytes), "Allocate C");
        checkCuda(cudaMemcpy(A, hostA.data(), bytes, cudaMemcpyHostToDevice), "Upload A");
        checkCuda(cudaMemcpy(B, hostB.data(), bytes, cudaMemcpyHostToDevice), "Upload B");
    }

    ~DeviceMatrices() {
        cudaFree(C);
        cudaFree(B);
        cudaFree(A);
    }

    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;
};

// Each block reuses input tiles in shared memory; each thread computes a
// 2x4 or 4x4 output tile in registers. Consecutive threads load/store columns.
// Keep the k loop ordered, with separate multiply and add operations, to
// preserve the original double-precision accumulation semantics.
template <int Rows>
__global__ void multiplyTiled(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, size_t N) {
    constexpr int Cols = 64;
    constexpr int Depth = 16;
    constexpr int ThreadRows = Rows / 16;
    __shared__ double tileA[Rows][Depth];
    __shared__ double tileB[Depth][Cols];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * Rows;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * Cols;
    double sums[ThreadRows][4] = {};

    for (size_t base = 0; base < N; base += Depth) {
        for (int index = tid; index < Rows * Depth; index += 256) {
            const int row = index / Depth;
            const int k = index % Depth;
            tileA[row][k] = rowBase + row < N && base + k < N
                         ? A[(rowBase + row) * N + base + k] : 0.0;
        }
        for (int index = tid; index < Depth * Cols; index += 256) {
            const int k = index / Cols;
            const int col = index % Cols;
            tileB[k][col] = base + k < N && colBase + col < N
                         ? B[(base + k) * N + colBase + col] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < Depth; ++k) {
            double a[ThreadRows], b[4];
            #pragma unroll
            for (int i = 0; i < ThreadRows; ++i)
                a[i] = tileA[ty + i * 16][k];
            #pragma unroll
            for (int j = 0; j < 4; ++j)
                b[j] = tileB[k][tx + j * 16];
            #pragma unroll
            for (int i = 0; i < ThreadRows; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j)
                    sums[i][j] += a[i] * b[j];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < ThreadRows; ++i) {
        const size_t row = rowBase + ty + i * 16;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t col = colBase + tx + j * 16;
            if (row < N && col < N)
                C[row * N + col] = sums[i][j];
        }
    }
}

void matrixMultiply(const DeviceMatrices& matrices,
                    std::vector<double>& C, const size_t N) {
    const dim3 threads(16, 16);
    // Smaller tiles supply more independent blocks for small matrices.
    if (N <= 1024) {
        const dim3 blocks((N + 63) / 64, (N + 31) / 32);
        multiplyTiled<32><<<blocks, threads>>>(matrices.A, matrices.B, matrices.C, N);
    } else {
        const dim3 blocks((N + 63) / 64, (N + 63) / 64);
        multiplyTiled<64><<<blocks, threads>>>(matrices.A, matrices.B, matrices.C, N);
    }
    checkCuda(cudaGetLastError(), "Launch matrix multiplication");
    // The blocking copy also waits for the kernel, so timing includes completed
    // computation and making the full result available to the caller.
    checkCuda(cudaMemcpy(C.data(), matrices.C, C.size() * sizeof(double),
                         cudaMemcpyDeviceToHost), "Download C");
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
            char* end = nullptr;
            const char* value = argv[++i];
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
            if (*value == '-' || *value == '\0' || *end != '\0' || parsed == 0 ||
                parsed > maxElements / parsed) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<size_t>(parsed);
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
    
    DeviceMatrices matrices(A, B);

    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(matrices, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / (milliseconds / 1000.0) / 1e9;
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
