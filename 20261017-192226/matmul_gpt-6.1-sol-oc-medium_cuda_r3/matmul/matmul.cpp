#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block computes 64x64 outputs. Coalesced cooperative loads reuse each
// input tile across 256 threads; each thread holds a 4x4 output tile in registers.
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, size_t N) {
    constexpr int tile = 64;
    constexpr int depth = 16;
    __shared__ double aTile[tile][depth + 1];
    __shared__ double bTile[depth][tile];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t blockRow = static_cast<size_t>(blockIdx.y) * tile;
    const size_t blockCol = static_cast<size_t>(blockIdx.x) * tile;
    double sums[4][4] = {};

    for (size_t base = 0; base < N; base += depth) {
        #pragma unroll
        for (int t = tid; t < tile * depth; t += 256) {
            const int ar = t / depth;
            const int ak = t % depth;
            const int bk = t / tile;
            const int bc = t % tile;
            aTile[ar][ak] = blockRow + ar < N && base + ak < N
                ? A[(blockRow + ar) * N + base + ak] : 0.0;
            bTile[bk][bc] = base + bk < N && blockCol + bc < N
                ? B[(base + bk) * N + blockCol + bc] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < depth; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                a[r] = aTile[ty + r * 16][k];
                b[r] = bTile[k][tx + r * 16];
            }
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    // --fmad=false retains the original multiply/add rounding.
                    sums[r][c] += a[r] * b[c];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t row = blockRow + ty + r * 16;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t col = blockCol + tx + c * 16;
            if (row < N && col < N) C[row * N + col] = sums[r][c];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes), "allocate A");
    checkCuda(cudaMalloc(&deviceB, bytes), "allocate B");
    checkCuda(cudaMalloc(&deviceC, bytes), "allocate C");
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice), "copy A");
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice), "copy B");

    const dim3 threads(16, 16);
    const dim3 blocks(static_cast<unsigned>((N + 63) / 64),
                      static_cast<unsigned>((N + 63) / 64));
    matrixMultiplyKernel<<<blocks, threads>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError(), "launch matrix multiplication");
    // The synchronous copy also waits for the kernel and reports execution errors.
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost), "copy C");
    checkCuda(cudaFree(deviceC), "free C");
    checkCuda(cudaFree(deviceB), "free B");
    checkCuda(cudaFree(deviceA), "free A");
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
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno != 0 || value[0] == '-' || end == value || *end != '\0' ||
                parsed == 0 || parsed > std::numeric_limits<size_t>::max()) {
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
    
    if (N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    // Initialize CUDA before timing, but include allocation and all transfers
    // in the measured multiplication time.
    checkCuda(cudaFree(nullptr), "initialize runtime");
    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "get device");
    checkCuda(cudaGetDeviceProperties(&properties, device), "get device properties");
    if ((N + 63) / 64 > static_cast<size_t>(properties.maxGridSize[1])) {
        fprintf(stderr, "Matrix size exceeds CUDA grid limits\n");
        return 1;
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
    auto start = std::chrono::high_resolution_clock::now();
    
    matrixMultiply(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) /
        std::chrono::duration<double>(end - start).count() / 1e9;
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
