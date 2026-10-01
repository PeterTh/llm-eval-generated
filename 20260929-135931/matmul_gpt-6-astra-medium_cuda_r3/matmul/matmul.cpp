#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cerrno>
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

static void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Each block computes 64x64 outputs; each thread holds a 4x4 register tile.
// Transposing A in shared memory avoids strided shared-memory reads.
template <bool fullTiles>
__global__ void multiplyKernel(const double* __restrict__ A,
                               const double* __restrict__ B,
                               double* __restrict__ C, size_t N) {
    __shared__ double tileA[16][65];
    __shared__ double tileB[16][64];
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const unsigned tid = ty * 16 + tx;
    const size_t rowBase = size_t(blockIdx.y) * 64;
    const size_t colBase = size_t(blockIdx.x) * 64;
    double sums[4][4] = {};
    for (size_t base = 0; base < N; base += 16) {
        #pragma unroll
        for (unsigned offset = 0; offset < 1024; offset += 256) {
            const unsigned index = tid + offset;
            const unsigned ar = index / 16, ak = index % 16;
            const unsigned bk = index / 64, bc = index % 64;
            tileA[ak][ar] = (fullTiles || (rowBase + ar < N && base + ak < N))
                ? A[(rowBase + ar) * N + base + ak] : 0.0;
            tileB[bk][bc] = (fullTiles || (base + bk < N && colBase + bc < N))
                ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < 16; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                a[i] = tileA[k][ty + i * 16];
                b[i] = tileB[k][tx + i * 16];
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j)
                    sums[i][j] = fma(a[i], b[j], sums[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const size_t row = rowBase + ty + i * 16;
            const size_t col = colBase + tx + j * 16;
            if (fullTiles || (row < N && col < N)) C[row * N + col] = sums[i][j];
        }
    }
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    const size_t bytes = N * N * sizeof(double);
    double *deviceA = nullptr, *deviceB = nullptr, *deviceC = nullptr;
    checkCuda(cudaMalloc(&deviceA, bytes));
    checkCuda(cudaMalloc(&deviceB, bytes));
    checkCuda(cudaMalloc(&deviceC, bytes));
    checkCuda(cudaMemcpy(deviceA, A.data(), bytes, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(deviceB, B.data(), bytes, cudaMemcpyHostToDevice));
    const dim3 block(16, 16);
    const dim3 grid((N + 63) / 64, (N + 63) / 64);
    if (N % 64 == 0)
        multiplyKernel<true><<<grid, block>>>(deviceA, deviceB, deviceC, N);
    else
        multiplyKernel<false><<<grid, block>>>(deviceA, deviceB, deviceC, N);
    checkCuda(cudaGetLastError());
    // The blocking copy also waits for kernel completion and reports failures.
    checkCuda(cudaMemcpy(C.data(), deviceC, bytes, cudaMemcpyDeviceToHost));
    checkCuda(cudaFree(deviceC));
    checkCuda(cudaFree(deviceB));
    checkCuda(cudaFree(deviceA));
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
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Matrix size must be a positive integer.\n");
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
    
    if (N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        fprintf(stderr, "Matrix size is too large.\n");
        return 1;
    }
    // Require a CUDA device and initialize its context before timing the work.
    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device));
    checkCuda(cudaGetDeviceProperties(&properties, device));
    if ((N + 63) / 64 > static_cast<size_t>(properties.maxGridSize[1])) {
        fprintf(stderr, "Matrix size exceeds the CUDA grid limit.\n");
        return 1;
    }
    checkCuda(cudaFree(nullptr));

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
