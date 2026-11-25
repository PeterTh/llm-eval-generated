#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initMatrix(double* matrix, size_t n) {
    const size_t row = blockIdx.x;
    for (size_t col = blockIdx.y * blockDim.x + threadIdx.x; col < n;
         col += gridDim.y * blockDim.x) {
        matrix[row * n + col] = getPseudoRndValue(n, row, col);
    }
}

constexpr int tile = 32;
constexpr int blockRows = 8;

// A warp covers consecutive columns. Each lane reuses one right-hand
// operand across four output rows, accumulating those rows in registers.
template<bool fullTiles, bool exactProducts = false>
__global__ void matrixMultiply(const double* input, double* output, size_t n) {
    __shared__ double left[tile][tile];
    __shared__ double right[tile][tile];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t col = size_t(blockIdx.x) * tile + x;
    const size_t firstRow = size_t(blockIdx.y) * tile + y;
    double sums[tile / blockRows] = {};

    const size_t paired = fullTiles ? n : n - n % 2;
    for (size_t base = 0; base < paired; base += tile) {
        #pragma unroll
        for (int r = 0; r < tile / blockRows; ++r) {
            const int offset = r * blockRows;
            const size_t row = firstRow + offset;
            left[y + offset][x] = (fullTiles || (row < n && base + x < paired))
                ? input[row * n + base + x] : 0.0;
            right[y + offset][x] = (fullTiles || (base + y + offset < paired && col < n))
                ? input[(base + y + offset) * n + col] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < tile; ++k) {
            #pragma unroll
            for (int r = 0; r < tile / blockRows; ++r) {
                const double a = left[y + r * blockRows][k];
                const double b = right[k][x];
                if constexpr (exactProducts)
                    sums[r] = __fma_rn(a, b, sums[r]);
                else
                    sums[r] += __dmul_rn(a, b);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < tile / blockRows; ++r) {
        const size_t row = firstRow + r * blockRows;
        if (fullTiles || (row < n && col < n)) {
            // Match the fused scalar tail of the original optimized dot product.
            if (!fullTiles && paired != n)
                sums[r] = __fma_rn(input[row * n + paired], input[paired * n + col], sums[r]);
            output[row * n + col] = sums[r];
        }
    }
}

bool validateResult(const std::vector<double>& C, const size_t N, size_t stride) {
    if (N == 0) return true;
    // Check the same leading rows and columns as the serial benchmark.
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }
            
            const double actual = C[i * stride + j];
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
    
    if (N && N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    const size_t bytes = N * N * sizeof(double);
    double* input = nullptr;
    double* output = nullptr;
    printf("Initializing matrices...\n");
    if (N) {
        checkCuda(cudaMalloc(&input, bytes), "Allocate input matrix");
        checkCuda(cudaMalloc(&output, bytes), "Allocate result matrix");
        initMatrix<<<dim3(N, std::min<size_t>((N + 255) / 256, 65535)), 256>>>(input, N);
        checkCuda(cudaGetLastError(), "Launch initialization");
        checkCuda(cudaDeviceSynchronize(), "Initialize matrix");
    }
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::steady_clock::now();
    if (N) {
        const dim3 grid((N + tile - 1) / tile, (N + tile - 1) / tile);
        const dim3 block(tile, blockRows);
        // For these powers of two each input is a binary fraction with at
        // most 26 significant bits. Its product is exact in double precision.
        if (N >= tile && N <= 8192 && (N & (N - 1)) == 0)
            matrixMultiply<true, true><<<grid, block>>>(input, output, N);
        else if (N % tile == 0)
            matrixMultiply<true><<<grid, block>>>(input, output, N);
        else
            matrixMultiply<false><<<grid, block>>>(input, output, N);
        checkCuda(cudaGetLastError(), "Launch multiplication");
        checkCuda(cudaDeviceSynchronize(), "Multiply matrices");
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    printf("Computation time: %.3f ms\n", elapsed.count() * 1000.0);
    const double gflops = elapsed.count() > 0 ? 2.0 * N * N * N / elapsed.count() / 1e9 : 0;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    std::vector<double> C;
    const size_t extent = printResults ? N : std::min<size_t>(N, 5);
    if (extent && (printResults || validate)) {
        C.resize(extent * extent);
        const size_t width = extent * sizeof(double);
        checkCuda(cudaMemcpy2D(C.data(), width, output, N * sizeof(double),
                               width, extent, cudaMemcpyDeviceToHost), "Copy result");
    }
    if (N) {
        checkCuda(cudaFree(output), "Free result matrix");
        checkCuda(cudaFree(input), "Free input matrix");
    }

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(C, N, extent);
        
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
