#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

__global__ void initMatrix(double* matrix, size_t n, size_t elements) {
    for (size_t index = blockIdx.x * size_t(blockDim.x) + threadIdx.x;
         index < elements; index += size_t(gridDim.x) * blockDim.x) {
        matrix[index] = getPseudoRndValue(n, index / n, index % n);
    }
}

// Each warp computes four rows of 32 adjacent columns. The block shares
// both input tiles, and keeps each output's running sum in a register.
constexpr unsigned tileRows = 32;
constexpr unsigned tileCols = 32;
constexpr unsigned tileDepth = 16;
constexpr unsigned threadRows = 8;

template<bool exactProducts>
__global__ void matrixMultiply(const double* input, double* output,
                               size_t n, size_t firstRow, size_t rows) {
    __shared__ double left[tileRows][tileDepth];
    __shared__ double right[tileDepth][tileCols];
    const unsigned x = threadIdx.x;
    const unsigned y = threadIdx.y;
    const unsigned tid = y * tileCols + x;
    const size_t baseRow = size_t(blockIdx.y) * tileRows;
    const size_t col = size_t(blockIdx.x) * tileCols + x;
    double sum[tileRows / threadRows] = {};

    // Match the original optimized dot product: pairs use rounded products
    // and an odd trailing element uses a scalar fused multiply-add.
    const size_t pairedDepth = n - n % 2;
    for (size_t baseK = 0; baseK < pairedDepth; baseK += tileDepth) {
        #pragma unroll
        for (unsigned t = tid; t < tileRows * tileDepth; t += threadRows * tileCols) {
            unsigned r = t / tileDepth;
            unsigned k = t % tileDepth;
            left[r][k] = baseRow + r < rows && baseK + k < pairedDepth
                ? input[(firstRow + baseRow + r) * n + baseK + k] : 0.0;
        }
        #pragma unroll
        for (unsigned k = y; k < tileDepth; k += threadRows) {
            right[k][x] = baseK + k < pairedDepth && col < n ? input[(baseK + k) * n + col] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < tileDepth; ++k) {
            #pragma unroll
            for (unsigned i = 0; i < tileRows / threadRows; ++i) {
                double a = left[y + i * threadRows][k];
                double b = right[k][x];
                if (exactProducts) {
                    sum[i] = fma(a, b, sum[i]);
                } else {
                    sum[i] += __dmul_rn(a, b);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned i = 0; i < tileRows / threadRows; ++i) {
        size_t r = baseRow + y + i * threadRows;
        if (r < rows && col < n) {
            if (n % 2) {
                sum[i] = fma(input[(firstRow + r) * n + pairedDepth],
                             input[pairedDepth * n + col], sum[i]);
            }
            output[r * n + col] = sum[i];
        }
    }
}

struct Partition {
    size_t firstRow;
    size_t rows;
    double* input;
    double* output;
};

bool validateResult(const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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
    
    if (N && N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "No CUDA device available\n");
        return 1;
    }
    // Use additional GPUs once each has a substantial band of output tiles.
    const int count = std::min<size_t>(devices, std::max<size_t>(1, N / 1024));
    std::vector<Partition> parts(count);
    std::vector<double> C;
    if (printResults || validate) C.resize(N * N);

    printf("Initializing matrices...\n");
    const size_t tileCount = (N + tileRows - 1) / tileRows;
    for (int d = 0; d < count && N; ++d) {
        checkCuda(cudaSetDevice(d));
        Partition& p = parts[d];
        p.firstRow = (tileCount * d / count) * tileRows;
        p.rows = std::min(N, (tileCount * (d + 1) / count) * tileRows) - p.firstRow;
        // Both operands have identical initialization and are read-only.
        checkCuda(cudaMalloc(&p.input, N * N * sizeof(double)));
        checkCuda(cudaMalloc(&p.output, p.rows * N * sizeof(double)));
        unsigned blocks = std::min<size_t>(65535, (N * N + 255) / 256);
        initMatrix<<<blocks, 256>>>(p.input, N, N * N);
        checkCuda(cudaGetLastError());
    }
    for (int d = 0; d < count; ++d) {
        checkCuda(cudaSetDevice(d));
        checkCuda(cudaDeviceSynchronize());
    }

    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::steady_clock::now();
    for (int d = 0; d < count && N; ++d) {
        checkCuda(cudaSetDevice(d));
        const Partition& p = parts[d];
        dim3 grid((N + tileCols - 1) / tileCols, (p.rows + tileRows - 1) / tileRows);
        dim3 block(tileCols, threadRows);
        // For these dyadic inputs each product fits exactly in a double.
        if ((N & (N - 1)) == 0 && N <= 8192) {
            matrixMultiply<true><<<grid, block>>>(p.input, p.output, N, p.firstRow, p.rows);
        } else {
            matrixMultiply<false><<<grid, block>>>(p.input, p.output, N, p.firstRow, p.rows);
        }
        checkCuda(cudaGetLastError());
    }
    for (int d = 0; d < count; ++d) {
        checkCuda(cudaSetDevice(d));
        checkCuda(cudaDeviceSynchronize());
    }
    std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    printf("Computation time: %ld ms\n",
           static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));
    printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / elapsed.count() / 1e9);

    for (int d = 0; d < count && N; ++d) {
        checkCuda(cudaSetDevice(d));
        const Partition& p = parts[d];
        if (!C.empty()) {
            checkCuda(cudaMemcpy(C.data() + p.firstRow * N, p.output, p.rows * N * sizeof(double),
                                 cudaMemcpyDeviceToHost));
        }
        checkCuda(cudaFree(p.output));
        checkCuda(cudaFree(p.input));
    }

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = (N == 0 || validateResult(C, N));
        
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
