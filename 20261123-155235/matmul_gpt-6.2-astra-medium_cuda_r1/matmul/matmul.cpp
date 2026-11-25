#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr int tile = 32;
constexpr int blockRows = 8;

__global__ void initMatrix(double* input, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n)
        input[row * n + col] = getPseudoRndValue(n, row, col);
}

// A warp shares one row of B. Each lane holds four output rows in registers.
template<bool exactProducts>
__global__ void matrixMultiply(const double* __restrict__ input, double* __restrict__ output,
                               size_t n, size_t firstRow) {
    __shared__ double a[tile][tile];
    __shared__ double b[tile][tile];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t localRow = blockIdx.y * tile + y;
    const size_t row = firstRow + localRow;
    const size_t col = blockIdx.x * tile + x;
    double sums[tile / blockRows] = {};

    // The scalar tail of an odd-length dot product is handled separately.
    const size_t paired = n - n % 2;
    for (size_t base = 0; base < paired; base += tile) {
        #pragma unroll
        for (int r = 0; r < tile; r += blockRows) {
            a[y + r][x] = row + r < n && base + x < paired ? input[(row + r) * n + base + x] : 0.0;
            b[y + r][x] = base + y + r < paired && col < n ? input[(base + y + r) * n + col] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < tile; ++k) {
            const double right = b[k][x];
            #pragma unroll
            for (int i = 0; i < tile / blockRows; ++i) {
                const double left = a[y + i * blockRows][k];
                if constexpr (exactProducts)
                    sums[i] = __fma_rn(left, right, sums[i]);
                else
                    sums[i] += __dmul_rn(left, right);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < tile / blockRows; ++i) {
        const size_t r = row + i * blockRows;
        if (r < n && col < n) {
            if (paired != n)
                sums[i] = __fma_rn(input[r * n + paired], input[paired * n + col], sums[i]);
            output[(localRow + i * blockRows) * n + col] = sums[i];
        }
    }
}

struct DeviceRows {
    double* input;
    double* output;
    size_t first;
    size_t count;
};

bool validateResult(const std::vector<double>& C, const size_t N) {
    if (N == 0) return true;

    // Check the first few rows and columns.
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
    
    if (N && N > std::vector<double>().max_size() / N) {
        fprintf(stderr, "Matrix size is too large.\n");
        return 1;
    }
    printf("Initializing matrices...\n");
    int available = 0;
    checkCuda(cudaGetDeviceCount(&available), "cudaGetDeviceCount");
    if (available == 0) {
        fprintf(stderr, "A CUDA device is required.\n");
        return 1;
    }
    // Use additional devices when each can receive at least 1024 output rows.
    const int devices = N == 0 ? 0 : std::min<size_t>(available, std::max<size_t>(1, N / 1024));
    std::vector<DeviceRows> partitions(devices);
    const size_t tiles = (N + tile - 1) / tile;
    const dim3 threads(tile, blockRows);
    for (int d = 0; d < devices; ++d) {
        auto& part = partitions[d];
        part.first = (tiles * d / devices) * tile;
        part.count = std::min(N, (tiles * (d + 1) / devices) * tile) - part.first;
        checkCuda(cudaSetDevice(d), "cudaSetDevice");
        checkCuda(cudaMalloc(&part.input, N * N * sizeof(double)), "allocate input");
        checkCuda(cudaMalloc(&part.output, part.count * N * sizeof(double)), "allocate output");
        initMatrix<<<dim3(tiles, (N + blockRows - 1) / blockRows), threads>>>(part.input, N);
        checkCuda(cudaGetLastError(), "initialize matrix");
    }
    auto synchronize = [devices]() {
        for (int d = 0; d < devices; ++d) {
            checkCuda(cudaSetDevice(d), "cudaSetDevice");
            checkCuda(cudaDeviceSynchronize(), "synchronize device");
        }
    };
    synchronize();

    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::steady_clock::now();
    for (int d = 0; d < devices; ++d) {
        const auto& part = partitions[d];
        checkCuda(cudaSetDevice(d), "cudaSetDevice");
        dim3 grid(tiles, (part.count + tile - 1) / tile);
        // For these binary fractions each product fits exactly in 53 bits.
        if ((N & (N - 1)) == 0 && N <= 8192)
            matrixMultiply<true><<<grid, threads>>>(part.input, part.output, N, part.first);
        else
            matrixMultiply<false><<<grid, threads>>>(part.input, part.output, N, part.first);
        checkCuda(cudaGetLastError(), "multiply matrices");
    }
    synchronize();
    auto elapsed = std::chrono::steady_clock::now() - start;
    double seconds = std::chrono::duration<double>(elapsed).count();
    printf("Computation time: %ld ms\n",
           static_cast<long>(std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()));
    printf("Performance: %.3f GFLOPS\n", (2.0 * N * N * N) / seconds / 1e9);

    std::vector<double> C;
    if (printResults || validate)
        C.resize(N * N);
    for (int d = 0; d < devices; ++d) {
        const auto& part = partitions[d];
        checkCuda(cudaSetDevice(d), "cudaSetDevice");
        if (!C.empty())
            checkCuda(cudaMemcpy(C.data() + part.first * N, part.output, part.count * N * sizeof(double),
                                 cudaMemcpyDeviceToHost), "copy output");
        checkCuda(cudaFree(part.output), "free output");
        checkCuda(cudaFree(part.input), "free input");
    }

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(C, N);
        
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
