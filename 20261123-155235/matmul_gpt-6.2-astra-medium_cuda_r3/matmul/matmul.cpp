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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initMatrix(double* input, size_t n) {
    for (size_t row = blockIdx.y; row < n; row += gridDim.y) {
        size_t col = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
        if (col < n) {
            input[row * n + col] = getPseudoRndValue(n, row, col);
        }
    }
}

constexpr unsigned tile = 32;
constexpr unsigned threadRows = 4;

// Each warp computes eight rows of a tile. Adjacent lanes load and store
// consecutive columns, and all four warps reuse both shared-memory tiles.
template<bool exactProducts, bool fullTiles>
__global__ void matrixMultiply(const double* input, double* output, size_t n,
                               size_t firstRow, size_t rows) {
    __shared__ double left[tile][tile];
    __shared__ double right[tile][tile];
    const unsigned x = threadIdx.x, y = threadIdx.y;
    const size_t rowBase = size_t(blockIdx.y) * tile;
    const size_t col = size_t(blockIdx.x) * tile + x;
    double sum[tile / threadRows] = {};
    for (size_t base = 0; base < n; base += tile) {
        #pragma unroll
        for (unsigned r = y; r < tile; r += threadRows) {
            left[r][x] = (fullTiles || (rowBase + r < rows && base + x < n - (n % 2)))
                ? input[(firstRow + rowBase + r) * n + base + x] : 0.0;
            right[r][x] = (fullTiles || (base + r < n && col < n))
                ? input[(base + r) * n + col] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (unsigned k = 0; k < tile; ++k) {
            #pragma unroll
            for (unsigned i = 0; i < tile / threadRows; ++i) {
                double a = left[y + i * threadRows][k], b = right[k][x];
                if (exactProducts) {
                    sum[i] = __fma_rn(a, b, sum[i]);
                } else {
                    sum[i] += __dmul_rn(a, b);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (unsigned i = 0; i < tile / threadRows; ++i) {
        size_t row = rowBase + y + i * threadRows;
        if (fullTiles || (row < rows && col < n)) {
            // Match the native CPU loop's fused scalar tail for an odd K.
            if (!fullTiles && n % 2) {
                sum[i] = __fma_rn(input[(firstRow + row) * n + n - 1],
                                  input[(n - 1) * n + col], sum[i]);
            }
            output[row * n + col] = sum[i];
        }
    }
}

struct Partition {
    size_t firstRow, rows;
    double* input;
    double* output;
};

// Independently recompute sampled elements on the host.
bool validateResult(const std::vector<double>& C, const size_t N) {
    if (N == 0) return true;
    // Check a few positions
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
    
    printf("Initializing matrices...\n");
    int devices = 0;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0 || (N > 0 && N > SIZE_MAX / sizeof(double) / N)) {
        fprintf(stderr, "No CUDA device or unsupported matrix size\n");
        return 1;
    }
    // Keep small jobs on one GPU. Split tiles, so each device writes whole rows.
    const size_t tiles = (N + tile - 1) / tile;
    devices = std::min<size_t>(devices, std::max<size_t>(1, tiles / 32));
    std::vector<Partition> parts(devices);
    for (int d = 0; d < devices && N; ++d) {
        auto& p = parts[d];
        p.firstRow = tiles * d / devices * tile;
        p.rows = std::min(N, tiles * (d + 1) / devices * tile) - p.firstRow;
        checkCuda(cudaSetDevice(d));
        checkCuda(cudaMalloc(&p.input, N * N * sizeof(double)));
        checkCuda(cudaMalloc(&p.output, p.rows * N * sizeof(double)));
        dim3 grid((N + 255) / 256, std::min<size_t>(N, 65535));
        initMatrix<<<grid, 256>>>(p.input, N);
        checkCuda(cudaGetLastError());
    }
    for (int d = 0; d < devices; ++d) {
        checkCuda(cudaSetDevice(d));
        checkCuda(cudaDeviceSynchronize());
    }

    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::steady_clock::now();
    for (int d = 0; d < devices && N; ++d) {
        checkCuda(cudaSetDevice(d));
        const auto& p = parts[d];
        dim3 grid(tiles, (p.rows + tile - 1) / tile), block(tile, threadRows);
        // For powers of two up to 8192 each product fits exactly in 52 bits.
        const bool exact = (N & (N - 1)) == 0 && N <= 8192;
        if (N % tile == 0) {
            if (exact) matrixMultiply<true, true><<<grid, block>>>(p.input, p.output, N, p.firstRow, p.rows);
            else matrixMultiply<false, true><<<grid, block>>>(p.input, p.output, N, p.firstRow, p.rows);
        } else {
            matrixMultiply<false, false><<<grid, block>>>(p.input, p.output, N, p.firstRow, p.rows);
        }
        checkCuda(cudaGetLastError());
    }
    for (int d = 0; d < devices; ++d) {
        checkCuda(cudaSetDevice(d));
        checkCuda(cudaDeviceSynchronize());
    }
    std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
    printf("Computation time: %.3f ms\n", duration.count() * 1000.0);
    double gflops = (2.0 * N * N * N) / duration.count() / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);

    std::vector<double> C;
    if (printResults || validate) C.resize(N * N);
    for (int d = 0; d < devices && N; ++d) {
        checkCuda(cudaSetDevice(d));
        const auto& p = parts[d];
        if (!C.empty()) {
            checkCuda(cudaMemcpy(C.data() + p.firstRow * N, p.output,
                                 p.rows * N * sizeof(double), cudaMemcpyDeviceToHost));
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
