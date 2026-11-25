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
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initMatrix(double* matrix, size_t n) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < n * n; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        matrix[index] = getPseudoRndValue(n, index / n, index % n);
    }
}

constexpr int tileRows = 32;
constexpr int tileCols = 64;
constexpr int tileDepth = 8;

template<bool exactProducts>
__global__ void matrixMultiply(const double* matrix, double* output, size_t n,
                               size_t firstRow, size_t rows) {
    __shared__ double left[tileRows][tileDepth];
    __shared__ double right[tileDepth][tileCols];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int lane = y * 32 + x;
    const size_t row = blockIdx.y * tileRows;
    const size_t col = blockIdx.x * tileCols;
    double sums[8][2] = {};
    const size_t paired = n - n % 2;

    for (size_t k = 0; k < paired; k += tileDepth) {
        #pragma unroll
        for (int t = lane; t < tileRows * tileDepth; t += 128) {
            int r = t / tileDepth, d = t % tileDepth;
            left[r][d] = row + r < rows && k + d < paired
                ? matrix[(firstRow + row + r) * n + k + d] : 0.0;
        }
        #pragma unroll
        for (int t = lane; t < tileDepth * tileCols; t += 128) {
            int d = t / tileCols, c = t % tileCols;
            right[d][c] = k + d < paired && col + c < n
                ? matrix[(k + d) * n + col + c] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int d = 0; d < tileDepth; ++d) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    double a = left[y + 4 * r][d], b = right[d][x + 32 * c];
                    // Dyadic inputs with an exact product allow a single FMA.
                    if constexpr (exactProducts)
                        sums[r][c] = __fma_rn(a, b, sums[r][c]);
                    else
                        sums[r][c] += __dmul_rn(a, b);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < 8; ++r) {
        size_t i = row + y + 4 * r;
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            size_t j = col + x + 32 * c;
            if (i < rows && j < n) {
                // Match the original optimized scalar tail for odd dimensions.
                if (paired < n)
                    sums[r][c] = __fma_rn(matrix[(firstRow + i) * n + paired],
                                          matrix[paired * n + j], sums[r][c]);
                output[i * n + j] = sums[r][c];
            }
        }
    }
}

struct Partition {
    size_t firstRow, rows;
    double* input;
    double* output;
};

// Simple validation: compute a single element and compare
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
    
    if (N > 0 && N > SIZE_MAX / sizeof(double) / N) {
        fprintf(stderr, "Matrix size exceeds addressable storage\n");
        return 1;
    }
    int available;
    checkCuda(cudaGetDeviceCount(&available), "Query CUDA devices");
    if (!available) {
        fprintf(stderr, "A CUDA device is required\n");
        return 1;
    }
    // Keep small products on one GPU to amortize device setup and launches.
    const size_t devices = std::min<size_t>(available, std::max<size_t>(1, N / 1024));
    std::vector<Partition> parts(devices);
    printf("Initializing matrices...\n");
    for (size_t d = 0; d < devices && N; ++d) {
        auto& p = parts[d];
        p.firstRow = N * d / devices;
        p.rows = N * (d + 1) / devices - p.firstRow;
        checkCuda(cudaSetDevice(d), "Select CUDA device");
        checkCuda(cudaMalloc(&p.input, N * N * sizeof(double)), "Allocate input");
        checkCuda(cudaMalloc(&p.output, p.rows * N * sizeof(double)), "Allocate output");
        initMatrix<<<std::min<size_t>(65535, (N * N + 255) / 256), 256>>>(p.input, N);
        checkCuda(cudaGetLastError(), "Launch initialization");
    }
    for (size_t d = 0; d < devices; ++d) {
        checkCuda(cudaSetDevice(d), "Select CUDA device");
        checkCuda(cudaDeviceSynchronize(), "Initialize matrix");
    }

    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::steady_clock::now();
    for (size_t d = 0; d < devices && N; ++d) {
        const auto& p = parts[d];
        checkCuda(cudaSetDevice(d), "Select CUDA device");
        dim3 grid((N + tileCols - 1) / tileCols, (p.rows + tileRows - 1) / tileRows);
        dim3 block(32, 4);
        if ((N & (N - 1)) == 0 && N <= 8192)
            matrixMultiply<true><<<grid, block>>>(p.input, p.output, N, p.firstRow, p.rows);
        else
            matrixMultiply<false><<<grid, block>>>(p.input, p.output, N, p.firstRow, p.rows);
        checkCuda(cudaGetLastError(), "Launch matrix multiplication");
    }
    for (size_t d = 0; d < devices; ++d) {
        checkCuda(cudaSetDevice(d), "Select CUDA device");
        checkCuda(cudaDeviceSynchronize(), "Complete matrix multiplication");
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    printf("Computation time: %ld ms\n",
           static_cast<long>(elapsed.count() * 1000));
    double gflops = (2.0 * N * N * N) / elapsed.count() / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    size_t hostRows = printResults ? N : (validate ? std::min<size_t>(5, N) : 0);
    std::vector<double> C(hostRows * N);
    for (size_t d = 0; d < devices && N; ++d) {
        const auto& p = parts[d];
        checkCuda(cudaSetDevice(d), "Select CUDA device");
        if (p.firstRow < hostRows) {
            size_t rows = std::min(p.rows, hostRows - p.firstRow);
            checkCuda(cudaMemcpy(C.data() + p.firstRow * N, p.output, rows * N * sizeof(double),
                                 cudaMemcpyDeviceToHost), "Read result");
        }
        checkCuda(cudaFree(p.output), "Free output");
        checkCuda(cudaFree(p.input), "Free input");
    }

    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = !N || validateResult(C, N);
        
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
