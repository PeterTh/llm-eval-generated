#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

__host__ __device__ double getPseudoRndValue(size_t elements, size_t i, size_t j,
                                             double binaryScale = 0.0) noexcept {
    const size_t value = (i + 1) * (i + j + 1) * 1299709;
    if (binaryScale != 0.0) {
        return static_cast<double>(value & (elements - 1)) * binaryScale;
    }
    return (value % elements) / static_cast<double>(elements);
}

__global__ void initMatrix(double* matrix, size_t N, double binaryScale) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        matrix[row * N + col] = getPseudoRndValue(N * N, row, col, binaryScale);
    }
}

constexpr int blockWidth = 16;
constexpr int blockHeight = 8;
constexpr int tileDepth = 16;

// Each block stages input tiles in shared memory. Each thread reuses those
// values across several independent dot products held in registers.
template <int rowsPerThread, int colsPerThread, bool exactProducts>
__global__ void matrixMultiply(const double* __restrict__ input,
                               double* __restrict__ output, size_t N) {
    constexpr int tileRows = blockHeight * rowsPerThread;
    constexpr int tileCols = blockWidth * colsPerThread;
    // Padding makes the left tile's shared-memory broadcasts use different banks.
    __shared__ double left[tileRows][tileDepth + 1];
    __shared__ double right[tileDepth][tileCols];

    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t rowBase = size_t(blockIdx.y) * tileRows;
    const size_t colBase = size_t(blockIdx.x) * tileCols;
    const bool fullOutputTile = rowBase + tileRows <= N && colBase + tileCols <= N;
    double sum[rowsPerThread][colsPerThread] = {};

    // The original optimized CPU loop rounds products separately in pairs
    // and uses a fused multiply-add for the scalar tail of an odd dot product.
    const size_t paired = exactProducts ? N : N - N % 2;
    for (size_t start = 0; start < paired; start += tileDepth) {
        const bool fullTile = exactProducts || (fullOutputTile && start + tileDepth <= paired);

        #pragma unroll
        for (int i = 0; i < rowsPerThread; ++i) {
            const size_t row = rowBase + y + i * blockHeight;
            const size_t k = start + x;
            left[y + i * blockHeight][x] =
                (fullTile || (row < N && k < paired)) ? input[row * N + k] : 0.0;
        }
        #pragma unroll
        for (int i = 0; i < tileDepth / blockHeight; ++i) {
            const size_t k = start + y + i * blockHeight;
            #pragma unroll
            for (int j = 0; j < colsPerThread; ++j) {
                const size_t col = colBase + x + j * blockWidth;
                right[y + i * blockHeight][x + j * blockWidth] =
                    (fullTile || (k < paired && col < N)) ? input[k * N + col] : 0.0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tileDepth; ++k) {
            double b[colsPerThread];
            #pragma unroll
            for (int j = 0; j < colsPerThread; ++j) {
                b[j] = right[k][x + j * blockWidth];
            }
            #pragma unroll
            for (int i = 0; i < rowsPerThread; ++i) {
                const double a = left[y + i * blockHeight][k];
                #pragma unroll
                for (int j = 0; j < colsPerThread; ++j) {
                    if constexpr (exactProducts) {
                        sum[i][j] = __fma_rn(a, b[j], sum[i][j]);
                    } else {
                        sum[i][j] = __dadd_rn(sum[i][j], __dmul_rn(a, b[j]));
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const size_t row = rowBase + y + i * blockHeight;
        if (exactProducts || row < N) {
            #pragma unroll
            for (int j = 0; j < colsPerThread; ++j) {
                const size_t col = colBase + x + j * blockWidth;
                if (exactProducts || col < N) {
                    double value = sum[i][j];
                    if (paired != N) {
                        value = __fma_rn(input[row * N + paired], input[paired * N + col], value);
                    }
                    output[row * N + col] = value;
                }
            }
        }
    }
}

unsigned int blocksFor(size_t count, unsigned int tileSize) {
    return static_cast<unsigned int>((count + tileSize - 1) / tileSize);
}

template <int rowsPerThread, int colsPerThread>
void launchMultiply(const double* input, double* output, size_t N, bool powerOfTwo) {
    constexpr unsigned int rows = blockHeight * rowsPerThread;
    constexpr unsigned int cols = blockWidth * colsPerThread;
    const dim3 blocks(blocksFor(N, cols), blocksFor(N, rows));
    const dim3 threads(blockWidth, blockHeight);
    // With a power-of-two N up to 8192 the inputs have at most 26
    // significant bits, so a fused multiply-add gives the same result.
    // These sizes also fill every tile when N is at least the tile dimensions.
    if (powerOfTwo && N <= 8192 && N >= std::max(rows, cols)) {
        matrixMultiply<rowsPerThread, colsPerThread, true><<<blocks, threads>>>(input, output, N);
    } else {
        matrixMultiply<rowsPerThread, colsPerThread, false><<<blocks, threads>>>(input, output, N);
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct CudaResources {
    double* matrices = nullptr;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;

    ~CudaResources() {
        if (stop) cudaEventDestroy(stop);
        if (start) cudaEventDestroy(start);
        if (matrices) cudaFree(matrices);
    }
};

constexpr size_t validationWidth = 5;

bool validateResult(const double* result, size_t stride, size_t N) {
    const size_t extent = std::min(N, validationWidth);
    const size_t elements = N * N;
    for (size_t i = 0; i < extent; ++i) {
        for (size_t j = 0; j < extent; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(elements, i, k) * getPseudoRndValue(elements, k, j);
            }

            const double actual = result[i * stride + j];
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

int runBenchmark(size_t N, bool validate, bool printResults) {
    if (N != 0 && N > std::numeric_limits<size_t>::max() / (2 * sizeof(double)) / N) {
        throw std::runtime_error("Matrix size is too large");
    }
    const size_t elements = N * N;
    const size_t bytes = elements * sizeof(double);
    const bool powerOfTwo = N != 0 && (N & (N - 1)) == 0;
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    CudaResources cuda;
    double* output = nullptr;
    printf("Initializing matrices...\n");
    if (N != 0) {
        // A and B use the same generator, so both operands share one allocation.
        checkCuda(cudaMalloc(&cuda.matrices, 2 * bytes), "Allocating device matrices");
        output = cuda.matrices + elements;
        checkCuda(cudaEventCreate(&cuda.start), "Creating start event");
        checkCuda(cudaEventCreate(&cuda.stop), "Creating stop event");
        const double binaryScale = powerOfTwo ? 1.0 / static_cast<double>(elements) : 0.0;
        initMatrix<<<dim3(blocksFor(N, 32), blocksFor(N, 8)), dim3(32, 8)>>>(
            cuda.matrices, N, binaryScale);
        checkCuda(cudaGetLastError(), "Launching matrix initialization");
    }

    printf("Computing matrix multiplication...\n");
    float milliseconds = 0.0f;
    if (N != 0) {
        checkCuda(cudaEventRecord(cuda.start), "Recording start event");
        if (N <= 256) {
            launchMultiply<2, 1>(cuda.matrices, output, N, powerOfTwo);
        } else if (N <= 768) {
            launchMultiply<4, 2>(cuda.matrices, output, N, powerOfTwo);
        } else {
            launchMultiply<4, 4>(cuda.matrices, output, N, powerOfTwo);
        }
        checkCuda(cudaGetLastError(), "Launching matrix multiplication");
        checkCuda(cudaEventRecord(cuda.stop), "Recording stop event");
        checkCuda(cudaEventSynchronize(cuda.stop), "Waiting for computation");
        checkCuda(cudaEventElapsedTime(&milliseconds, cuda.start, cuda.stop), "Measuring computation");
    }
    printf("Computation time: %ld ms\n", static_cast<long>(milliseconds));

    const double gflops = milliseconds > 0.0f ? (2.0 * N * N * N) / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);

    std::vector<double> C;
    if (printResults) {
        C.resize(elements);
        if (N != 0) {
            checkCuda(cudaMemcpy(C.data(), output, bytes, cudaMemcpyDeviceToHost), "Copying result matrix");
        }
        print_results(C, "MatrixC");
    }

    if (validate) {
        printf("Validating result...\n");
        double samples[validationWidth * validationWidth];
        const size_t extent = std::min(N, validationWidth);
        if (!printResults && extent != 0) {
            checkCuda(cudaMemcpy2D(samples, validationWidth * sizeof(double),
                                   output, N * sizeof(double), extent * sizeof(double),
                                   extent, cudaMemcpyDeviceToHost), "Copying validation samples");
        }
        const bool valid = printResults ? validateResult(C.data(), N, N)
                                        : validateResult(samples, validationWidth, N);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;

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

    try {
        return runBenchmark(N, validate, printResults);
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
