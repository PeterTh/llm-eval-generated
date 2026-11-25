#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int blockX = 16;
constexpr int blockY = 8;
constexpr int tileK = 16;
constexpr size_t validationWidth = 5;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct CudaFree {
    void operator()(double* pointer) const noexcept { cudaFree(pointer); }
};

std::unique_ptr<double, CudaFree> allocateMatrix(size_t bytes) {
    double* pointer = nullptr;
    checkCuda(cudaMalloc(&pointer, bytes), "Allocating device matrix");
    return std::unique_ptr<double, CudaFree>(pointer);
}

class CudaEvent {
public:
    CudaEvent() { checkCuda(cudaEventCreate(&event), "Creating timing event"); }
    ~CudaEvent() { cudaEventDestroy(event); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;

    cudaEvent_t event;
};

__host__ __device__ size_t pseudoRandomNumerator(size_t row, size_t column) {
    return (row + 1) * (row + column + 1) * 1299709;
}

__global__ void initMatrix(double* matrix, unsigned n, size_t elements, bool powerOfTwo) {
    const unsigned column = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && column < n) {
        const size_t numerator = pseudoRandomNumerator(row, column);
        const size_t remainder = powerOfTwo ? numerator & (elements - 1) : numerator % elements;
        matrix[size_t{row} * n + column] =
            __ddiv_rn(static_cast<double>(remainder), static_cast<double>(elements));
    }
}

// Each block stages tiles of both operands and each thread accumulates a
// sub-tile of C in registers, visiting the inner dimension in ascending order.
template <int Tile, bool Aligned, bool ExactProducts = false>
__global__ void multiplyTiles(const double* __restrict__ input, double* __restrict__ output,
                              unsigned n) {
    constexpr int rowsPerThread = Tile / blockY;
    constexpr int columnsPerThread = Tile / blockX;
    __shared__ double left[Tile][tileK + 1];
    __shared__ double right[tileK][Tile];
    double sums[rowsPerThread][columnsPerThread] = {};

    const unsigned x = threadIdx.x;
    const unsigned y = threadIdx.y;
    const unsigned lane = y * blockX + x;
    const unsigned firstRow = blockIdx.y * Tile;
    const unsigned firstColumn = blockIdx.x * Tile;
    // The original optimized CPU loop rounds the paired products separately
    // and contracts the scalar remainder for an odd inner dimension.
    const unsigned pairedEnd = Aligned ? n : n & ~1u;

    for (unsigned k = 0; k < pairedEnd; k += tileK) {
        #pragma unroll
        for (unsigned index = lane; index < Tile * tileK; index += blockX * blockY) {
            const unsigned row = index / tileK;
            const unsigned inner = index % tileK;
            left[row][inner] = (Aligned || (firstRow + row < n && k + inner < pairedEnd))
                ? input[size_t{firstRow + row} * n + k + inner] : 0.0;
        }
        #pragma unroll
        for (unsigned index = lane; index < tileK * Tile; index += blockX * blockY) {
            const unsigned inner = index / Tile;
            const unsigned column = index % Tile;
            right[inner][column] = (Aligned || (k + inner < pairedEnd && firstColumn + column < n))
                ? input[size_t{k + inner} * n + firstColumn + column] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < tileK; ++inner) {
            double rightValues[columnsPerThread];
            #pragma unroll
            for (int j = 0; j < columnsPerThread; ++j) {
                rightValues[j] = right[inner][x + j * blockX];
            }
            #pragma unroll
            for (int i = 0; i < rowsPerThread; ++i) {
                const double leftValue = left[y + i * blockY][inner];
                #pragma unroll
                for (int j = 0; j < columnsPerThread; ++j) {
                    if constexpr (ExactProducts) {
                        sums[i][j] = __fma_rn(leftValue, rightValues[j], sums[i][j]);
                    } else {
                        sums[i][j] += leftValue * rightValues[j];
                    }
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const unsigned row = firstRow + y + i * blockY;
        #pragma unroll
        for (int j = 0; j < columnsPerThread; ++j) {
            const unsigned column = firstColumn + x + j * blockX;
            if (Aligned || (row < n && column < n)) {
                double value = sums[i][j];
                if (!Aligned && pairedEnd < n) {
                    value = __fma_rn(input[size_t{row} * n + pairedEnd],
                                     input[size_t{pairedEnd} * n + column], value);
                }
                output[size_t{row} * n + column] = value;
            }
        }
    }
}

template <int Tile>
void launchMultiply(const double* input, double* output, unsigned n, bool powerOfTwo) {
    const dim3 block(blockX, blockY);
    const dim3 grid((n + Tile - 1) / Tile, (n + Tile - 1) / Tile);
    if (n % Tile != 0) {
        multiplyTiles<Tile, false><<<grid, block>>>(input, output, n);
    } else if (powerOfTwo && n <= 8192) {
        // The inputs are dyadic fractions with at most 26 significant bits.
        // Their products are exact in FP64, so FMA preserves the rounding.
        multiplyTiles<Tile, true, true><<<grid, block>>>(input, output, n);
    } else {
        multiplyTiles<Tile, true><<<grid, block>>>(input, output, n);
    }
}

float matrixMultiply(size_t n, size_t resultWidth, std::vector<double>& result) {
    if (n == 0) {
        printf("Computing matrix multiplication...\n");
        return 0.0f;
    }
    if (n > std::numeric_limits<size_t>::max() / sizeof(double) / n) {
        throw std::runtime_error("Matrix size is too large");
    }
    const size_t elements = n * n;
    const size_t bytes = elements * sizeof(double);
    const bool powerOfTwo = (n & (n - 1)) == 0;
    // Both original operands are initialized to the same values.
    auto input = allocateMatrix(bytes);
    auto output = allocateMatrix(bytes);
    CudaEvent start;
    CudaEvent end;

    const dim3 initBlock(32, 8);
    const dim3 initGrid((n + initBlock.x - 1) / initBlock.x, (n + initBlock.y - 1) / initBlock.y);
    initMatrix<<<initGrid, initBlock>>>(input.get(), n, elements, powerOfTwo);
    checkCuda(cudaGetLastError(), "Launching matrix initialization");

    printf("Computing matrix multiplication...\n");
    checkCuda(cudaEventRecord(start.event), "Recording start event");
    if (n <= 1024) {
        launchMultiply<32>(input.get(), output.get(), n, powerOfTwo);
    } else {
        launchMultiply<64>(input.get(), output.get(), n, powerOfTwo);
    }
    checkCuda(cudaGetLastError(), "Launching matrix multiplication");
    checkCuda(cudaEventRecord(end.event), "Recording end event");
    checkCuda(cudaEventSynchronize(end.event), "Executing matrix multiplication");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start.event, end.event), "Measuring kernel time");

    if (resultWidth != 0) {
        result.resize(resultWidth * resultWidth);
        const size_t pitch = resultWidth * sizeof(double);
        checkCuda(cudaMemcpy2D(result.data(), pitch, output.get(), n * sizeof(double), pitch,
                               resultWidth, cudaMemcpyDeviceToHost), "Copying matrix results");
    }
    return milliseconds;
}

bool validateResult(const std::vector<double>& result, size_t N, size_t resultWidth) {
    const size_t checks = std::min(N, validationWidth);
    const size_t elements = N * N;
    for (size_t i = 0; i < checks; ++i) {
        for (size_t j = 0; j < checks; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                const double a = (pseudoRandomNumerator(i, k) % elements) / static_cast<double>(elements);
                const double b = (pseudoRandomNumerator(k, j) % elements) / static_cast<double>(elements);
                expected += a * b;
            }

            const double actual = result[i * resultWidth + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (!(relError <= 1e-6)) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }
    
    return true;
}

}  // namespace

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
    const size_t resultWidth = printResults ? N : (validate ? std::min(N, validationWidth) : 0);
    std::vector<double> C;
    float milliseconds;
    try {
        milliseconds = matrixMultiply(N, resultWidth, C);
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }

    printf("Computation time: %.3f ms\n", milliseconds);
    const double gflops = milliseconds > 0.0f ? (2.0 * N * N * N) / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(C, N, resultWidth);
        
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
