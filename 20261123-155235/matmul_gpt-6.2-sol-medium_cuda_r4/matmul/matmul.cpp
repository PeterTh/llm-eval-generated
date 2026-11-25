#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int BlockX = 16;
constexpr int BlockY = 8;
constexpr int BlockThreads = BlockX * BlockY;
constexpr int TileK = 16;
constexpr size_t ValidationSize = 5;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__host__ __device__ double getPseudoRndValue(size_t N, size_t i, size_t j) {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

__global__ void initMatrix(double* matrix, size_t N) {
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        matrix[row * N + col] = getPseudoRndValue(N, row, col);
    }
}

template <bool Fused>
__device__ double accumulate(double a, double b, double sum) {
    if constexpr (Fused) {
        return __fma_rn(a, b, sum);
    }
    return __dadd_rn(sum, __dmul_rn(a, b));
}

// GCC vectorizes pairs of products and may fuse the final scalar operation.
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
constexpr bool FuseScalarTail = true;
#else
constexpr bool FuseScalarTail = false;
#endif

template <int RowsPerThread, int ColsPerThread, bool ExactProducts>
__global__ void matrixMultiply(const double* __restrict__ input,
                               double* __restrict__ output, int N) {
    constexpr int TileRows = BlockY * RowsPerThread;
    constexpr int TileCols = BlockX * ColsPerThread;

    // Padding spreads the A row reads across shared-memory banks.
    __shared__ double aTile[TileRows][TileK + 1];
    __shared__ double bTile[TileK][TileCols];

    const int rowBase = blockIdx.y * TileRows;
    const int colBase = blockIdx.x * TileCols;
    const int lane = threadIdx.y * BlockX + threadIdx.x;
    const int unpaired = N % 2;
    const int pairedEnd = N - unpaired;
    double sums[RowsPerThread][ColsPerThread] = {};

    for (int kBase = 0; kBase < pairedEnd; kBase += TileK) {
        #pragma unroll
        for (int offset = lane; offset < TileRows * TileK; offset += BlockThreads) {
            const int row = rowBase + offset / TileK;
            const int k = kBase + offset % TileK;
            aTile[offset / TileK][offset % TileK] =
                row < N && k < pairedEnd ? input[static_cast<size_t>(row) * N + k] : 0.0;
        }

        #pragma unroll
        for (int offset = lane; offset < TileK * TileCols; offset += BlockThreads) {
            const int k = kBase + offset / TileCols;
            const int col = colBase + offset % TileCols;
            bTile[offset / TileCols][offset % TileCols] =
                k < pairedEnd && col < N ? input[static_cast<size_t>(k) * N + col] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TileK; ++k) {
            double b[ColsPerThread];
            #pragma unroll
            for (int j = 0; j < ColsPerThread; ++j) {
                b[j] = bTile[k][threadIdx.x + j * BlockX];
            }
            #pragma unroll
            for (int i = 0; i < RowsPerThread; ++i) {
                const double a = aTile[threadIdx.y + i * BlockY][k];
                #pragma unroll
                for (int j = 0; j < ColsPerThread; ++j) {
                    sums[i][j] = accumulate<ExactProducts>(a, b[j], sums[i][j]);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < RowsPerThread; ++i) {
        const int row = rowBase + threadIdx.y + i * BlockY;
        #pragma unroll
        for (int j = 0; j < ColsPerThread; ++j) {
            const int col = colBase + threadIdx.x + j * BlockX;
            if (row < N && col < N) {
                double sum = sums[i][j];
                if (unpaired) {
                    const double a = input[static_cast<size_t>(row) * N + pairedEnd];
                    const double b = input[static_cast<size_t>(pairedEnd) * N + col];
                    sum = accumulate<FuseScalarTail>(a, b, sum);
                }
                output[static_cast<size_t>(row) * N + col] = sum;
            }
        }
    }
}

template <int RowsPerThread, int ColsPerThread>
void launchMultiply(const double* input, double* output, size_t N, cudaEvent_t start) {
    constexpr int TileRows = BlockY * RowsPerThread;
    constexpr int TileCols = BlockX * ColsPerThread;
    const dim3 block(BlockX, BlockY);
    const dim3 grid((N + TileCols - 1) / TileCols, (N + TileRows - 1) / TileRows);

    // These power-of-two sizes yield exactly representable double products.
    const bool exactProducts = N <= 8192 && (N & (N - 1)) == 0;
    const auto kernel = exactProducts ? matrixMultiply<RowsPerThread, ColsPerThread, true>
                                      : matrixMultiply<RowsPerThread, ColsPerThread, false>;

    // CUDA may load the kernel lazily. Keep that work outside the timed interval.
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncGetAttributes(&attributes, kernel), "Loading matrix multiplication kernel");
    checkCuda(cudaEventRecord(start), "Recording start event");
    kernel<<<grid, block>>>(input, output, N);
    checkCuda(cudaGetLastError(), "Launching matrix multiplication");
}

bool validateResult(const double* C, size_t stride, size_t N) {
    const size_t checks = std::min(N, ValidationSize);
    for (size_t i = 0; i < checks; ++i) {
        for (size_t j = 0; j < checks; ++j) {
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

} // namespace

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

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    if (N != 0 && N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    const size_t elements = N * N;
    const size_t bytes = elements * sizeof(double);
    double* input = nullptr;
    double* output = nullptr;
    cudaEvent_t start, end;

    printf("Initializing matrices...\n");
    if (N != 0) {
        // Both input matrices are initialized to the same values.
        checkCuda(cudaMalloc(&input, bytes), "Allocating input on GPU");
        checkCuda(cudaMalloc(&output, bytes), "Allocating output on GPU");
        checkCuda(cudaEventCreate(&start), "Creating start event");
        checkCuda(cudaEventCreate(&end), "Creating end event");
        const dim3 block(32, 8);
        const dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        initMatrix<<<grid, block>>>(input, N);
        checkCuda(cudaGetLastError(), "Launching matrix initialization");
    }

    printf("Computing matrix multiplication...\n");
    float milliseconds = 0.0f;
    if (N != 0) {
        if (N < 384) {
            launchMultiply<2, 1>(input, output, N, start);
        } else {
            launchMultiply<4, 4>(input, output, N, start);
        }
        checkCuda(cudaEventRecord(end), "Recording end event");
        checkCuda(cudaEventSynchronize(end), "Waiting for matrix multiplication");
        checkCuda(cudaEventElapsedTime(&milliseconds, start, end), "Measuring GPU time");
        checkCuda(cudaEventDestroy(start), "Destroying start event");
        checkCuda(cudaEventDestroy(end), "Destroying end event");
        checkCuda(cudaFree(input), "Freeing GPU input");
    }

    printf("Computation time: %.3f ms\n", milliseconds);
    const double gflops = milliseconds > 0.0f ? (2.0 * N * N * N) / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);

    std::vector<double> C;
    if (printResults) {
        C.resize(elements);
        if (N != 0) {
            checkCuda(cudaMemcpy(C.data(), output, bytes, cudaMemcpyDeviceToHost),
                      "Copying matrix results to host");
        }
        print_results(C, "MatrixC");
    }

    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        if (printResults) {
            valid = validateResult(C.data(), N, N);
        } else if (N != 0) {
            const size_t checks = std::min(N, ValidationSize);
            const size_t width = checks * sizeof(double);
            double sample[ValidationSize * ValidationSize];
            checkCuda(cudaMemcpy2D(sample, width, output, N * sizeof(double), width, checks,
                                   cudaMemcpyDeviceToHost), "Copying validation samples to host");
            valid = validateResult(sample, checks, N);
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    if (output != nullptr) {
        checkCuda(cudaFree(output), "Freeing GPU output");
    }
    return valid ? 0 : 1;
}
