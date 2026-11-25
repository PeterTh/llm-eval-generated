#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

struct CudaBuffer {
    double* data = nullptr;

    ~CudaBuffer() {
        if (data) {
            cudaFree(data);
        }
    }

    void allocate(size_t elements) {
        if (elements > std::numeric_limits<size_t>::max() / sizeof(double)) {
            throw std::length_error("Matrix allocation is too large");
        }
        checkCuda(cudaMalloc(&data, elements * sizeof(double)));
    }
};

struct CudaEvent {
    cudaEvent_t event = nullptr;

    ~CudaEvent() {
        if (event) {
            cudaEventDestroy(event);
        }
    }

    void create() { checkCuda(cudaEventCreate(&event)); }
};

// Both input matrices use the same deterministic values.
__host__ __device__ double getPseudoRndValue(size_t N, size_t i, size_t j) {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

__global__ void initMatrix(double* matrix, size_t N, size_t stride) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t rowStep = gridDim.y * blockDim.y;
    for (size_t row = blockIdx.y * blockDim.y + threadIdx.y; row < stride; row += rowStep) {
        matrix[row * stride + col] = row < N && col < N ? getPseudoRndValue(N, row, col) : 0.0;
    }
}

constexpr int threadsX = 16;
constexpr int threadsY = 8;
constexpr int tileK = 16;

// The original GCC build with native FMA vectorizes pairs of products and
// contracts only the scalar remainder. Match its rounding for the result hash.
#if defined(__GNUC__) && !defined(__clang__) && defined(__FMA__)
constexpr bool fuseOddTail = true;
#else
constexpr bool fuseOddTail = false;
#endif

template <int rowsPerThread, int colsPerThread, bool oddTail>
__global__ void matrixMultiply(const double* __restrict__ input, double* __restrict__ output,
                               int N, int stride, int rowOffset, int numRows) {
    constexpr int tileRows = threadsY * rowsPerThread;
    constexpr int tileCols = threadsX * colsPerThread;
    __shared__ double left[tileRows][tileK + 1];
    __shared__ double right[tileK][tileCols];

    const int firstRow = rowOffset + blockIdx.y * tileRows;
    const int firstCol = blockIdx.x * tileCols;
    const int thread = threadIdx.y * threadsX + threadIdx.x;
    double sums[rowsPerThread][colsPerThread] = {};
    const int regularTerms = oddTail ? N - 1 : N;

    for (int firstK = 0; firstK < regularTerms; firstK += tileK) {
        for (int i = thread; i < tileRows * tileK; i += threadsX * threadsY) {
            const int row = i / tileK;
            const int k = firstK + i % tileK;
            left[row][i % tileK] = oddTail && k == regularTerms ? 0.0
                : input[static_cast<size_t>(firstRow + row) * stride + k];
        }
        for (int i = thread; i < tileK * tileCols; i += threadsX * threadsY) {
            const int k = i / tileCols;
            const int col = i % tileCols;
            right[k][col] = input[static_cast<size_t>(firstK + k) * stride + firstCol + col];
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tileK; ++k) {
            double rhs[colsPerThread];
            #pragma unroll
            for (int c = 0; c < colsPerThread; ++c) {
                rhs[c] = right[k][threadIdx.x + c * threadsX];
            }
            #pragma unroll
            for (int r = 0; r < rowsPerThread; ++r) {
                const double lhs = left[threadIdx.y + r * threadsY][k];
                #pragma unroll
                for (int c = 0; c < colsPerThread; ++c) {
                    sums[r][c] = __dadd_rn(sums[r][c], __dmul_rn(lhs, rhs[c]));
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < rowsPerThread; ++r) {
        const int row = firstRow + threadIdx.y + r * threadsY;
        #pragma unroll
        for (int c = 0; c < colsPerThread; ++c) {
            const int col = firstCol + threadIdx.x + c * threadsX;
            if (row - rowOffset < numRows && col < N) {
                double sum = sums[r][c];
                if constexpr (oddTail) {
                    const double lhs = input[static_cast<size_t>(row) * stride + regularTerms];
                    const double rhs = input[static_cast<size_t>(regularTerms) * stride + col];
                    sum = __fma_rn(lhs, rhs, sum);
                }
                output[static_cast<size_t>(row - rowOffset) * N + col] = sum;
            }
        }
    }
}

template <int rowsPerThread, int colsPerThread>
void launchMultiply(const double* input, double* output, int N, int stride, int rowOffset, int numRows) {
    constexpr int tileRows = threadsY * rowsPerThread;
    constexpr int tileCols = threadsX * colsPerThread;
    const dim3 threads(threadsX, threadsY);
    const dim3 blocks((N + tileCols - 1) / tileCols, (numRows + tileRows - 1) / tileRows);
    if (fuseOddTail && (N & 1)) {
        matrixMultiply<rowsPerThread, colsPerThread, true><<<blocks, threads>>>(
            input, output, N, stride, rowOffset, numRows);
    } else {
        matrixMultiply<rowsPerThread, colsPerThread, false><<<blocks, threads>>>(
            input, output, N, stride, rowOffset, numRows);
    }
    checkCuda(cudaGetLastError());
}

struct DeviceWork {
    int device = 0;
    int rowOffset = 0;
    int numRows = 0;
    CudaBuffer input, output;

    ~DeviceWork() {
        if (input.data) {
            cudaSetDevice(device);
        }
    }

    void activate() const { checkCuda(cudaSetDevice(device)); }
};

double runBenchmark(int N, std::vector<double>& result, bool printResults, bool validate) {
    printf("Initializing matrices...\n");
    int visibleDevices;
    checkCuda(cudaGetDeviceCount(&visibleDevices));
    if (visibleDevices == 0) {
        throw std::runtime_error("No CUDA devices available");
    }
    if (N == 0) {
        printf("Computing matrix multiplication...\n");
        return 0.0;
    }

    // Pad the inputs so all threads can load full, aligned tiles.
    const int stride = (N + 63) / 64 * 64;
    const size_t paddedElements = static_cast<size_t>(stride) * stride;
    const dim3 initThreads(32, 8);
    const dim3 initBlocks(stride / initThreads.x, std::min(stride / initThreads.y, 65535u));

    // Use additional GPUs only when there is enough work to amortize setup.
    const int devices = std::min(visibleDevices, std::max(1, N / 2048));
    // Split at tile boundaries so only the final device has a partial row tile.
    constexpr int rowsPerBlock = 4 * threadsY;
    const int rowBlocks = (N + rowsPerBlock - 1) / rowsPerBlock;
    std::vector<DeviceWork> work(devices);
    for (int i = 0; i < devices; ++i) {
        auto& part = work[i];
        part.device = i;
        part.rowOffset = static_cast<int>(static_cast<long long>(rowBlocks) * i / devices) * rowsPerBlock;
        const int lastBlock = static_cast<long long>(rowBlocks) * (i + 1) / devices;
        part.numRows = std::min(N, lastBlock * rowsPerBlock) - part.rowOffset;
        part.activate();
        part.input.allocate(paddedElements);
        part.output.allocate(static_cast<size_t>(part.numRows) * N);
        initMatrix<<<initBlocks, initThreads>>>(part.input.data, N, stride);
        checkCuda(cudaGetLastError());
    }

    printf("Computing matrix multiplication...\n");
    double milliseconds;
    if (devices == 1) {
        const auto& part = work.front();
        CudaEvent start, stop;
        start.create();
        stop.create();
        checkCuda(cudaEventRecord(start.event));
        if (N <= 512) {
            launchMultiply<2, 2>(part.input.data, part.output.data, N, stride, 0, N);
        } else {
            launchMultiply<4, 4>(part.input.data, part.output.data, N, stride, 0, N);
        }
        checkCuda(cudaEventRecord(stop.event));
        checkCuda(cudaEventSynchronize(stop.event));
        float elapsed;
        checkCuda(cudaEventElapsedTime(&elapsed, start.event, stop.event));
        milliseconds = elapsed;
    } else {
        for (const auto& part : work) {
            part.activate();
            checkCuda(cudaDeviceSynchronize());
        }
        const auto start = std::chrono::steady_clock::now();
        for (const auto& part : work) {
            part.activate();
            launchMultiply<4, 4>(part.input.data, part.output.data, N, stride, part.rowOffset, part.numRows);
        }
        for (const auto& part : work) {
            part.activate();
            checkCuda(cudaDeviceSynchronize());
        }
        milliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    if (printResults) {
        result.resize(static_cast<size_t>(N) * N);
        for (const auto& part : work) {
            part.activate();
            const size_t offset = static_cast<size_t>(part.rowOffset) * N;
            const size_t elements = static_cast<size_t>(part.numRows) * N;
            checkCuda(cudaMemcpy(result.data() + offset, part.output.data,
                                 elements * sizeof(double), cudaMemcpyDeviceToHost));
        }
    } else if (validate) {
        const auto& part = work.front();
        part.activate();
        const size_t checks = std::min(N, 5);
        const size_t checkWidth = checks * sizeof(double);
        result.resize(checks * checks);
        checkCuda(cudaMemcpy2D(result.data(), checkWidth, part.output.data, N * sizeof(double),
                              checkWidth, checks, cudaMemcpyDeviceToHost));
    }
    return milliseconds;
}

bool validateResult(const std::vector<double>& C, size_t N, size_t resultStride) {
    const size_t checks = std::min<size_t>(N, 5);
    for (size_t i = 0; i < checks; ++i) {
        for (size_t j = 0; j < checks; ++j) {
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
            }

            const double actual = C[i * resultStride + j];
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
    int N = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            const char* value = argv[++i];
            const long size = std::strtol(value, &end, 10);
            const auto maxElements = std::numeric_limits<size_t>::max() / sizeof(double);
            if (end == value || *end || size < 0 || size > std::numeric_limits<int>::max() - 63 ||
                (size > 0 && static_cast<size_t>(size) > maxElements / static_cast<size_t>(size))) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            N = static_cast<int>(size);
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
    printf("Matrix size: %d x %d\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> C;
    double milliseconds;
    try {
        milliseconds = runBenchmark(N, C, printResults, validate);
    } catch (const std::exception& error) {
        fprintf(stderr, "Matrix multiplication failed: %s\n", error.what());
        return 1;
    }

    printf("Computation time: %.3f ms\n", milliseconds);
    const double gflops = milliseconds > 0 ? (2.0 * N * N * N) / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);

    if (printResults) {
        print_results(C, "MatrixC");
    }

    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(C, N, printResults ? N : std::min(N, 5));
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
