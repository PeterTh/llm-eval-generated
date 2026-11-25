#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

__host__ __device__ constexpr double getPseudoRndValue(size_t n, size_t i, size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (n * n)) / static_cast<double>(n * n);
}

__global__ void initMatrix(double* input, size_t n) {
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (col < n) {
        for (size_t row = blockIdx.y; row < n; row += gridDim.y) {
            input[row * n + col] = getPseudoRndValue(n, row, col);
        }
    }
}

constexpr int blockRows = 32;
constexpr int tileDepth = 8;
constexpr int threadsX = 16;
constexpr int threadsY = 8;

// Each thread reuses shared-memory operands across a register tile of C.
template <int blockCols, bool fullTiles>
__global__ void matrixMultiply(const double* __restrict__ input, double* __restrict__ output,
                               int n, int firstRow, int rows) {
    constexpr int threadRows = blockRows / threadsY;
    constexpr int threadCols = blockCols / threadsX;
    __shared__ double left[blockRows][tileDepth + 1];
    __shared__ double right[tileDepth][blockCols];
    const int localRow = blockIdx.y * blockRows;
    const int row = firstRow + localRow;
    const int col = blockIdx.x * blockCols;
    const int thread = threadIdx.y * threadsX + threadIdx.x;
    double sums[threadRows][threadCols] = {};

#ifdef MATMUL_FUSED_TAIL
    const int paired = n - n % 2;
#else
    const int paired = n;
#endif

    for (int base = 0; base < paired; base += tileDepth) {
        #pragma unroll
        for (int index = thread; index < blockRows * tileDepth; index += threadsX * threadsY) {
            const int r = index / tileDepth;
            const int k = index % tileDepth;
            left[r][k] = (fullTiles || (localRow + r < rows && base + k < paired))
                ? input[static_cast<size_t>(row + r) * n + base + k] : 0.0;
        }
        #pragma unroll
        for (int index = thread; index < tileDepth * blockCols; index += threadsX * threadsY) {
            const int k = index / blockCols;
            const int c = index % blockCols;
            right[k][c] = (fullTiles || (base + k < paired && col + c < n))
                ? input[static_cast<size_t>(base + k) * n + col + c] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < tileDepth; ++k) {
            double a[threadRows];
            double b[threadCols];
            #pragma unroll
            for (int r = 0; r < threadRows; ++r) {
                a[r] = left[threadIdx.y + r * threadsY][k];
            }
            #pragma unroll
            for (int c = 0; c < threadCols; ++c) {
                b[c] = right[k][threadIdx.x + c * threadsX];
            }
            #pragma unroll
            for (int r = 0; r < threadRows; ++r) {
                #pragma unroll
                for (int c = 0; c < threadCols; ++c) {
                    // Keep the original multiply and add rounding for the result hash.
                    sums[r][c] = __dadd_rn(sums[r][c], __dmul_rn(a[r], b[c]));
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < threadRows; ++r) {
        const int i = localRow + threadIdx.y + r * threadsY;
        #pragma unroll
        for (int c = 0; c < threadCols; ++c) {
            const int j = col + threadIdx.x + c * threadsX;
            if (fullTiles || (i < rows && j < n)) {
                double sum = sums[r][c];
#ifdef MATMUL_FUSED_TAIL
                if (!fullTiles && paired != n) {
                    sum = __fma_rn(input[static_cast<size_t>(firstRow + i) * n + paired],
                                   input[static_cast<size_t>(paired) * n + j], sum);
                }
#endif
                output[static_cast<size_t>(i) * n + j] = sum;
            }
        }
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

struct DeviceWork {
    int device = 0;
    int firstRow = 0;
    int rows = 0;
    double* input = nullptr;
    double* output = nullptr;

    DeviceWork() = default;
    DeviceWork(const DeviceWork&) = delete;
    DeviceWork& operator=(const DeviceWork&) = delete;

    ~DeviceWork() {
        if (input) {
            cudaSetDevice(device);
            if (output) cudaFree(output);
            cudaFree(input);
        }
    }
};

void waitForDevices(const std::vector<DeviceWork>& devices, const char* operation) {
    for (const auto& work : devices) {
        checkCuda(cudaSetDevice(work.device), "selecting CUDA device");
        checkCuda(cudaDeviceSynchronize(), operation);
    }
}

int tileCount(int size, int tile) {
    return size / tile + (size % tile != 0);
}

template <int blockCols>
void launchMultiply(const DeviceWork& work, int n) {
    const dim3 threads(threadsX, threadsY);
    const dim3 blocks(tileCount(n, blockCols), tileCount(work.rows, blockRows));
    if (n % blockCols == 0 && work.rows % blockRows == 0) {
        matrixMultiply<blockCols, true><<<blocks, threads>>>(work.input, work.output, n,
                                                            work.firstRow, work.rows);
    } else {
        matrixMultiply<blockCols, false><<<blocks, threads>>>(work.input, work.output, n,
                                                             work.firstRow, work.rows);
    }
    checkCuda(cudaGetLastError(), "launching matrix multiplication");
}

bool validateResult(const double* samples, int n, int stride) {
    for (int pi = 0; pi < 5; ++pi) {
        for (int pj = 0; pj < 5; ++pj) {
            const size_t i = pi % n;
            const size_t j = pj % n;
            double expected = 0.0;
            for (size_t k = 0; k < static_cast<size_t>(n); ++k) {
                expected += getPseudoRndValue(n, i, k) * getPseudoRndValue(n, k, j);
            }
            const double actual = samples[i * stride + j];
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

int runBenchmark(int n, bool validate, bool printResults) {
    const size_t elements = static_cast<size_t>(n) * n;
    const size_t rowBytes = static_cast<size_t>(n) * sizeof(double);

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %d x %d\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int availableDevices = 0;
    checkCuda(cudaGetDeviceCount(&availableDevices), "enumerating CUDA devices");
    if (availableDevices == 0) {
        throw std::runtime_error("No CUDA device available");
    }

    // Row blocks are independent. Keep small jobs on one GPU to avoid the
    // extra context setup and input replication for work that fits on one.
    const int tiles = tileCount(n, blockRows);
    const int deviceCount = n == 0 ? 0 : std::min(availableDevices, std::max(1, n / 1024));
    std::vector<DeviceWork> devices(deviceCount);

    printf("Initializing matrices...\n");
    for (int device = 0, nextTile = 0; device < deviceCount; ++device) {
        auto& work = devices[device];
        work.device = device;
        work.firstRow = nextTile * blockRows;
        nextTile += tileCount(tiles - nextTile, deviceCount - device);
        work.rows = std::min(n - work.firstRow, (nextTile * blockRows) - work.firstRow);
        checkCuda(cudaSetDevice(device), "selecting CUDA device");
        checkCuda(cudaMalloc(&work.input, elements * sizeof(double)), "allocating CUDA input");
        checkCuda(cudaMalloc(&work.output, work.rows * rowBytes), "allocating CUDA output");

        // A and B use the same initializer, so they share one read-only allocation.
        constexpr int threads = 256;
        const dim3 blocks(tileCount(n, threads), std::min(n, 65535));
        initMatrix<<<blocks, threads>>>(work.input, n);
        checkCuda(cudaGetLastError(), "launching CUDA matrix initialization");
    }
    waitForDevices(devices, "waiting for CUDA matrix initialization");

    printf("Computing matrix multiplication...\n");
    const auto start = std::chrono::steady_clock::now();
    for (const auto& work : devices) {
        checkCuda(cudaSetDevice(work.device), "selecting CUDA device");
        if (n <= 768) {
            launchMultiply<32>(work, n);
        } else {
            launchMultiply<64>(work, n);
        }
    }
    waitForDevices(devices, "waiting for CUDA multiplication");
    const auto end = std::chrono::steady_clock::now();
    const double milliseconds = n == 0 ? 0.0 :
        std::chrono::duration<double, std::milli>(end - start).count();
    printf("Computation time: %.3f ms\n", milliseconds);
    const double gflops = milliseconds > 0.0 ? (2.0 * n * n * n) / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);

    std::vector<double> result;
    if (printResults) {
        result.resize(elements);
        for (const auto& work : devices) {
            checkCuda(cudaSetDevice(work.device), "selecting CUDA device");
            checkCuda(cudaMemcpy(result.data() + static_cast<size_t>(work.firstRow) * n,
                                 work.output, work.rows * rowBytes, cudaMemcpyDeviceToHost),
                      "copying CUDA matrix result");
        }
        print_results(result, "MatrixC");
    }
    if (validate) {
        printf("Validating result...\n");
        bool valid = true;
        if (n != 0) {
            if (printResults) {
                valid = validateResult(result.data(), n, n);
            } else {
                double samples[25];
                const int width = std::min(n, 5);
                checkCuda(cudaSetDevice(devices.front().device), "selecting CUDA device");
                const size_t sampleBytes = width * sizeof(double);
                checkCuda(cudaMemcpy2D(samples, sampleBytes, devices.front().output, rowBytes,
                                       sampleBytes, width, cudaMemcpyDeviceToHost),
                          "copying CUDA validation samples");
                valid = validateResult(samples, n, width);
            }
        }
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
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
    int n = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            const char* end = value + strlen(value);
            const auto parsed = std::from_chars(value, end, n);
            if (parsed.ec != std::errc{} || parsed.ptr != end || n < 0 ||
                (n != 0 && std::numeric_limits<size_t>::max() / sizeof(double) / n <
                               static_cast<size_t>(n))) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
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
        return runBenchmark(n, validate, printResults);
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
