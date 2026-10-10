#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
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

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void initMatrix(std::vector<double>& mat, const size_t N) {
    for (size_t i = 0; i < N; ++i) {
        for (size_t j = 0; j < N; ++j) {
            mat[i * N + j] = getPseudoRndValue(N, i, j);
        }
    }
}

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// A 16x16 thread block computes a 32x32 or 64x64 output tile. Each thread
// keeps several independent double-precision dot products in registers.
// Transposing A's shared tile and padding its stride avoids bank conflicts.
template <int Tile, bool FullTiles>
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     size_t N, size_t rows) {
    constexpr int KTile = 16;
    constexpr int ThreadTile = Tile / 16;
    __shared__ double tileA[KTile][Tile + 1];
    __shared__ double tileB[KTile][Tile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * Tile;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * Tile;
    double sums[ThreadTile][ThreadTile] = {};

    for (size_t base = 0; base < N; base += KTile) {
#pragma unroll
        for (int load = 0; load < Tile * KTile / 256; ++load) {
            const int index = tid + load * 256;
            const int ar = index / KTile;
            const int ak = index % KTile;
            const int bk = index / Tile;
            const int bc = index % Tile;
            tileA[ak][ar] = (FullTiles || (rowBase + ar < rows && base + ak < N))
                               ? A[(rowBase + ar) * N + base + ak] : 0.0;
            tileB[bk][bc] = (FullTiles || (base + bk < N && colBase + bc < N))
                               ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < KTile; ++k) {
            double a[ThreadTile];
            double b[ThreadTile];
#pragma unroll
            for (int r = 0; r < ThreadTile; ++r) {
                a[r] = tileA[k][ty + r * 16];
            }
#pragma unroll
            for (int c = 0; c < ThreadTile; ++c) {
                b[c] = tileB[k][tx + c * 16];
            }
#pragma unroll
            for (int r = 0; r < ThreadTile; ++r) {
#pragma unroll
                for (int c = 0; c < ThreadTile; ++c) {
                    sums[r][c] = __fma_rn(a[r], b[c], sums[r][c]);
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ThreadTile; ++r) {
        const size_t row = rowBase + ty + r * 16;
#pragma unroll
        for (int c = 0; c < ThreadTile; ++c) {
            const size_t col = colBase + tx + c * 16;
            if (FullTiles || (row < rows && col < N)) {
                C[row * N + col] = sums[r][c];
            }
        }
    }
}

struct DeviceMatrices {
    int device;
    size_t firstRow;
    size_t rows;
    int tile;
    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;
    cudaStream_t stream = nullptr;

    DeviceMatrices(int id, size_t first, size_t count, size_t N)
        : device(id), firstRow(first), rows(count), tile(64) {
        try {
            checkCuda(cudaSetDevice(device), "Selecting CUDA device");
            cudaDeviceProp properties{};
            checkCuda(cudaGetDeviceProperties(&properties, device), "Querying CUDA device");
            // Smaller tiles expose enough blocks to occupy the GPU on small inputs.
            if (((rows + 63) / 64) * ((N + 63) / 64) <
                static_cast<size_t>(properties.multiProcessorCount) * 2) {
                tile = 32;
            }
            if ((N + tile - 1) / tile > static_cast<size_t>(properties.maxGridSize[0]) ||
                (rows + tile - 1) / tile > static_cast<size_t>(properties.maxGridSize[1])) {
                throw std::runtime_error("Matrix exceeds CUDA grid limits");
            }
            checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                      "Creating CUDA stream");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&A), rows * N * sizeof(double)),
                      "Allocating device matrix A");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&B), N * N * sizeof(double)),
                      "Allocating device matrix B");
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&C), rows * N * sizeof(double)),
                      "Allocating device matrix C");
        } catch (...) {
            release();
            throw;
        }
    }

    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;
    ~DeviceMatrices() { release(); }

    void release() noexcept {
        cudaSetDevice(device);
        // cudaFree also waits for outstanding uses of the allocation on error paths.
        if (A) cudaFree(A);
        if (B) cudaFree(B);
        if (C) cudaFree(C);
        if (stream) cudaStreamDestroy(stream);
        A = B = C = nullptr;
        stream = nullptr;
    }

    template <int Tile>
    void launch(size_t N) const {
        const dim3 block(16, 16);
        const dim3 grid(static_cast<unsigned>((N + Tile - 1) / Tile),
                        static_cast<unsigned>((rows + Tile - 1) / Tile));
        if (N % Tile == 0 && rows % Tile == 0) {
            matrixMultiplyKernel<Tile, true><<<grid, block, 0, stream>>>(A, B, C, N, rows);
        } else {
            matrixMultiplyKernel<Tile, false><<<grid, block, 0, stream>>>(A, B, C, N, rows);
        }
        checkCuda(cudaGetLastError(), "Launching CUDA matrix multiplication");
    }
};

// Pin the output for concurrent, asynchronous downloads from multiple GPUs.
// Registration is optional: CUDA can still copy to ordinary host memory.
struct HostRegistration {
    double* data;
    bool registered;

    explicit HostRegistration(std::vector<double>& C) : data(C.data()), registered(false) {
        if (C.size() * sizeof(double) >= 1024 * 1024) {
            registered = cudaHostRegister(data, C.size() * sizeof(double),
                                          cudaHostRegisterPortable) == cudaSuccess;
            if (!registered) cudaGetLastError();
        }
    }

    ~HostRegistration() {
        if (registered) cudaHostUnregister(data);
    }
    HostRegistration(const HostRegistration&) = delete;
    HostRegistration& operator=(const HostRegistration&) = delete;
};

class CudaMatrixMultiply {
    size_t N;
    HostRegistration outputRegistration;
    std::vector<std::unique_ptr<DeviceMatrices>> devices;

public:
    CudaMatrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                       std::vector<double>& C, size_t size)
        : N(size), outputRegistration(C) {
        int available = 0;
        checkCuda(cudaGetDeviceCount(&available), "Detecting CUDA devices");
        if (available == 0) throw std::runtime_error("A CUDA GPU is required");
        if (N == 0) return;

        // Avoid the cost of replicating B on multiple devices for small problems.
        const size_t wanted = N < 1024 ? 1 : N / 512;
        const size_t count = std::min(static_cast<size_t>(available), wanted);
        const size_t rowTiles = (N + 31) / 32;
        size_t first = 0;
        for (size_t d = 0; d < count; ++d) {
            const size_t end = std::min(N, ((d + 1) * rowTiles / count) * 32);
            devices.push_back(std::make_unique<DeviceMatrices>(
                static_cast<int>(d), first, end - first, N));
            const auto& device = *devices.back();
            checkCuda(cudaMemcpyAsync(device.A, A.data() + first * N,
                                      device.rows * N * sizeof(double),
                                      cudaMemcpyHostToDevice, device.stream), "Uploading matrix A");
            checkCuda(cudaMemcpyAsync(device.B, B.data(), N * N * sizeof(double),
                                      cudaMemcpyHostToDevice, device.stream), "Uploading matrix B");
            first = end;
        }
        for (const auto& device : devices) {
            checkCuda(cudaSetDevice(device->device), "Selecting CUDA device");
            checkCuda(cudaStreamSynchronize(device->stream), "Preparing CUDA matrices");
        }
    }

    void multiply(std::vector<double>& C) const {
        // Enqueue every GPU's kernel before downloading or waiting for any result.
        for (const auto& device : devices) {
            checkCuda(cudaSetDevice(device->device), "Selecting CUDA device");
            if (device->tile == 64) device->launch<64>(N);
            else device->launch<32>(N);
        }
        for (const auto& device : devices) {
            checkCuda(cudaSetDevice(device->device), "Selecting CUDA device");
            checkCuda(cudaMemcpyAsync(C.data() + device->firstRow * N, device->C,
                                      device->rows * N * sizeof(double),
                                      cudaMemcpyDeviceToHost, device->stream), "Downloading matrix C");
        }
        for (const auto& device : devices) {
            checkCuda(cudaSetDevice(device->device), "Selecting CUDA device");
            checkCuda(cudaStreamSynchronize(device->stream), "Completing CUDA matrix multiplication");
        }
    }
};

} // namespace

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    if (N == 0) return true;
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += A[i * N + k] * B[k * N + j];
            }
            
            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));
            
            if (!std::isfinite(actual) || relError > 1e-6) {
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

int main(int argc, char** argv) try {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end != '\0' || errno == ERANGE ||
                parsed > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("Invalid matrix size");
            }
            N = static_cast<size_t>(parsed);
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
    
    // Reject size arithmetic overflow before either host or device allocation.
    if (N != 0 && N > std::numeric_limits<size_t>::max() / sizeof(double) / N) {
        throw std::runtime_error("Matrix size is too large");
    }

    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);

    // Prepare GPU resources and inputs outside the computation interval, just as
    // host allocation and input initialization are outside it. The timed operation
    // includes all kernels and the completed download of C to host memory.
    CudaMatrixMultiply multiplier(A, B, C, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::steady_clock::now();
    
    multiplier.multiply(C);
    
    auto end = std::chrono::steady_clock::now();
    const double milliseconds = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate GFLOPS
    double gflops = milliseconds > 0.0 ? (2.0 * N * N * N) / (milliseconds * 1e6) : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(C, "MatrixC");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(A, B, C, N);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
} catch (const std::exception& error) {
    fprintf(stderr, "Error: %s\n", error.what());
    return 1;
}
