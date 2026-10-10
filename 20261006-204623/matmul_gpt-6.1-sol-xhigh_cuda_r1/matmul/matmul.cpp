#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

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

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
    }
}

constexpr size_t tileSize = 64;
constexpr unsigned tileDepth = 16;

// Each thread keeps a 4x4 output tile in registers, and all threads reuse
// input tiles in shared memory. Shorter row tiles expose more blocks for
// small matrices; large matrices use 64x64 tiles for greater input reuse.
template <bool aligned, unsigned tileRows>
__global__ void multiplyTiles(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, size_t N, size_t rows) {
    __shared__ double tileA[tileRows][tileDepth + 1];
    __shared__ double tileB[tileDepth][tileSize];
    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    constexpr unsigned rowStride = tileRows / 4;
    constexpr unsigned threads = 16 * rowStride;
    const unsigned tid = ty * 16 + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * tileRows;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * tileSize;
    double sums[4][4] = {};

    for (size_t base = 0; base < N; base += tileDepth) {
        #pragma unroll
        for (unsigned p = 0; p < 4; ++p) {
            const unsigned aRow = (tid + p * threads) / tileDepth;
            const unsigned aCol = (tid + p * threads) % tileDepth;
            tileA[aRow][aCol] = (aligned || (rowBase + aRow < rows && base + aCol < N))
                ? A[(rowBase + aRow) * N + base + aCol] : 0.0;
        }
        #pragma unroll
        for (unsigned p = 0; p < tileDepth * tileSize / threads; ++p) {
            const unsigned bRow = (tid + p * threads) / tileSize;
            const unsigned bCol = (tid + p * threads) % tileSize;
            tileB[bRow][bCol] = (aligned || (base + bRow < N && colBase + bCol < N))
                ? B[(base + bRow) * N + colBase + bCol] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (unsigned k = 0; k < tileDepth; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (unsigned p = 0; p < 4; ++p) {
                a[p] = tileA[ty + p * rowStride][k];
                b[p] = tileB[k][tx + p * 16];
            }
            #pragma unroll
            for (unsigned i = 0; i < 4; ++i) {
                #pragma unroll
                for (unsigned j = 0; j < 4; ++j) {
                    // Keep the original double-precision multiply/add rounding
                    // and increasing-k accumulation order (no reduced precision).
                    sums[i][j] = __dadd_rn(sums[i][j], __dmul_rn(a[i], b[j]));
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (unsigned i = 0; i < 4; ++i) {
        #pragma unroll
        for (unsigned j = 0; j < 4; ++j) {
            const size_t row = rowBase + ty + i * rowStride;
            const size_t col = colBase + tx + j * 16;
            if (aligned || (row < rows && col < N)) {
                C[row * N + col] = sums[i][j];
            }
        }
    }
}

class CudaMatmul {
    struct Device {
        int id = 0;
        size_t firstRow = 0;
        size_t rows = 0;
        cudaStream_t stream = nullptr;
        double* a = nullptr;
        double* b = nullptr;
        double* c = nullptr;
    };

    size_t n;
    size_t rowTile;
    std::vector<Device> devices;
    double* pinnedOutput = nullptr;

    template <unsigned tileRows>
    void launch(const Device& device) {
        const dim3 block(16, tileRows / 4);
        const dim3 grid(static_cast<unsigned>((n + tileSize - 1) / tileSize),
                        static_cast<unsigned>((device.rows + tileRows - 1) / tileRows));
        if (n % tileSize == 0) {
            multiplyTiles<true, tileRows><<<grid, block, 0, device.stream>>>(device.a, device.b, device.c, n, device.rows);
        } else {
            multiplyTiles<false, tileRows><<<grid, block, 0, device.stream>>>(device.a, device.b, device.c, n, device.rows);
        }
        checkCuda(cudaGetLastError(), "Launch matrix multiplication");
    }

    template <unsigned tileRows>
    void prepareKernel() {
        cudaFuncAttributes attributes{};
        // Load the kernel before timing, including any driver-side PTX JIT.
        if (n % tileSize == 0) {
            checkCuda(cudaFuncGetAttributes(&attributes, multiplyTiles<true, tileRows>), "Load CUDA kernel");
        } else {
            checkCuda(cudaFuncGetAttributes(&attributes, multiplyTiles<false, tileRows>), "Load CUDA kernel");
        }
    }

    void release() noexcept {
        for (auto& device : devices) {
            cudaSetDevice(device.id);
            if (device.stream) cudaStreamSynchronize(device.stream);
            if (device.a) cudaFree(device.a);
            if (device.b) cudaFree(device.b);
            if (device.c) cudaFree(device.c);
            if (device.stream) cudaStreamDestroy(device.stream);
        }
        if (pinnedOutput) cudaHostUnregister(pinnedOutput);
    }

public:
    // Setup and input uploads are outside the computation timer. The output is
    // pinned so row slabs can be downloaded concurrently from different GPUs.
    CudaMatmul(const std::vector<double>& A, const std::vector<double>& B,
               std::vector<double>& C, size_t N) : n(N), rowTile(N < 1024 ? 16 : tileSize) {
        try {
            int deviceCount = 0;
            checkCuda(cudaGetDeviceCount(&deviceCount), "Get CUDA device count");
            if (deviceCount == 0) throw std::runtime_error("A CUDA GPU is required");

            cudaDeviceProp properties{};
            checkCuda(cudaGetDeviceProperties(&properties, 0), "Get CUDA device properties");
            const size_t tiles = (n + tileSize - 1) / tileSize;
            // Give each GPU enough blocks to occupy its multiprocessors. This
            // avoids spreading small problems over otherwise idle devices.
            const size_t usefulDevices = std::max<size_t>(1,
                tiles * tiles / (2 * static_cast<size_t>(properties.multiProcessorCount)));
            const size_t rowTiles = (n + rowTile - 1) / rowTile;
            const size_t count = std::min({static_cast<size_t>(deviceCount), usefulDevices, rowTiles});
            devices.resize(count);

            checkCuda(cudaHostRegister(C.data(), C.size() * sizeof(double), cudaHostRegisterPortable),
                      "Pin result buffer");
            pinnedOutput = C.data();
            for (size_t i = 0; i < count; ++i) {
                auto& device = devices[i];
                device.id = static_cast<int>(i);
                device.firstRow = (rowTiles * i / count) * rowTile;
                const size_t end = std::min(n, (rowTiles * (i + 1) / count) * rowTile);
                device.rows = end - device.firstRow;
                checkCuda(cudaSetDevice(device.id), "Select CUDA device");
                checkCuda(cudaStreamCreateWithFlags(&device.stream, cudaStreamNonBlocking),
                          "Create CUDA stream");
                const size_t slabBytes = device.rows * n * sizeof(double);
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.a), slabBytes), "Allocate device A");
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.b), B.size() * sizeof(double)),
                          "Allocate device B");
                checkCuda(cudaMalloc(reinterpret_cast<void**>(&device.c), slabBytes), "Allocate device C");
                checkCuda(cudaMemcpyAsync(device.a, A.data() + device.firstRow * n, slabBytes,
                                          cudaMemcpyHostToDevice, device.stream), "Upload A");
                checkCuda(cudaMemcpyAsync(device.b, B.data(), B.size() * sizeof(double),
                                          cudaMemcpyHostToDevice, device.stream), "Upload B");
                if (rowTile == 16) prepareKernel<16>();
                else prepareKernel<tileSize>();
            }
            for (const auto& device : devices) {
                checkCuda(cudaSetDevice(device.id), "Select CUDA device");
                checkCuda(cudaStreamSynchronize(device.stream), "Finish input uploads");
            }
        } catch (...) {
            release();
            throw;
        }
    }

    CudaMatmul(const CudaMatmul&) = delete;
    CudaMatmul& operator=(const CudaMatmul&) = delete;
    ~CudaMatmul() { release(); }

    void multiply(std::vector<double>& C) {
        // Submit every GPU's work before waiting for any of them.
        for (const auto& device : devices) {
            checkCuda(cudaSetDevice(device.id), "Select CUDA device");
            if (rowTile == 16) launch<16>(device);
            else launch<tileSize>(device);
            checkCuda(cudaMemcpyAsync(C.data() + device.firstRow * n, device.c,
                                      device.rows * n * sizeof(double), cudaMemcpyDeviceToHost,
                                      device.stream), "Download result");
        }
        for (const auto& device : devices) {
            checkCuda(cudaSetDevice(device.id), "Select CUDA device");
            checkCuda(cudaStreamSynchronize(device.stream), "Finish matrix multiplication");
        }
    }
};

} // namespace

// Recompute representative elements on the CPU, including boundary rows.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    const size_t checkPoints[] = {0, N / 4, N / 2, 3 * N / 4, N - 1};
    
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
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Matrix size must be a positive integer\n");
                return 1;
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
    
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrices
    std::vector<double> A(N * N);
    std::vector<double> B(N * N);
    std::vector<double> C(N * N);
    
    // Initialize matrices
    printf("Initializing matrices...\n");
    initMatrix(A, N);
    initMatrix(B, N);
    CudaMatmul multiplication(A, B, C, N);
    
    // Perform matrix multiplication
    printf("Computing matrix multiplication...\n");
    auto start = std::chrono::steady_clock::now();
    
    multiplication.multiply(C);
    
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %lld ms\n", static_cast<long long>(duration.count()));
    
    // Calculate GFLOPS
    double gflops = (2.0 * N * N * N) / std::chrono::duration<double>(end - start).count() / 1e9;
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
