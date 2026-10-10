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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// A block reuses two shared-memory tiles. Each thread accumulates a small
// register tile, retaining double precision and increasing k in order.
template <int Tile, bool FullTiles>
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C,
                                     size_t N, size_t rows) {
    constexpr int Threads = 16;
    constexpr int Depth = 16;
    constexpr int Results = Tile / Threads;
    __shared__ double tileA[Tile][Depth + 1];
    __shared__ double tileB[Depth][Tile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int thread = ty * Threads + tx;
    const size_t firstRow = static_cast<size_t>(blockIdx.y) * Tile;
    const size_t firstCol = static_cast<size_t>(blockIdx.x) * Tile;
    double sums[Results][Results] = {};

    for (size_t base = 0; base < N; base += Depth) {
        #pragma unroll
        for (int index = thread; index < Tile * Depth; index += Threads * Threads) {
            const int row = index / Depth;
            const int k = index % Depth;
            if constexpr (FullTiles) {
                tileA[row][k] = A[(firstRow + row) * N + base + k];
            } else {
                tileA[row][k] = firstRow + row < rows && base + k < N
                    ? A[(firstRow + row) * N + base + k] : 0.0;
            }
            const int bk = index / Tile;
            const int col = index % Tile;
            if constexpr (FullTiles) {
                tileB[bk][col] = B[(base + bk) * N + firstCol + col];
            } else {
                tileB[bk][col] = base + bk < N && firstCol + col < N
                    ? B[(base + bk) * N + firstCol + col] : 0.0;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < Depth; ++k) {
            double a[Results];
            double b[Results];
            #pragma unroll
            for (int i = 0; i < Results; ++i) {
                a[i] = tileA[ty + i * Threads][k];
                b[i] = tileB[k][tx + i * Threads];
            }
            #pragma unroll
            for (int i = 0; i < Results; ++i) {
                #pragma unroll
                for (int j = 0; j < Results; ++j) {
                    sums[i][j] += a[i] * b[j];
                }
            }
        }
        // Every reader must finish before the next tiles overwrite shared memory.
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < Results; ++i) {
        const size_t row = firstRow + ty + i * Threads;
        #pragma unroll
        for (int j = 0; j < Results; ++j) {
            const size_t col = firstCol + tx + j * Threads;
            if (FullTiles || (row < rows && col < N)) {
                C[row * N + col] = sums[i][j];
            }
        }
    }
}

struct HostRegistration {
    double* data = nullptr;

    ~HostRegistration() {
        if (data) cudaHostUnregister(data);
    }

    void registerBuffer(double* buffer, size_t bytes) {
        checkCuda(cudaHostRegister(buffer, bytes, cudaHostRegisterPortable),
                  "Registering result memory");
        data = buffer;
    }
};

struct DeviceWork {
    int device = -1;
    size_t firstRow = 0;
    size_t rows = 0;
    double* A = nullptr;
    double* B = nullptr;
    double* C = nullptr;
    cudaStream_t stream = nullptr;

    ~DeviceWork() {
        if (device < 0) return;
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (A) cudaFree(A);
        if (B) cudaFree(B);
        if (C) cudaFree(C);
        if (stream) cudaStreamDestroy(stream);
    }
};

class CudaMatmul {
public:
    CudaMatmul(const std::vector<double>& A, const std::vector<double>& B,
               std::vector<double>& C, size_t N) : N_(N), C_(C.data()) {
        int available = 0;
        checkCuda(cudaGetDeviceCount(&available), "Enumerating CUDA devices");
        if (available == 0) throw std::runtime_error("A CUDA GPU is required");

        cudaDeviceProp properties{};
        checkCuda(cudaGetDeviceProperties(&properties, 0), "Querying CUDA device");
        const size_t rowTiles = (N + 63) / 64;
        // Small matrices stay on one GPU. Larger matrices split whole row tiles
        // across visible GPUs, with enough blocks per GPU to keep its SMs busy.
        const size_t usefulDevices = N < 2048 ? 1 : std::max<size_t>(
            1, rowTiles * rowTiles / (2 * properties.multiProcessorCount));
        const int devices = static_cast<int>(std::min({static_cast<size_t>(available),
                                                      rowTiles, usefulDevices}));
        resultMemory_.registerBuffer(C_, N * N * sizeof(double));
        work_.reserve(devices);
        for (int device = 0; device < devices; ++device) {
            work_.emplace_back(std::make_unique<DeviceWork>());
            DeviceWork& work = *work_.back();
            work.device = device;
            work.firstRow = rowTiles * device / devices * 64;
            const size_t lastRow = std::min(N, rowTiles * (device + 1) / devices * 64);
            work.rows = lastRow - work.firstRow;
            checkCuda(cudaSetDevice(device), "Selecting CUDA device");
            checkCuda(cudaStreamCreateWithFlags(&work.stream, cudaStreamNonBlocking),
                      "Creating CUDA stream");
            const size_t chunkBytes = work.rows * N * sizeof(double);
            checkCuda(cudaMalloc(&work.A, chunkBytes), "Allocating device matrix A");
            checkCuda(cudaMalloc(&work.B, N * N * sizeof(double)), "Allocating device matrix B");
            checkCuda(cudaMalloc(&work.C, chunkBytes), "Allocating device matrix C");
            checkCuda(cudaMemcpyAsync(work.A, A.data() + work.firstRow * N, chunkBytes,
                                      cudaMemcpyHostToDevice, work.stream), "Uploading matrix A");
            checkCuda(cudaMemcpyAsync(work.B, B.data(), N * N * sizeof(double),
                                      cudaMemcpyHostToDevice, work.stream), "Uploading matrix B");
            checkCuda(cudaStreamSynchronize(work.stream), "Preparing device matrices");
        }
    }

    void matrixMultiply() {
        // Submit all devices before waiting, so independent row slabs run concurrently.
        for (const auto& item : work_) {
            const DeviceWork& work = *item;
            checkCuda(cudaSetDevice(work.device), "Selecting CUDA device");
            if (N_ < 128) launch<16>(work);
            else if (N_ < 1024) launch<32>(work);
            else launch<64>(work);
            checkCuda(cudaGetLastError(), "Launching matrix multiplication");
            checkCuda(cudaMemcpyAsync(C_ + work.firstRow * N_, work.C,
                                      work.rows * N_ * sizeof(double),
                                      cudaMemcpyDeviceToHost, work.stream), "Downloading matrix C");
        }
        for (const auto& item : work_) {
            checkCuda(cudaSetDevice(item->device), "Selecting CUDA device");
            checkCuda(cudaStreamSynchronize(item->stream), "Completing matrix multiplication");
        }
    }

private:
    template <int Tile>
    void launch(const DeviceWork& work) {
        const dim3 threads(16, 16);
        const dim3 blocks(static_cast<unsigned>((N_ + Tile - 1) / Tile),
                          static_cast<unsigned>((work.rows + Tile - 1) / Tile));
        if (N_ % Tile == 0 && work.rows % Tile == 0) {
            matrixMultiplyKernel<Tile, true><<<blocks, threads, 0, work.stream>>>(
                work.A, work.B, work.C, N_, work.rows);
        } else {
            matrixMultiplyKernel<Tile, false><<<blocks, threads, 0, work.stream>>>(
                work.A, work.B, work.C, N_, work.rows);
        }
    }

    size_t N_;
    double* C_;
    // Device streams are destroyed before unregistering their host destination.
    HostRegistration resultMemory_;
    std::vector<std::unique_ptr<DeviceWork>> work_;
};

} // namespace

// Validate both the original positions and samples across the GPU row slabs.
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
    const size_t checkPoints[] = {0, 1, 2, 3, 4, N / 4, N / 2, 3 * (N / 4), N - 1};
    
    for (size_t row : checkPoints) {
        for (size_t col : checkPoints) {
            const size_t i = row % N;
            const size_t j = col % N;
            
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

int main(int argc, char** argv) {
    size_t N = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || end == value || *end != '\0' || *value == '-' || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Matrix size must be a positive integer.\n");
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
    
    printf("Matrix Multiplication Benchmark\n");
    printf("Matrix size: %zu x %zu\n", N, N);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (N > std::numeric_limits<size_t>::max() / N / sizeof(double)) {
        fprintf(stderr, "Matrix size is too large.\n");
        return 1;
    }

    try {
        // Allocate matrices
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        std::vector<double> C(N * N);

        // Initialize matrices
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);
        // Allocate and upload outside the computation timer, just like host setup.
        CudaMatmul gpu(A, B, C, N);

        // Perform matrix multiplication
        printf("Computing matrix multiplication...\n");
        auto start = std::chrono::high_resolution_clock::now();

        gpu.matrixMultiply();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS
        const double seconds = std::chrono::duration<double>(end - start).count();
        double gflops = (2.0 * N * N * N) / seconds / 1e9;
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

    } catch (const std::exception& error) {
        fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
    
    return 0;
}
