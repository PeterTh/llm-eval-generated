#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
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

// A block cooperatively loads coalesced tiles, then each thread accumulates
// several output elements in registers. All accumulation remains in double
// precision and visits k in the same order as the original multiplication.
template<int Tile, bool FullTiles>
__global__ void multiplyTiles(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, int N, int rows) {
    constexpr int KTile = 16;
    constexpr int Threads = 16;
    constexpr int Results = Tile / Threads;
    __shared__ double tileA[Tile][KTile + 1];
    __shared__ double tileB[KTile][Tile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * Threads + tx;
    const int rowBase = blockIdx.y * Tile;
    const int colBase = blockIdx.x * Tile;
    double sums[Results][Results] = {};

    for (int base = 0; base < N; base += KTile) {
        #pragma unroll
        for (int offset = tid; offset < Tile * KTile; offset += Threads * Threads) {
            const int ar = offset / KTile;
            const int ak = offset % KTile;
            const int bk = offset / Tile;
            const int bc = offset % Tile;
            tileA[ar][ak] = (FullTiles || (rowBase + ar < rows && base + ak < N))
                ? A[static_cast<size_t>(rowBase + ar) * N + base + ak] : 0.0;
            tileB[bk][bc] = (FullTiles || (base + bk < N && colBase + bc < N))
                ? B[static_cast<size_t>(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < KTile; ++k) {
            double a[Results];
            double b[Results];
            #pragma unroll
            for (int r = 0; r < Results; ++r) {
                a[r] = tileA[ty + r * Threads][k];
            }
            #pragma unroll
            for (int c = 0; c < Results; ++c) {
                b[c] = tileB[k][tx + c * Threads];
            }
            #pragma unroll
            for (int r = 0; r < Results; ++r) {
                #pragma unroll
                for (int c = 0; c < Results; ++c) {
                    sums[r][c] += a[r] * b[c];
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < Results; ++r) {
        const int row = rowBase + ty + r * Threads;
        #pragma unroll
        for (int c = 0; c < Results; ++c) {
            const int col = colBase + tx + c * Threads;
            if (FullTiles || (row < rows && col < N)) {
                C[static_cast<size_t>(row) * N + col] = sums[r][c];
            }
        }
    }
}

struct DevicePartition {
    int device;
    int firstRow;
    int rows;
    cudaStream_t stream = nullptr;
    double* a = nullptr;
    double* b = nullptr;
    double* c = nullptr;

    DevicePartition(int device, int firstRow, int rows)
        : device(device), firstRow(firstRow), rows(rows) {}

    DevicePartition(const DevicePartition&) = delete;
    DevicePartition& operator=(const DevicePartition&) = delete;

    ~DevicePartition() {
        cudaSetDevice(device);
        if (stream) cudaStreamSynchronize(stream);
        if (a) cudaFree(a);
        if (b) cudaFree(b);
        if (c) cudaFree(c);
        if (stream) cudaStreamDestroy(stream);
    }

    void prepare(const std::vector<double>& A, const std::vector<double>& B, int N) {
        checkCuda(cudaSetDevice(device), "Select CUDA device");
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Create CUDA stream");
        const size_t rowBytes = static_cast<size_t>(rows) * N * sizeof(double);
        const size_t matrixBytes = static_cast<size_t>(N) * N * sizeof(double);
        checkCuda(cudaMalloc(&a, rowBytes), "Allocate CUDA matrix A");
        checkCuda(cudaMalloc(&b, matrixBytes), "Allocate CUDA matrix B");
        checkCuda(cudaMalloc(&c, rowBytes), "Allocate CUDA matrix C");
        checkCuda(cudaMemcpyAsync(a, A.data() + static_cast<size_t>(firstRow) * N,
                                  rowBytes, cudaMemcpyHostToDevice, stream), "Upload matrix A");
        checkCuda(cudaMemcpyAsync(b, B.data(), matrixBytes,
                                  cudaMemcpyHostToDevice, stream), "Upload matrix B");
    }

    template<int Tile>
    void loadKernel(int N) {
        // Load the selected kernel before starting the computation timer.
        cudaFuncAttributes attributes;
        if (N % Tile == 0 && rows % Tile == 0) {
            checkCuda(cudaFuncGetAttributes(&attributes, multiplyTiles<Tile, true>), "Load CUDA kernel");
        } else {
            checkCuda(cudaFuncGetAttributes(&attributes, multiplyTiles<Tile, false>), "Load CUDA kernel");
        }
    }

    template<int Tile>
    void launch(int N) {
        const dim3 block(16, 16);
        const dim3 grid((N + Tile - 1) / Tile, (rows + Tile - 1) / Tile);
        if (N % Tile == 0 && rows % Tile == 0) {
            multiplyTiles<Tile, true><<<grid, block, 0, stream>>>(a, b, c, N, rows);
        } else {
            multiplyTiles<Tile, false><<<grid, block, 0, stream>>>(a, b, c, N, rows);
        }
        checkCuda(cudaGetLastError(), "Launch CUDA matrix multiplication");
    }

    void synchronize() {
        checkCuda(cudaSetDevice(device), "Select CUDA device");
        checkCuda(cudaStreamSynchronize(stream), "Complete CUDA operation");
    }
};

class CudaMatrixMultiply {
    // Smaller tiles expose more parallel blocks for modest matrix sizes.
    static constexpr int SmallTileLimit = 2048;
    int N;
    std::vector<std::unique_ptr<DevicePartition>> partitions;

public:
    CudaMatrixMultiply(const std::vector<double>& A, const std::vector<double>& B, int N) : N(N) {
        int deviceCount = 0;
        checkCuda(cudaGetDeviceCount(&deviceCount), "Query CUDA devices");
        if (deviceCount == 0) {
            throw std::runtime_error("CUDA matrix multiplication requires a CUDA-capable GPU");
        }

        // Use about 1024 rows per GPU to amortize multi-device overhead.
        // Row partitions require no communication during multiplication.
        const int devices = std::min(deviceCount, std::max(1, N / 1024));
        constexpr int RowTile = 64;
        const int tiles = (N + RowTile - 1) / RowTile;
        int firstTile = 0;
        for (int device = 0; device < devices; ++device) {
            const int count = tiles / devices + (device < tiles % devices ? 1 : 0);
            const int firstRow = firstTile * RowTile;
            const int rows = std::min(count * RowTile, N - firstRow);
            auto partition = std::make_unique<DevicePartition>(device, firstRow, rows);
            partition->prepare(A, B, N);
            if (N <= SmallTileLimit) partition->loadKernel<32>(N);
            else partition->loadKernel<64>(N);
            partitions.push_back(std::move(partition));
            firstTile += count;
        }
        for (auto& partition : partitions) partition->synchronize();
    }

    void multiply() {
        // Enqueue every GPU before waiting, so their kernels execute concurrently.
        for (auto& partition : partitions) {
            checkCuda(cudaSetDevice(partition->device), "Select CUDA device");
            if (N <= SmallTileLimit) partition->launch<32>(N);
            else partition->launch<64>(N);
        }
        for (auto& partition : partitions) partition->synchronize();
    }

    void copyResult(std::vector<double>& C) {
        for (auto& partition : partitions) {
            checkCuda(cudaSetDevice(partition->device), "Select CUDA device");
            const size_t offset = static_cast<size_t>(partition->firstRow) * N;
            const size_t bytes = static_cast<size_t>(partition->rows) * N * sizeof(double);
            checkCuda(cudaMemcpyAsync(C.data() + offset, partition->c, bytes,
                                      cudaMemcpyDeviceToHost, partition->stream), "Download matrix C");
        }
        for (auto& partition : partitions) partition->synchronize();
    }
};

} // namespace

// Simple validation: compute a single element and compare
bool validateResult(const std::vector<double>& A, const std::vector<double>& B,
                   const std::vector<double>& C, const size_t N) {
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
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed == 0 || parsed > INT_MAX ||
                parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(double)) {
                fprintf(stderr, "Matrix size must be a positive integer that fits in memory.\n");
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

    try {
        // Allocate matrices
        std::vector<double> A(N * N);
        std::vector<double> B(N * N);
        std::vector<double> C(N * N);

        // Initialize matrices
        printf("Initializing matrices...\n");
        initMatrix(A, N);
        initMatrix(B, N);

        // Allocation, input transfers and CUDA module loading are preparation;
        // time the computation after every device has its input matrices.
        CudaMatrixMultiply multiplication(A, B, static_cast<int>(N));

        // Perform matrix multiplication
        printf("Computing matrix multiplication...\n");
        auto start = std::chrono::steady_clock::now();

        multiplication.multiply();

        auto end = std::chrono::steady_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();

        printf("Computation time: %.3f ms\n", seconds * 1000.0);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / seconds / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        multiplication.copyResult(C);

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
        fprintf(stderr, "Matrix multiplication failed: %s\n", error.what());
        return 1;
    }

    return 0;
}
