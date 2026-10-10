#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <thread>
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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(status));
    }
}

// Each block computes a 64 x 64 tile; each thread holds 16 outputs.
// Cooperatively loaded tiles reuse each input across many output elements.
__global__ void multiplyTiles(const double* __restrict__ A,
                              const double* __restrict__ B,
                              double* __restrict__ C, size_t N, size_t rows) {
    constexpr int tile = 64;
    constexpr int depth = 16;
    __shared__ double aTile[tile][depth + 1];
    __shared__ double bTile[depth][tile];
    // Keep the original ascending-k accumulation in double precision.
    double sums[4][4] = {};
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * tile;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * tile;

    for (size_t base = 0; base < N; base += depth) {
        #pragma unroll
        for (int offset = tid; offset < tile * depth; offset += 256) {
            const int ar = offset / depth;
            const int ak = offset % depth;
            const int bk = offset / tile;
            const int bc = offset % tile;
            aTile[ar][ak] = rowBase + ar < rows && base + ak < N
                ? A[(rowBase + ar) * N + base + ak] : 0.0;
            bTile[bk][bc] = base + bk < N && colBase + bc < N
                ? B[(base + bk) * N + colBase + bc] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < depth; ++k) {
            double av[4], bv[4];
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                av[t] = aTile[ty + t * 16][k];
                bv[t] = bTile[k][tx + t * 16];
            }
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    sums[r][c] += av[r] * bv[c];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t row = rowBase + ty + r * 16;
            const size_t col = colBase + tx + c * 16;
            if (row < rows && col < N) {
                C[row * N + col] = sums[r][c];
            }
        }
    }
}

struct DeviceBuffer {
    double* data = nullptr;
    explicit DeviceBuffer(size_t bytes) {
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&data), bytes));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

void multiplyOnDevice(const std::vector<double>& A, const std::vector<double>& B,
                      std::vector<double>& C, size_t N, int device,
                      size_t firstRow, size_t rows) {
    checkCuda(cudaSetDevice(device));
    const size_t chunkBytes = rows * N * sizeof(double);
    const size_t matrixBytes = N * N * sizeof(double);
    DeviceBuffer a(chunkBytes), b(matrixBytes), c(chunkBytes);
    checkCuda(cudaMemcpy(a.data, A.data() + firstRow * N, chunkBytes,
                         cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(b.data, B.data(), matrixBytes, cudaMemcpyHostToDevice));
    multiplyTiles<<<dim3((N + 63) / 64, (rows + 63) / 64), dim3(16, 16)>>>(
        a.data, b.data, c.data, N, rows);
    checkCuda(cudaGetLastError());
    // The synchronous copy also waits for the kernel and reports device errors.
    checkCuda(cudaMemcpy(C.data() + firstRow * N, c.data, chunkBytes,
                         cudaMemcpyDeviceToHost));
}

void matrixMultiply(const std::vector<double>& A, const std::vector<double>& B,
                    std::vector<double>& C, const size_t N) {
    int deviceCount = 0;
    checkCuda(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        throw std::runtime_error("CUDA matrix multiplication requires a GPU");
    }
    // Smaller problems cannot amortize additional device setup and B copies.
    const int workers = N < 8192 ? 1
        : static_cast<int>(std::min<size_t>(deviceCount, N / 512));
    if (workers == 1) {
        multiplyOnDevice(A, B, C, N, 0, 0, N);
        return;
    }
    std::vector<std::thread> threads;
    std::vector<std::exception_ptr> errors(workers);
    threads.reserve(workers);
    try {
        for (int device = 0; device < workers; ++device) {
            // Partition whole tiles, including the final partial tile.
            const size_t tiles = (N + 63) / 64;
            const size_t firstRow = tiles * device / workers * 64;
            const size_t endRow = std::min(N, tiles * (device + 1) / workers * 64);
            threads.emplace_back([&, device, firstRow, endRow] {
                try {
                    multiplyOnDevice(A, B, C, N, device, firstRow, endRow - firstRow);
                } catch (...) {
                    errors[device] = std::current_exception();
                }
            });
        }
    } catch (...) {
        for (auto& thread : threads) thread.join();
        throw;
    }
    for (auto& thread : threads) thread.join();
    for (const auto& error : errors) {
        if (error) std::rethrow_exception(error);
    }
}

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
            const unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '-' || end == value || *end != '\0' || errno == ERANGE || parsed == 0 ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
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
        fprintf(stderr, "Matrix size is too large\n");
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
    
        // Perform matrix multiplication
        printf("Computing matrix multiplication...\n");
        auto start = std::chrono::high_resolution_clock::now();
    
        matrixMultiply(A, B, C, N);
    
        auto end = std::chrono::high_resolution_clock::now();
        const double seconds = std::chrono::duration<double>(end - start).count();
    
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
        // Calculate GFLOPS
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
    
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "Matrix multiplication failed: %s\n", error.what());
        return 1;
    }
}
