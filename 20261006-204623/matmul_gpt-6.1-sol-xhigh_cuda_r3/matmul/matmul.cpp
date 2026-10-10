#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

// Generate pseudo-random values for matrix initialization
__host__ __device__ constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

// Both inputs use the same deterministic initialization as the CPU benchmark.
__global__ void initMatrix(double* A, double* B, const size_t N) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < N * N; index += stride) {
        const double value = getPseudoRndValue(N, index / N, index % N);
        A[index] = value;
        B[index] = value;
    }
}

constexpr unsigned tileRows = 64;
constexpr unsigned tileCols = 64;
constexpr unsigned tileDepth = 16;
constexpr unsigned threadsX = 32;
constexpr unsigned threadsY = 8;

// Each warp accesses consecutive columns, and each thread keeps an 8x2 output
// tile in registers. Shared input tiles are reused for 4096 output elements.
template <bool fullTiles>
__global__ void matrixMultiplyKernel(const double* __restrict__ A,
                                     const double* __restrict__ B,
                                     double* __restrict__ C, const size_t N) {
    __shared__ double aTile[tileRows][tileDepth + 1];
    __shared__ double bTile[tileDepth][tileCols];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned thread = ty * threadsX + tx;
    const size_t rowBase = static_cast<size_t>(blockIdx.y) * tileRows;
    const size_t colBase = static_cast<size_t>(blockIdx.x) * tileCols;
    double sums[tileRows / threadsY][tileCols / threadsX] = {};

    for (size_t base = 0; base < N; base += tileDepth) {
        // Coalesced global loads; edge tiles are zero padded.
        #pragma unroll
        for (unsigned index = thread; index < tileRows * tileDepth;
             index += threadsX * threadsY) {
            const unsigned row = index / tileDepth;
            const unsigned k = index % tileDepth;
            aTile[row][k] = (fullTiles || (rowBase + row < N && base + k < N))
                ? A[(rowBase + row) * N + base + k] : 0.0;
        }
        #pragma unroll
        for (unsigned index = thread; index < tileDepth * tileCols;
             index += threadsX * threadsY) {
            const unsigned k = index / tileCols;
            const unsigned col = index % tileCols;
            bTile[k][col] = (fullTiles || (base + k < N && colBase + col < N))
                ? B[(base + k) * N + colBase + col] : 0.0;
        }
        __syncthreads();

        // Accumulate in increasing k order, using full-precision double FMA.
        #pragma unroll
        for (unsigned k = 0; k < tileDepth; ++k) {
            double aValues[tileRows / threadsY];
            double bValues[tileCols / threadsX];
            #pragma unroll
            for (unsigned row = 0; row < tileRows / threadsY; ++row) {
                aValues[row] = aTile[ty + row * threadsY][k];
            }
            #pragma unroll
            for (unsigned col = 0; col < tileCols / threadsX; ++col) {
                bValues[col] = bTile[k][tx + col * threadsX];
            }
            #pragma unroll
            for (unsigned row = 0; row < tileRows / threadsY; ++row) {
                #pragma unroll
                for (unsigned col = 0; col < tileCols / threadsX; ++col) {
                    sums[row][col] = fma(aValues[row], bValues[col], sums[row][col]);
                }
            }
        }
        // All readers must finish before the next tile overwrites shared memory.
        __syncthreads();
    }

    #pragma unroll
    for (unsigned row = 0; row < tileRows / threadsY; ++row) {
        const size_t i = rowBase + ty + row * threadsY;
        #pragma unroll
        for (unsigned col = 0; col < tileCols / threadsX; ++col) {
            const size_t j = colBase + tx + col * threadsX;
            if (fullTiles || (i < N && j < N)) {
                C[i * N + j] = sums[row][col];
            }
        }
    }
}

// Allocate once, outside the timed multiplication, and release on every exit.
class DeviceMatrices {
public:
    explicit DeviceMatrices(size_t N) : elements_(N * N) {
        checkCuda(cudaSetDevice(0), "Selecting CUDA device");
        if (elements_ != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                 3 * elements_ * sizeof(double)), "Allocating GPU matrices");
        }
    }

    ~DeviceMatrices() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }

    DeviceMatrices(const DeviceMatrices&) = delete;
    DeviceMatrices& operator=(const DeviceMatrices&) = delete;

    double* A() const { return data_; }
    double* B() const { return data_ == nullptr ? nullptr : data_ + elements_; }
    double* C() const { return data_ == nullptr ? nullptr : data_ + 2 * elements_; }

private:
    size_t elements_;
    double* data_ = nullptr;
};

void matrixMultiply(const DeviceMatrices& matrices, const size_t N) {
    if (N == 0) {
        return;
    }
    const dim3 threads(threadsX, threadsY);
    const dim3 blocks(static_cast<unsigned>((N + tileCols - 1) / tileCols),
                      static_cast<unsigned>((N + tileRows - 1) / tileRows));
    if (N % tileRows == 0 && N % tileCols == 0 && N % tileDepth == 0) {
        matrixMultiplyKernel<true><<<blocks, threads>>>(matrices.A(), matrices.B(), matrices.C(), N);
    } else {
        matrixMultiplyKernel<false><<<blocks, threads>>>(matrices.A(), matrices.B(), matrices.C(), N);
    }
    checkCuda(cudaGetLastError(), "Launching CUDA multiplication");
    checkCuda(cudaDeviceSynchronize(), "Completing CUDA multiplication");
}

// Validate the same 25 positions against independently computed CPU results.
bool validateResult(const std::vector<double>& C, const size_t N) {
    if (N == 0) {
        return true;
    }
    // Check the original fixed positions.
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};
    
    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;
            
            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * getPseudoRndValue(N, k, j);
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
            if (errno != 0 || end == value || *end != '\0' || *value == '-' ||
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
    
    // Check the combined allocation before forming N*N or multiplying bytes.
    if (N != 0 && N > std::numeric_limits<size_t>::max() / sizeof(double) / 3 / N) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    try {
        DeviceMatrices matrices(N);

        // Initialize matrices
        printf("Initializing matrices...\n");
        if (N != 0) {
            const unsigned blocks = static_cast<unsigned>(std::min<size_t>((N * N + 255) / 256, 4096));
            initMatrix<<<blocks, 256>>>(matrices.A(), matrices.B(), N);
            checkCuda(cudaGetLastError(), "Launching CUDA initialization");
            checkCuda(cudaDeviceSynchronize(), "Completing CUDA initialization");
        }

        // Perform matrix multiplication
        printf("Computing matrix multiplication...\n");
        auto start = std::chrono::steady_clock::now();

        matrixMultiply(matrices, N);

        auto end = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

        printf("Computation time: %lld ms\n", static_cast<long long>(duration.count()));

        // Calculate GFLOPS with sub-millisecond precision for small matrices.
        const double seconds = std::chrono::duration<double>(end - start).count();
        double gflops = seconds > 0.0 ? (2.0 * N * N * N) / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Transfer results only when the existing output/validation options need them.
        std::vector<double> C;
        if (printResults || validate) {
            C.resize(N * N);
            if (!C.empty()) {
                checkCuda(cudaMemcpy(C.data(), matrices.C(), C.size() * sizeof(double),
                                     cudaMemcpyDeviceToHost), "Copying CUDA results");
            }
        }

        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(C, N);

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
