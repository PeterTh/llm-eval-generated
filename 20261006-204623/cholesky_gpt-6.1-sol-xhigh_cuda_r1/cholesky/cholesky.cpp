#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_),
                                 count * sizeof(T)), "Allocating GPU memory");
        }
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

constexpr int Panel = 32;
constexpr int Tile = 64;
constexpr int Depth = 32;
constexpr int Threads = 256;

// glibc rand_r advances this 32-bit LCG three times per output. Skip ahead
// with affine transformations so CUDA threads reproduce the serial seed stream.
__host__ __device__ uint32_t advanceSeed(uint32_t seed, size_t steps) {
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    while (steps) {
        if (steps & 1) seed = multiplier * seed + increment;
        increment *= multiplier + 1u;
        multiplier *= multiplier;
        steps >>= 1;
    }
    return seed;
}

__global__ void randomMatrixKernel(double* matrix, size_t count,
                                   uint32_t jumpMultiplier,
                                   uint32_t jumpIncrement) {
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    uint32_t seed = advanceSeed(42u, 3 * id);
    for (size_t i = id; i < count; i += stride) {
        uint32_t next = seed * 1103515245u + 12345u;
        uint32_t value = (next >> 16) & 2047u;
        next = next * 1103515245u + 12345u;
        value = (value << 10) ^ ((next >> 16) & 1023u);
        next = next * 1103515245u + 12345u;
        value = (value << 10) ^ ((next >> 16) & 1023u);
        matrix[i] = __dadd_rn(__ddiv_rn(static_cast<double>(value),
                                       2147483647.0), -0.5);
        seed = jumpMultiplier * seed + jumpIncrement;
    }
}

enum class Product { Generate, Update, Validate };

// Each 256-thread block computes a 64 x 64 tile, with a 4 x 4 register
// accumulator per thread. Both operands are reused from shared memory.
// Only lower tiles are evaluated; Gram products are symmetric.
template <Product Operation>
__global__ void productKernel(const double* input,
                              double* output,
                              const double* __restrict__ original,
                              double* __restrict__ errors, size_t n,
                              size_t offset, size_t begin, size_t length) {
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const size_t errorIndex =
        2 * (static_cast<size_t>(blockIdx.y) * gridDim.x + blockIdx.x);
    if (blockIdx.y < blockIdx.x) {
        if constexpr (Operation == Product::Validate) {
            if (tid == 0) errors[errorIndex] = errors[errorIndex + 1] = 0.0;
        }
        return;
    }

    __shared__ double left[Depth][Tile + 1];
    __shared__ double right[Depth][Tile + 1];
    double sums[4][4] = {};
    const size_t firstRow = offset + static_cast<size_t>(blockIdx.y) * Tile;
    const size_t firstCol = offset + static_cast<size_t>(blockIdx.x) * Tile;

    for (size_t base = 0; base < length; base += Depth) {
        for (int item = tid; item < Tile * Depth; item += Threads) {
            const int row = item / Depth;
            const int k = item % Depth;
            const size_t col = begin + base + k;
            left[k][row] = (firstRow + row < n && base + k < length)
                ? input[(firstRow + row) * n + col] : 0.0;
            right[k][row] = (firstCol + row < n && base + k < length)
                ? input[(firstCol + row) * n + col] : 0.0;
        }
        __syncthreads();

        #pragma unroll 4
        for (int k = 0; k < Depth; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (int r = 0; r < 4; ++r)
                a[r] = left[k][threadIdx.y + 16 * r];
            #pragma unroll
            for (int c = 0; c < 4; ++c)
                b[c] = right[k][threadIdx.x + 16 * c];
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                #pragma unroll
                for (int c = 0; c < 4; ++c)
                    sums[r][c] = fma(a[r], b[c], sums[r][c]);
            }
        }
        __syncthreads();
    }

    double maxAbsolute = 0.0, maxRelative = 0.0;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        const size_t row = firstRow + threadIdx.y + 16 * r;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t col = firstCol + threadIdx.x + 16 * c;
            if (row < n && col < n && row >= col) {
                const double value = sums[r][c];
                if constexpr (Operation == Product::Update) {
                    output[row * n + col] -= value;
                } else if constexpr (Operation == Product::Generate) {
                    const double entry = value + (row == col ? double(n) : 0.0);
                    output[row * n + col] = entry;
                    if (row != col) output[col * n + row] = entry;
                } else {
                    const double reference = original[row * n + col];
                    double error = fabs(value - reference);
                    if (!isfinite(error)) error = CUDART_INF;
                    maxAbsolute = fmax(maxAbsolute, error);
                    maxRelative = fmax(maxRelative,
                                       error / (fabs(reference) + 1e-10));
                    if (row != col) {
                        const double upper = original[col * n + row];
                        error = fabs(value - upper);
                        if (!isfinite(error)) error = CUDART_INF;
                        maxAbsolute = fmax(maxAbsolute, error);
                        maxRelative = fmax(maxRelative,
                                           error / (fabs(upper) + 1e-10));
                    }
                }
            }
        }
    }

    if constexpr (Operation == Product::Validate) {
        __shared__ double absolute[Threads], relative[Threads];
        absolute[tid] = maxAbsolute;
        relative[tid] = maxRelative;
        __syncthreads();
        for (int step = Threads / 2; step > 0; step /= 2) {
            if (tid < step) {
                absolute[tid] = fmax(absolute[tid], absolute[tid + step]);
                relative[tid] = fmax(relative[tid], relative[tid + step]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            errors[errorIndex] = absolute[0];
            errors[errorIndex + 1] = relative[0];
        }
    }
}

// Factor a small diagonal tile cooperatively. The enclosing stream orders
// this dependency before the panel solve and the parallel trailing update.
__global__ void diagonalKernel(double* matrix, size_t n, size_t base,
                               int width, int* failure) {
    if (*failure >= 0) return;
    __shared__ double diagonal[Panel][Panel + 1];
    __shared__ int failed;
    const int tid = threadIdx.x;
    if (tid == 0) failed = 0;
    for (int item = tid; item < Panel * Panel; item += blockDim.x) {
        const int row = item / Panel, col = item % Panel;
        diagonal[row][col] = (row < width && col <= row)
            ? matrix[(base + row) * n + base + col] : 0.0;
    }
    __syncthreads();

    for (int k = 0; k < width; ++k) {
        if (tid == 0) {
            const double value = diagonal[k][k];
            if (value <= 0.0) {
                *failure = static_cast<int>(base + k);
                failed = 1;
            } else {
                diagonal[k][k] = sqrt(value);
            }
        }
        __syncthreads();
        if (failed) return;

        if (tid > k && tid < width)
            diagonal[tid][k] /= diagonal[k][k];
        __syncthreads();

        for (int item = tid; item < Panel * Panel; item += blockDim.x) {
            const int row = item / Panel, col = item % Panel;
            if (row > k && col > k && col <= row && row < width)
                diagonal[row][col] = fma(-diagonal[row][k],
                                         diagonal[col][k], diagonal[row][col]);
        }
        __syncthreads();
    }

    for (int item = tid; item < Panel * Panel; item += blockDim.x) {
        const int row = item / Panel, col = item % Panel;
        if (row < width && col <= row)
            matrix[(base + row) * n + base + col] = diagonal[row][col];
    }
}

// Solve each independent 64-row panel against the factored diagonal tile.
__global__ void panelKernel(double* matrix, size_t n, size_t base,
                            int width, const int* failure) {
    if (*failure >= 0) return;
    __shared__ double diagonal[Panel][Panel + 1];
    __shared__ double panel[Tile][Panel + 1];
    const int tid = threadIdx.x;
    const size_t firstRow = base + width + static_cast<size_t>(blockIdx.x) * Tile;
    for (int item = tid; item < Panel * Panel; item += blockDim.x) {
        const int row = item / Panel, col = item % Panel;
        diagonal[row][col] = (row < width && col <= row)
            ? matrix[(base + row) * n + base + col] : 0.0;
    }
    for (int item = tid; item < Tile * Panel; item += blockDim.x) {
        const int row = item / Panel, col = item % Panel;
        panel[row][col] = (firstRow + row < n && col < width)
            ? matrix[(firstRow + row) * n + base + col] : 0.0;
    }
    __syncthreads();

    for (int k = 0; k < width; ++k) {
        if (tid < Tile) panel[tid][k] /= diagonal[k][k];
        __syncthreads();
        for (int item = tid; item < Tile * Panel; item += blockDim.x) {
            const int row = item / Panel, col = item % Panel;
            if (col > k && col < width)
                panel[row][col] = fma(-panel[row][k], diagonal[col][k],
                                      panel[row][col]);
        }
        __syncthreads();
    }

    for (int item = tid; item < Tile * Panel; item += blockDim.x) {
        const int row = item / Panel, col = item % Panel;
        if (firstRow + row < n && col < width)
            matrix[(firstRow + row) * n + base + col] = panel[row][col];
    }
}

__global__ void clearUpperKernel(double* matrix, size_t n) {
    if (blockIdx.x < blockIdx.y) return;
    const size_t col = static_cast<size_t>(blockIdx.x) * 32 + threadIdx.x;
    for (int r = threadIdx.y; r < 32; r += blockDim.y) {
        const size_t row = static_cast<size_t>(blockIdx.y) * 32 + r;
        if (col < n && row < n && col > row) matrix[row * n + col] = 0.0;
    }
}

dim3 productGrid(size_t n) {
    const unsigned int tiles = static_cast<unsigned int>((n + Tile - 1) / Tile);
    return dim3(tiles, tiles);
}

} // namespace

// A resident matrix lets the benchmark retain its generated input on the GPU.
// Standalone calls still accept and return the original row-major host vector.
bool choleskyDecomposition(std::vector<double>& A, const size_t n,
                           double* resident = nullptr) {
    if (n == 0) return true;
    DeviceBuffer<double> storage(resident ? 0 : n * n);
    double* matrix = resident ? resident : storage.get();
    if (!resident)
        checkCuda(cudaMemcpy(matrix, A.data(), n * n * sizeof(double),
                             cudaMemcpyHostToDevice), "Uploading Cholesky input");

    DeviceBuffer<int> status(1);
    checkCuda(cudaMemset(status.get(), 0xff, sizeof(int)), "Resetting pivot status");
    for (size_t base = 0; base < n; base += Panel) {
        const int width = static_cast<int>(std::min(size_t(Panel), n - base));
        diagonalKernel<<<1, Threads>>>(matrix, n, base, width, status.get());
        const size_t trailing = n - base - width;
        if (trailing != 0) {
            panelKernel<<<static_cast<unsigned int>((trailing + Tile - 1) / Tile),
                          Threads>>>(matrix, n, base, width, status.get());
            productKernel<Product::Update><<<productGrid(trailing), dim3(16, 16)>>>(
                matrix, matrix, nullptr, nullptr, n, base + width, base, width);
        }
    }
    checkCuda(cudaGetLastError(), "Launching blocked Cholesky kernels");
    int failure = -1;
    checkCuda(cudaMemcpy(&failure, status.get(), sizeof(int), cudaMemcpyDeviceToHost),
              "Reading Cholesky pivot status");
    if (failure >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n",
               failure);
        return false;
    }
    const unsigned int tiles = static_cast<unsigned int>((n + 31) / 32);
    clearUpperKernel<<<dim3(tiles, tiles), dim3(32, 8)>>>(matrix, n);
    checkCuda(cudaGetLastError(), "Clearing the upper triangle");
    checkCuda(cudaMemcpy(A.data(), matrix, n * n * sizeof(double),
                         cudaMemcpyDeviceToHost), "Downloading Cholesky factor");
    return true;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n,
                                   double* resident = nullptr) {
    if (n == 0) return;
    DeviceBuffer<double> storage(resident ? 0 : n * n);
    double* matrix = resident ? resident : storage.get();
    DeviceBuffer<double> random(n * n);

#if defined(__GLIBC__) && RAND_MAX == 2147483647
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(size_t(256), (n * n + Threads - 1) / Threads));
    const size_t stride = static_cast<size_t>(blocks) * Threads;
    const uint32_t increment = advanceSeed(0u, 3 * stride);
    const uint32_t multiplier = advanceSeed(1u, 3 * stride) - increment;
    randomMatrixKernel<<<blocks, Threads>>>(random.get(), n * n,
                                           multiplier, increment);
#else
    // Keep the platform's exact rand_r sequence on non-glibc systems.
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (double& value : B)
        value = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
    checkCuda(cudaMemcpy(random.get(), B.data(), n * n * sizeof(double),
                         cudaMemcpyHostToDevice), "Uploading random matrix");
#endif

    productKernel<Product::Generate><<<productGrid(n), dim3(16, 16)>>>(
        random.get(), matrix, nullptr, nullptr, n, 0, 0, n);
    checkCuda(cudaGetLastError(), "Generating positive definite matrix");
    if (!resident)
        checkCuda(cudaMemcpy(A.data(), matrix, n * n * sizeof(double),
                             cudaMemcpyDeviceToHost), "Downloading generated matrix");
    // random's cudaFree waits for its outstanding uses before releasing it.
}

bool validateCholesky(const std::vector<double>& L,
                      const std::vector<double>& A_orig, const size_t n,
                      const double* resident = nullptr) {
    double maxError = 0.0, relError = 0.0;
    if (n != 0) {
        DeviceBuffer<double> factor(resident ? 0 : n * n);
        const double* matrix = resident ? resident : factor.get();
        DeviceBuffer<double> original(n * n);
        const dim3 grid = productGrid(n);
        const size_t errorCount = 2 * static_cast<size_t>(grid.x) * grid.y;
        DeviceBuffer<double> errors(errorCount);
        if (!resident)
            checkCuda(cudaMemcpy(factor.get(), L.data(), n * n * sizeof(double),
                                 cudaMemcpyHostToDevice), "Uploading factor for validation");
        checkCuda(cudaMemcpy(original.get(), A_orig.data(), n * n * sizeof(double),
                             cudaMemcpyHostToDevice), "Uploading validation reference");
        productKernel<Product::Validate><<<grid, dim3(16, 16)>>>(
            matrix, nullptr, original.get(), errors.get(), n, 0, 0, n);
        checkCuda(cudaGetLastError(), "Launching Cholesky validation");
        std::vector<double> partial(errorCount);
        checkCuda(cudaMemcpy(partial.data(), errors.get(), errorCount * sizeof(double),
                             cudaMemcpyDeviceToHost), "Reading validation errors");
        for (size_t i = 0; i < errorCount; i += 2) {
            maxError = std::max(maxError, partial[i]);
            relError = std::max(relError, partial[i + 1]);
        }
    }
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) try {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' ||
                parsed > std::numeric_limits<size_t>::max()) {
                fprintf(stderr, "Invalid matrix size: %s\n", value);
                return 1;
            }
            n = static_cast<size_t>(parsed);
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (n != 0 && n > std::numeric_limits<size_t>::max() / sizeof(double) / n) {
        fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }
    // Initialize CUDA before timing, and keep the matrix resident between stages.
    checkCuda(cudaFree(nullptr), "Initializing CUDA");
    DeviceBuffer<double> deviceA(n * n);
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, deviceA.get());
    
    if (validate) {
        A_orig.resize(n * n);
        if (n != 0)
            checkCuda(cudaMemcpy(A_orig.data(), deviceA.get(), n * n * sizeof(double),
                                 cudaMemcpyDeviceToHost), "Saving validation reference");
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, deviceA.get());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n, deviceA.get());
        
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
