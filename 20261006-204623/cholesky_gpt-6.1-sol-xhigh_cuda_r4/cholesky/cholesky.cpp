#include <algorithm>
#include <cerrno>
#include <chrono>
#include <climits>
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

namespace {

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(error));
    }
}

// All buffers stay on the GPU through generation, factorization and validation.
// Destruction also releases allocations when an operation throws.
template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                      "cudaMalloc");
        }
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

constexpr int panelSize = 32;
constexpr int productSize = 64;
constexpr int threadsPerBlock = 256;
enum class Product { Generate, Update, Validate };

// Pack the lower triangular tiles into a one-dimensional launch, so there are
// no idle blocks above the diagonal.
__device__ void lowerTile(int& row, int& col) {
    row = static_cast<int>((sqrt(8.0 * blockIdx.x + 1.0) - 1.0) * 0.5);
    col = static_cast<int>(blockIdx.x) - row * (row + 1) / 2;
}

unsigned int productBlocks(int extent, int tileSize = productSize) {
    const size_t tiles = (static_cast<size_t>(extent) + tileSize - 1) / tileSize;
    return static_cast<unsigned int>(tiles * (tiles + 1) / 2);
}

// A register-blocked double precision matrix product. Each thread calculates
// a 4x4 or 2x2 tile; coalesced loads and shared memory reuse both panel operands.
// The same kernel generates B*B^T, updates the Schur complement, and validates
// L*L^T without materializing a second reconstruction matrix.
template <Product operation, int tileSize = productSize>
__global__ void productKernel(const double* input, double* output,
                              const double* original, double* errors,
                              int n, int offset, const int* info) {
    if constexpr (operation == Product::Update) {
        if (*info >= 0) return;
    }
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;
    int tileRow, tileCol;
    lowerTile(tileRow, tileCol);
    const int start = operation == Product::Update ? offset + panelSize : 0;
    const int rowBase = start + tileRow * tileSize;
    const int colBase = start + tileCol * tileSize;
    const size_t stride = static_cast<size_t>(n);
    __shared__ double left[tileSize][panelSize + 1];
    __shared__ double right[tileSize][panelSize + 1];
    constexpr int cells = tileSize / 16;
    double sums[cells][cells] = {};
    int end = n;
    if constexpr (operation == Product::Update) end = offset + panelSize;
    if constexpr (operation == Product::Validate) end = min(n, colBase + tileSize);
    const int first = operation == Product::Update ? offset : 0;
    for (int base = first; base < end; base += panelSize) {
        for (int index = tid; index < tileSize * panelSize; index += threadsPerBlock) {
            const int row = index / panelSize;
            const int k = index % panelSize;
            const int column = base + k;
            left[row][k] = rowBase + row < n && column < n
                ? input[static_cast<size_t>(rowBase + row) * stride + column] : 0.0;
            right[row][k] = colBase + row < n && column < n
                ? input[static_cast<size_t>(colBase + row) * stride + column] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < panelSize; ++k) {
            double a[cells], b[cells];
            #pragma unroll
            for (int i = 0; i < cells; ++i) {
                a[i] = left[ty + 16 * i][k];
                b[i] = right[tx + 16 * i][k];
            }
            #pragma unroll
            for (int i = 0; i < cells; ++i) {
                #pragma unroll
                for (int j = 0; j < cells; ++j) {
                    sums[i][j] = fma(a[i], b[j], sums[i][j]);
                }
            }
        }
        __syncthreads();
    }
    double maxAbsolute = 0.0, maxRelative = 0.0;
    #pragma unroll
    for (int i = 0; i < cells; ++i) {
        const int row = rowBase + ty + 16 * i;
        #pragma unroll
        for (int j = 0; j < cells; ++j) {
            const int col = colBase + tx + 16 * j;
            if (row < n && col < n && col <= row) {
                const size_t index = static_cast<size_t>(row) * stride + col;
                if constexpr (operation == Product::Generate) {
                    const double value = sums[i][j] + (row == col ? n : 0.0);
                    output[index] = value;
                    if (row != col) output[static_cast<size_t>(col) * stride + row] = value;
                } else if constexpr (operation == Product::Update) {
                    output[index] -= sums[i][j];
                } else {
                    const double value = original[index];
                    const double error = fabs(sums[i][j] - value);
                    maxAbsolute = fmax(maxAbsolute, error);
                    maxRelative = fmax(maxRelative, error / (fabs(value) + 1e-10));
                    // Also compare the symmetric entry, as the original validator did.
                    const double other = original[static_cast<size_t>(col) * stride + row];
                    const double otherError = fabs(sums[i][j] - other);
                    maxAbsolute = fmax(maxAbsolute, otherError);
                    maxRelative = fmax(maxRelative, otherError / (fabs(other) + 1e-10));
                    if (!isfinite(sums[i][j])) {
                        maxAbsolute = maxRelative = INFINITY;
                    }
                }
            }
        }
    }
    if constexpr (operation == Product::Validate) {
        __shared__ double absolute[threadsPerBlock];
        __shared__ double relative[threadsPerBlock];
        absolute[tid] = maxAbsolute;
        relative[tid] = maxRelative;
        __syncthreads();
        for (int step = threadsPerBlock / 2; step > 0; step /= 2) {
            if (tid < step) {
                absolute[tid] = fmax(absolute[tid], absolute[tid + step]);
                relative[tid] = fmax(relative[tid], relative[tid + step]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            errors[2 * blockIdx.x] = absolute[0];
            errors[2 * blockIdx.x + 1] = relative[0];
        }
    }
}

// A diagonal panel fits in shared memory. Only this small tile has sequential
// pivot dependencies; the row scaling and rank-one updates are parallel.
__global__ void factorDiagonal(double* matrix, int n, int offset, int* info) {
    if (*info >= 0) return;
    __shared__ double tile[panelSize][panelSize + 1];
    __shared__ int failed;
    const int tid = threadIdx.x;
    const int width = min(panelSize, n - offset);
    const size_t stride = static_cast<size_t>(n);
    for (int index = tid; index < panelSize * panelSize; index += blockDim.x) {
        const int row = index / panelSize, col = index % panelSize;
        tile[row][col] = row < width && col <= row
            ? matrix[static_cast<size_t>(offset + row) * stride + offset + col] : 0.0;
    }
    if (tid == 0) failed = 0;
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (tid == 0) {
            const double value = tile[k][k];
            if (value <= 0.0) {
                *info = offset + k;
                failed = 1;
            } else {
                tile[k][k] = sqrt(value);
            }
        }
        __syncthreads();
        if (failed) return;
        if (tid > k && tid < width) tile[tid][k] /= tile[k][k];
        __syncthreads();
        for (int index = tid; index < panelSize * panelSize; index += blockDim.x) {
            const int row = index / panelSize, col = index % panelSize;
            if (row < width && col > k && col <= row) {
                tile[row][col] = fma(-tile[row][k], tile[col][k], tile[row][col]);
            }
        }
        __syncthreads();
    }
    for (int index = tid; index < panelSize * panelSize; index += blockDim.x) {
        const int row = index / panelSize, col = index % panelSize;
        if (row < width && col <= row) {
            matrix[static_cast<size_t>(offset + row) * stride + offset + col] = tile[row][col];
        }
    }
}

// Solve all rows below a diagonal panel independently. Shared-memory staging
// makes global reads/writes coalesced despite the rowwise triangular solves.
__global__ void solvePanel(double* matrix, int n, int offset, const int* info) {
    if (*info >= 0) return;
    __shared__ double diagonal[panelSize][panelSize + 1];
    __shared__ double rows[panelSize][panelSize + 1];
    const int tid = threadIdx.x;
    const int rowBase = offset + panelSize + blockIdx.x * panelSize;
    const size_t stride = static_cast<size_t>(n);
    for (int index = tid; index < panelSize * panelSize; index += blockDim.x) {
        const int row = index / panelSize, col = index % panelSize;
        diagonal[row][col] = col <= row
            ? matrix[static_cast<size_t>(offset + row) * stride + offset + col] : 0.0;
        rows[row][col] = rowBase + row < n
            ? matrix[static_cast<size_t>(rowBase + row) * stride + offset + col] : 0.0;
    }
    __syncthreads();
    if (tid < panelSize && rowBase + tid < n) {
        double values[panelSize];
        #pragma unroll
        for (int k = 0; k < panelSize; ++k) values[k] = rows[tid][k];
        #pragma unroll
        for (int col = 0; col < panelSize; ++col) {
            double value = values[col];
            #pragma unroll
            for (int k = 0; k < col; ++k) {
                value = fma(-values[k], diagonal[col][k], value);
            }
            values[col] = value / diagonal[col][col];
        }
        #pragma unroll
        for (int k = 0; k < panelSize; ++k) rows[tid][k] = values[k];
    }
    __syncthreads();
    for (int index = tid; index < panelSize * panelSize; index += blockDim.x) {
        const int row = index / panelSize, col = index % panelSize;
        if (rowBase + row < n) {
            matrix[static_cast<size_t>(rowBase + row) * stride + offset + col] = rows[row][col];
        }
    }
}

__global__ void zeroUpper(double* matrix, int n, const int* info) {
    if (*info >= 0) return;
    const size_t count = static_cast<size_t>(n) * n;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (index % n > index / n) matrix[index] = 0.0;
    }
}

__global__ void reduceErrors(const double* partial, size_t count, double* result) {
    const int tid = threadIdx.x;
    double a = 0.0, r = 0.0;
    for (size_t i = tid; i < count; i += blockDim.x) {
        a = fmax(a, partial[2 * i]);
        r = fmax(r, partial[2 * i + 1]);
    }
    __shared__ double absolute[threadsPerBlock], relative[threadsPerBlock];
    absolute[tid] = a;
    relative[tid] = r;
    __syncthreads();
    for (int step = threadsPerBlock / 2; step > 0; step /= 2) {
        if (tid < step) {
            absolute[tid] = fmax(absolute[tid], absolute[tid + step]);
            relative[tid] = fmax(relative[tid], relative[tid + step]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        result[0] = absolute[0];
        result[1] = relative[0];
    }
}

} // namespace

bool choleskyDecomposition(double* matrix, const size_t n) {
    if (n == 0) return true;
    DeviceBuffer<int> info(1);
    checkCuda(cudaMemset(info.get(), 0xff, sizeof(int)), "Initialize Cholesky status");
    const int size = static_cast<int>(n);
    for (int offset = 0; offset < size; offset += panelSize) {
        factorDiagonal<<<1, threadsPerBlock>>>(matrix, size, offset, info.get());
        checkCuda(cudaGetLastError(), "Launch diagonal factorization");
        const int trailing = size - offset - panelSize;
        if (trailing > 0) {
            solvePanel<<<(trailing + panelSize - 1) / panelSize, threadsPerBlock>>>(
                matrix, size, offset, info.get());
            checkCuda(cudaGetLastError(), "Launch panel solve");
            // Smaller tiles expose more independent blocks near the end of the
            // factorization, when the remaining matrix cannot fill the GPU.
            if (trailing <= 1024) {
                productKernel<Product::Update, 32><<<productBlocks(trailing, 32), dim3(16, 16)>>>(
                    matrix, matrix, nullptr, nullptr, size, offset, info.get());
            } else {
                productKernel<Product::Update><<<productBlocks(trailing), dim3(16, 16)>>>(
                    matrix, matrix, nullptr, nullptr, size, offset, info.get());
            }
            checkCuda(cudaGetLastError(), "Launch trailing matrix update");
        }
    }
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<size_t>((n * n + threadsPerBlock - 1) / threadsPerBlock, 65535));
    zeroUpper<<<blocks, threadsPerBlock>>>(matrix, size, info.get());
    checkCuda(cudaGetLastError(), "Launch upper triangle clearing");
    int failedAt;
    // This transfer waits for every factorization kernel, keeping timing honest.
    checkCuda(cudaMemcpy(&failedAt, info.get(), sizeof(int), cudaMemcpyDeviceToHost),
              "Read Cholesky status");
    if (failedAt >= 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", failedAt);
        return false;
    }
    return true;
}

void generatePositiveDefiniteMatrix(double* matrix, const size_t n) {
    if (n == 0) return;
    std::vector<double> random(n * n);
    unsigned int seed = 42;
    // Preserve the original rand_r sequence exactly; only the cubic product is
    // moved to the GPU. Replacing the generator would change the benchmark input.
    for (size_t i = 0; i < random.size(); ++i) {
        random[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    }
    DeviceBuffer<double> input(random.size());
    checkCuda(cudaMemcpy(input.get(), random.data(), random.size() * sizeof(double),
                         cudaMemcpyHostToDevice), "Upload random matrix");
    productKernel<Product::Generate><<<productBlocks(static_cast<int>(n)), dim3(16, 16)>>>(
        input.get(), matrix, nullptr, nullptr, static_cast<int>(n), 0, nullptr);
    checkCuda(cudaGetLastError(), "Launch positive definite matrix generation");
    checkCuda(cudaDeviceSynchronize(), "Generate positive definite matrix");
}

bool validateCholesky(const double* matrix, const double* original, const size_t n) {
    double errors[2] = {};
    if (n != 0) {
        const unsigned int blocks = productBlocks(static_cast<int>(n));
        DeviceBuffer<double> partial(2 * static_cast<size_t>(blocks));
        DeviceBuffer<double> result(2);
        productKernel<Product::Validate><<<blocks, dim3(16, 16)>>>(
            matrix, nullptr, original, partial.get(), static_cast<int>(n), 0, nullptr);
        checkCuda(cudaGetLastError(), "Launch Cholesky validation");
        reduceErrors<<<1, threadsPerBlock>>>(partial.get(), blocks, result.get());
        checkCuda(cudaGetLastError(), "Launch validation error reduction");
        checkCuda(cudaMemcpy(errors, result.get(), sizeof(errors), cudaMemcpyDeviceToHost),
                  "Read validation errors");
    }
    printf("Max absolute error: %.10e\n", errors[0]);
    printf("Max relative error: %.10e\n", errors[1]);
    if (errors[1] > 1e-6) {
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
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || end == value || *end != '\0' || errno == ERANGE ||
                parsed > INT_MAX ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() /
                                            sizeof(double) / parsed)) {
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
    
    // CUDA is required; failures are reported instead of selecting a CPU path.
    checkCuda(cudaFree(nullptr), "Initialize CUDA");
    DeviceBuffer<double> A(n * n);
    DeviceBuffer<double> A_orig(validate ? n * n : 0);
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A.get(), n);
    
    if (validate) {
        if (n != 0) {
            checkCuda(cudaMemcpy(A_orig.get(), A.get(), n * n * sizeof(double),
                                 cudaMemcpyDeviceToDevice), "Save original matrix");
        }
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A.get(), n);
    
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    
    if (!success) {
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> result(n * n);
        if (n != 0) {
            checkCuda(cudaMemcpy(result.data(), A.get(), result.size() * sizeof(double),
                                 cudaMemcpyDeviceToHost), "Read Cholesky result");
        }
        print_results(result, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A.get(), A_orig.get(), n);
        
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
