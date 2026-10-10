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

#include <cuda_runtime.h>
#include <math_constants.h>

#include "../common/results_output.hpp"

namespace {

void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count != 0) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
                      "Allocating GPU memory");
        }
    }
    ~DeviceBuffer() { if (data_ != nullptr) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

constexpr int factorTile = 32;
constexpr int productTile = 64;
constexpr int productDepth = 16;
constexpr int productThreads = 256;
constexpr int panelRows = 64;

enum class Product { Generate, Update, Validate };

// Only lower-triangular tiles are scheduled. A 1D grid avoids launching empty
// upper-triangular blocks and the grid.y limit for large matrices.
unsigned int triangularTiles(int extent) {
    const size_t tiles = (static_cast<size_t>(extent) + productTile - 1) / productTile;
    const size_t count = tiles * (tiles + 1) / 2;
    if (count > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Matrix exceeds the CUDA grid limit");
    }
    return static_cast<unsigned int>(count);
}

__device__ double warpMax(double value) {
    for (int delta = 16; delta > 0; delta /= 2) {
        value = fmax(value, __shfl_down_sync(0xffffffffu, value, delta));
    }
    return value;
}

// Each thread accumulates a 4x4 register tile. Shared-memory tiles reuse each
// input across 64 output rows/columns; global loads follow row-major storage.
// This is used for B*B^T, the Schur complement, and L*L^T validation.
template <Product mode>
__global__ void symmetricProduct(const double* input,
                                 double* output,
                                 const double* __restrict__ original,
                                 double* __restrict__ errors,
                                 int n, int base, int first, int depth,
                                 const int* status) {
    if constexpr (mode == Product::Update) {
        if (*status != 0) return;
    }
    const size_t tile = blockIdx.x;
    int tileRow = static_cast<int>((sqrtf(8.0f * static_cast<float>(tile) + 1.0f) - 1.0f) * 0.5f);
    while (static_cast<size_t>(tileRow) * (tileRow + 1) / 2 > tile) --tileRow;
    while (static_cast<size_t>(tileRow + 1) * (tileRow + 2) / 2 <= tile) ++tileRow;
    const int tileCol = static_cast<int>(tile - static_cast<size_t>(tileRow) * (tileRow + 1) / 2);
    const int rowBase = base + tileRow * productTile;
    const int colBase = base + tileCol * productTile;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * 16 + tx;

    __shared__ double left[productDepth][productTile + 1];
    __shared__ double right[productDepth][productTile + 1];
    double sum[4][4] = {};

    int end = first + depth;
    if constexpr (mode == Product::Validate) {
        // L is triangular: terms past the smaller row index are exactly zero.
        end = min(end, colBase + productTile);
    }
    for (int kBase = first; kBase < end; kBase += productDepth) {
        for (int index = tid; index < productTile * productDepth; index += productThreads) {
            const int row = index / productDepth;
            const int k = kBase + index % productDepth;
            left[index % productDepth][row] =
                rowBase + row < n && k < end ? input[static_cast<size_t>(rowBase + row) * n + k] : 0.0;
            right[index % productDepth][row] =
                colBase + row < n && k < end ? input[static_cast<size_t>(colBase + row) * n + k] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < productDepth; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = left[k][ty + i * 16];
                b[i] = right[k][tx + i * 16];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) sum[i][j] += a[i] * b[j];
            }
        }
        __syncthreads();
    }

    double maxAbsolute = 0.0;
    double maxRelative = 0.0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = rowBase + ty + i * 16;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int col = colBase + tx + j * 16;
            if (row < n && col < n && row >= col) {
                const size_t index = static_cast<size_t>(row) * n + col;
                if constexpr (mode == Product::Generate) {
                    const double value = sum[i][j] + (row == col ? n : 0.0);
                    output[index] = value;
                    if (row != col) output[static_cast<size_t>(col) * n + row] = value;
                } else if constexpr (mode == Product::Update) {
                    output[index] -= sum[i][j];
                } else {
                    const double error = fabs(sum[i][j] - original[index]);
                    const double relative = error / (fabs(original[index]) + 1e-10);
                    maxAbsolute = fmax(maxAbsolute, isfinite(error) ? error : CUDART_INF);
                    maxRelative = fmax(maxRelative, isfinite(relative) ? relative : CUDART_INF);
                }
            }
        }
    }

    if constexpr (mode == Product::Validate) {
        __shared__ double absolute[8], relative[8];
        maxAbsolute = warpMax(maxAbsolute);
        maxRelative = warpMax(maxRelative);
        if ((tid & 31) == 0) {
            absolute[tid / 32] = maxAbsolute;
            relative[tid / 32] = maxRelative;
        }
        __syncthreads();
        if (tid < 32) {
            maxAbsolute = warpMax(tid < 8 ? absolute[tid] : 0.0);
            maxRelative = warpMax(tid < 8 ? relative[tid] : 0.0);
            if (tid == 0) {
                errors[tile * 2] = maxAbsolute;
                errors[tile * 2 + 1] = maxRelative;
            }
        }
    }
}

__global__ void factorDiagonal(double* A, int n, int base, int width, int* status) {
    if (*status != 0) return;
    __shared__ double diagonal[factorTile][factorTile + 1];
    __shared__ bool failed;
    const int tid = threadIdx.x;
    if (tid == 0) failed = false;
    // Constant tile strides compile to shifts, avoiding integer division in
    // the serial diagonal path while still handling a partial final tile.
    for (int index = tid; index < factorTile * factorTile; index += blockDim.x) {
        const int row = index / factorTile, col = index % factorTile;
        if (row < width && col < width) {
            diagonal[row][col] = A[static_cast<size_t>(base + row) * n + base + col];
        }
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (tid == 0) {
            const double value = diagonal[k][k];
            if (value <= 0.0) {
                *status = base + k + 1;
                failed = true;
            }
            else diagonal[k][k] = sqrt(value);
        }
        __syncthreads();
        if (failed) return;
        if (tid > k && tid < width) diagonal[tid][k] /= diagonal[k][k];
        __syncthreads();
        for (int index = tid; index < factorTile * factorTile; index += blockDim.x) {
            const int row = index / factorTile, col = index % factorTile;
            if (row < width && row >= col && col > k) {
                diagonal[row][col] -= diagonal[row][k] * diagonal[col][k];
            }
        }
        __syncthreads();
    }
    for (int index = tid; index < factorTile * factorTile; index += blockDim.x) {
        const int row = index / factorTile, col = index % factorTile;
        if (row < width && row >= col) A[static_cast<size_t>(base + row) * n + base + col] = diagonal[row][col];
    }
}

// Independent rows solve A21 * inv(L11^T) concurrently. The panel is staged
// transposed in shared memory so both global transfers and row solves coalesce.
__global__ void solvePanel(double* A, int n, int base, int width, const int* status) {
    if (*status != 0) return;
    __shared__ double diagonal[factorTile][factorTile + 1];
    __shared__ double panel[factorTile][panelRows + 1];
    const int tid = threadIdx.x;
    const int rowBase = base + width + blockIdx.x * panelRows;
    for (int index = tid; index < factorTile * factorTile; index += blockDim.x) {
        const int row = index / factorTile, col = index % factorTile;
        if (row < width && col < width) {
            const double value = A[static_cast<size_t>(base + row) * n + base + col];
            // Reuse one reciprocal per diagonal across every panel row.
            diagonal[row][col] = row == col ? 1.0 / value : value;
        }
    }
    for (int index = tid; index < panelRows * factorTile; index += blockDim.x) {
        const int row = index / factorTile, col = index % factorTile;
        panel[col][row] = rowBase + row < n && col < width ? A[static_cast<size_t>(rowBase + row) * n + base + col] : 0.0;
    }
    __syncthreads();
    if (tid < panelRows && rowBase + tid < n) {
        for (int col = 0; col < width; ++col) {
            double sum = 0.0;
            for (int k = 0; k < col; ++k) sum += panel[k][tid] * diagonal[col][k];
            panel[col][tid] = (panel[col][tid] - sum) * diagonal[col][col];
        }
    }
    __syncthreads();
    for (int index = tid; index < panelRows * factorTile; index += blockDim.x) {
        const int row = index / factorTile, col = index % factorTile;
        if (rowBase + row < n && col < width) A[static_cast<size_t>(rowBase + row) * n + base + col] = panel[col][row];
    }
}

__global__ void zeroUpper(double* A, int n) {
    const size_t count = static_cast<size_t>(n) * n;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        if (index % n > index / n) A[index] = 0.0;
    }
}

__global__ void reduceErrors(const double* errors, unsigned int tiles, double* result) {
    double absolute = 0.0, relative = 0.0;
    for (unsigned int tile = threadIdx.x; tile < tiles; tile += blockDim.x) {
        absolute = fmax(absolute, errors[static_cast<size_t>(tile) * 2]);
        relative = fmax(relative, errors[static_cast<size_t>(tile) * 2 + 1]);
    }
    absolute = warpMax(absolute);
    relative = warpMax(relative);
    __shared__ double absWarps[8], relWarps[8];
    if ((threadIdx.x & 31) == 0) {
        absWarps[threadIdx.x / 32] = absolute;
        relWarps[threadIdx.x / 32] = relative;
    }
    __syncthreads();
    if (threadIdx.x < 32) {
        absolute = warpMax(threadIdx.x < 8 ? absWarps[threadIdx.x] : 0.0);
        relative = warpMax(threadIdx.x < 8 ? relWarps[threadIdx.x] : 0.0);
        if (threadIdx.x == 0) {
            result[0] = absolute;
            result[1] = relative;
        }
    }
}

// Matrices stay resident on the GPU across generation, factorization and
// validation. All kernels use the same stream, so panel dependencies are ordered
// without a device/host synchronization at every column or tile.
bool choleskyDecomposition(double* A, int n, int* status) {
    cudaCheck(cudaMemset(status, 0, sizeof(int)), "Initializing factorization status");
    for (int base = 0; base < n; base += factorTile) {
        const int width = std::min(factorTile, n - base);
        factorDiagonal<<<1, 256>>>(A, n, base, width, status);
        cudaCheck(cudaGetLastError(), "Launching diagonal factorization");
        const int remaining = n - base - width;
        if (remaining > 0) {
            solvePanel<<<(remaining + panelRows - 1) / panelRows, 128>>>(A, n, base, width, status);
            cudaCheck(cudaGetLastError(), "Launching panel solve");
            symmetricProduct<Product::Update><<<triangularTiles(remaining), dim3(16, 16)>>>(
                A, A, nullptr, nullptr, n, base + width, base, width, status);
            cudaCheck(cudaGetLastError(), "Launching trailing matrix update");
        }
    }
    if (n != 0) {
        const size_t count = static_cast<size_t>(n) * n;
        const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>((count + 255) / 256, 65535));
        zeroUpper<<<blocks, 256>>>(A, n);
        cudaCheck(cudaGetLastError(), "Launching upper triangle clearing");
    }
    int hostStatus = 0;
    cudaCheck(cudaMemcpy(&hostStatus, status, sizeof(int), cudaMemcpyDeviceToHost), "Completing factorization");
    if (hostStatus != 0) {
        printf("Error: Matrix is not positive definite at diagonal element %d\n", hostStatus - 1);
        return false;
    }
    return true;
}

void generatePositiveDefiniteMatrix(double* A, int n) {
    if (n == 0) return;
    const size_t count = static_cast<size_t>(n) * n;
    std::vector<double> B(count);
    unsigned int seed = 42;
    // Preserve the original rand_r sequence exactly; only the cubic product
    // is parallelized, without introducing another random-number dependency.
    for (double& value : B) value = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    DeviceBuffer<double> deviceB(count);
    cudaCheck(cudaMemcpy(deviceB.get(), B.data(), count * sizeof(double), cudaMemcpyHostToDevice),
              "Uploading random matrix");
    symmetricProduct<Product::Generate><<<triangularTiles(n), dim3(16, 16)>>>(
        deviceB.get(), A, nullptr, nullptr, n, 0, 0, n, nullptr);
    cudaCheck(cudaGetLastError(), "Launching matrix generation");
    cudaCheck(cudaDeviceSynchronize(), "Completing matrix generation");
}

bool validateCholesky(const double* L, const double* original, int n) {
    double error[2] = {};
    if (n != 0) {
        const unsigned int tiles = triangularTiles(n);
        DeviceBuffer<double> partial(static_cast<size_t>(tiles) * 2);
        DeviceBuffer<double> result(2);
        symmetricProduct<Product::Validate><<<tiles, dim3(16, 16)>>>(
            L, nullptr, original, partial.get(), n, 0, 0, n, nullptr);
        cudaCheck(cudaGetLastError(), "Launching validation");
        reduceErrors<<<1, 256>>>(partial.get(), tiles, result.get());
        cudaCheck(cudaGetLastError(), "Launching error reduction");
        cudaCheck(cudaMemcpy(error, result.get(), sizeof(error), cudaMemcpyDeviceToHost), "Completing validation");
    }
    printf("Max absolute error: %.10e\n", error[0]);
    printf("Max relative error: %.10e\n", error[1]);
    if (error[1] > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

} // namespace

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long parsed = strtoull(argument, &end, 10);
            if (argument[0] == '-' || end == argument || *end != '\0' || errno == ERANGE ||
                parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()) ||
                (parsed != 0 && parsed > std::numeric_limits<size_t>::max() / sizeof(double) / parsed)) {
                fprintf(stderr, "Invalid matrix size: %s\n", argument);
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
    
    try {
        // CUDA is required even for an empty matrix; no sequential fallback.
        cudaCheck(cudaFree(nullptr), "Initializing CUDA");
        DeviceBuffer<double> A(n * n);
        DeviceBuffer<double> A_orig(validate ? n * n : 0);
        DeviceBuffer<int> status(1);
        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A.get(), static_cast<int>(n));
        if (validate && n != 0) {
            cudaCheck(cudaMemcpy(A_orig.get(), A.get(), n * n * sizeof(double), cudaMemcpyDeviceToDevice),
                      "Saving original matrix");
        }
        cudaCheck(cudaDeviceSynchronize(), "Preparing factorization");

        printf("Computing Cholesky decomposition...\n");
        const auto start = std::chrono::steady_clock::now();
        const bool success = choleskyDecomposition(A.get(), static_cast<int>(n), status.get());
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        if (!success) {
            printf("Cholesky decomposition failed\n");
            return 1;
        }
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double ops = static_cast<double>(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);

        if (printResults) {
            std::vector<double> result(n * n);
            if (n != 0) {
                cudaCheck(cudaMemcpy(result.data(), A.get(), n * n * sizeof(double), cudaMemcpyDeviceToHost),
                          "Downloading Cholesky factor");
            }
            print_results(result, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A.get(), A_orig.get(), static_cast<int>(n));
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
    return 0;
}
