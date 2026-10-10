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

namespace {

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(result));
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

class CudaStream {
public:
    CudaStream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~CudaStream() { cudaStreamDestroy(stream_); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    operator cudaStream_t() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T)) {
            throw std::runtime_error("Matrix is too large");
        }
        if (count) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
        }
    }
    ~DeviceBuffer() { cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

constexpr int Panel = 32;
constexpr int Tile = 64;
constexpr int Threads = 16;

// Each block computes a 64 x 64 tile, with sixteen double accumulators per
// thread. The two row-major operands are staged in padded shared memory.
// Only lower tiles are scheduled; generation also writes the symmetric half.
enum GramOperation { Generate, Update, Validate };

template <GramOperation Operation>
__global__ void gramKernel(const double* input, double* output,
                           const double* original, double* errors,
                           size_t n, size_t first, size_t begin, size_t end,
                           const int* info, bool lowerTriangular = false) {
    if (blockIdx.y < blockIdx.x) return;
    if (Operation == Update && *info != 0) return;

    __shared__ double left[Tile][Panel + 1];
    __shared__ double right[Tile][Panel + 1];
    __shared__ double maxima[2][8];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * Threads + tx;
    const size_t rowBase = first + size_t(blockIdx.y) * Tile;
    const size_t colBase = first + size_t(blockIdx.x) * Tile;
    // L is triangular: no terms beyond the last column of this tile can
    // contribute to L * L^T.
    if (Operation == Validate && lowerTriangular) end = min(end, colBase + Tile);

    double sums[4][4] = {};
    for (size_t base = begin; base < end; base += Panel) {
        const int width = int(min(size_t(Panel), end - base));
        for (int index = tid; index < Tile * Panel; index += Threads * Threads) {
            const int row = index / Panel;
            const int k = index % Panel;
            left[row][k] = rowBase + row < n && k < width
                ? input[(rowBase + row) * n + base + k] : 0.0;
            right[row][k] = colBase + row < n && k < width
                ? input[(colBase + row) * n + base + k] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Panel; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = left[ty + i * Threads][k];
                b[i] = right[tx + i * Threads][k];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    sums[i][j] = fma(a[i], b[j], sums[i][j]);
                }
            }
        }
        __syncthreads();
    }

    double maxAbsolute = 0.0;
    double maxRelative = 0.0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t row = rowBase + ty + i * Threads;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t col = colBase + tx + j * Threads;
            if (row < n && col < n && row >= col) {
                const size_t index = row * n + col;
                if (Operation == Generate) {
                    const double value = sums[i][j] + (row == col ? double(n) : 0.0);
                    output[index] = value;
                    if (row != col) output[col * n + row] = value;
                } else if (Operation == Update) {
                    output[index] -= sums[i][j];
                } else {
                    const double value = original[index];
                    const double error = fabs(sums[i][j] - value);
                    maxAbsolute = fmax(maxAbsolute, error);
                    maxRelative = fmax(maxRelative, error / (fabs(value) + 1e-10));
                    // Check both halves even for an asymmetric reference.
                    const double opposite = original[col * n + row];
                    const double oppositeError = fabs(sums[i][j] - opposite);
                    maxAbsolute = fmax(maxAbsolute, oppositeError);
                    maxRelative = fmax(maxRelative,
                                       oppositeError / (fabs(opposite) + 1e-10));
                }
            }
        }
    }

    if (Operation == Validate) {
        const int lane = tid & 31;
        const int warp = tid / 32;
        for (int offset = 16; offset; offset /= 2) {
            maxAbsolute = fmax(maxAbsolute, __shfl_down_sync(0xffffffff, maxAbsolute, offset));
            maxRelative = fmax(maxRelative, __shfl_down_sync(0xffffffff, maxRelative, offset));
        }
        if (lane == 0) {
            maxima[0][warp] = maxAbsolute;
            maxima[1][warp] = maxRelative;
        }
        __syncthreads();
        if (warp == 0) {
            maxAbsolute = lane < 8 ? maxima[0][lane] : 0.0;
            maxRelative = lane < 8 ? maxima[1][lane] : 0.0;
            for (int offset = 16; offset; offset /= 2) {
                maxAbsolute = fmax(maxAbsolute, __shfl_down_sync(0xffffffff, maxAbsolute, offset));
                maxRelative = fmax(maxRelative, __shfl_down_sync(0xffffffff, maxRelative, offset));
            }
            if (lane == 0) {
                const size_t index = 2 * (size_t(blockIdx.y) * gridDim.x + blockIdx.x);
                errors[index] = maxAbsolute;
                errors[index + 1] = maxRelative;
            }
        }
    }
}

// A single warp handles a diagonal tile. All lanes participate in warp
// barriers, including the lanes outside a short final tile.
__global__ void factorDiagonal(double* matrix, size_t n, size_t first,
                               int width, int* info) {
    if (*info != 0) return;
    __shared__ double diagonal[Panel][Panel + 1];
    __shared__ int failed;
    const int row = threadIdx.x;
    for (int col = 0; col < width; ++col) {
        diagonal[row][col] = row < width && col <= row
            ? matrix[(first + row) * n + first + col] : 0.0;
    }
    if (row == 0) failed = 0;
    __syncwarp();

    for (int col = 0; col < width; ++col) {
        if (row == col) {
            double sum = 0.0;
            for (int k = 0; k < col; ++k) {
                sum = fma(diagonal[col][k], diagonal[col][k], sum);
            }
            const double value = diagonal[col][col] - sum;
            if (value <= 0.0) {
                *info = int(first + col + 1);
                failed = 1;
            } else {
                diagonal[col][col] = sqrt(value);
            }
        }
        __syncwarp();
        if (failed) return;
        if (row > col && row < width) {
            double sum = 0.0;
            for (int k = 0; k < col; ++k) {
                sum = fma(diagonal[row][k], diagonal[col][k], sum);
            }
            diagonal[row][col] = (diagonal[row][col] - sum) / diagonal[col][col];
        }
        __syncwarp();
    }
    for (int index = row; index < width * width; index += Panel) {
        const int r = index / width;
        const int c = index % width;
        if (c <= r) matrix[(first + r) * n + first + c] = diagonal[r][c];
    }
}

// Solve L_panel * L_diagonal^T = A_panel. Rows are independent; cooperative
// loads/stores make the row-major global-memory transfers coalesced.
__global__ void solvePanel(double* matrix, size_t n, size_t first,
                           int width, const int* info) {
    if (*info != 0) return;
    __shared__ double diagonal[Panel][Panel + 1];
    __shared__ double rows[Tile][Panel + 1];
    const int tid = threadIdx.x;
    const size_t rowBase = first + width + size_t(blockIdx.x) * Tile;

    for (int index = tid; index < Panel * Panel; index += Tile) {
        const int row = index / Panel;
        const int col = index % Panel;
        diagonal[row][col] = row < width && col <= row
            ? matrix[(first + row) * n + first + col] : 0.0;
    }
    for (int index = tid; index < Tile * Panel; index += Tile) {
        const int row = index / Panel;
        const int col = index % Panel;
        rows[row][col] = rowBase + row < n && col < width
            ? matrix[(rowBase + row) * n + first + col] : 0.0;
    }
    __syncthreads();

    if (rowBase + tid < n) {
        #pragma unroll
        for (int col = 0; col < Panel; ++col) {
            if (col < width) {
                double sum = 0.0;
                #pragma unroll
                for (int k = 0; k < col; ++k) {
                    sum = fma(rows[tid][k], diagonal[col][k], sum);
                }
                rows[tid][col] = (rows[tid][col] - sum) / diagonal[col][col];
            }
        }
    }
    __syncthreads();
    for (int index = tid; index < Tile * Panel; index += Tile) {
        const int row = index / Panel;
        const int col = index % Panel;
        if (rowBase + row < n && col < width) {
            matrix[(rowBase + row) * n + first + col] = rows[row][col];
        }
    }
}

__global__ void zeroUpper(double* matrix, size_t n) {
    const size_t count = n * n;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(blockDim.x) * gridDim.x) {
        if (index % n > index / n) matrix[index] = 0.0;
    }
}

void generatePositiveDefiniteMatrix(double* matrix, size_t n, cudaStream_t stream) {
    if (n == 0) return;
    // Preserve the original rand_r sequence exactly, rather than substituting
    // a GPU random generator with a different matrix for the same seed.
    std::vector<double> random(n * n);
    unsigned int seed = 42;
    for (double& value : random) {
        value = rand_r(&seed) / double(RAND_MAX) - 0.5;
    }
    DeviceBuffer<double> deviceRandom(random.size());
    CUDA_CHECK(cudaMemcpyAsync(deviceRandom.get(), random.data(),
                               random.size() * sizeof(double),
                               cudaMemcpyHostToDevice, stream));
    const unsigned tiles = unsigned((n + Tile - 1) / Tile);
    gramKernel<Generate><<<dim3(tiles, tiles), dim3(Threads, Threads), 0, stream>>>(
        deviceRandom.get(), matrix, nullptr, nullptr, n, 0, 0, n, nullptr);
    CUDA_CHECK(cudaGetLastError());
    // The random operand and its host storage must outlive the kernel.
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool choleskyDecomposition(double* matrix, size_t n, int* info, cudaStream_t stream) {
    CUDA_CHECK(cudaMemsetAsync(info, 0, sizeof(int), stream));
    for (size_t first = 0; first < n; first += Panel) {
        const int width = int(std::min(size_t(Panel), n - first));
        factorDiagonal<<<1, Panel, 0, stream>>>(matrix, n, first, width, info);
        CUDA_CHECK(cudaGetLastError());
        const size_t next = first + width;
        if (next < n) {
            const unsigned tiles = unsigned((n - next + Tile - 1) / Tile);
            solvePanel<<<tiles, Tile, 0, stream>>>(matrix, n, first, width, info);
            CUDA_CHECK(cudaGetLastError());
            gramKernel<Update><<<dim3(tiles, tiles), dim3(Threads, Threads), 0, stream>>>(
                matrix, matrix, nullptr, nullptr, n, next, first, next, info);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (n) {
        const unsigned blocks = unsigned(std::min(size_t(65535), (n * n + 255) / 256));
        zeroUpper<<<blocks, 256, 0, stream>>>(matrix, n);
        CUDA_CHECK(cudaGetLastError());
    }
    int hostInfo = 0;
    CUDA_CHECK(cudaMemcpyAsync(&hostInfo, info, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (hostInfo) {
        printf("Error: Matrix is not positive definite at diagonal element %zu\n",
               size_t(hostInfo - 1));
        return false;
    }
    return true;
}

bool validateCholesky(const double* matrix, const double* original, size_t n,
                      cudaStream_t stream, bool lowerTriangular = false) {
    double maxError = 0.0;
    double relError = 0.0;
    if (n) {
        const unsigned tiles = unsigned((n + Tile - 1) / Tile);
        std::vector<double> errors(2 * size_t(tiles) * tiles);
        DeviceBuffer<double> deviceErrors(errors.size());
        CUDA_CHECK(cudaMemsetAsync(deviceErrors.get(), 0, errors.size() * sizeof(double), stream));
        gramKernel<Validate><<<dim3(tiles, tiles), dim3(Threads, Threads), 0, stream>>>(
            matrix, nullptr, original, deviceErrors.get(), n, 0, 0, n, nullptr, lowerTriangular);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(errors.data(), deviceErrors.get(),
                                   errors.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (size_t i = 0; i < errors.size(); i += 2) {
            maxError = std::max(maxError, errors[i]);
            relError = std::max(relError, errors[i + 1]);
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

} // namespace

// Retain the original host-vector interfaces; these also always execute on CUDA.
bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    CudaStream stream;
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<int> info(1);
    if (n) CUDA_CHECK(cudaMemcpyAsync(matrix.get(), A.data(), n * n * sizeof(double),
                                     cudaMemcpyHostToDevice, stream));
    const bool success = choleskyDecomposition(matrix.get(), n, info.get(), stream);
    if (n) CUDA_CHECK(cudaMemcpyAsync(A.data(), matrix.get(), n * n * sizeof(double),
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return success;
}

void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    CudaStream stream;
    DeviceBuffer<double> matrix(n * n);
    generatePositiveDefiniteMatrix(matrix.get(), n, stream);
    if (n) CUDA_CHECK(cudaMemcpyAsync(A.data(), matrix.get(), n * n * sizeof(double),
                                     cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig,
                      const size_t n) {
    CudaStream stream;
    DeviceBuffer<double> matrix(n * n);
    DeviceBuffer<double> original(n * n);
    if (n) {
        CUDA_CHECK(cudaMemcpyAsync(matrix.get(), L.data(), n * n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(original.get(), A_orig.data(), n * n * sizeof(double),
                                   cudaMemcpyHostToDevice, stream));
    }
    return validateCholesky(matrix.get(), original.get(), n, stream);
}

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

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
        if (n > size_t(std::numeric_limits<int>::max()) ||
            (n && n > std::numeric_limits<size_t>::max() / n / sizeof(double))) {
            throw std::runtime_error("Matrix is too large");
        }
        // Keep the matrix resident through generation, factorization and
        // validation. Only -r requires downloading the complete factor.
        CudaStream stream;
        DeviceBuffer<double> matrix(n * n);
        DeviceBuffer<double> original(validate ? n * n : 0);
        DeviceBuffer<int> info(1);

        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(matrix.get(), n, stream);
        if (validate && n) {
            CUDA_CHECK(cudaMemcpyAsync(original.get(), matrix.get(), n * n * sizeof(double),
                                       cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        printf("Computing Cholesky decomposition...\n");
        const auto start = std::chrono::high_resolution_clock::now();
        const bool success = choleskyDecomposition(matrix.get(), n, info.get(), stream);
        const auto end = std::chrono::high_resolution_clock::now();

        if (!success) {
            printf("Cholesky decomposition failed\n");
            return 1;
        }
        const double seconds = std::chrono::duration<double>(end - start).count();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        const double ops = double(n) * n * n / 3.0;
        printf("Performance: %.3f GFLOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);

        if (printResults) {
            std::vector<double> A(n * n);
            if (n) CUDA_CHECK(cudaMemcpyAsync(A.data(), matrix.get(), n * n * sizeof(double),
                                             cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            print_results(A, "CholeskyL");
        }

        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(matrix.get(), original.get(), n, stream, true);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Error: %s\n", error.what());
        return 1;
    }
    return 0;
}
