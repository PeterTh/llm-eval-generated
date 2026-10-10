#include <algorithm>
#include <cerrno>
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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA: ") + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        if (count) checkCuda(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_) cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return data_; }
private:
    T* data_ = nullptr;
};

class Stream {
public:
    Stream() { checkCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    operator cudaStream_t() const { return stream_; }
private:
    cudaStream_t stream_ = nullptr;
};

class Event {
public:
    Event() { checkCuda(cudaEventCreate(&event_)); }
    ~Event() { cudaEventDestroy(event_); }
    operator cudaEvent_t() const { return event_; }
private:
    cudaEvent_t event_ = nullptr;
};

constexpr int Threads = 256;
constexpr int Tile = 64;
constexpr int Inner = 32;
constexpr int Panel = 32;
constexpr int SolveRows = 64;

// Jump ahead in the 32-bit LCG used by glibc rand_r. Unsigned overflow
// implements arithmetic modulo 2^32, so each thread reproduces its own
// portion of the original seed-42 sequence without serial RNG work.
__device__ unsigned advanceRandom(unsigned state, size_t steps) {
    unsigned multiplier = 1103515245u, increment = 12345u;
    unsigned accumulatedMultiplier = 1u, accumulatedIncrement = 0u;
    while (steps) {
        if (steps & 1) {
            accumulatedMultiplier *= multiplier;
            accumulatedIncrement = accumulatedIncrement * multiplier + increment;
        }
        increment *= multiplier + 1u;
        multiplier *= multiplier;
        steps >>= 1;
    }
    return accumulatedMultiplier * state + accumulatedIncrement;
}

__global__ void generateRandom(double* B, size_t count) {
    const size_t thread = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    // Each warp fills 256 consecutive elements with coalesced stores.
    size_t index = (thread / 32) * 256 + thread % 32;
    unsigned state = advanceRandom(42u, 3 * index);
    #pragma unroll
    for (int batch = 0; batch < 8; ++batch, index += 32) {
        state = state * 1103515245u + 12345u;
        unsigned result = (state >> 16) & 2047u;
        state = state * 1103515245u + 12345u;
        result = (result << 10) ^ ((state >> 16) & 1023u);
        state = state * 1103515245u + 12345u;
        result = (result << 10) ^ ((state >> 16) & 1023u);
        if (index < count) B[index] = double(result) / 2147483647.0 - 0.5;
        state = advanceRandom(state, 3 * 31);
    }
}

// Modes: 0 = B B^T + n I; 1 = trailing A -= Lpanel Lpanel^T;
// 2 = reconstruct L L^T. All matrices keep the original row-major layout.
// A 64x64 output tile uses a 4x4 register tile per thread and padded,
// transposed shared-memory panels to coalesce loads and avoid bank conflicts.
// Only triangular tiles run; generation/reconstruction mirror their output.
template <int Mode>
__global__ void rankProduct(double* output, const double* input, size_t n,
                            size_t firstRow, size_t firstColumn, size_t rank,
                            const int* info) {
    if constexpr (Mode == 1) {
        if (*info) return;
    }
    const size_t tileIndex = blockIdx.x;
    size_t tileRow = size_t((sqrt(double(8 * tileIndex + 1)) - 1.0) * 0.5);
    while ((tileRow + 1) * (tileRow + 2) / 2 <= tileIndex) ++tileRow;
    while (tileRow * (tileRow + 1) / 2 > tileIndex) --tileRow;
    const size_t tileColumn = tileIndex - tileRow * (tileRow + 1) / 2;
    const size_t rowBase = firstRow + tileRow * Tile;
    const size_t columnBase = firstRow + tileColumn * Tile;
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int thread = ty * 16 + tx;
    __shared__ double left[Inner][Tile + 1];
    __shared__ double right[Inner][Tile + 1];
    double sums[4][4] = {};

    for (size_t start = 0; start < rank; start += Inner) {
        #pragma unroll
        for (int element = thread; element < Tile * Inner; element += Threads) {
            const int row = element / Inner, k = element % Inner;
            const size_t column = firstColumn + start + k;
            left[k][row] = rowBase + row < n && start + k < rank
                ? input[(rowBase + row) * n + column] : 0.0;
            right[k][row] = columnBase + row < n && start + k < rank
                ? input[(columnBase + row) * n + column] : 0.0;
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < Inner; ++k) {
            double a[4], b[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = left[k][ty + i * 16];
                b[i] = right[k][tx + i * 16];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) sums[i][j] = fma(a[i], b[j], sums[i][j]);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const size_t row = rowBase + ty + i * 16;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t column = columnBase + tx + j * 16;
            if (row < n && column < n && row >= column) {
                if constexpr (Mode == 1) {
                    output[row * n + column] -= sums[i][j];
                } else {
                    double value = sums[i][j];
                    if constexpr (Mode == 0) {
                        if (row == column) value += double(n);
                    }
                    output[row * n + column] = value;
                    if (row != column) output[column * n + row] = value;
                }
            }
        }
    }
}

// The diagonal panel's dependencies stay inside one thread block. Rank-one
// updates parallelize every remaining lower-triangular element in shared memory.
template <int Width>
__global__ void factorDiagonal(double* A, size_t n, size_t offset, int* info) {
    __shared__ double panel[Width][Width + 1];
    __shared__ double inverseDiagonal;
    __shared__ int failed;
    const int thread = threadIdx.x;
    const int width = int(n - offset < Width ? n - offset : Width);
    if (thread == 0) failed = *info;
    for (int element = thread; element < Width * Width; element += Threads) {
        const int row = element / Width, column = element % Width;
        panel[row][column] = row < width && column < width
            ? A[(offset + row) * n + offset + column] : 0.0;
    }
    __syncthreads();
    if (failed) return;
    for (int k = 0; k < width; ++k) {
        if (thread == 0) {
            const double diagonal = panel[k][k];
            if (diagonal <= 0.0 || !isfinite(diagonal)) {
                failed = int(offset + k + 1);
                *info = failed;
            } else {
                panel[k][k] = sqrt(diagonal);
                inverseDiagonal = 1.0 / panel[k][k];
            }
        }
        __syncthreads();
        if (failed) return;
        if (thread > k && thread < width) panel[thread][k] *= inverseDiagonal;
        __syncthreads();
        for (int element = thread; element < Width * Width; element += Threads) {
            const int row = element / Width, column = element % Width;
            if (row < width && column > k && column <= row) {
                panel[row][column] = fma(-panel[row][k], panel[column][k], panel[row][column]);
            }
        }
        __syncthreads();
    }
    for (int element = thread; element < Width * Width; element += Threads) {
        const int row = element / Width, column = element % Width;
        if (row < width && column < width) {
            A[(offset + row) * n + offset + column] = column <= row ? panel[row][column] : 0.0;
        }
    }
}

// Each thread solves one row of A21 L11^-T in registers. Shared memory
// transposes the row-major input/output so global accesses remain coalesced.
__global__ void solvePanel(double* A, size_t n, size_t offset, int width, const int* info) {
    if (*info) return;
    __shared__ double diagonal[Panel][Panel + 1];
    __shared__ double inverseDiagonal[Panel];
    __shared__ double rhs[Panel][SolveRows + 1];
    const int thread = threadIdx.x;
    const size_t rowBase = offset + width + size_t(blockIdx.x) * SolveRows;
    // Reuse one reciprocal per column instead of serial divisions in every row.
    if (thread < width) {
        inverseDiagonal[thread] = 1.0 / A[(offset + thread) * n + offset + thread];
    }
    for (int element = thread; element < Panel * Panel; element += Threads) {
        const int row = element / Panel, column = element % Panel;
        diagonal[row][column] = row < width && column < width
            ? A[(offset + row) * n + offset + column] : 0.0;
    }
    for (int element = thread; element < SolveRows * Panel; element += Threads) {
        const int row = element / Panel, column = element % Panel;
        rhs[column][row] = rowBase + row < n && column < width
            ? A[(rowBase + row) * n + offset + column] : 0.0;
    }
    __syncthreads();
    if (thread < SolveRows && rowBase + thread < n) {
        double values[Panel];
        #pragma unroll
        for (int column = 0; column < Panel; ++column) {
            if (column < width) {
                double value = rhs[column][thread];
                #pragma unroll
                for (int k = 0; k < column; ++k) {
                    value = fma(-diagonal[column][k], values[k], value);
                }
                values[column] = value * inverseDiagonal[column];
                rhs[column][thread] = values[column];
            }
        }
    }
    __syncthreads();
    for (int element = thread; element < SolveRows * Panel; element += Threads) {
        const int row = element / Panel, column = element % Panel;
        if (rowBase + row < n && column < width) {
            A[(rowBase + row) * n + offset + column] = rhs[column][row];
        }
    }
}

__global__ void clearUpper(double* A, size_t n) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < n * n; index += size_t(blockDim.x) * gridDim.x) {
        if (index % n > index / n) A[index] = 0.0;
    }
}

__global__ void measureErrors(const double* reconstructed, const double* original,
                               size_t count, double* errors) {
    __shared__ double absolute[Threads], relative[Threads];
    const int thread = threadIdx.x;
    double maxAbsolute = 0.0, maxRelative = 0.0;
    for (size_t index = size_t(blockIdx.x) * blockDim.x + thread;
         index < count; index += size_t(blockDim.x) * gridDim.x) {
        const double error = fabs(reconstructed[index] - original[index]);
        const double rel = error / (fabs(original[index]) + 1e-10);
        maxAbsolute = fmax(maxAbsolute, isfinite(error) ? error : CUDART_INF);
        maxRelative = fmax(maxRelative, isfinite(rel) ? rel : CUDART_INF);
    }
    absolute[thread] = maxAbsolute;
    relative[thread] = maxRelative;
    __syncthreads();
    for (int stride = Threads / 2; stride; stride >>= 1) {
        if (thread < stride) {
            absolute[thread] = fmax(absolute[thread], absolute[thread + stride]);
            relative[thread] = fmax(relative[thread], relative[thread + stride]);
        }
        __syncthreads();
    }
    if (thread == 0) {
        errors[2 * blockIdx.x] = absolute[0];
        errors[2 * blockIdx.x + 1] = relative[0];
    }
}

unsigned triangularBlocks(size_t rows) {
    const size_t tiles = (rows + Tile - 1) / Tile;
    return unsigned(tiles * (tiles + 1) / 2);
}

void prepareKernels(size_t n, bool validate) {
    // Load factorization code before the timed region: CUDA's lazy module
    // loading can otherwise charge first-use setup to the factorization.
    cudaFuncAttributes attributes;
    checkCuda(cudaFuncSetCacheConfig(rankProduct<0>, cudaFuncCachePreferShared));
    if (validate) checkCuda(cudaFuncSetCacheConfig(rankProduct<2>, cudaFuncCachePreferShared));
    if (n <= 64) {
        checkCuda(cudaFuncGetAttributes(&attributes, factorDiagonal<64>));
    } else {
        checkCuda(cudaFuncGetAttributes(&attributes, factorDiagonal<Panel>));
        checkCuda(cudaFuncGetAttributes(&attributes, solvePanel));
        checkCuda(cudaFuncGetAttributes(&attributes, clearUpper));
        checkCuda(cudaFuncSetCacheConfig(rankProduct<1>, cudaFuncCachePreferShared));
        checkCuda(cudaFuncGetAttributes(&attributes, rankProduct<1>));
    }
}

void generatePositiveDefiniteMatrix(double* A, double* scratch, size_t n, cudaStream_t stream) {
    if (!n) return;
#if defined(__GLIBC__) && RAND_MAX == 2147483647
    generateRandom<<<unsigned((n * n + Threads * 8 - 1) / (Threads * 8)), Threads, 0, stream>>>(scratch, n * n);
    checkCuda(cudaGetLastError());
#else
    // Preserve rand_r semantics on libc implementations with a different RNG.
    std::vector<double> B(n * n);
    unsigned seed = 42;
    for (double& value : B) value = double(rand_r(&seed)) / RAND_MAX - 0.5;
    checkCuda(cudaMemcpyAsync(scratch, B.data(), B.size() * sizeof(double), cudaMemcpyHostToDevice, stream));
    checkCuda(cudaStreamSynchronize(stream));
#endif
    rankProduct<0><<<triangularBlocks(n), dim3(16, 16), 0, stream>>>(A, scratch, n, 0, 0, n, nullptr);
    checkCuda(cudaGetLastError());
}

void choleskyDecomposition(double* A, size_t n, int* info, cudaStream_t stream) {
    checkCuda(cudaMemsetAsync(info, 0, sizeof(int), stream));
    if (!n) return;
    if (n <= 64) {
        factorDiagonal<64><<<1, Threads, 0, stream>>>(A, n, 0, info);
        checkCuda(cudaGetLastError());
        return;
    }
    for (size_t offset = 0; offset < n; offset += Panel) {
        const int width = int(std::min(size_t(Panel), n - offset));
        factorDiagonal<Panel><<<1, Threads, 0, stream>>>(A, n, offset, info);
        checkCuda(cudaGetLastError());
        const size_t remaining = n - offset - width;
        if (remaining) {
            solvePanel<<<unsigned((remaining + SolveRows - 1) / SolveRows), Threads, 0, stream>>>(A, n, offset, width, info);
            checkCuda(cudaGetLastError());
            rankProduct<1><<<triangularBlocks(remaining), dim3(16, 16), 0, stream>>>(A, A, n, offset + width, offset, width, info);
            checkCuda(cudaGetLastError());
        }
    }
    clearUpper<<<unsigned(std::min(size_t(65535), (n * n + Threads - 1) / Threads)), Threads, 0, stream>>>(A, n);
    checkCuda(cudaGetLastError());
}

bool validateCholesky(const double* L, const double* original, double* scratch,
                      double* deviceErrors, size_t n, cudaStream_t stream) {
    double maxError = 0.0, relError = 0.0;
    if (n) {
        rankProduct<2><<<triangularBlocks(n), dim3(16, 16), 0, stream>>>(scratch, L, n, 0, 0, n, nullptr);
        checkCuda(cudaGetLastError());
        const unsigned blocks = unsigned(std::min(size_t(1024), (n * n + Threads - 1) / Threads));
        measureErrors<<<blocks, Threads, 0, stream>>>(scratch, original, n * n, deviceErrors);
        checkCuda(cudaGetLastError());
        std::vector<double> errors(2 * blocks);
        checkCuda(cudaMemcpyAsync(errors.data(), deviceErrors, errors.size() * sizeof(double), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
        for (unsigned block = 0; block < blocks; ++block) {
            maxError = std::max(maxError, errors[2 * block]);
            relError = std::max(relError, errors[2 * block + 1]);
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

} // namespace

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(argument, &end, 10);
            if (errno || end == argument || *end || *argument == '-' ||
                value > unsigned(std::numeric_limits<int>::max()) ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(double) / value)) {
                fprintf(stderr, "Invalid matrix size: %s\n", argument);
                return 1;
            }
            n = size_t(value);
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
        const size_t count = n * n;
        Stream stream;
        Event start, end;
        DeviceBuffer<double> A(count), scratch(count), original(validate ? count : 0);
        DeviceBuffer<int> info(1);
        const size_t errorBlocks = std::min(size_t(1024), (count + Threads - 1) / Threads);
        DeviceBuffer<double> errors(validate ? 2 * errorBlocks : 0);
        prepareKernels(n, validate);

        printf("Generating positive definite matrix...\n");
        generatePositiveDefiniteMatrix(A.get(), scratch.get(), n, stream);
        if (validate && count) {
            checkCuda(cudaMemcpyAsync(original.get(), A.get(), count * sizeof(double), cudaMemcpyDeviceToDevice, stream));
        }

        printf("Computing Cholesky decomposition...\n");
        // Events on the same stream exclude input generation and include all
        // factorization work. Synchronization prevents timing only the launches.
        checkCuda(cudaEventRecord(start, stream));
        choleskyDecomposition(A.get(), n, info.get(), stream);
        checkCuda(cudaEventRecord(end, stream));
        checkCuda(cudaEventSynchronize(end));
        float milliseconds = 0.0f;
        checkCuda(cudaEventElapsedTime(&milliseconds, start, end));
        int failure = 0;
        checkCuda(cudaMemcpyAsync(&failure, info.get(), sizeof(int), cudaMemcpyDeviceToHost, stream));
        checkCuda(cudaStreamSynchronize(stream));
        if (failure) {
            printf("Error: Matrix is not positive definite at diagonal element %d\n", failure - 1);
            printf("Cholesky decomposition failed\n");
            return 1;
        }
        printf("Computation time: %.3f ms\n", double(milliseconds));
        const double ops = double(n) * double(n) * double(n) / 3.0;
        const double gflops = milliseconds > 0.0f ? ops / (double(milliseconds) * 1e6) : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);

        if (printResults) {
            std::vector<double> result(count);
            if (count) {
                checkCuda(cudaMemcpyAsync(result.data(), A.get(), count * sizeof(double), cudaMemcpyDeviceToHost, stream));
                checkCuda(cudaStreamSynchronize(stream));
            }
            print_results(result, "CholeskyL");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateCholesky(A.get(), original.get(), scratch.get(), errors.get(), n, stream);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            return valid ? 0 : 1;
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
