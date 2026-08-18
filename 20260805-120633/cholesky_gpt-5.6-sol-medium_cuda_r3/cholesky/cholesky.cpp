#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// A 32-column panel gives enough work to amortize launches, while keeping the
// dot products and diagonal panel in registers/cache.
constexpr int PANEL = 32;
constexpr int TILE = 16;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t error__ = (call);                                      \
        if (error__ != cudaSuccess) {                                            \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                         __LINE__, cudaGetErrorString(error__));                 \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                       \
    } while (false)

// The panel is already updated by preceding trailing-update kernels.  It is
// small and dependency-heavy, so one thread is faster than repeated launches.
__global__ void factorDiagonalPanel(double* __restrict__ a, size_t n, size_t p,
                                    size_t width, int* __restrict__ failure) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || *failure >= 0) return;

    for (size_t jj = 0; jj < width; ++jj) {
        const size_t j = p + jj;
        double diagonal = a[j * n + j];
        for (size_t k = p; k < j; ++k) {
            const double v = a[j * n + k];
            diagonal -= v * v;
        }
        if (!(diagonal > 0.0)) {
            *failure = static_cast<int>(j);
            return;
        }
        const double ljj = sqrt(diagonal);
        a[j * n + j] = ljj;

        for (size_t i = j + 1; i < p + width; ++i) {
            double value = a[i * n + j];
            for (size_t k = p; k < j; ++k)
                value -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = value / ljj;
        }
    }
}

// One warp cooperatively solves one row below the diagonal panel.  The shuffle
// reduction removes shared-memory traffic and all rows execute independently.
__global__ void solvePanelRows(double* __restrict__ a, size_t n, size_t p,
                               size_t width, const int* __restrict__ failure) {
    const size_t row = p + width + blockIdx.x;
    const unsigned lane = threadIdx.x;
    if (row >= n || *failure >= 0) return;

    for (size_t jj = 0; jj < width; ++jj) {
        const size_t j = p + jj;
        double part = 0.0;
        for (size_t kk = lane; kk < jj; kk += 32)
            part += a[row * n + p + kk] * a[j * n + p + kk];
        for (int offset = 16; offset > 0; offset >>= 1)
            part += __shfl_down_sync(0xffffffffu, part, offset);
        if (lane == 0)
            a[row * n + j] = (a[row * n + j] - part) / a[j * n + j];
        __syncwarp();
    }
}

// Rank-k update of the lower trailing matrix.  Adjacent threads access adjacent
// columns, yielding coalesced loads/stores; each output remains in a register.
__global__ void updateTrailing(double* __restrict__ a, size_t n, size_t p,
                               size_t width, const int* __restrict__ failure) {
    __shared__ double row_panel[TILE][PANEL];
    __shared__ double col_panel[TILE][PANEL];
    const size_t col_base = p + width + blockIdx.x * TILE;
    const size_t row_base = p + width + blockIdx.y * TILE;
    // Entire tiles strictly above the diagonal have no lower-triangle output.
    if (col_base > row_base + TILE - 1 || *failure >= 0) return;

    const unsigned linear = threadIdx.y * TILE + threadIdx.x;
#pragma unroll
    for (unsigned offset = 0; offset < TILE * PANEL; offset += TILE * TILE) {
        const unsigned item = linear + offset;
        const unsigned local_row = item / PANEL;
        const unsigned kk = item % PANEL;
        const size_t global_row = row_base + local_row;
        const size_t global_col = col_base + local_row;
        row_panel[local_row][kk] = (global_row < n && kk < width)
                                      ? a[global_row * n + p + kk] : 0.0;
        col_panel[local_row][kk] = (global_col < n && kk < width)
                                      ? a[global_col * n + p + kk] : 0.0;
    }
    __syncthreads();

    const size_t col = col_base + threadIdx.x;
    const size_t row = row_base + threadIdx.y;
    if (row >= n || col >= n || col > row) return;

    double value = a[row * n + col];
#pragma unroll
    for (size_t kk = 0; kk < PANEL; ++kk)
        value -= row_panel[threadIdx.y][kk] * col_panel[threadIdx.x][kk];
    a[row * n + col] = value;
}

__global__ void clearUpper(double* __restrict__ a, size_t n) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n && col > row) a[row * n + col] = 0.0;
}

// Tiled A = B * B^T construction.  The original benchmark spent O(n^3) scalar
// CPU work preparing an O(n^3) GPU benchmark; doing it here keeps setup scalable
// and reuses each input tile from shared memory.
__global__ void formPositiveDefinite(double* __restrict__ a,
                                     const double* __restrict__ b, size_t n) {
    __shared__ double row_tile[TILE][TILE];
    __shared__ double col_tile[TILE][TILE];
    const size_t row = blockIdx.y * TILE + threadIdx.y;
    const size_t col = blockIdx.x * TILE + threadIdx.x;
    double sum = 0.0;
    for (size_t base = 0; base < n; base += TILE) {
        const size_t row_k = base + threadIdx.x;
        const size_t col_k = base + threadIdx.x;
        row_tile[threadIdx.y][threadIdx.x] =
            (row < n && row_k < n) ? b[row * n + row_k] : 0.0;
        // Loading with threadIdx.x as the reduction dimension is coalesced.
        const size_t other_row = blockIdx.x * TILE + threadIdx.y;
        col_tile[threadIdx.y][threadIdx.x] =
            (other_row < n && col_k < n) ? b[other_row * n + col_k] : 0.0;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += row_tile[threadIdx.y][k] * col_tile[threadIdx.x][k];
        __syncthreads();
    }
    if (row < n && col < n)
        a[row * n + col] = sum + (row == col ? static_cast<double>(n) : 0.0);
}

__global__ void validationErrors(const double* __restrict__ l,
                                 const double* __restrict__ original, size_t n,
                                 unsigned long long* max_absolute,
                                 unsigned long long* max_relative) {
    const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= n || col >= n) return;

    double reconstructed = 0.0;
    const size_t end = min(row, col);
    for (size_t k = 0; k <= end; ++k)
        reconstructed += l[row * n + k] * l[col * n + k];
    const double error = fabs(reconstructed - original[row * n + col]);
    const double relative = error / (fabs(original[row * n + col]) + 1e-10);
    // Both values are nonnegative, for which IEEE-754 bit ordering is integer
    // ordering.  This provides a fast native reduction without another library.
    atomicMax(max_absolute, __double_as_longlong(error));
    atomicMax(max_relative, __double_as_longlong(relative));
}

class DeviceMatrix {
public:
    explicit DeviceMatrix(size_t count) {
        if (count) CUDA_CHECK(cudaMalloc(&data_, count * sizeof(double)));
    }
    ~DeviceMatrix() { if (data_) cudaFree(data_); }
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;
    double* get() const { return data_; }
private:
    double* data_ = nullptr;
};

bool choleskyDecomposition(double* device_a, size_t n, float& elapsed_ms) {
    if (n == 0) {
        elapsed_ms = 0.0f;
        return true;
    }

    int* device_failure = nullptr;
    CUDA_CHECK(cudaMalloc(&device_failure, sizeof(int)));
    CUDA_CHECK(cudaMemset(device_failure, 0xff, sizeof(int)));
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (size_t p = 0; p < n; p += PANEL) {
        const size_t width = std::min<size_t>(PANEL, n - p);
        factorDiagonalPanel<<<1, 1>>>(device_a, n, p, width, device_failure);
        const size_t rows = n - p - width;
        if (rows) {
            solvePanelRows<<<static_cast<unsigned>(rows), 32>>>(
                device_a, n, p, width, device_failure);
            const unsigned tiles = static_cast<unsigned>((rows + TILE - 1) / TILE);
            updateTrailing<<<dim3(tiles, tiles), dim3(TILE, TILE)>>>(
                device_a, n, p, width, device_failure);
        }
    }
    const dim3 block(TILE, TILE);
    const dim3 grid(static_cast<unsigned>((n + TILE - 1) / TILE),
                    static_cast<unsigned>((n + TILE - 1) / TILE));
    clearUpper<<<grid, block>>>(device_a, n);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CUDA_CHECK(cudaGetLastError());

    int failure = -1;
    CUDA_CHECK(cudaMemcpy(&failure, device_failure, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device_failure));
    if (failure >= 0)
        std::printf("Error: Matrix is not positive definite at diagonal element %d\n", failure);
    return failure < 0;
}

void generatePositiveDefiniteMatrix(double* device_a, size_t n) {
    std::vector<double> b(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i)
        b[i] = (rand_r(&seed) / static_cast<double>(RAND_MAX)) - 0.5;
    if (!n) return;
    DeviceMatrix device_b(b.size());
    CUDA_CHECK(cudaMemcpy(device_b.get(), b.data(), b.size() * sizeof(double),
                          cudaMemcpyHostToDevice));
    const dim3 block(TILE, TILE);
    const dim3 grid(static_cast<unsigned>((n + TILE - 1) / TILE),
                    static_cast<unsigned>((n + TILE - 1) / TILE));
    formPositiveDefinite<<<grid, block>>>(device_a, device_b.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

bool validateCholesky(const double* device_l, const double* device_original, size_t n) {
    if (n == 0) {
        std::printf("Max absolute error: 0.0000000000e+00\n");
        std::printf("Max relative error: 0.0000000000e+00\n");
        return true;
    }
    unsigned long long* device_errors = nullptr;
    CUDA_CHECK(cudaMalloc(&device_errors, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(device_errors, 0, 2 * sizeof(unsigned long long)));
    const dim3 block(TILE, TILE);
    const dim3 grid(static_cast<unsigned>((n + TILE - 1) / TILE),
                    static_cast<unsigned>((n + TILE - 1) / TILE));
    validationErrors<<<grid, block>>>(device_l, device_original, n,
                                      device_errors, device_errors + 1);
    unsigned long long errors[2]{};
    CUDA_CHECK(cudaMemcpy(errors, device_errors, sizeof(errors), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(device_errors));
    double max_error, relative_error;
    std::memcpy(&max_error, &errors[0], sizeof(double));
    std::memcpy(&relative_error, &errors[1], sizeof(double));
    std::printf("Max absolute error: %.10e\n", max_error);
    std::printf("Max relative error: %.10e\n", relative_error);
    if (relative_error > 1e-6) {
        std::printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool print_results_requested = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (!end || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
                std::printf("Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) print_results_requested = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (n && n > std::numeric_limits<size_t>::max() / n) {
        std::fprintf(stderr, "Matrix size is too large\n");
        return 1;
    }

    std::printf("Cholesky Decomposition Benchmark\n");
    std::printf("Matrix size: %zu x %zu\n", n, n);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
    const size_t element_count = n * n;
    std::vector<double> a(print_results_requested ? element_count : 0);
    DeviceMatrix device_a(element_count);
    DeviceMatrix device_original(validate ? element_count : 0);
    std::printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(device_a.get(), n);
    if (validate && element_count)
        CUDA_CHECK(cudaMemcpy(device_original.get(), device_a.get(),
                              element_count * sizeof(double), cudaMemcpyDeviceToDevice));

    std::printf("Computing Cholesky decomposition...\n");
    float elapsed_ms = 0.0f;
    if (!choleskyDecomposition(device_a.get(), n, elapsed_ms)) {
        std::printf("Cholesky decomposition failed\n");
        return 1;
    }
    std::printf("Computation time: %.3f ms\n", elapsed_ms);
    const double operations = static_cast<double>(n) * n * n / 3.0;
    const double gflops = elapsed_ms > 0.0 ? operations / (elapsed_ms * 1e6) : 0.0;
    std::printf("Performance: %.3f GFLOPS\n", gflops);

    if (print_results_requested) {
        if (!a.empty())
            CUDA_CHECK(cudaMemcpy(a.data(), device_a.get(), a.size() * sizeof(double),
                                  cudaMemcpyDeviceToHost));
        print_results(a, "CholeskyL");
    }
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateCholesky(device_a.get(), device_original.get(), n);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
