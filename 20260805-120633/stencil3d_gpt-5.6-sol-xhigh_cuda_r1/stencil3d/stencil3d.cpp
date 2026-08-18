#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

[[noreturn]] void cudaFailure(const cudaError_t error, const char* expression,
                              const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while evaluating %s: %s\n",
                 file, line, expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                   \
    do {                                                                         \
        const cudaError_t cuda_check_error = (expression);                       \
        if (cuda_check_error != cudaSuccess) {                                   \
            cudaFailure(cuda_check_error, #expression, __FILE__, __LINE__);      \
        }                                                                        \
    } while (false)

// X is a full warp wide for coalesced accesses. Each thread advances through a
// short Z slab, retaining the two Z-neighbour values in registers. This lowers
// the steady-state input traffic from seven loads to five per update while the
// independent slabs provide enough blocks to saturate large GPUs.
constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 4;
constexpr unsigned kZSlab = 8;

__global__ void initializeGrids(Real* const first, Real* const second,
                                const size_t count, const bool initializeSecond) {
    const size_t firstIndex = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = firstIndex; index < count; index += stride) {
        const Real value = static_cast<Real>(index % 19);
        first[index] = value;
        second[index] = initializeSecond ? value : 0.0;
    }
}

__global__ __launch_bounds__(kBlockX * kBlockY)
void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                   const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = 1 + static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x;
    const size_t y = 1 + static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y;
    const size_t zBegin = 1 + static_cast<size_t>(blockIdx.z) * kZSlab;

    if (x >= nx - 1 || y >= ny - 1 || zBegin >= nz - 1) {
        return;
    }

    const size_t plane = nx * ny;
    size_t index = zBegin * plane + y * nx + x;
    const size_t zEnd = min(zBegin + static_cast<size_t>(kZSlab), nz - 1);
    Real below = input[index - plane];
    Real center = input[index];

#pragma unroll 4
    for (size_t z = zBegin; z < zEnd; ++z, index += plane) {
        const Real above = input[index + plane];
        const Real sum = center + input[index - 1] + input[index + 1]
                       + input[index - nx] + input[index + nx]
                       + below + above;
        output[index] = sum / 7.0;
        below = center;
        center = above;
    }
}

class DeviceGrid {
  public:
    explicit DeviceGrid(const size_t bytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes));
    }

    DeviceGrid(const DeviceGrid&) = delete;
    DeviceGrid& operator=(const DeviceGrid&) = delete;

    ~DeviceGrid() {
        if (data_ != nullptr) {
            // All checked CUDA work is complete before normal destruction.
            cudaFree(data_);
        }
    }

    Real* get() const noexcept { return data_; }

  private:
    Real* data_ = nullptr;
};

class CudaEvent {
  public:
    CudaEvent() { CUDA_CHECK(cudaEventCreate(&event_)); }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    ~CudaEvent() { cudaEventDestroy(event_); }
    cudaEvent_t get() const noexcept { return event_; }

  private:
    cudaEvent_t event_{};
};

size_t divideRoundUp(const size_t value, const size_t divisor) {
    return value / divisor + (value % divisor != 0);
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minValue = grid[0];
    Real maxValue = grid[0];
    for (const Real value : grid) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }

    std::printf("Value range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 1e6 || minValue < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* const programName) {
    std::printf("Usage: %s [options]\n", programName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int argument = 1; argument < argc; ++argument) {
        if (std::strcmp(argv[argument], "-x") == 0 && argument + 1 < argc) {
            nx = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-y") == 0 && argument + 1 < argc) {
            ny = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-z") == 0 && argument + 1 < argc) {
            nz = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-i") == 0 && argument + 1 < argc) {
            iterations = std::atoi(argv[++argument]);
        } else if (std::strcmp(argv[argument], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[argument], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[argument], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[argument]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    if (nx != 0 && ny > std::numeric_limits<size_t>::max() / nx) {
        std::fprintf(stderr, "Grid dimensions overflow addressable memory\n");
        return 1;
    }
    const size_t plane = nx * ny;
    if (plane != 0 && nz > std::numeric_limits<size_t>::max() / plane) {
        std::fprintf(stderr, "Grid dimensions overflow addressable memory\n");
        return 1;
    }
    const size_t gridSize = plane * nz;
    if (gridSize == 0 || gridSize > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid must contain at least one addressable element\n");
        return 1;
    }
    const size_t bytes = gridSize * sizeof(Real);

    DeviceGrid grid1(bytes);
    DeviceGrid grid2(bytes);

    std::printf("Initializing grid...\n");
    constexpr unsigned initializationThreads = 256;
    const size_t neededBlocks = divideRoundUp(gridSize, initializationThreads);
    const unsigned initializationBlocks = static_cast<unsigned>(
        std::min<size_t>(neededBlocks, 65535));
    initializeGrids<<<initializationBlocks, initializationThreads>>>(
        grid1.get(), grid2.get(), gridSize, iterations > 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFuncSetCacheConfig(stencilKernel, cudaFuncCachePreferL1));

    std::printf("Running stencil computation...\n");
    CudaEvent start;
    CudaEvent stop;
    CUDA_CHECK(cudaEventRecord(start.get()));

    Real* input = grid1.get();
    Real* output = grid2.get();
    if (iterations > 0 && nx > 2 && ny > 2 && nz > 2) {
        const dim3 threads(kBlockX, kBlockY, 1);
        const dim3 blocks(
            static_cast<unsigned>(divideRoundUp(nx - 2, kBlockX)),
            static_cast<unsigned>(divideRoundUp(ny - 2, kBlockY)),
            static_cast<unsigned>(divideRoundUp(nz - 2, kZSlab)));

        for (int iteration = 0; iteration < iterations; ++iteration) {
            stencilKernel<<<blocks, threads>>>(input, output, nx, ny, nz);
            std::swap(input, output);
        }
        CUDA_CHECK(cudaGetLastError());
    } else if (iterations > 0 && (iterations & 1) != 0) {
        // With no interior points, each iteration only copies the boundaries.
        // Both device buffers already contain those invariant values.
        std::swap(input, output);
    } else if (iterations < 0 && (iterations % 2) != 0) {
        // Preserve the original program's selection of its untouched zeroed
        // second buffer for a negative odd iteration count.
        std::swap(input, output);
    }

    CUDA_CHECK(cudaEventRecord(stop.get()));
    CUDA_CHECK(cudaEventSynchronize(stop.get()));
    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start.get(), stop.get()));

    std::printf("Computation time: %ld ms\n",
                static_cast<long>(elapsedMilliseconds));
    const double interiorCells = (nx > 2 && ny > 2 && nz > 2)
        ? static_cast<double>(nx - 2) * static_cast<double>(ny - 2)
            * static_cast<double>(nz - 2)
        : 0.0;
    const double cellUpdates = interiorCells * static_cast<double>(iterations);
    const double mcups = cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1e3);
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(finalGrid)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
