#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(error_));                               \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (0)

__global__ void initializeGrid(Real* grid1, Real* grid2, const size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const Real value = static_cast<Real>(index % 19);
        grid1[index] = value;
        grid2[index] = value;
    }
}

// A 3-D launch avoids costly per-cell integer division, preserves contiguous X
// accesses, and skips work on the six unchanged boundary planes.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 const size_t nx, const size_t ny,
                                 const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) return;

    const size_t plane = nx * ny;
    const size_t index = z * plane + y * nx + x;

    output[index] = (input[index] + input[index - 1] + input[index + 1] +
                     input[index - nx] + input[index + nx] +
                     input[index - plane] + input[index + plane]) / 7.0;
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto [minIt, maxIt] = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *minIt, *maxIt);
    if (*maxIt > 1e6 || *minIt < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions must be at least 3, iterations nonnegative, and sizes must not overflow.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real *deviceGrid1 = nullptr, *deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, bytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, bytes));

    constexpr unsigned threads = 256;
    auto blockCount = [](size_t count) {
        return static_cast<unsigned>((count + threads - 1) / threads);
    };
    std::printf("Initializing grid...\n");
    initializeGrid<<<blockCount(gridSize), threads>>>(deviceGrid1, deviceGrid2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const size_t interiorX = nx - 2, interiorY = ny - 2;
    const size_t interiorCount = interiorX * interiorY * (nz - 2);
    // A full warp spans X for coalesced loads/stores; the modest Y/Z extent
    // exposes enough independent warps without inflating boundary overlaunch.
    const dim3 stencilBlock(32, 4, 2);
    const dim3 stencilGrid(static_cast<unsigned>((interiorX + stencilBlock.x - 1) / stencilBlock.x),
                           static_cast<unsigned>((interiorY + stencilBlock.y - 1) / stencilBlock.y),
                           static_cast<unsigned>(((nz - 2) + stencilBlock.z - 1) / stencilBlock.z));
    Real *input = deviceGrid1, *output = deviceGrid2;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < iterations; ++iteration) {
        stencilIteration<<<stencilGrid, stencilBlock>>>(input, output, nx, ny, nz);
        std::swap(input, output);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double updates = static_cast<double>(interiorCount) * iterations;
    const double mcups = elapsedMs > 0.0f ? updates / (elapsedMs * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    int result = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost));
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(finalGrid);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));
    return result;
}
