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

namespace {

constexpr int BX = 32;
constexpr int BY = 4;
constexpr int BZ = 2;

[[noreturn]] void cudaFailure(cudaError_t error, const char* operation) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call) do { const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) cudaFailure(error_, #call); } while (false)

__global__ __launch_bounds__(BX * BY * BZ)
void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                   size_t nx, size_t ny, size_t nz) {
    // The haloed shared tile turns seven global reads per output into about
    // (BX+2)(BY+2)(BZ+2)/(BX*BY*BZ) global reads per output.
    __shared__ Real tile[BZ + 2][BY + 2][BX + 2];
    constexpr int tileSize = (BX + 2) * (BY + 2) * (BZ + 2);
    constexpr int threads = BX * BY * BZ;
    const int tid = threadIdx.x + BX * (threadIdx.y + BY * threadIdx.z);
    const size_t plane = nx * ny;

    for (int linear = tid; linear < tileSize; linear += threads) {
        const int lx = linear % (BX + 2);
        const int q = linear / (BX + 2);
        const int ly = q % (BY + 2);
        const int lz = q / (BY + 2);
        size_t x = static_cast<size_t>(blockIdx.x) * BX + lx;
        size_t y = static_cast<size_t>(blockIdx.y) * BY + ly;
        size_t z = static_cast<size_t>(blockIdx.z) * BZ + lz;
        x = min(x, nx - 1);
        y = min(y, ny - 1);
        z = min(z, nz - 1);
        tile[lz][ly][lx] = input[z * plane + y * nx + x];
    }
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * BX + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * BY + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * BZ + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) return;

    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int lz = threadIdx.z + 1;
    output[z * plane + y * nx + x] =
        (tile[lz][ly][lx] + tile[lz][ly][lx - 1] + tile[lz][ly][lx + 1] +
         tile[lz][ly - 1][lx] + tile[lz][ly + 1][lx] +
         tile[lz - 1][ly][lx] + tile[lz + 1][ly][lx]) / Real{7.0};
}

void initializeGrid(std::vector<Real>& grid) {
    for (size_t i = 0; i < grid.size(); ++i) grid[i] = static_cast<Real>(i % 19);
}

bool validateResult(const std::vector<Real>& grid) {
    Real minVal = grid.front();
    Real maxVal = grid.front();
    for (Real val : grid) {
        if (!std::isfinite(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

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
        nx > std::numeric_limits<size_t>::max() / ny || nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions must be at least 3, iterations nonnegative, and grid size representable.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    std::vector<Real> hostGrid(gridSize);
    std::printf("Initializing grid...\n");
    initializeGrid(hostGrid);

    Real *deviceA = nullptr, *deviceB = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceA, bytes));
    CUDA_CHECK(cudaMalloc(&deviceB, bytes));
    CUDA_CHECK(cudaMemcpy(deviceA, hostGrid.data(), bytes, cudaMemcpyHostToDevice));
    // Both buffers need the fixed boundary values; interiors are overwritten.
    CUDA_CHECK(cudaMemcpy(deviceB, hostGrid.data(), bytes, cudaMemcpyHostToDevice));

    const dim3 block(BX, BY, BZ);
    const dim3 blocks(static_cast<unsigned>((nx - 2 + BX - 1) / BX),
                      static_cast<unsigned>((ny - 2 + BY - 1) / BY),
                      static_cast<unsigned>((nz - 2 + BZ - 1) / BZ));
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        stencilKernel<<<blocks, block>>>(deviceA, deviceB, nx, ny, nz);
        std::swap(deviceA, deviceB);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double updates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    const double mcups = elapsedMs > 0.0f ? updates / (static_cast<double>(elapsedMs) * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) CUDA_CHECK(cudaMemcpy(hostGrid.data(), deviceA, bytes, cudaMemcpyDeviceToHost));
    if (printResults) print_results(hostGrid, "Grid");
    int result = 0;
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(hostGrid);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        result = valid ? 0 : 1;
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceA));
    CUDA_CHECK(cudaFree(deviceB));
    return result;
}
