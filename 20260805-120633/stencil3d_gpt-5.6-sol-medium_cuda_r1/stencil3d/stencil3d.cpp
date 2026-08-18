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

constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 4;
constexpr unsigned kBlockZ = 4;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeKernel(Real* grid, size_t count) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

// The center of the tile and its six faces are loaded exactly once per block.
// Edges and corners are unnecessary for a seven-point stencil.
__global__ __launch_bounds__(kBlockX * kBlockY * kBlockZ)
void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                   size_t nx, size_t ny, size_t nz) {
    __shared__ Real tile[kBlockZ + 2][kBlockY + 2][kBlockX + 2];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tz = threadIdx.z;
    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + tx + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + ty + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * kBlockZ + tz + 1;
    const size_t plane = nx * ny;

    const bool inGrid = x < nx && y < ny && z < nz;
    const size_t index = z * plane + y * nx + x;
    tile[tz + 1][ty + 1][tx + 1] = inGrid ? input[index] : 0.0;

    if (inGrid && tx == 0)
        tile[tz + 1][ty + 1][0] = input[index - 1];
    if (inGrid && tx == kBlockX - 1 && x + 1 < nx)
        tile[tz + 1][ty + 1][kBlockX + 1] = input[index + 1];
    if (inGrid && ty == 0)
        tile[tz + 1][0][tx + 1] = input[index - nx];
    if (inGrid && ty == kBlockY - 1 && y + 1 < ny)
        tile[tz + 1][kBlockY + 1][tx + 1] = input[index + nx];
    if (inGrid && tz == 0)
        tile[0][ty + 1][tx + 1] = input[index - plane];
    if (inGrid && tz == kBlockZ - 1 && z + 1 < nz)
        tile[kBlockZ + 1][ty + 1][tx + 1] = input[index + plane];

    __syncthreads();

    if (x < nx - 1 && y < ny - 1 && z < nz - 1) {
        const Real sum = tile[tz + 1][ty + 1][tx + 1]
                       + tile[tz + 1][ty + 1][tx]
                       + tile[tz + 1][ty + 1][tx + 2]
                       + tile[tz + 1][ty][tx + 1]
                       + tile[tz + 1][ty + 2][tx + 1]
                       + tile[tz][ty + 1][tx + 1]
                       + tile[tz + 2][ty + 1][tx + 1];
        output[index] = sum / 7.0;
    }
}

bool validateResult(const std::vector<Real>& grid) {
    for (const Real value : grid) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto bounds = std::minmax_element(grid.begin(), grid.end());
    std::printf("Value range: [%.6f, %.6f]\n", *bounds.first, *bounds.second);
    if (*bounds.second > 1e6 || *bounds.first < -1e6) {
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

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::strtoull(argv[++i], nullptr, 10);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid dimensions must be at least 3 and fit in memory; iterations must be nonnegative.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    cudaCheck(cudaMalloc(&deviceGrid1, bytes), "allocating the first grid");
    cudaCheck(cudaMalloc(&deviceGrid2, bytes), "allocating the second grid");

    std::printf("Initializing grid...\n");
    constexpr unsigned initThreads = 256;
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>(
        (gridSize + initThreads - 1) / initThreads, 65535));
    initializeKernel<<<initBlocks, initThreads>>>(deviceGrid1, gridSize);
    cudaCheck(cudaGetLastError(), "launching grid initialization");
    cudaCheck(cudaMemcpy(deviceGrid2, deviceGrid1, bytes, cudaMemcpyDeviceToDevice),
              "initializing the second grid");

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 blocks(static_cast<unsigned>((nx - 2 + kBlockX - 1) / kBlockX),
                      static_cast<unsigned>((ny - 2 + kBlockY - 1) / kBlockY),
                      static_cast<unsigned>((nz - 2 + kBlockZ - 1) / kBlockZ));
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaCheck(cudaEventCreate(&start), "creating the start event");
    cudaCheck(cudaEventCreate(&stop), "creating the stop event");

    std::printf("Running stencil computation...\n");
    cudaCheck(cudaEventRecord(start), "recording the start event");
    Real* input = deviceGrid1;
    Real* output = deviceGrid2;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        stencilKernel<<<blocks, block>>>(input, output, nx, ny, nz);
        std::swap(input, output);
    }
    cudaCheck(cudaGetLastError(), "launching the stencil kernels");
    cudaCheck(cudaEventRecord(stop), "recording the stop event");
    cudaCheck(cudaEventSynchronize(stop), "waiting for stencil completion");

    float elapsedMs = 0.0f;
    cudaCheck(cudaEventElapsedTime(&elapsedMs, start, stop), "measuring computation time");
    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMs));
    const double cellUpdates = static_cast<double>(nx - 2) *
                               static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * iterations;
    const double mcups = elapsedMs > 0.0f ? cellUpdates / (elapsedMs * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        cudaCheck(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost),
                  "copying the final grid to the host");
    }
    if (printResults)
        print_results(finalGrid, "Grid");

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(finalGrid);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(deviceGrid1);
    cudaFree(deviceGrid2);
    return valid ? 0 : 1;
}
