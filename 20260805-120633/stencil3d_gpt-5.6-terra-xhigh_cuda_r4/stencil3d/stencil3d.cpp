#include <algorithm>
#include <chrono>
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

constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 4;
constexpr int kThreadsPerBlock = kBlockX * kBlockY * kBlockZ;
constexpr int kTileX = kBlockX + 2;
constexpr int kTileY = kBlockY + 2;
constexpr int kTileZ = kBlockZ + 2;
constexpr int kTileElements = kTileX * kTileY * kTileZ;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t status_ = (call);                                               \
        if (status_ != cudaSuccess) {                                                     \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,     \
                         cudaGetErrorString(status_));                                    \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (false)

__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void initializeGrid(Real* const grid, const size_t gridSize) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < gridSize) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

// Each block stages a 32 x 4 x 4 interior region, including its one-cell halo.
// Boundaries are never written by this kernel; both buffers receive the same
// initialized boundary values before the timed iteration loop begins.
__global__ __launch_bounds__(kThreadsPerBlock, 2)
void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                      const size_t nx, const size_t ny, const size_t nz) {
    __shared__ Real tile[kTileElements];

    const size_t baseX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t baseZ = static_cast<size_t>(blockIdx.z) * kBlockZ;

    const int threadLinear = (threadIdx.z * kBlockY + threadIdx.y) * kBlockX + threadIdx.x;
    for (int tileIndex = threadLinear; tileIndex < kTileElements; tileIndex += kThreadsPerBlock) {
        const int sx = tileIndex % kTileX;
        const int sy = (tileIndex / kTileX) % kTileY;
        const int sz = tileIndex / (kTileX * kTileY);
        const size_t x = baseX + sx;
        const size_t y = baseY + sy;
        const size_t z = baseZ + sz;

        // The out-of-range portion of a tail tile is not consumed by a valid
        // output point, but assigning it keeps the cooperative load defined.
        tile[tileIndex] = (x < nx && y < ny && z < nz) ? input[idx3(x, y, z, nx, ny)] : 0.0;
    }
    __syncthreads();

    const size_t x = baseX + threadIdx.x + 1;
    const size_t y = baseY + threadIdx.y + 1;
    const size_t z = baseZ + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) {
        return;
    }

    const int sx = threadIdx.x + 1;
    const int sy = threadIdx.y + 1;
    const int sz = threadIdx.z + 1;
    const int centerIndex = (sz * kTileY + sy) * kTileX + sx;
    const Real center = tile[centerIndex];
    const Real left = tile[centerIndex - 1];
    const Real right = tile[centerIndex + 1];
    const Real front = tile[centerIndex - kTileX];
    const Real back = tile[centerIndex + kTileX];
    const Real bottom = tile[centerIndex - kTileX * kTileY];
    const Real top = tile[centerIndex + kTileX * kTileY];

    output[idx3(x, y, z, nx, ny)] =
        (center + left + right + front + back + bottom + top) / 7.0;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
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

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

bool checkedGridSize(const size_t nx, const size_t ny, const size_t nz, size_t& gridSize) {
    if (nx > std::numeric_limits<size_t>::max() / ny) {
        return false;
    }
    const size_t xy = nx * ny;
    if (xy > std::numeric_limits<size_t>::max() / nz) {
        return false;
    }
    gridSize = xy * nz;
    return true;
}

unsigned int blocksFor(const size_t extent, const int blockExtent) {
    const size_t blocks = (extent + static_cast<size_t>(blockExtent) - 1) /
                          static_cast<size_t>(blockExtent);
    if (blocks > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Grid dimension is too large for CUDA launch configuration\n");
        std::exit(EXIT_FAILURE);
    }
    return static_cast<unsigned int>(blocks);
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::atoi(argv[++i]);
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

    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be at least 3 and iterations must be non-negative\n");
        return 1;
    }

    size_t gridSize = 0;
    if (!checkedGridSize(nx, ny, nz, gridSize) ||
        gridSize > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        std::fprintf(stderr, "Grid size is too large\n");
        return 1;
    }
    const size_t gridBytes = gridSize * sizeof(Real);

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    Real* grid1Device = nullptr;
    Real* grid2Device = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1Device, gridBytes));
    CUDA_CHECK(cudaMalloc(&grid2Device, gridBytes));

    std::printf("Initializing grid...\n");
    constexpr int initThreads = 256;
    const unsigned int initBlocks = blocksFor(gridSize, initThreads);
    initializeGrid<<<initBlocks, initThreads>>>(grid1Device, gridSize);
    CUDA_CHECK(cudaGetLastError());
    // Initialize the second buffer once. The stencil kernel writes interiors
    // only, so this preserves the original immutable-boundary behavior.
    CUDA_CHECK(cudaMemcpy(grid2Device, grid1Device, gridBytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(blocksFor(nx - 2, kBlockX), blocksFor(ny - 2, kBlockY),
                    blocksFor(nz - 2, kBlockZ));

    std::printf("Running stencil computation...\n");
    cudaEvent_t start{};
    cudaEvent_t end{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencilIteration<<<grid, block>>>(grid1Device, grid2Device, nx, ny, nz);
        } else {
            stencilIteration<<<grid, block>>>(grid2Device, grid1Device, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    std::printf("Computation time: %ld ms\n", static_cast<long>(elapsedMilliseconds));

    const double cellUpdates = static_cast<double>(nx - 2) * static_cast<double>(ny - 2) *
                               static_cast<double>(nz - 2) * static_cast<double>(iterations);
    const double mcups = elapsedMilliseconds > 0.0F
                             ? cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1.0e3)
                             : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* const finalDevice = (iterations % 2 == 0) ? grid1Device : grid2Device;
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDevice, gridBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(grid1Device));
    CUDA_CHECK(cudaFree(grid2Device));

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(finalGrid, nx, ny, nz)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
