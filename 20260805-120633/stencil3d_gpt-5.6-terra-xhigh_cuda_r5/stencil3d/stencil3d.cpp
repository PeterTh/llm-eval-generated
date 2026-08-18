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

// The tile is deliberately wide in X, which keeps both global-memory loads and
// stores coalesced.  Each block computes four Z planes, increasing reuse of
// the two halo planes while keeping shared-memory use low enough for high
// occupancy on current CUDA GPUs.
constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 8;
constexpr unsigned int kBlockZ = 4;
constexpr unsigned int kThreadsPerBlock = kBlockX * kBlockY;
constexpr unsigned int kTileX = kBlockX + 2;
constexpr unsigned int kTileY = kBlockY + 2;
constexpr unsigned int kTileZ = kBlockZ + 2;
constexpr unsigned int kTileElements = kTileX * kTileY * kTileZ;

[[noreturn]] void cudaCheckFailure(const cudaError_t error, const char* expression,
                                   const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                    \
    do {                                                                          \
        const cudaError_t cudaStatus = (expression);                             \
        if (cudaStatus != cudaSuccess) {                                          \
            cudaCheckFailure(cudaStatus, #expression, __FILE__, __LINE__);       \
        }                                                                         \
    } while (false)

__global__ void initializeGridKernel(Real* const grid, const size_t gridSize) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < gridSize) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

// Every point, including the boundary, is written each iteration.  Keeping the
// boundary copy in this kernel avoids a second launch and lets the two grids be
// used with ordinary ping-pong buffering.
__global__ void stencilIterationKernel(const Real* __restrict__ input,
                                       Real* __restrict__ output, const size_t nx,
                                       const size_t ny, const size_t nz) {
    __shared__ Real tile[kTileZ][kTileY][kTileX];

    const size_t blockBaseX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t blockBaseY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t blockBaseZ = static_cast<size_t>(blockIdx.z) * kBlockZ;
    const size_t planeSize = nx * ny;
    const unsigned int lane = threadIdx.y * kBlockX + threadIdx.x;

    // Cooperatively stage one-cell halos around all four output Z planes.
    // Invalid portions of an edge tile are set to zero; no valid interior point
    // reads them because every such point has all six neighbors in range.
    for (unsigned int linear = lane; linear < kTileElements; linear += kThreadsPerBlock) {
        const unsigned int localX = linear % kTileX;
        const unsigned int yz = linear / kTileX;
        const unsigned int localY = yz % kTileY;
        const unsigned int localZ = yz / kTileY;

        const long long globalX = static_cast<long long>(blockBaseX) + localX - 1;
        const long long globalY = static_cast<long long>(blockBaseY) + localY - 1;
        const long long globalZ = static_cast<long long>(blockBaseZ) + localZ - 1;

        if (globalX >= 0 && globalX < static_cast<long long>(nx) && globalY >= 0 &&
            globalY < static_cast<long long>(ny) && globalZ >= 0 &&
            globalZ < static_cast<long long>(nz)) {
            const size_t globalIndex = static_cast<size_t>(globalZ) * planeSize +
                                       static_cast<size_t>(globalY) * nx +
                                       static_cast<size_t>(globalX);
            tile[localZ][localY][localX] = input[globalIndex];
        } else {
            tile[localZ][localY][localX] = 0.0;
        }
    }
    __syncthreads();

    const size_t x = blockBaseX + threadIdx.x;
    const size_t y = blockBaseY + threadIdx.y;
    const unsigned int tileX = threadIdx.x + 1;
    const unsigned int tileY = threadIdx.y + 1;

#pragma unroll
    for (unsigned int localZ = 0; localZ < kBlockZ; ++localZ) {
        const size_t z = blockBaseZ + localZ;
        if (x >= nx || y >= ny || z >= nz) {
            continue;
        }

        const size_t outputIndex = z * planeSize + y * nx + x;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
            output[outputIndex] = input[outputIndex];
        } else {
            const unsigned int tileZ = localZ + 1;
            const Real sum = tile[tileZ][tileY][tileX] +
                             tile[tileZ][tileY][tileX - 1] +
                             tile[tileZ][tileY][tileX + 1] +
                             tile[tileZ][tileY - 1][tileX] +
                             tile[tileZ][tileY + 1][tileX] +
                             tile[tileZ - 1][tileY][tileX] +
                             tile[tileZ + 1][tileY][tileX];
            output[outputIndex] = sum / 7.0;
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

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

    // After averaging, values should be somewhat bounded
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

bool checkedGridSize(const size_t nx, const size_t ny, const size_t nz, size_t* const gridSize) {
    if (nx == 0 || ny == 0 || nz == 0 || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        return false;
    }
    *gridSize = nx * ny * nz;
    return true;
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

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }

    size_t gridSize = 0;
    if (!checkedGridSize(nx, ny, nz, &gridSize)) {
        std::fprintf(stderr, "Grid dimensions must be non-zero and fit in addressable memory.\n");
        return 1;
    }

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // CUDA grid dimensions are 32-bit.  Values outside this range cannot be
    // represented by a CUDA launch even when their allocation happened to fit.
    const size_t maxGridDimension = std::numeric_limits<unsigned int>::max();
    if ((nx + kBlockX - 1) / kBlockX > maxGridDimension ||
        (ny + kBlockY - 1) / kBlockY > maxGridDimension ||
        (nz + kBlockZ - 1) / kBlockZ > maxGridDimension) {
        std::fprintf(stderr, "Grid dimensions exceed CUDA launch limits.\n");
        return 1;
    }

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&deviceGrid1, bytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, bytes));

    std::printf("Initializing grid...\n");
    const unsigned int initThreads = 256;
    const size_t initBlocks = (gridSize + initThreads - 1) / initThreads;
    if (initBlocks > maxGridDimension) {
        std::fprintf(stderr, "Grid is too large for CUDA initialization.\n");
        CUDA_CHECK(cudaFree(deviceGrid2));
        CUDA_CHECK(cudaFree(deviceGrid1));
        return 1;
    }
    initializeGridKernel<<<static_cast<unsigned int>(initBlocks), initThreads>>>(deviceGrid1, gridSize);
    // The second buffer is intentionally zeroed to match the original host
    // allocation before its first stencil update.
    CUDA_CHECK(cudaMemset(deviceGrid2, 0, bytes));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 block(kBlockX, kBlockY, 1);
    const dim3 grid(static_cast<unsigned int>((nx + kBlockX - 1) / kBlockX),
                    static_cast<unsigned int>((ny + kBlockY - 1) / kBlockY),
                    static_cast<unsigned int>((nz + kBlockZ - 1) / kBlockZ));

    // The tiled kernel has negligible L1 reuse beyond the explicit halo tile.
    // Favoring shared memory permits more resident tiles and therefore more
    // memory-level parallelism on Ampere.
    CUDA_CHECK(cudaFuncSetCacheConfig(stencilIterationKernel, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetAttribute(stencilIterationKernel,
                                    cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t endEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&endEvent));

    std::printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    for (int iter = 0; iter < iterations; ++iter) {
        if ((iter & 1) == 0) {
            stencilIterationKernel<<<grid, block>>>(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilIterationKernel<<<grid, block>>>(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
    }
    CUDA_CHECK(cudaEventRecord(endEvent));
    CUDA_CHECK(cudaEventSynchronize(endEvent));
    CUDA_CHECK(cudaGetLastError());

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, endEvent));
    CUDA_CHECK(cudaEventDestroy(endEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));

    std::printf("Computation time: %ld ms\n", static_cast<long>(elapsedMilliseconds));

    // Calculate performance metrics from the GPU event duration, which measures
    // exactly the ping-pong stencil kernel sequence.
    const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    const double elapsedSeconds = static_cast<double>(elapsedMilliseconds) / 1000.0;
    const double mcups = cellUpdates / elapsedSeconds / 1e6;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const bool finalInGrid1 = (iterations % 2 == 0);
    const Real* const finalDeviceGrid = finalInGrid1 ? deviceGrid1 : deviceGrid2;
    bool valid = true;
    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, bytes, cudaMemcpyDeviceToHost));

        if (printResults) {
            print_results(finalGrid, "Grid");
        }

        if (validate) {
            std::printf("Validating result...\n");
            valid = validateResult(finalGrid, nx, ny, nz);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }

    CUDA_CHECK(cudaFree(deviceGrid2));
    CUDA_CHECK(cudaFree(deviceGrid1));
    return valid ? 0 : 1;
}
