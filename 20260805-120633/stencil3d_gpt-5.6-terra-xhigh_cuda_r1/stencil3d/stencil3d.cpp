#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 4;
constexpr unsigned int kBlockZ = 4;
constexpr unsigned int kThreadsPerBlock = kBlockX * kBlockY * kBlockZ;
constexpr unsigned int kTileX = kBlockX + 2;
constexpr unsigned int kTileY = kBlockY + 2;
constexpr unsigned int kTileZ = kBlockZ + 2;
constexpr unsigned int kTileElements = kTileX * kTileY * kTileZ;

[[noreturn]] void cudaFail(const cudaError_t status, const char* expression, const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\\n",
                 file, line, expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call)                                                                            \
    do {                                                                                            \
        const cudaError_t cudaStatus_ = (call);                                                     \
        if (cudaStatus_ != cudaSuccess) {                                                           \
            cudaFail(cudaStatus_, #call, __FILE__, __LINE__);                                      \
        }                                                                                           \
    } while (false)

// Initializing both device buffers keeps every boundary cell valid.  The stencil
// kernel can then update only interior cells, so no boundary-copy kernel or
// per-iteration host/device traffic is needed.
__global__ void initializeGrids(Real* const grid1, Real* const grid2, const size_t gridSize) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < gridSize;
         index += stride) {
        const Real value = static_cast<Real>(index % 19);
        grid1[index] = value;
        grid2[index] = value;
    }
}

// A 32x4x4 output tile gives coalesced x-direction accesses.  Shared memory
// stages the one-cell halo so that the seven values used by neighbouring output
// cells are loaded once per block rather than repeatedly from global memory.
__global__ __launch_bounds__(kThreadsPerBlock, 2)
void stencilIteration(const Real* __restrict__ input,
                      Real* __restrict__ output,
                      const size_t nx,
                      const size_t ny,
                      const size_t nz) {
    __shared__ Real tile[kTileElements];

    const unsigned int threadLinear =
        (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const size_t plane = nx * ny;
    const size_t interiorX = nx - 2;
    const size_t interiorY = ny - 2;
    const size_t interiorZ = nz - 2;

    // The grid-stride tile loops also cover dimensions larger than CUDA's grid
    // limits.  For ordinary benchmark sizes every loop has one trip, so they
    // impose no extra work on the hot path.
    for (size_t tileZBase = static_cast<size_t>(blockIdx.z) * kBlockZ;
         tileZBase < interiorZ;
         tileZBase += static_cast<size_t>(gridDim.z) * kBlockZ) {
        for (size_t tileYBase = static_cast<size_t>(blockIdx.y) * kBlockY;
             tileYBase < interiorY;
             tileYBase += static_cast<size_t>(gridDim.y) * kBlockY) {
            for (size_t tileXBase = static_cast<size_t>(blockIdx.x) * kBlockX;
                 tileXBase < interiorX;
                 tileXBase += static_cast<size_t>(gridDim.x) * kBlockX) {
                for (unsigned int linear = threadLinear; linear < kTileElements; linear += kThreadsPerBlock) {
                    const unsigned int localX = linear % kTileX;
                    const unsigned int localY = (linear / kTileX) % kTileY;
                    const unsigned int localZ = linear / (kTileX * kTileY);
                    const size_t x = tileXBase + localX;
                    const size_t y = tileYBase + localY;
                    const size_t z = tileZBase + localZ;

                    tile[linear] = (x < nx && y < ny && z < nz)
                        ? input[z * plane + y * nx + x]
                        : Real{0};
                }
                __syncthreads();

                const size_t x = tileXBase + threadIdx.x + 1;
                const size_t y = tileYBase + threadIdx.y + 1;
                const size_t z = tileZBase + threadIdx.z + 1;
                if (x < nx - 1 && y < ny - 1 && z < nz - 1) {
                    const unsigned int center =
                        ((threadIdx.z + 1) * kTileY + (threadIdx.y + 1)) * kTileX + threadIdx.x + 1;
                    output[z * plane + y * nx + x] =
                        (tile[center] + tile[center - 1] + tile[center + 1] +
                         tile[center - kTileX] + tile[center + kTileX] +
                         tile[center - kTileX * kTileY] + tile[center + kTileX * kTileY]) / 7.0;
                }
                __syncthreads();
            }
        }
    }
}

size_t ceilDivide(const size_t numerator, const size_t denominator) {
    return numerator / denominator + (numerator % denominator != 0);
}

}  // namespace

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
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
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
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
    
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate host grids for the existing result and validation interfaces.
    // The working copies remain on the GPU for the complete iteration loop.
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    printf("Initializing grid...\n");

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;

    CUDA_CHECK(cudaMalloc(&deviceGrid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, gridSize * sizeof(Real)));

    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));
    const size_t maxGridX = static_cast<size_t>(deviceProperties.maxGridSize[0]);
    const size_t initBlocks = std::min(ceilDivide(gridSize, static_cast<size_t>(256)), maxGridX);
    initializeGrids<<<static_cast<unsigned int>(initBlocks), 256>>>(deviceGrid1, deviceGrid2, gridSize);
    CUDA_CHECK(cudaGetLastError());

    dim3 stencilGrid{};
    if (nx > 2 && ny > 2 && nz > 2) {
        const size_t blocksX = std::min(ceilDivide(nx - 2, static_cast<size_t>(kBlockX)), maxGridX);
        const size_t blocksY = std::min(ceilDivide(ny - 2, static_cast<size_t>(kBlockY)),
                                        static_cast<size_t>(deviceProperties.maxGridSize[1]));
        const size_t blocksZ = std::min(ceilDivide(nz - 2, static_cast<size_t>(kBlockZ)),
                                        static_cast<size_t>(deviceProperties.maxGridSize[2]));
        stencilGrid = dim3(static_cast<unsigned int>(blocksX),
                           static_cast<unsigned int>(blocksY),
                           static_cast<unsigned int>(blocksZ));
    }

    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    printf("Running stencil computation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));

    Real* currentGrid = deviceGrid1;
    Real* nextGrid = deviceGrid2;
    if (nx > 2 && ny > 2 && nz > 2) {
        constexpr dim3 stencilBlock{kBlockX, kBlockY, kBlockZ};
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<stencilGrid, stencilBlock>>>(currentGrid, nextGrid, nx, ny, nz);
            CUDA_CHECK(cudaGetLastError());
            std::swap(currentGrid, nextGrid);
        }
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));

    printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMilliseconds));
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy just the final buffer back after all GPU work has completed.
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), currentGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));

    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
