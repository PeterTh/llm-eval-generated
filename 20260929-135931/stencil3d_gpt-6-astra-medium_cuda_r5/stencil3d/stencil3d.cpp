#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// Fail explicitly if CUDA is unavailable; this benchmark has no CPU fallback.
void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = Real(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

constexpr unsigned TileX = 32;
constexpr unsigned TileY = 4;
constexpr unsigned TileZ = 8;

// Each warp accesses consecutive X coordinates. Streaming a short Z column
// reuses the center and both Z neighbors in registers across iterations.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesX, size_t tilesY, size_t tileCount) {
    const size_t plane = nx * ny;
    for (size_t tile = blockIdx.x; tile < tileCount; tile += gridDim.x) {
        const size_t tx = tile % tilesX;
        const size_t tyz = tile / tilesX;
        const size_t x = tx * TileX + threadIdx.x + 1;
        const size_t y = (tyz % tilesY) * TileY + threadIdx.y + 1;
        const size_t zBegin = (tyz / tilesY) * TileZ + 1;
        if (x >= nx - 1 || y >= ny - 1) continue;
        const size_t zEnd = (zBegin + TileZ < nz - 1) ? zBegin + TileZ : nz - 1;
        size_t i = zBegin * plane + y * nx + x;
        Real bottom = input[i - plane];
        Real center = input[i];
        for (size_t z = zBegin; z < zEnd; ++z, i += plane) {
            const Real top = input[i + plane];
            // Preserve the original double precision summation order.
            output[i] = (center + input[i - 1] + input[i + 1]
                         + input[i - nx] + input[i + nx] + bottom + top) / 7.0;
            bottom = center;
            center = top;
        }
    }
}

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
    
    const size_t maxCount = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > maxCount / ny || nx * ny > maxCount / nz) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    checkCuda(cudaMalloc(&grid1, bytes), "Allocating first CUDA grid");
    checkCuda(cudaMalloc(&grid2, bytes), "Allocating second CUDA grid");

    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min(size_t(65535), (gridSize + 255) / 256));
    initializeGrid<<<initBlocks, 256>>>(grid1, grid2, gridSize);
    checkCuda(cudaGetLastError(), "Launching grid initialization");
    checkCuda(cudaDeviceSynchronize(), "Initializing CUDA grids");

    // Both buffers already contain the immutable boundaries, so no per-step
    // boundary copy or host/device transfer is necessary.
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t tilesX = hasInterior ? (nx - 2 + TileX - 1) / TileX : 0;
    const size_t tilesY = hasInterior ? (ny - 2 + TileY - 1) / TileY : 0;
    const size_t tilesZ = hasInterior ? (nz - 2 + TileZ - 1) / TileZ : 0;
    const size_t tileCount = tilesX * tilesY * tilesZ;
    const unsigned blocks = static_cast<unsigned>(std::min(size_t(65535), tileCount));
    const dim3 threads(TileX, TileY);

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations && hasInterior; ++iter) {
        stencilIteration<<<blocks, threads>>>(grid1, grid2, nx, ny, nz,
                                              tilesX, tilesY, tileCount);
        std::swap(grid1, grid2);
    }
    checkCuda(cudaGetLastError(), "Launching stencil iterations");
    checkCuda(cudaDeviceSynchronize(), "Completing stencil iterations");
    auto end = std::chrono::high_resolution_clock::now();
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %.3f ms\n", seconds * 1000.0);
    const double cellUpdates = hasInterior
        ? double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? cellUpdates / seconds / 1e6 : 0.0);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), grid1, bytes, cudaMemcpyDeviceToHost),
                  "Reading final CUDA grid");
    }
    checkCuda(cudaFree(grid2), "Freeing second CUDA grid");
    checkCuda(cudaFree(grid1), "Freeing first CUDA grid");

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
