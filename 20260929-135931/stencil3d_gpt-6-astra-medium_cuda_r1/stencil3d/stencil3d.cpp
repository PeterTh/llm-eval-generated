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

// CUDA is required: report allocation, launch, and execution failures explicitly.
inline void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

__global__ void initializeGrids(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

// Each warp accesses consecutive X coordinates. Each thread advances through
// a short Z column, retaining the previous and current planes in registers.
// Both buffers already contain the fixed boundaries, so only interiors are written.
constexpr unsigned tileX = 32;
constexpr unsigned tileY = 8;
constexpr unsigned tileZ = 4;
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesX, size_t tilesY, size_t tileCount) {
    const size_t plane = nx * ny;
    for (size_t tile = blockIdx.x; tile < tileCount; tile += gridDim.x) {
        const size_t tx = tile % tilesX;
        const size_t yz = tile / tilesX;
        const size_t x = tx * tileX + threadIdx.x;
        const size_t y = (yz % tilesY) * tileY + threadIdx.y + 1;
        const size_t zBegin = (yz / tilesY) * tileZ + 1;
        if (x == 0 || x >= nx - 1 || y >= ny - 1) continue;
        size_t index = zBegin * plane + y * nx + x;
        Real bottom = input[index - plane];
        Real center = input[index];
        #pragma unroll
        for (unsigned dz = 0; dz < tileZ; ++dz) {
            if (zBegin + dz >= nz - 1) break;
            const Real top = input[index + plane];
            // Preserve the original order of floating-point additions and division.
            output[index] = (center + input[index - 1] + input[index + 1]
                             + input[index - nx] + input[index + nx]
                             + bottom + top) / 7.0;
            bottom = center;
            center = top;
            index += plane;
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
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1, bytes));
    CUDA_CHECK(cudaMalloc(&grid2, bytes));

    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(
        std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeGrids<<<initBlocks, 256>>>(grid1, grid2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t tilesX = hasInterior ? (nx - 1 + tileX - 1) / tileX : 0;
    const size_t tilesY = hasInterior ? (ny - 2 + tileY - 1) / tileY : 0;
    const size_t tilesZ = hasInterior ? (nz - 2 + tileZ - 1) / tileZ : 0;
    const size_t tileCount = tilesX * tilesY * tilesZ;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(tileCount, 65535));
    const dim3 threads(tileX, tileY);
    Real* input = grid1;
    Real* output = grid2;

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (hasInterior) {
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<blocks, threads>>>(input, output, nx, ny, nz,
                                                  tilesX, tilesY, tileCount);
            std::swap(input, output);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));

    const double cellUpdates = hasInterior
        ? double(nx - 2) * double(ny - 2) * double(nz - 2) * iterations : 0.0;
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Transfer only the final grid, and only when the caller requests its values.
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(grid2));
    CUDA_CHECK(cudaFree(grid1));

    // Print results for external validation
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
