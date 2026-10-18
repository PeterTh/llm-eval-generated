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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}
#define CUDA_CHECK(operation) checkCuda((operation), #operation)

__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

// Each warp accesses consecutive X coordinates. Rolling Z values in registers
// avoids reloading the center and bottom for each successive plane. Z tiles
// also expose parallelism for grids with a small XY cross-section.
constexpr unsigned blockX = 32;
constexpr unsigned blockY = 8;
constexpr unsigned tileZ = 8;
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t yTiles, size_t totalTiles) {
    const size_t x = size_t(blockIdx.x) * blockX + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t tile = blockIdx.y; tile < totalTiles; tile += gridDim.y) {
        const size_t y = (tile % yTiles) * blockY + threadIdx.y + 1;
        if (y >= ny - 1) continue;
        const size_t zStart = (tile / yTiles) * tileZ + 1;
        const size_t zEnd = zStart + tileZ < nz - 1 ? zStart + tileZ : nz - 1;
        size_t i = zStart * plane + y * nx + x;
        Real bottom = input[i - plane];
        Real center = input[i];
        for (size_t z = zStart; z < zEnd; ++z, i += plane) {
            const Real top = input[i + plane];
            output[i] = (center + input[i - 1] + input[i + 1] +
                         input[i - nx] + input[i + nx] + bottom + top) / 7.0;
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

    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Device double buffering keeps every iteration on the GPU. Both buffers
    // start with the fixed boundary values, so no boundary-copy pass is needed.
    std::vector<Real> grid1(gridSize);
    Real* deviceInput = nullptr;
    Real* deviceOutput = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&deviceInput, bytes));
    CUDA_CHECK(cudaMalloc(&deviceOutput, bytes));
    
    // Initialize
    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeGrid<<<initBlocks, 256>>>(deviceInput, deviceOutput, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t yTiles = hasInterior ? (ny - 2 + blockY - 1) / blockY : 0;
    const size_t totalTiles = hasInterior ? yTiles * ((nz - 2 + tileZ - 1) / tileZ) : 0;
    const dim3 block(blockX, blockY);
    const dim3 launchGrid(hasInterior ? static_cast<unsigned>((nx - 2 + blockX - 1) / blockX) : 1,
                          static_cast<unsigned>(std::min<size_t>(totalTiles, 65535)));
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (hasInterior) {
            stencilIteration<<<launchGrid, block>>>(deviceInput, deviceOutput,
                                                    nx, ny, nz, yTiles, totalTiles);
        }
        std::swap(deviceInput, deviceOutput);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = hasInterior ? double(nx-2) * double(ny-2) * double(nz-2) * iterations : 0.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / seconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    CUDA_CHECK(cudaMemcpy(grid1.data(), deviceInput, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceOutput));
    CUDA_CHECK(cudaFree(deviceInput));
    const std::vector<Real>& finalGrid = grid1;
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
