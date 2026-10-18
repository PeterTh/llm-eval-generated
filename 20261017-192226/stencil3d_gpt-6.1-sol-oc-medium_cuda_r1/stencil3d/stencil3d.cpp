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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

// Each warp spans X for coalesced loads/stores. Each thread sweeps a short
// Z column, retaining adjacent planes in registers. Independent Z tiles keep
// enough blocks available even for grids with small X/Y dimensions.
constexpr unsigned tileX = 32;
constexpr unsigned tileY = 4;
constexpr unsigned tileZ = 8;

__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesX, size_t tilesY, size_t tileCount) {
    for (size_t tile = blockIdx.x; tile < tileCount; tile += gridDim.x) {
        const size_t x = (tile % tilesX) * tileX + threadIdx.x + 1;
        const size_t yz = tile / tilesX;
        const size_t y = (yz % tilesY) * tileY + threadIdx.y + 1;
        const size_t z = (yz / tilesY) * tileZ + 1;
        if (x >= nx - 1 || y >= ny - 1) continue;

        const size_t plane = nx * ny;
        size_t idx = z * plane + y * nx + x;
        Real bottom = input[idx - plane];
        Real center = input[idx];
        #pragma unroll
        for (unsigned dz = 0; dz < tileZ; ++dz) {
            if (z + dz >= nz - 1) break;
            const Real top = input[idx + plane];
            // Preserve the original double-precision addition order.
            output[idx] = (center + input[idx - 1] + input[idx + 1]
                         + input[idx - nx] + input[idx + nx] + bottom + top) / 7.0;
            bottom = center;
            center = top;
            idx += plane;
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
    
    // Both buffers stay resident on the GPU for the complete computation.
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    checkCuda(cudaMalloc(&grid1, bytes), "allocate first grid");
    checkCuda(cudaMalloc(&grid2, bytes), "allocate second grid");
    
    // Initialize
    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min<size_t>((gridSize + 255) / 256, 65535));
    initializeGrid<<<initBlocks, 256>>>(grid1, grid2, gridSize);
    checkCuda(cudaGetLastError(), "initialize grids");
    // Fixed boundary values are initialized in both buffers and never written
    // by the stencil, avoiding a boundary-copy pass on every iteration.
    checkCuda(cudaDeviceSynchronize(), "finish initialization");

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t tilesX = hasInterior ? (nx - 2 + tileX - 1) / tileX : 0;
    const size_t tilesY = hasInterior ? (ny - 2 + tileY - 1) / tileY : 0;
    const size_t tileCount = hasInterior ? tilesX * tilesY * ((nz - 2 + tileZ - 1) / tileZ) : 0;
    const unsigned blocks = static_cast<unsigned>(std::min<size_t>(tileCount, 65535));
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (hasInterior) {
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<blocks, dim3(tileX, tileY)>>>(
                grid1, grid2, nx, ny, nz, tilesX, tilesY, tileCount);
            std::swap(grid1, grid2);
        }
        checkCuda(cudaGetLastError(), "launch stencil");
    }
    checkCuda(cudaDeviceSynchronize(), "finish stencil computation");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = hasInterior ? double(nx-2) * double(ny-2) * double(nz-2) * iterations : 0.0;
    double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), grid1, bytes, cudaMemcpyDeviceToHost), "read final grid");
    }
    checkCuda(cudaFree(grid1), "free first grid");
    checkCuda(cudaFree(grid2), "free second grid");
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
