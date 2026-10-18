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
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        // Both buffers retain these fixed boundaries throughout all iterations.
        second[i] = value;
    }
}

// Each warp accesses adjacent X coordinates. Register rolling along Z reuses
// the center and Z neighbors without shared-memory barriers. Independent Z
// tiles retain enough parallelism even for long, narrow grids.
constexpr unsigned tileX = 32;
constexpr unsigned tileY = 4;
constexpr unsigned tileZ = 8;

__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesX, size_t tilesY, size_t tileCount) {
    const size_t plane = nx * ny;
    for (size_t tile = blockIdx.x; tile < tileCount; tile += gridDim.x) {
        const size_t tx = tile % tilesX;
        const size_t yz = tile / tilesX;
        const size_t x = tx * tileX + threadIdx.x + 1;
        const size_t y = (yz % tilesY) * tileY + threadIdx.y + 1;
        const size_t zBegin = (yz / tilesY) * tileZ + 1;
        if (x >= nx - 1 || y >= ny - 1) continue;

        size_t i = zBegin * plane + y * nx + x;
        Real bottom = input[i - plane];
        Real center = input[i];
        #pragma unroll
        for (unsigned dz = 0; dz < tileZ; ++dz, i += plane) {
            if (zBegin + dz >= nz - 1) break;
            const Real top = input[i + plane];
            // Preserve the original order of double-precision operations.
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
    
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > maxElements / ny || nx * ny > maxElements / nz) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);

    int device = 0;
    cudaDeviceProp properties{};
    checkCuda(cudaGetDevice(&device), "getting CUDA device");
    checkCuda(cudaGetDeviceProperties(&properties, device), "getting device properties");
    const size_t maxBlocks = size_t(properties.multiProcessorCount) * 32;
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    checkCuda(cudaMalloc(&grid1, bytes), "allocating first grid");
    checkCuda(cudaMalloc(&grid2, bytes), "allocating second grid");
    
    // Initialize
    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(std::min((gridSize + 255) / 256, maxBlocks));
    initializeGrid<<<initBlocks, 256>>>(grid1, grid2, gridSize);
    checkCuda(cudaGetLastError(), "launching grid initialization");
    checkCuda(cudaDeviceSynchronize(), "initializing grids");

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const size_t tilesX = hasInterior ? (nx - 2 + tileX - 1) / tileX : 0;
    const size_t tilesY = hasInterior ? (ny - 2 + tileY - 1) / tileY : 0;
    const size_t tilesZ = hasInterior ? (nz - 2 + tileZ - 1) / tileZ : 0;
    const size_t tileCount = tilesX * tilesY * tilesZ;
    const unsigned blocks = static_cast<unsigned>(std::min(tileCount, maxBlocks));
    const dim3 threads(tileX, tileY);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    Real* input = grid1;
    Real* output = grid2;
    if (hasInterior) {
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<blocks, threads>>>(input, output, nx, ny, nz,
                                                  tilesX, tilesY, tileCount);
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "launching stencil iterations");
    }
    checkCuda(cudaDeviceSynchronize(), "running stencil iterations");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = hasInterior ? double(nx-2) * double(ny-2) * double(nz-2) * iterations : 0.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost),
                  "copying final grid");
    }
    checkCuda(cudaFree(grid1), "freeing first grid");
    checkCuda(cudaFree(grid2), "freeing second grid");
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
