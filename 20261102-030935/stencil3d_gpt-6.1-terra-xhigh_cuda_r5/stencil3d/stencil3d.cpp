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

// Abort on CUDA errors, including asynchronous kernel failures at synchronization.
static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// Both buffers start with the same boundary values; the stencil never writes them.
__global__ void initializeGrids(Real* grid1, Real* grid2, size_t count) {
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (; i < count; i += stride) {
        const Real value = static_cast<Real>(i % 19);
        grid1[i] = value;
        grid2[i] = value;
    }
}

// Adjacent lanes traverse X, giving coalesced loads and stores. A block covers
// four rows; the grid-stride loop also supports arbitrarily many Y/Z tiles.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t rowTiles) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    const size_t tileCount = rowTiles * (nz - 2);
    for (size_t tile = blockIdx.y; tile < tileCount; tile += gridDim.y) {
        const size_t y = (tile % rowTiles) * blockDim.y + threadIdx.y + 1;
        const size_t z = tile / rowTiles + 1;
        if (y < ny - 1) {
            const size_t i = z * plane + y * nx + x;
            const Real center = input[i];
            const Real left = input[i - 1];
            const Real right = input[i + 1];
            const Real front = input[i - nx];
            const Real back = input[i + nx];
            const Real bottom = input[i - plane];
            const Real top = input[i + plane];
            output[i] = (center + left + right + front + back + bottom + top) / 7.0;
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
    
    if (nx == 0 || iterations < 0 ||
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
    checkCuda(cudaMalloc(&grid1, bytes), "allocating first grid");
    checkCuda(cudaMalloc(&grid2, bytes), "allocating second grid");

    printf("Initializing grid...\n");
    constexpr unsigned int initThreads = 256;
    const unsigned int initBlocks = static_cast<unsigned int>(
        std::min<size_t>((gridSize - 1) / initThreads + 1, 65535));
    initializeGrids<<<initBlocks, initThreads>>>(grid1, grid2, gridSize);
    checkCuda(cudaGetLastError(), "launching initialization");
    checkCuda(cudaDeviceSynchronize(), "initializing grids");

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const dim3 threads(32, 4);
    dim3 blocks(1, 1);
    size_t rowTiles = 0;
    if (hasInterior) {
        const size_t xBlocks = (nx - 3) / threads.x + 1;
        rowTiles = (ny - 3) / threads.y + 1;
        const size_t tileCount = rowTiles * (nz - 2);
        int device = 0;
        cudaDeviceProp properties{};
        checkCuda(cudaGetDevice(&device), "querying device");
        checkCuda(cudaGetDeviceProperties(&properties, device), "querying device properties");
        if (xBlocks > static_cast<size_t>(properties.maxGridSize[0])) {
            fprintf(stderr, "Grid X dimension exceeds CUDA launch limits\n");
            return 1;
        }
        blocks = dim3(static_cast<unsigned int>(xBlocks),
                      static_cast<unsigned int>(std::min<size_t>(tileCount, properties.maxGridSize[1])));
    }

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    Real* input = grid1;
    Real* output = grid2;
    if (hasInterior) {
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<blocks, threads>>>(input, output, nx, ny, nz, rowTiles);
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "launching stencil");
        checkCuda(cudaDeviceSynchronize(), "running stencil");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %ld ms\n", duration.count());

    double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                         static_cast<double>(ny > 2 ? ny - 2 : 0) *
                         static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost), "copying result");
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
