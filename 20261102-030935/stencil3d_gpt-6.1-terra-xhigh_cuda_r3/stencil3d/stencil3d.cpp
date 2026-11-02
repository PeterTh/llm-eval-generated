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

// Report CUDA failures at the point where they occur, including asynchronous errors.
static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrid(Real* grid1, Real* grid2, size_t count,
                               bool setBoundaries) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < count; idx += stride) {
        const Real value = static_cast<Real>(idx % 19);
        grid1[idx] = value;
        // Every interior cell is overwritten in the first iteration.
        grid2[idx] = setBoundaries ? value : 0.0;
    }
}

// Each warp covers consecutive x coordinates for coalesced accesses.  The
// boundary values were initialized in both buffers and never need rewriting.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = nx * ny;
    for (size_t z = static_cast<size_t>(blockIdx.z) + 1; z < nz - 1; z += gridDim.z) {
        for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
             y < ny - 1; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
            if (x > 0 && x < nx - 1) {
                const size_t idx = z * plane + y * nx + x;
                const Real sum = (((((input[idx] + input[idx - 1]) + input[idx + 1])
                                  + input[idx - nx]) + input[idx + nx])
                                  + input[idx - plane]) + input[idx + plane];
                output[idx] = sum / 7.0;
            }
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
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real *grid1 = nullptr, *grid2 = nullptr;
    cudaCheck(cudaMalloc(&grid1, bytes), "allocate first grid");
    cudaCheck(cudaMalloc(&grid2, bytes), "allocate second grid");

    printf("Initializing grid...\n");
    const unsigned int initBlocks = static_cast<unsigned int>(
        std::min<size_t>((gridSize - 1) / 256 + 1, 65535));
    initializeGrid<<<initBlocks, 256>>>(grid1, grid2, gridSize, iterations > 0);
    cudaCheck(cudaGetLastError(), "launch initialization");
    cudaCheck(cudaDeviceSynchronize(), "initialize grids");

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (nx > 2 && ny > 2 && nz > 2) {
        const size_t xBlocks = (nx - 1) / 32 + 1;
        if (xBlocks > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fprintf(stderr, "Grid X dimension is too large for CUDA launch\n");
            return 1;
        }
        const dim3 block(32, 4);
        const dim3 grid(static_cast<unsigned int>(xBlocks),
                        static_cast<unsigned int>(std::min<size_t>((ny - 3) / 4 + 1, 65535)),
                        static_cast<unsigned int>(std::min<size_t>(nz - 2, 65535)));
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<grid, block>>>((iter % 2 == 0) ? grid1 : grid2,
                                               (iter % 2 == 0) ? grid2 : grid1,
                                               nx, ny, nz);
        }
        cudaCheck(cudaGetLastError(), "launch stencil");
        cudaCheck(cudaDeviceSynchronize(), "run stencil");
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    const double cellUpdates = nx > 2 && ny > 2 && nz > 2
        ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * std::max(iterations, 0)
        : 0.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double mcups = cellUpdates / seconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        cudaCheck(cudaMemcpy(finalGrid.data(), (iterations % 2 == 0) ? grid1 : grid2,
                             bytes, cudaMemcpyDeviceToHost), "copy result to host");
    }
    cudaCheck(cudaFree(grid1), "free first grid");
    cudaCheck(cudaFree(grid2), "free second grid");
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
