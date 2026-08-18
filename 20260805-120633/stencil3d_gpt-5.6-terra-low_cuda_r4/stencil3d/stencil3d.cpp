#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// x is the contiguous dimension, so each warp issues coalesced global loads.
__global__ void initializeGrid(Real* grid, size_t count) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < count) grid[idx] = static_cast<Real>(idx % 19);
}

// Boundaries are copied in the same launch as the interior calculation.  This
// avoids a separate full-grid pass and preserves the original ping-pong semantics.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t plane = nx * ny;
    const size_t idx = z * plane + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
        output[idx] = input[idx];
    } else {
        output[idx] = (input[idx] + input[idx - 1] + input[idx + 1]
                     + input[idx - nx] + input[idx + nx]
                     + input[idx - plane] + input[idx + plane]) / 7.0;
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
    
    size_t gridSize = nx * ny * nz;
    
    // Initialize
    printf("Initializing grid...\n");
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    checkCuda(cudaMalloc(&grid1, gridSize * sizeof(Real)), "allocating first grid");
    checkCuda(cudaMalloc(&grid2, gridSize * sizeof(Real)), "allocating second grid");

    constexpr unsigned int initBlockSize = 256;
    const unsigned int initBlocks = static_cast<unsigned int>((gridSize + initBlockSize - 1) / initBlockSize);
    initializeGrid<<<initBlocks, initBlockSize>>>(grid1, gridSize);
    checkCuda(cudaGetLastError(), "launching initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "initializing grid");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    checkCuda(cudaEventRecord(start), "recording start event");

    const dim3 block(32, 8, 1);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(nz));
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration<<<grid, block>>>(grid1, grid2, nx, ny, nz);
        } else {
            stencilIteration<<<grid, block>>>(grid2, grid1, nx, ny, nz);
        }
    }
    
    checkCuda(cudaGetLastError(), "launching stencil kernel");
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "running stencil computation");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "measuring elapsed time");
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (elapsedMs / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const Real* finalGridDevice = (iterations % 2 == 0) ? grid1 : grid2;
    
    // Validation
    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), finalGridDevice, gridSize * sizeof(Real), cudaMemcpyDeviceToHost),
                  "copying final grid to host");
        if (printResults) print_results(finalGrid, "Grid");
        if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        checkCuda(cudaFree(grid1), "freeing first grid");
        checkCuda(cudaFree(grid2), "freeing second grid");
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
        }
    }
    checkCuda(cudaFree(grid1), "freeing first grid");
    checkCuda(cudaFree(grid2), "freeing second grid");
    
    return 0;
}
