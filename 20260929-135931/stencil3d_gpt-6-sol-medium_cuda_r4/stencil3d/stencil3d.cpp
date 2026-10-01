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

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrid(Real* grid, size_t count) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = start; i < count; i += stride) {
        grid[i] = static_cast<Real>(i % 19);
    }
}

// Each thread owns one X/Y position and four consecutive Z positions.
// The three Z values in registers are advanced after each output cell.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = nx * ny;
    const size_t firstZ = static_cast<size_t>(blockIdx.z) * 4;
    if (firstZ >= nz) return;
    const size_t lastZ = min(firstZ + 4, nz);
    size_t index = firstZ * plane + y * nx + x;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny) {
        for (size_t z = firstZ; z < lastZ; ++z, index += plane)
            output[index] = input[index];
        return;
    }

    Real below = firstZ ? input[index - plane] : 0.0;
    Real center = input[index];
    for (size_t z = firstZ; z < lastZ; ++z, index += plane) {
        const Real above = z + 1 < nz ? input[index + plane] : 0.0;
        if (z == 0 || z + 1 == nz) {
            output[index] = center;
        } else {
            output[index] = (center + input[index - 1] + input[index + 1]
                             + input[index - nx] + input[index + nx]
                             + below + above) / 7.0;
        }
        below = center;
        center = above;
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
    
    if (nx < 2 || ny < 2 || nz < 2 ||
        nx > SIZE_MAX / ny || nx * ny > SIZE_MAX / nz ||
        nx * ny * nz > SIZE_MAX / sizeof(Real)) {
        fprintf(stderr, "Invalid grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    checkCuda(cudaMalloc(&grid1, bytes), "Allocate first GPU grid");
    checkCuda(cudaMalloc(&grid2, bytes), "Allocate second GPU grid");

    // Initialize the first buffer on the GPU.
    printf("Initializing grid...\n");
    initializeGrid<<<static_cast<unsigned int>(std::min<size_t>((gridSize + 255) / 256, 65535)), 256>>>(grid1, gridSize);
    checkCuda(cudaGetLastError(), "Launch initialization");
    checkCuda(cudaDeviceSynchronize(), "Initialize grid");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    const dim3 block(32, 8);
    const dim3 launchGrid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                          static_cast<unsigned int>((ny + block.y - 1) / block.y),
                          static_cast<unsigned int>((nz + 3) / 4));
    for (int iter = 0; iter < iterations; ++iter) {
        const Real* input = (iter % 2 == 0) ? grid1 : grid2;
        Real* output = (iter % 2 == 0) ? grid2 : grid1;
        stencilIteration<<<launchGrid, block>>>(input, output, nx, ny, nz);
        checkCuda(cudaGetLastError(), "Launch stencil iteration");
    }
    checkCuda(cudaDeviceSynchronize(), "Run stencil iterations");
    
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %.3f ms\n", elapsedSeconds * 1000.0);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / elapsedSeconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        const Real* finalDeviceGrid = (iterations % 2 == 0) ? grid1 : grid2;
        checkCuda(cudaMemcpy(finalGrid.data(), finalDeviceGrid, bytes, cudaMemcpyDeviceToHost),
                  "Copy final grid to host");
    }
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    int result = 0;
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    checkCuda(cudaFree(grid1), "Free first GPU grid");
    checkCuda(cudaFree(grid2), "Free second GPU grid");
    return result;
}
