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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

static void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// x is the contiguous dimension.  A full warp therefore issues coalesced loads
// and stores, while adjacent warps reuse most y/z-neighbor cache lines.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) return;

    const size_t plane = nx * ny;
    const size_t i = z * plane + y * nx + x;
    output[i] = (input[i] + input[i - 1] + input[i + 1] +
                 input[i - nx] + input[i + nx] +
                 input[i - plane] + input[i + plane]) / 7.0;
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
    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0) {
        fprintf(stderr, "Grid dimensions must be at least 3 and iterations non-negative\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    const size_t gridBytes = gridSize * sizeof(Real);
    checkCuda(cudaMalloc(&deviceGrid1, gridBytes), "allocating first grid");
    checkCuda(cudaMalloc(&deviceGrid2, gridBytes), "allocating second grid");
    checkCuda(cudaMemcpy(deviceGrid1, grid1.data(), gridBytes, cudaMemcpyHostToDevice), "uploading first grid");
    // Both grids start identically, so immutable boundary cells are already correct.
    checkCuda(cudaMemcpy(deviceGrid2, deviceGrid1, gridBytes, cudaMemcpyDeviceToDevice), "initializing second grid");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    const dim3 block(32, 4, 2);
    const dim3 launchGrid(static_cast<unsigned>((nx - 2 + block.x - 1) / block.x),
                          static_cast<unsigned>((ny - 2 + block.y - 1) / block.y),
                          static_cast<unsigned>((nz - 2 + block.z - 1) / block.z));
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter & 1) ? deviceGrid2 : deviceGrid1;
        Real* output = (iter & 1) ? deviceGrid1 : deviceGrid2;
        stencilIteration<<<launchGrid, block>>>(input, output, nx, ny, nz);
    }
    checkCuda(cudaGetLastError(), "launching stencil kernel");
    checkCuda(cudaDeviceSynchronize(), "executing stencil kernels");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / elapsedSeconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    Real* finalDeviceGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
    if (printResults || validate) {
        checkCuda(cudaMemcpy(finalGrid.data(), finalDeviceGrid, gridBytes, cudaMemcpyDeviceToHost), "downloading result");
    }
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            checkCuda(cudaFree(deviceGrid1), "freeing first grid");
            checkCuda(cudaFree(deviceGrid2), "freeing second grid");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            checkCuda(cudaFree(deviceGrid1), "freeing first grid");
            checkCuda(cudaFree(deviceGrid2), "freeing second grid");
            return 1;
        }
    }
    
    checkCuda(cudaFree(deviceGrid1), "freeing first grid");
    checkCuda(cudaFree(deviceGrid2), "freeing second grid");
    return 0;
}
