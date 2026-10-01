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

// Both buffers start with the same boundary values, which never change.
// Each launch therefore only needs to write the interior.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) return;

    const size_t plane = nx * ny;
    for (size_t z = 1 + blockIdx.z; z < nz - 1; z += gridDim.z) {
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

bool checkCuda(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return true;
    fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
    return false;
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
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    if (!checkCuda(cudaMalloc(&deviceGrid1, bytes), "allocation of first grid") ||
        !checkCuda(cudaMalloc(&deviceGrid2, bytes), "allocation of second grid") ||
        !checkCuda(cudaMemcpy(deviceGrid1, grid1.data(), bytes, cudaMemcpyHostToDevice), "initial grid upload") ||
        !checkCuda(cudaMemcpy(deviceGrid2, deviceGrid1, bytes, cudaMemcpyDeviceToDevice), "boundary initialization")) {
        cudaFree(deviceGrid1);
        cudaFree(deviceGrid2);
        return 1;
    }

    Real* deviceInput = deviceGrid1;
    Real* deviceOutput = deviceGrid2;
    const dim3 block(32, 8);
    const size_t interiorX = nx > 2 ? nx - 2 : 0;
    const size_t interiorY = ny > 2 ? ny - 2 : 0;
    const size_t interiorZ = nz > 2 ? nz - 2 : 0;
    const dim3 grid(static_cast<unsigned int>((interiorX + block.x - 1) / block.x),
                    static_cast<unsigned int>((interiorY + block.y - 1) / block.y),
                    static_cast<unsigned int>(std::min<size_t>(interiorZ, 65535)));
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (interiorX && interiorY && interiorZ) {
            stencilIteration<<<grid, block>>>(deviceInput, deviceOutput, nx, ny, nz);
        }
        std::swap(deviceInput, deviceOutput);
    }
    if (!checkCuda(cudaGetLastError(), "stencil launch") ||
        !checkCuda(cudaDeviceSynchronize(), "stencil execution")) {
        cudaFree(deviceGrid1);
        cudaFree(deviceGrid2);
        return 1;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if ((printResults || validate) &&
        !checkCuda(cudaMemcpy(finalGrid.data(), deviceInput, bytes, cudaMemcpyDeviceToHost), "final grid download")) {
        cudaFree(deviceGrid1);
        cudaFree(deviceGrid2);
        return 1;
    }
    cudaFree(deviceGrid1);
    cudaFree(deviceGrid2);
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
