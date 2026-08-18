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

__global__ void initializeKernel(Real* grid, const size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        grid[i] = static_cast<Real>(i % 19);
    }
}

// 7-point stencil computation
__global__ void stencilKernel(const Real* input, Real* output, const size_t count,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t plane = nx * ny;
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; idx < count;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t z = idx / plane;
        const size_t rem = idx - z * plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
            output[idx] = input[idx];
        } else {
            output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                           input[idx - nx] + input[idx + nx] +
                           input[idx - plane] + input[idx + plane]) / 7.0;
        }
    }
}

static void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
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
    
    // Allocate host grids and device double buffers.
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    cudaCheck(cudaMalloc(&deviceGrid1, gridSize * sizeof(Real)), "cudaMalloc");
    cudaCheck(cudaMalloc(&deviceGrid2, gridSize * sizeof(Real)), "cudaMalloc");
    
    // Initialize
    printf("Initializing grid...\n");
    const int threads = 256;
    const int blocks = static_cast<int>(std::min<size_t>((gridSize + threads - 1) / threads, 2147483647ULL));
    initializeKernel<<<blocks, threads>>>(deviceGrid1, gridSize);
    cudaCheck(cudaGetLastError(), "initialize kernel launch");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    cudaCheck(cudaDeviceSynchronize(), "initialization");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter % 2 == 0) ? deviceGrid1 : deviceGrid2;
        Real* output = (iter % 2 == 0) ? deviceGrid2 : deviceGrid1;
        stencilKernel<<<blocks, threads>>>(input, output, gridSize, nx, ny, nz);
        cudaCheck(cudaGetLastError(), "stencil kernel launch");
    }

    cudaCheck(cudaDeviceSynchronize(), "stencil computation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    cudaCheck(cudaMemcpy(finalGrid.data(), (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2,
                         gridSize * sizeof(Real), cudaMemcpyDeviceToHost), "cudaMemcpy");
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
    
    cudaCheck(cudaFree(deviceGrid1), "cudaFree");
    cudaCheck(cudaFree(deviceGrid2), "cudaFree");
    return 0;
}
