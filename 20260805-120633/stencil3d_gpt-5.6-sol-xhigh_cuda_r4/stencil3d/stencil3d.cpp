#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 2;

__global__ void initializeKernel(Real* first, Real* second,
                                 const size_t elementCount) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elementCount; index += stride) {
        const Real value = static_cast<Real>(index % 19);
        first[index] = value;
        second[index] = value;
    }
}

// X spans one complete warp for coalesced loads and stores.  The y/z extent
// supplies enough independent warps to hide memory latency while neighboring
// blocks naturally reuse stencil planes through the read-only/L2 caches.
__global__ __launch_bounds__(kBlockX * kBlockY * kBlockZ)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output,
                   const int nx, const int ny, const int nz) {
    const int x = 1 + static_cast<int>(blockIdx.x) * kBlockX + threadIdx.x;
    const int y = 1 + static_cast<int>(blockIdx.y) * kBlockY + threadIdx.y;
    const int z = 1 + static_cast<int>(blockIdx.z) * kBlockZ + threadIdx.z;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) {
        return;
    }

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t index = static_cast<size_t>(z) * plane
                       + static_cast<size_t>(y) * static_cast<size_t>(nx)
                       + static_cast<size_t>(x);
    Real value = input[index];
    value += input[index - 1];
    value += input[index + 1];
    value += input[index - nx];
    value += input[index + nx];
    value += input[index - plane];
    value += input[index + plane];
    output[index] = value / 7.0;
}

bool cudaSucceeded(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) {
        return true;
    }
    fprintf(stderr, "CUDA error during %s: %s\n", operation,
            cudaGetErrorString(status));
    return false;
}

} // namespace

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

    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > static_cast<size_t>(INT_MAX) ||
        ny > static_cast<size_t>(INT_MAX) ||
        nz > static_cast<size_t>(INT_MAX) || iterations < 0) {
        fprintf(stderr, "Grid dimensions must be positive and iterations must be non-negative.\n");
        return 1;
    }
    if (ny > std::numeric_limits<size_t>::max() / nx ||
        nz > std::numeric_limits<size_t>::max() / (nx * ny) ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Requested grid is too large.\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);
    
    // Both working grids remain device-resident throughout all iterations.
    printf("Initializing grid...\n");
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    if (!cudaSucceeded(cudaMalloc(&deviceGrid1, gridBytes), "first grid allocation") ||
        !cudaSucceeded(cudaMalloc(&deviceGrid2, gridBytes), "second grid allocation")) {
        cudaFree(deviceGrid1);
        cudaFree(deviceGrid2);
        return 1;
    }

    constexpr unsigned int initializationThreads = 256;
    const unsigned int initializationBlocks = static_cast<unsigned int>(
        std::min<size_t>((gridSize + initializationThreads - 1) / initializationThreads,
                         4096));
    initializeKernel<<<initializationBlocks, initializationThreads>>>(
        deviceGrid1, deviceGrid2, gridSize);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess) {
        cudaStatus = cudaDeviceSynchronize();
    }
    if (!cudaSucceeded(cudaStatus, "grid initialization")) {
        cudaFree(deviceGrid1);
        cudaFree(deviceGrid2);
        return 1;
    }

    // Every boundary remains equal to its initialized value.  Initializing
    // both buffers once avoids a boundary-copy kernel on every iteration.
    Real* input = deviceGrid1;
    Real* output = deviceGrid2;
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 blocks(hasInterior ? static_cast<unsigned int>((nx - 2 + kBlockX - 1) / kBlockX) : 1U,
                      hasInterior ? static_cast<unsigned int>((ny - 2 + kBlockY - 1) / kBlockY) : 1U,
                      hasInterior ? static_cast<unsigned int>((nz - 2 + kBlockZ - 1) / kBlockZ) : 1U);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (hasInterior) {
            stencilKernel<<<blocks, block>>>(input, output,
                                             static_cast<int>(nx),
                                             static_cast<int>(ny),
                                             static_cast<int>(nz));
        }
        std::swap(input, output);
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus == cudaSuccess) {
        cudaStatus = cudaDeviceSynchronize();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    if (!cudaSucceeded(cudaStatus, "stencil execution")) {
        cudaFree(deviceGrid1);
        cudaFree(deviceGrid2);
        return 1;
    }
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double interiorX = nx > 2 ? static_cast<double>(nx - 2) : 0.0;
    const double interiorY = ny > 2 ? static_cast<double>(ny - 2) : 0.0;
    const double interiorZ = nz > 2 ? static_cast<double>(nz - 2) : 0.0;
    const double cellUpdates = interiorX * interiorY * interiorZ * iterations;
    const double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        if (!cudaSucceeded(cudaMemcpy(finalGrid.data(), input, gridBytes,
                                      cudaMemcpyDeviceToHost), "final device-to-host copy")) {
            cudaFree(deviceGrid1);
            cudaFree(deviceGrid2);
            return 1;
        }
    }
    cudaFree(deviceGrid1);
    cudaFree(deviceGrid2);
    
    // Print results for external validation
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
