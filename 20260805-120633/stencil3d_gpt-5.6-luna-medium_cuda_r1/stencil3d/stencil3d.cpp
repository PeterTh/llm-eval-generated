#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        cudaFailure(error, operation);
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

// One thread owns one output cell.  The x dimension is contiguous, which
// gives coalesced accesses for the center and all x-neighbors.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t firstZ = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    const size_t zStride = static_cast<size_t>(blockDim.z) * gridDim.z;
    const size_t plane = nx * ny;
    if (x >= nx || y >= ny) return;

    // The z grid is capped by the device grid-dimension limit and traversed
    // in strides, so very deep domains remain supported as well.
    for (size_t z = firstZ; z < nz; z += zStride) {
        const size_t index = z * plane + y * nx + x;
        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
            output[index] = input[index];
        } else {
            output[index] = (input[index] + input[index - 1] + input[index + 1]
                + input[index - nx] + input[index + nx]
                + input[index - plane] + input[index + plane]) / 7.0;
        }
    }
}

__global__ void initializeKernel(Real* __restrict__ grid, const size_t elementCount) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < elementCount; index += stride) {
        grid[index] = static_cast<Real>(index % 19);
    }
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Keep the double buffers resident on the GPU for the complete stencil.
    std::vector<Real> finalGrid(gridSize);
    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&grid2, gridSize * sizeof(Real)));
    
    // Initialize
    printf("Initializing grid...\n");
    constexpr unsigned int threadsPerBlock = 256;
    const unsigned int initBlocks = static_cast<unsigned int>(std::min<size_t>(
        (gridSize + threadsPerBlock - 1) / threadsPerBlock, 4096));
    initializeKernel<<<initBlocks, threadsPerBlock>>>(grid1, gridSize);
    CUDA_CHECK(cudaGetLastError());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    const dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>(std::min<size_t>(
                        (nz + block.z - 1) / block.z, 65535)));
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<grid, block>>>(grid1, grid2, nx, ny, nz);
        } else {
            stencilKernel<<<grid, block>>>(grid2, grid1, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), (iterations % 2 == 0) ? grid1 : grid2,
                          gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));
    const long durationMilliseconds = static_cast<long>(elapsedMilliseconds);
    
    printf("Computation time: %ld ms\n", durationMilliseconds);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (elapsedMilliseconds / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
