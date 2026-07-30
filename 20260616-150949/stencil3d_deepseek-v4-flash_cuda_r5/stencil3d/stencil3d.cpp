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

// CUDA error checking helper
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// CUDA kernel: initialize grid
__global__ void initializeKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = z * (nx * ny) + y * nx + x;
    grid[idx] = (idx % 19) * 1.0;
}

// CUDA kernel: 7-point stencil computation with boundary handling
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = z * (nx * ny) + y * nx + x;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[idx] = input[idx];
    } else {
        const Real center = input[idx];
        const Real left   = input[idx - 1];
        const Real right  = input[idx + 1];
        const Real front  = input[idx - nx];
        const Real back   = input[idx + nx];
        const Real bottom = input[idx - nx * ny];
        const Real top    = input[idx + nx * ny];

        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// Host wrapper: launch stencil kernel with 3D block decomposition
void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // 512 threads per block: 32 x 4 x 4 for good coalescing in x-dimension
    const dim3 blockDim(32, 4, 4);
    const dim3 gridDim((nx + blockDim.x - 1) / blockDim.x,
                       (ny + blockDim.y - 1) / blockDim.y,
                       (nz + blockDim.z - 1) / blockDim.z);

    stencilKernel<<<gridDim, blockDim>>>(d_input, d_output, nx, ny, nz);
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
    
    const size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);
    
    // Allocate device memory (double buffering)
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes));
    
    // Initialize grid on device
    {
        printf("Initializing grid on GPU...\n");
        const dim3 initBlockDim(32, 8, 4);  // 1024 threads
        const dim3 initGridDim((nx + initBlockDim.x - 1) / initBlockDim.x,
                               (ny + initBlockDim.y - 1) / initBlockDim.y,
                               (nz + initBlockDim.z - 1) / initBlockDim.z);
        initializeKernel<<<initGridDim, initBlockDim>>>(d_grid1, nx, ny, nz);
        CUDA_CHECK(cudaPeekAtLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Run stencil iterations with CUDA event timing
    printf("Running stencil computation on GPU...\n");
    
    cudaEvent_t startEvent, stopEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    
    CUDA_CHECK(cudaEventRecord(startEvent));
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    
    float gpuMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuMs, startEvent, stopEvent));
    
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    
    printf("Computation time: %.0f ms\n", gpuMs);
    
    // Copy final result from device to host
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridBytes, cudaMemcpyDeviceToHost));
    
    // Calculate performance metrics
    const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    const double mcups = cellUpdates / (gpuMs / 1000.0) / 1e6;
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
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
    // Return validation result if validation was run
    if (validate) {
        return validateResult(finalGrid, nx, ny, nz) ? 0 : 1;
    }
    
    return 0;
}
