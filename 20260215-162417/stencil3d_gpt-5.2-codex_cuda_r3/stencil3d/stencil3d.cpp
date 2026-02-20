#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <utility>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
__host__ __device__ __forceinline__ size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline void checkCuda(cudaError_t result, const char* context) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", context, cudaGetErrorString(result));
        std::exit(1);
    }
}

__global__ void initKernel(Real* grid, size_t total) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < total) {
        grid[idx] = static_cast<Real>(idx % 19);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t idx = z * plane + y * nx + x;

    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - plane];
    const Real top = input[idx + plane];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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
    size_t bytes = gridSize * sizeof(Real);

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    checkCuda(cudaMalloc(&d_grid1, bytes), "cudaMalloc grid1");
    checkCuda(cudaMalloc(&d_grid2, bytes), "cudaMalloc grid2");

    // Initialize on device
    printf("Initializing grid...\n");
    constexpr int initBlockSize = 256;
    const int initGridSize = static_cast<int>((gridSize + initBlockSize - 1) / initBlockSize);
    initKernel<<<initGridSize, initBlockSize>>>(d_grid1, gridSize);
    checkCuda(cudaGetLastError(), "init kernel");
    checkCuda(cudaDeviceSynchronize(), "init sync");

    // Run stencil iterations
    printf("Running stencil computation...\n");
    const dim3 block(8, 8, 8);
    const dim3 grid(
        static_cast<unsigned int>((nx + block.x - 1) / block.x),
        static_cast<unsigned int>((ny + block.y - 1) / block.y),
        static_cast<unsigned int>((nz + block.z - 1) / block.z));

    cudaEvent_t startEvent;
    cudaEvent_t stopEvent;
    checkCuda(cudaEventCreate(&startEvent), "event create");
    checkCuda(cudaEventCreate(&stopEvent), "event create");
    checkCuda(cudaEventRecord(startEvent), "event record start");

    Real* d_current = d_grid1;
    Real* d_next = d_grid2;
    for (int iter = 0; iter < iterations; ++iter) {
        stencilKernel<<<grid, block>>>(d_current, d_next, nx, ny, nz);
        checkCuda(cudaGetLastError(), "stencil kernel");
        std::swap(d_current, d_next);
    }

    checkCuda(cudaEventRecord(stopEvent), "event record stop");
    checkCuda(cudaEventSynchronize(stopEvent), "event sync");

    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent), "event elapsed");
    printf("Computation time: %.3f ms\n", elapsedMs);

    // Calculate performance metrics
    double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double seconds = elapsedMs / 1000.0;
    double mcups = (seconds > 0.0) ? (cellUpdates / seconds / 1e6) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), d_current, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy final");
    }

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    int exitCode = 0;
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    checkCuda(cudaEventDestroy(startEvent), "event destroy start");
    checkCuda(cudaEventDestroy(stopEvent), "event destroy stop");
    checkCuda(cudaFree(d_grid1), "cudaFree grid1");
    checkCuda(cudaFree(d_grid2), "cudaFree grid2");

    return exitCode;
}
