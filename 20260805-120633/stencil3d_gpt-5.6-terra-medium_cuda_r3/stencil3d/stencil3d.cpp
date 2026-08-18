#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;

    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t idx = z * plane + y * nx + x;
    output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                   input[idx - nx] + input[idx + nx] +
                   input[idx - plane] + input[idx + plane]) / 7.0;
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

    if (nx < 3 || ny < 3 || nz < 3 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        fprintf(stderr, "Grid dimensions must each be at least 3 and their product must fit in size_t; iterations must be non-negative.\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate and initialize the host grid.  Both device buffers receive this
    // data, so their boundary cells remain unchanged without a per-iteration
    // boundary-copy kernel.
    std::vector<Real> grid1(gridSize);
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    checkCuda(cudaMalloc(&deviceGrid1, gridSize * sizeof(Real)), "allocating first grid");
    checkCuda(cudaMalloc(&deviceGrid2, gridSize * sizeof(Real)), "allocating second grid");
    checkCuda(cudaMemcpy(deviceGrid1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice), "uploading first grid");
    checkCuda(cudaMemcpy(deviceGrid2, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice), "uploading second grid");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    cudaEvent_t start{};
    cudaEvent_t end{};
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    checkCuda(cudaEventRecord(start), "recording start event");

    constexpr dim3 blockSize(32, 4, 2);
    const dim3 gridSize3D(
        static_cast<unsigned int>((nx - 2 + blockSize.x - 1) / blockSize.x),
        static_cast<unsigned int>((ny - 2 + blockSize.y - 1) / blockSize.y),
        static_cast<unsigned int>((nz - 2 + blockSize.z - 1) / blockSize.z));
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<gridSize3D, blockSize>>>(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilKernel<<<gridSize3D, blockSize>>>(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
        checkCuda(cudaGetLastError(), "launching stencil kernel");
    }
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "synchronizing stencil computation");
    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, end), "measuring stencil computation");
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = elapsedMilliseconds > 0.0F ? cellUpdates / elapsedMilliseconds / 1e3 : 0.0;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults || validate) {
        checkCuda(cudaMemcpy(grid1.data(), (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2,
                             gridSize * sizeof(Real), cudaMemcpyDeviceToHost), "downloading final grid");
    }
    checkCuda(cudaFree(deviceGrid1), "freeing first grid");
    checkCuda(cudaFree(deviceGrid2), "freeing second grid");

    if (printResults) {
        print_results(grid1, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(grid1, nx, ny, nz);
        
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
