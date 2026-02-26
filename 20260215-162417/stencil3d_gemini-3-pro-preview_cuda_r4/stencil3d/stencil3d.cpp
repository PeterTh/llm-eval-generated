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

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__); \
        fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// CUDA Kernel
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output, 
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    const size_t idx = idx3(x, y, z, nx, ny);

    if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1 && z >= 1 && z < nz - 1) {
        const Real center = input[idx];
        const Real left = input[idx3(x-1, y, z, nx, ny)];
        const Real right = input[idx3(x+1, y, z, nx, ny)];
        const Real front = input[idx3(x, y-1, z, nx, ny)];
        const Real back = input[idx3(x, y+1, z, nx, ny)];
        const Real bottom = input[idx3(x, y, z-1, nx, ny)];
        const Real top = input[idx3(x, y, z+1, nx, ny)];
        
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    } else if (x < nx && y < ny && z < nz) {
        output[idx] = input[idx];
    }
}

void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    dim3 block(8, 8, 8);
    dim3 grid((nx + block.x - 1) / block.x, 
              (ny + block.y - 1) / block.y, 
              (nz + block.z - 1) / block.z);
    
    stencilKernel<<<grid, block>>>(d_input, d_output, nx, ny, nz);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
}

bool validateResult(const Real* grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz, size_t gridSize) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (size_t i = 0; i < gridSize; ++i) {
        const auto val = grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value at index %zu\n", i);
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (size_t i = 0; i < gridSize; ++i) {
        const auto val = grid[i];
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
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
    // Host memory
    std::vector<Real> h_grid1(gridSize);
    std::vector<Real> h_grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(h_grid1.data(), nx, ny, nz);
    
    // Device memory
    Real *d_grid1, *d_grid2;
    CHECK_CUDA(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CHECK_CUDA(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    
    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_grid1, h_grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    // Initialize d_grid2 with same values just in case, though it will be overwritten
    CHECK_CUDA(cudaMemcpy(d_grid2, h_grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));

    // Run stencil iterations
    printf("Running stencil computation...\n");
    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy back results
    // If iterations is even, result is in d_grid1 (last write was iter % 2 != 0 -> write to grid1)
    // If iterations is odd, result is in d_grid2 (last write was iter % 2 == 0 -> write to grid2)
    const Real* final_d_ptr = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real>& final_h_grid = (iterations % 2 == 0) ? h_grid1 : h_grid2;
    
    CHECK_CUDA(cudaMemcpy(final_h_grid.data(), final_d_ptr, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    
    // Print results for external validation
    if (printResults) {
        print_results(final_h_grid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(final_h_grid.data(), nx, ny, nz, gridSize);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            CHECK_CUDA(cudaFree(d_grid1));
            CHECK_CUDA(cudaFree(d_grid2));
            return 1;
        }
    }
    
    CHECK_CUDA(cudaFree(d_grid1));
    CHECK_CUDA(cudaFree(d_grid2));
    
    return 0;
}
