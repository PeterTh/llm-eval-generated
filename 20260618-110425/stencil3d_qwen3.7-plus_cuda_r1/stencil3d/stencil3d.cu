#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// 3D index calculation (device)
__device__ __forceinline__ size_t idx3(const size_t x, const size_t y, const size_t z, 
                                        const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA kernel for grid initialization
__global__ void initializeGridKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        grid[idx] = (idx % 19) * 1.0;
    }
}

// CUDA kernel for 7-point stencil computation (all points including boundaries)
__global__ void stencilIterationKernel(const Real* __restrict__ input, 
                                        Real* __restrict__ output,
                                        const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3(x, y, z, nx, ny);
        
        // Boundary check - copy boundary values directly
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            output[idx] = input[idx];
        } else {
            // Interior point - apply 7-point stencil
            const Real center = input[idx];
            const Real left = input[idx3(x-1, y, z, nx, ny)];
            const Real right = input[idx3(x+1, y, z, nx, ny)];
            const Real front = input[idx3(x, y-1, z, nx, ny)];
            const Real back = input[idx3(x, y+1, z, nx, ny)];
            const Real bottom = input[idx3(x, y, z-1, nx, ny)];
            const Real top = input[idx3(x, y, z+1, nx, ny)];
            
            // Simple averaging stencil
            output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

// Host wrapper for stencil iteration
void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    // Configure thread blocks for optimal occupancy on RTX 3090
    dim3 blockDim(16, 4, 4);
    
    // Launch combined stencil kernel
    dim3 gridDim(
        (nx + blockDim.x - 1) / blockDim.x,
        (ny + blockDim.y - 1) / blockDim.y,
        (nz + blockDim.z - 1) / blockDim.z
    );
    stencilIterationKernel<<<gridDim, blockDim>>>(d_input, d_output, nx, ny, nz);
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
    
    printf("3D Stencil Benchmark (CUDA)\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Print GPU info
    int device;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    
    size_t gridSize = nx * ny * nz;
    size_t gridBytes = gridSize * sizeof(Real);
    
    // Allocate device grids (double buffering)
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes));
    
    // Initialize grid on device
    printf("Initializing grid...\n");
    {
        dim3 block(16, 8, 8);
        dim3 grid(
            (nx + block.x - 1) / block.x,
            (ny + block.y - 1) / block.y,
            (nz + block.z - 1) / block.z
        );
        initializeGridKernel<<<grid, block>>>(d_grid1, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    
    // Warm up and synchronize before timing
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy final result back to host for validation/printing
    std::vector<Real> h_finalGrid(gridSize);
    if (iterations % 2 == 0) {
        CUDA_CHECK(cudaMemcpy(h_finalGrid.data(), d_grid1, gridBytes, cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy(h_finalGrid.data(), d_grid2, gridBytes, cudaMemcpyDeviceToHost));
    }
    
    // Print results for external validation
    if (printResults) {
        print_results(h_finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(h_finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaDeviceReset());
    
    return 0;
}
