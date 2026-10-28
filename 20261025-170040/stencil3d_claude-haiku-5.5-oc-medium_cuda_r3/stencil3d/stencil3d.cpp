#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// RAII wrapper for device memory
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), count * sizeof(Real)));
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    Real* get() const { return ptr_; }

  private:
    Real* ptr_ = nullptr;
};

// Block shape used for all kernels: 32 threads along x (coalesced access).
static constexpr dim3 kBlock(32, 4, 4);

// Grid shape covering nx x ny x nz; y/z are capped and handled with grid-stride loops.
static dim3 launchGrid(const size_t nx, const size_t ny, const size_t nz) {
    const size_t gx = (nx + kBlock.x - 1) / kBlock.x;
    const size_t gy = std::min<size_t>((ny + kBlock.y - 1) / kBlock.y, 65535);
    const size_t gz = std::min<size_t>((nz + kBlock.z - 1) / kBlock.z, 65535);
    return dim3(static_cast<unsigned int>(gx), static_cast<unsigned int>(gy), static_cast<unsigned int>(gz));
}

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t yStride = static_cast<size_t>(gridDim.y) * blockDim.y;
    const size_t zStride = static_cast<size_t>(gridDim.z) * blockDim.z;
    for (size_t z = blockIdx.z * blockDim.z + threadIdx.z; z < nz; z += zStride) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += yStride) {
            const size_t idx = idx3(x, y, z, nx, ny);
            grid[idx] = (idx % 19) * 1.0;
        }
    }
}

// 7-point stencil with copy-through on the domain boundary
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t yStride = static_cast<size_t>(gridDim.y) * blockDim.y;
    const size_t zStride = static_cast<size_t>(gridDim.z) * blockDim.z;
    for (size_t z = blockIdx.z * blockDim.z + threadIdx.z; z < nz; z += zStride) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += yStride) {
            const size_t idx = idx3(x, y, z, nx, ny);
            if (x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1) {
                const Real center = input[idx];
                const Real left = input[idx3(x-1, y, z, nx, ny)];
                const Real right = input[idx3(x+1, y, z, nx, ny)];
                const Real front = input[idx3(x, y-1, z, nx, ny)];
                const Real back = input[idx3(x, y+1, z, nx, ny)];
                const Real bottom = input[idx3(x, y, z-1, nx, ny)];
                const Real top = input[idx3(x, y, z+1, nx, ny)];

                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            } else {
                output[idx] = input[idx];
            }
        }
    }
}

void initializeGrid(Real* d_grid, const size_t nx, const size_t ny, const size_t nz) {
    initializeGridKernel<<<launchGrid(nx, ny, nz), kBlock>>>(d_grid, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
}

void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    stencilKernel<<<launchGrid(nx, ny, nz), kBlock>>>(d_input, d_output, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
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
    
    // Allocate grids on the device (double buffering)
    DeviceBuffer d_grid1(gridSize);
    DeviceBuffer d_grid2(gridSize);
    CUDA_CHECK(cudaMemset(d_grid2.get(), 0, gridSize * sizeof(Real)));
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(d_grid1.get(), nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1.get(), d_grid2.get(), nx, ny, nz);
        } else {
            stencilIteration(d_grid2.get(), d_grid1.get(), nx, ny, nz);
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
    
    // Copy final result back to the host only when it is needed
    const Real* d_final = (iterations % 2 == 0) ? d_grid1.get() : d_grid2.get();
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }

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
