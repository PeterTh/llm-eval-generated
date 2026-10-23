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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

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

// Thread-block shape for the stencil sweep. 512 threads per block gives the
// best occupancy / cache behaviour on current GPUs for this access pattern.
constexpr int kBlockX = 16;
constexpr int kBlockY = 8;
constexpr int kBlockZ = 4;

// 7-point stencil over the interior of the grid.
// Boundary values are already resident in the output buffer (they are copied
// once before the iteration loop) which removes the need to touch them.
// Neighbour accesses along x are fully coalesced; the y/z strides are served
// from the read-only/L2 caches.
__global__ void __launch_bounds__(kBlockX * kBlockY * kBlockZ)
stencilKernel(const Real* __restrict__ input,
              Real* __restrict__ output,
              const int nx, const int ny, const int nz) {
    const int x = static_cast<int>(blockIdx.x) * kBlockX +
                  static_cast<int>(threadIdx.x);
    const int y = static_cast<int>(blockIdx.y) * kBlockY +
                  static_cast<int>(threadIdx.y);
    const int z = static_cast<int>(blockIdx.z) * kBlockZ +
                  static_cast<int>(threadIdx.z);

    if (x < 1 || y < 1 || z < 1 || x >= nx - 1 || y >= ny - 1 || z >= nz - 1) {
        return;
    }

    const size_t layer = static_cast<size_t>(nx) * ny;
    const size_t idx = static_cast<size_t>(z) * layer + static_cast<size_t>(y) * nx + x;

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - nx];
    const Real back   = input[idx + nx];
    const Real bottom = input[idx - layer];
    const Real top    = input[idx + layer];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

void stencilIteration(const Real* input,
                      Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) {
        return;
    }

    const int gx = static_cast<int>(nx);
    const int gy = static_cast<int>(ny);
    const int gz = static_cast<int>(nz);

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid((gx + kBlockX - 1) / kBlockX,
                    (gy + kBlockY - 1) / kBlockY,
                    (gz + kBlockZ - 1) / kBlockZ);

    stencilKernel<<<grid, block>>>(input, output,
                                   static_cast<int>(nx),
                                   static_cast<int>(ny),
                                   static_cast<int>(nz));
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
    size_t gridBytes = gridSize * sizeof(Real);

    // Host staging buffer for initialization and final output.
    std::vector<Real> hostGrid(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(hostGrid, nx, ny, nz);

    // Device grids (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes));
    CUDA_CHECK(cudaMemcpy(d_grid1, hostGrid.data(), gridBytes, cudaMemcpyHostToDevice));
    // Seed the second buffer so that its boundary ring holds the initial
    // values. The stencil kernel only rewrites interior points, and interior
    // results never feed back into the boundary, so the ring stays constant.
    CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridBytes, cudaMemcpyDeviceToDevice));

    // Run stencil iterations
    printf("Running stencil computation...\n");
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

    // Fetch the final grid from the device.
    const Real* finalDevice = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), finalDevice, gridBytes, cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(hostGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(hostGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            CUDA_CHECK(cudaFree(d_grid1));
            CUDA_CHECK(cudaFree(d_grid2));
            return 0;
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d_grid1));
            CUDA_CHECK(cudaFree(d_grid2));
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
    return 0;
}
