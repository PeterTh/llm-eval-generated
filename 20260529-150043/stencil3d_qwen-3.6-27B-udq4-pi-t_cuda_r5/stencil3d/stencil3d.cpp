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

// ============================================================================
// CUDA Kernels
// ============================================================================

__device__ __forceinline__ size_t idx3d(const size_t x, const size_t y, const size_t z,
                                         const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize grid on device: grid[idx] = (idx % 19) * 1.0
__global__ void initGridKernel(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        const size_t idx = idx3d(x, y, z, nx, ny);
        grid[idx] = static_cast<Real>(static_cast<int>(idx % 19)) * 1.0;
    }
}

// 7-point stencil kernel with shared memory tiling for maximum performance.
//
// Each thread block loads a (blockDim+2)^3 tile into shared memory,
// including 1-element halo on all sides for stencil neighbor access.
// Threads cooperatively load shared memory using a strided pattern.
//
// Boundary points simply copy their input value to output.
// Interior points compute: (center + left + right + front + back + bottom + top) / 7.0
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    // Shared memory tile: (blockDim.x+2) x (blockDim.y+2) x (blockDim.z+2)
    extern __shared__ Real s_tile[];

    const unsigned int tx = threadIdx.x;
    const unsigned int ty = threadIdx.y;
    const unsigned int tz = threadIdx.z;

    const unsigned int sx = blockDim.x + 2;
    const unsigned int sy = blockDim.y + 2;
    const unsigned int sz_dim = blockDim.z + 2;

    // Cooperatively load shared memory tile with halo using strided access
    const unsigned int threadId = tz * blockDim.x * blockDim.y + ty * blockDim.x + tx;
    const unsigned int tileSize = sx * sy * sz_dim;
    const unsigned int stride = blockDim.x * blockDim.y * blockDim.z;

    for (unsigned int i = threadId; i < tileSize; i += stride) {
        const unsigned int szi = i / (sx * sy);
        const unsigned int ssi = (i % (sx * sy)) / sx;
        const unsigned int ssi_x = i % sx;

        const size_t gx = static_cast<size_t>(blockIdx.x) * blockDim.x + ssi_x - 1;
        const size_t gy = static_cast<size_t>(blockIdx.y) * blockDim.y + ssi - 1;
        const size_t gz = static_cast<size_t>(blockIdx.z) * blockDim.z + szi - 1;

        Real val = 0.0;
        if (gx < nx && gy < ny && gz < nz) {
            val = __ldg(&input[idx3d(gx, gy, gz, nx, ny)]);
        }
        s_tile[i] = val;
    }

    __syncthreads();

    const size_t gx = static_cast<size_t>(blockIdx.x) * blockDim.x + tx;
    const size_t gy = static_cast<size_t>(blockIdx.y) * blockDim.y + ty;
    const size_t gz = static_cast<size_t>(blockIdx.z) * blockDim.z + tz;

    if (gx < nx && gy < ny && gz < nz) {
        const size_t idx = idx3d(gx, gy, gz, nx, ny);

        if (gx == 0 || gx == nx - 1 || gy == 0 || gy == ny - 1 || gz == 0 || gz == nz - 1) {
            output[idx] = input[idx];
        } else {
            const unsigned int stx = tx + 1;
            const unsigned int sty = ty + 1;
            const unsigned int stz = tz + 1;

            const Real center = s_tile[stz * sx * sy + sty * sx + stx];
            const Real left   = s_tile[stz * sx * sy + sty * sx + (stx - 1)];
            const Real right  = s_tile[stz * sx * sy + sty * sx + (stx + 1)];
            const Real front  = s_tile[stz * sx * sy + (sty - 1) * sx + stx];
            const Real back   = s_tile[stz * sx * sy + (sty + 1) * sx + stx];
            const Real bottom = s_tile[(stz - 1) * sx * sy + sty * sx + stx];
            const Real top    = s_tile[(stz + 1) * sx * sy + sty * sx + stx];

            output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

// ============================================================================
// Host-side helpers
// ============================================================================

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

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

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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

    // Host-side vectors for validation / results printing
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    // Device-side double-buffered grids (pinned memory for fast transfers)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;

    cudaMalloc(&d_grid1, gridSize * sizeof(Real));
    cudaMalloc(&d_grid2, gridSize * sizeof(Real));

    // Thread and block configuration
    const dim3 blockSize(8, 8, 8);
    const dim3 gridSizeInit(
        (nx + blockSize.x - 1) / blockSize.x,
        (ny + blockSize.y - 1) / blockSize.y,
        (nz + blockSize.z - 1) / blockSize.z
    );

    // Shared memory size: (blockDim.x+2) * (blockDim.y+2) * (blockDim.z+2) * sizeof(Real)
    // For 8x8x8 blocks: 10*10*10*8 = 8000 bytes
    const size_t sharedMemBytes = (blockSize.x + 2) * (blockSize.y + 2) * (blockSize.z + 2) * sizeof(Real);

    // Initialize grid on device
    printf("Initializing grid...\n");
    initGridKernel<<<gridSizeInit, blockSize>>>(d_grid1, nx, ny, nz);

    // Run stencil iterations with double buffering on GPU
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<gridSizeInit, blockSize, sharedMemBytes>>>(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilKernel<<<gridSizeInit, blockSize, sharedMemBytes>>>(d_grid2, d_grid1, nx, ny, nz);
        }
    }

    // Synchronize all GPU work before timing
    cudaDeviceSynchronize();

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy final result back to host for validation / results
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    cudaMemcpy(grid1.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);

    // Print results for external validation
    const std::vector<Real>& finalGrid = grid1;
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

    // Cleanup
    cudaFree(d_grid1);
    cudaFree(d_grid2);

    return 0;
}
