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

// Thread block shape: x is the fastest-varying index, so a 32-wide x block
// gives coalesced access across each warp
constexpr unsigned BLOCK_X = 32;
constexpr unsigned BLOCK_Y = 4;
constexpr unsigned BLOCK_Z = 4;
// CUDA limits gridDim.y and gridDim.z to 65535; larger extents are covered by grid-stride loops
constexpr unsigned MAX_GRID_YZ = 65535;

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

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    for (size_t z = blockIdx.z * blockDim.z + threadIdx.z; z < nz; z += gridDim.z * blockDim.z) {
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
            const size_t idx = idx3(x, y, z, nx, ny);
            grid[idx] = (idx % 19) * 1.0;
        }
    }
}

// 7-point stencil computation, one thread per cell.
// Interior points are averaged; boundary points are copied from the input.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const bool xBoundary = (x == 0 || x == nx - 1);
    for (size_t z = blockIdx.z * blockDim.z + threadIdx.z; z < nz; z += gridDim.z * blockDim.z) {
        const bool zBoundary = (z == 0 || z == nz - 1);
        for (size_t y = blockIdx.y * blockDim.y + threadIdx.y; y < ny; y += gridDim.y * blockDim.y) {
            const size_t idx = idx3(x, y, z, nx, ny);

            if (xBoundary || zBoundary || y == 0 || y == ny - 1) {
                output[idx] = input[idx];
                continue;
            }

            const Real center = input[idx];
            const Real left = input[idx - 1];
            const Real right = input[idx + 1];
            const Real front = input[idx - nx];
            const Real back = input[idx + nx];
            const Real bottom = input[idx - nx * ny];
            const Real top = input[idx + nx * ny];

            // Simple averaging stencil
            output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
    }
}

// Grid covering the x extent with one thread per column; y and z are capped and looped over in-kernel
static dim3 launchGrid(const size_t nx, const size_t ny, const size_t nz) {
    const size_t gx = (nx + BLOCK_X - 1) / BLOCK_X;
    const size_t gy = std::min<size_t>((ny + BLOCK_Y - 1) / BLOCK_Y, MAX_GRID_YZ);
    const size_t gz = std::min<size_t>((nz + BLOCK_Z - 1) / BLOCK_Z, MAX_GRID_YZ);
    return dim3(static_cast<unsigned>(gx), static_cast<unsigned>(gy), static_cast<unsigned>(gz));
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
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    if (gridSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
        CUDA_CHECK(cudaMemset(d_grid2, 0, gridSize * sizeof(Real)));
    }

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid = launchGrid(nx, ny, nz);

    // Initialize
    printf("Initializing grid...\n");
    if (gridSize > 0) {
        initializeGridKernel<<<grid, block>>>(d_grid1, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        const Real* src = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* dst = (iter % 2 == 0) ? d_grid2 : d_grid1;
        if (gridSize > 0) {
            stencilKernel<<<grid, block>>>(src, dst, nx, ny, nz);
            CUDA_CHECK(cudaGetLastError());
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

    // Copy the final grid back to the host only when it is needed
    const bool needHost = printResults || validate;
    std::vector<Real> finalGrid;
    if (needHost && gridSize > 0) {
        Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
        finalGrid.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    // Print results for external validation
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    int status = 0;

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            status = 1;
        }
    }

    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    return status;
}
