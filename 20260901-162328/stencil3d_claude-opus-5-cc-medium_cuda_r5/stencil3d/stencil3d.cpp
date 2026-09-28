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

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err__), __FILE__,          \
                   __LINE__, cudaGetErrorString(err__));                                       \
            exit(1);                                                                           \
        }                                                                                      \
    } while (0)

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
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

// Thread block tile covering the x/y plane; each thread streams along z.
static constexpr int BLOCK_X = 32;
static constexpr int BLOCK_Y = 2;

// 7-point stencil over the interior points. Boundary points are untouched: they are
// identical in both buffers (see setup in main), which matches the reference code where
// every iteration copies the boundary values through unchanged.
__global__ __launch_bounds__(BLOCK_X* BLOCK_Y) void stencilKernel(const Real* __restrict__ input,
                                                                  Real* __restrict__ output,
                                                                  const int nx, const int ny,
                                                                  const int nz) {
    __shared__ Real tile[BLOCK_Y + 2][BLOCK_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = 1 + blockIdx.x * BLOCK_X + tx;
    const int y = 1 + blockIdx.y * BLOCK_Y + ty;

    const bool active = (x < nx - 1) && (y < ny - 1);

    // Clamp out-of-range threads onto the boundary: their loads stay in bounds and the
    // first inactive thread in each direction still supplies the correct halo value for
    // its active neighbour inside the tile.
    const int cx = min(x, nx - 1);
    const int cy = min(y, ny - 1);

    const size_t slice = (size_t)nx * (size_t)ny;
    const Real* col = input + (size_t)cy * nx + cx;  // column base at z == 0

    Real below = col[0];
    Real center = col[slice];

    for (int z = 1; z < nz - 1; ++z) {
        const Real above = col[(size_t)(z + 1) * slice];

        // Stage the current z-plane in shared memory, including a one-element halo.
        __syncthreads();
        const Real* plane = input + (size_t)z * slice;
        tile[ty + 1][tx + 1] = center;
        if (tx == 0) {
            tile[ty + 1][0] = plane[(size_t)cy * nx + (cx - 1)];
        }
        if (tx == BLOCK_X - 1) {
            tile[ty + 1][BLOCK_X + 1] = plane[(size_t)cy * nx + min(cx + 1, nx - 1)];
        }
        if (ty == 0) {
            tile[0][tx + 1] = plane[(size_t)(cy - 1) * nx + cx];
        }
        if (ty == BLOCK_Y - 1) {
            tile[BLOCK_Y + 1][tx + 1] = plane[(size_t)min(cy + 1, ny - 1) * nx + cx];
        }
        __syncthreads();

        if (active) {
            const Real left = tile[ty + 1][tx];
            const Real right = tile[ty + 1][tx + 2];
            const Real front = tile[ty][tx + 1];
            const Real back = tile[ty + 2][tx + 1];

            // Simple averaging stencil
            output[(size_t)z * slice + (size_t)y * nx + x] =
                (center + left + right + front + back + below + above) / 7.0;
        }

        below = center;
        center = above;
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

    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    // Device buffers; grid2 starts as a copy of grid1 so that the boundary values (which
    // the stencil never modifies) are already in place in both buffers.
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_grid1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridSize * sizeof(Real), cudaMemcpyDeviceToDevice));

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(nx > 2 ? (unsigned)(((nx - 2) + BLOCK_X - 1) / BLOCK_X) : 0u,
                    ny > 2 ? (unsigned)(((ny - 2) + BLOCK_Y - 1) / BLOCK_Y) : 0u);
    const bool hasInterior = (nx > 2 && ny > 2 && nz > 2);

    // Warm up the context so the timed region measures the computation itself.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        const Real* in = (iter % 2 == 0) ? d_grid1 : d_grid2;
        Real* out = (iter % 2 == 0) ? d_grid2 : d_grid1;
        if (hasInterior) {
            stencilKernel<<<grid, block>>>(in, out, (int)nx, (int)ny, (int)nz);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy the result back to the host
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

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
