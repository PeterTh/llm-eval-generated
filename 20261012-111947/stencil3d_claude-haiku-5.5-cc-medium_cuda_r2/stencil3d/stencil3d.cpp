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

// Thread block tile: x is the fastest-varying dimension for coalesced access.
// BLOCK_Z is the number of z-planes each block sweeps.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 16;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err_ = (call);                                          \
        if (err_ != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err_));                              \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ inline int clampIndex(const int v, const int hi) noexcept {
    return min(max(v, 0), hi);
}

__global__ void initializeGridKernel(Real* grid, const size_t gridSize) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < gridSize) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

void initializeGrid(Real* grid, const size_t gridSize) {
    constexpr unsigned int threads = 256;
    const size_t blocks = (gridSize + threads - 1) / threads;
    if (blocks == 0) return;
    initializeGridKernel<<<static_cast<unsigned int>(blocks), threads>>>(grid, gridSize);
    CUDA_CHECK(cudaGetLastError());
}

// 7-point stencil. Each block covers a BLOCK_X x BLOCK_Y column of (x, y) and sweeps
// BLOCK_Z planes in z. The x/y neighbours of the current plane are staged in shared
// memory, and the z neighbours are carried in registers, so each input cell is read
// from global memory roughly once per iteration.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz) {
    __shared__ Real tile[BLOCK_Y + 2][BLOCK_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int x = blockIdx.x * BLOCK_X + tx;
    const int y = blockIdx.y * BLOCK_Y + ty;
    const int zBegin = blockIdx.z * BLOCK_Z;
    const int zEnd = min(zBegin + BLOCK_Z, nz);

    // Clamped loads keep halo fetches in bounds for edge threads. Clamped values only
    // feed boundary cells, which are copied rather than computed.
    auto load = [&](const int xx, const int yy, const int zz) {
        return input[idx3(clampIndex(xx, nx - 1), clampIndex(yy, ny - 1), clampIndex(zz, nz - 1), nx, ny)];
    };

    Real below = load(x, y, zBegin - 1);
    Real center = load(x, y, zBegin);
    for (int z = zBegin; z < zEnd; ++z) {
        const Real above = load(x, y, z + 1);

        tile[ty + 1][tx + 1] = center;
        if (tx == 0) tile[ty + 1][0] = load(x - 1, y, z);
        if (tx == BLOCK_X - 1) tile[ty + 1][BLOCK_X + 1] = load(x + 1, y, z);
        if (ty == 0) tile[0][tx + 1] = load(x, y - 1, z);
        if (ty == BLOCK_Y - 1) tile[BLOCK_Y + 1][tx + 1] = load(x, y + 1, z);
        __syncthreads();

        if (x < nx && y < ny) {
            const size_t idx = idx3(x, y, z, nx, ny);
            if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
                // Boundary values are copied unchanged
                output[idx] = center;
            } else {
                // Same summation order as the sequential reference
                output[idx] = (center + tile[ty + 1][tx] + tile[ty + 1][tx + 2] + tile[ty][tx + 1]
                               + tile[ty + 2][tx + 1] + below + above) / 7.0;
            }
        }
        __syncthreads();

        below = center;
        center = above;
    }
}

// 7-point stencil computation
void stencilIteration(const Real* input, Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) return;

    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((nz + BLOCK_Z - 1) / BLOCK_Z));
    stencilKernel<<<grid, block>>>(input, output, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
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

    // Allocate device grids (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));

    // Initialize (second buffer zeroed, matching the host vector semantics)
    printf("Initializing grid...\n");
    initializeGrid(d_grid1, gridSize);
    CUDA_CHECK(cudaMemset(d_grid2, 0, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaDeviceSynchronize());

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

    // Copy the final grid back to the host for reporting and validation
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
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
