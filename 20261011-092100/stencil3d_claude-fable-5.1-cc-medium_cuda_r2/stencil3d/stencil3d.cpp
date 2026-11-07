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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                  const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Kernel configuration
// ---------------------------------------------------------------------------
constexpr int BLOCK_X = 32;   // threads along x (one warp -> coalesced rows)
constexpr int BLOCK_Y = 8;    // threads along y
constexpr int TARGET_BLOCKS = 2048;  // desired number of blocks for good occupancy

// Grid initialisation on the device: grid[idx] = (idx % 19) * 1.0
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Correctly rounded x / 7.0 without a full IEEE division sequence.
//
// With zh = RN(1/7):  q = RN(x*zh) is within 1.5 ulp of x/7, so the
// remainder r = x - 7*q is exactly representable and the FMA computes it
// exactly.  q + r*zh then differs from x/7 by at most ~2^-53 ulp, while
// x/7 (x dyadic, 7 odd) is never closer than 1/14 ulp to a rounding
// midpoint; hence the final FMA rounds to exactly RN(x/7), bit-identical
// to the reference's x / 7.0, at a fraction of the FP64 instruction cost.
__device__ __forceinline__ Real div7(const Real x) {
    constexpr Real zh = 1.0 / 7.0;  // RN(1/7)
    const Real q = __dmul_rn(x, zh);
    const Real r = __fma_rn(-q, 7.0, x);
    return __fma_rn(r, zh, q);
}

// 7-point stencil on the interior points.
//
// Each thread owns one (x, y) column and marches along z over a chunk of
// planes.  The values at z-1, z and z+1 of its own column are kept in
// registers so every input element is fetched from global memory only once
// per column for the vertical neighbours; the four horizontal neighbours are
// served from L1 since the neighbouring threads of the same block load them.
//
// Boundary cells are never written: they are copied once before the time
// loop (see main) which is equivalent to copying them every iteration since
// the boundary of the input never changes.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz, const int zChunk) {
    const int x = 1 + blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = 1 + blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) return;

    const int z0 = 1 + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, nz - 1);
    if (z0 >= z1) return;

    const size_t plane = (size_t)nx * ny;
    size_t idx = idx3(x, y, z0, nx, ny);

    Real bottom = input[idx - plane];
    Real center = input[idx];

    for (int z = z0; z < z1; ++z, idx += plane) {
        const Real top = input[idx + plane];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];

        // Simple averaging stencil (same evaluation order as the reference)
        output[idx] = div7(center + left + right + front + back + bottom + top);

        bottom = center;
        center = top;
    }
}

// Launch one stencil iteration on the device
void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) return;  // no interior points

    const int inx = (int)(nx - 2);
    const int iny = (int)(ny - 2);
    const int inz = (int)(nz - 2);

    const int gx = (inx + BLOCK_X - 1) / BLOCK_X;
    const int gy = (iny + BLOCK_Y - 1) / BLOCK_Y;

    // Split z so that the launch has enough blocks to fill the GPU
    int zChunks = (TARGET_BLOCKS + gx * gy - 1) / (gx * gy);
    zChunks = std::max(1, std::min(zChunks, inz));
    const int zChunk = (inz + zChunks - 1) / zChunks;
    const int gz = (inz + zChunk - 1) / zChunk;

    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 grid(gx, gy, gz);
    stencilKernel<<<grid, block>>>(d_input, d_output, (int)nx, (int)ny, (int)nz, zChunk);
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
    const size_t gridBytes = gridSize * sizeof(Real);

    // Allocate grids on the device (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, std::max(gridBytes, (size_t)1)));
    CUDA_CHECK(cudaMalloc(&d_grid2, std::max(gridBytes, (size_t)1)));

    // Initialize
    printf("Initializing grid...\n");
    if (gridSize > 0) {
        const int initBlock = 256;
        const size_t initGridWanted = (gridSize + initBlock - 1) / initBlock;
        const int initGrid = (int)std::min(initGridWanted, (size_t)65536);
        initializeGridKernel<<<initGrid, initBlock>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
        // The stencil only writes interior points; boundary values are
        // copied from the input, so seed the second buffer with the full
        // initial grid once (the boundary never changes afterwards).
        CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridBytes, cudaMemcpyDeviceToDevice));
    }
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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy the final grid back to the host
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridBytes, cudaMemcpyDeviceToHost));
    }
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
