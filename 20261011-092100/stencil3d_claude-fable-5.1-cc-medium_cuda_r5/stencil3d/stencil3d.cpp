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
// Kernel configuration: a 2D tile of threads in the XY plane, each thread
// marches along Z keeping bottom/center/top in registers.  The XY plane of the
// current slice is staged in shared memory (with a one-cell halo) so the
// left/right/front/back neighbours are served from on-chip memory.
// ---------------------------------------------------------------------------
constexpr int TILE_X = 32;
constexpr int TILE_Y = 8;
constexpr int Z_CHUNK = 32;

// Correctly rounded division by the constant 7.0 without a hardware divide.
// FP64 division on consumer GPUs is a long multi-instruction sequence and was
// the dominant cost of the stencil.  Using a double-double reciprocal
// zh + zl = 1/7 (|zh + zl - 1/7| < 2^-105):
//   u = RN(s*zl)                     (|u| ~ 2^-56 s, rounding error < 2^-109 s)
//   q = RN(s*zh + u)                 single rounding of  s/7 + err,  |err| < 2^-50 ulp(q)
// For s with 53-bit significand M, s/7 in ulp(q) units equals M*2^j/7 (j >= 2),
// so 2*M*2^j = 7*(2k+1) is impossible (even = odd): s/7 is never a rounding
// midpoint and is always >= 1/14 ulp away from one.  The tiny error therefore
// cannot change the rounding, and q == RN(s/7) bit-for-bit for all normal s.
// (Also verified on the host against true division for >2.6e9 inputs.)
constexpr Real INV7_HI = 0x1.2492492492492p-3;   // RN(1/7)
constexpr Real INV7_LO = 0x1.2492492492492p-57;  // RN(1/7 - INV7_HI)
__device__ __forceinline__ Real div7(const Real s) {
    return fma(s, INV7_HI, s * INV7_LO);
}

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil on interior points.  Boundary cells are never touched by this
// kernel; they are kept identical in both buffers (see main), which is exactly
// equivalent to the original "copy boundary values" step every iteration.
__global__ void __launch_bounds__(TILE_X * TILE_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz) {
    __shared__ Real tile[TILE_Y + 2][TILE_X + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Global interior coordinates handled by this thread.
    const int x = blockIdx.x * TILE_X + tx + 1;
    const int y = blockIdx.y * TILE_Y + ty + 1;

    const int z_begin = blockIdx.z * Z_CHUNK + 1;
    const int z_end = min(z_begin + Z_CHUNK, nz - 1);

    const bool active = (x < nx - 1) && (y < ny - 1);
    // Clamp coordinates for inactive threads so that their (unused) loads stay in bounds.
    const int xc = active ? x : 1;
    const int yc = active ? y : 1;

    const size_t plane = (size_t)nx * ny;
    const Real* __restrict__ col = input + (size_t)yc * nx + xc;
    Real* __restrict__ ocol = output + (size_t)yc * nx + xc;

    // Halo loads: flags for which threads load which halo cells.
    const bool loadL = (tx == 0);
    const bool loadR = (tx == TILE_X - 1) || (x == nx - 2);
    const bool loadF = (ty == 0);
    const bool loadB = (ty == TILE_Y - 1) || (y == ny - 2);

    Real bottom = col[(size_t)(z_begin - 1) * plane];
    Real center = col[(size_t)z_begin * plane];

    for (int z = z_begin; z < z_end; ++z) {
        const Real top = col[(size_t)(z + 1) * plane];
        const Real* __restrict__ slab = input + (size_t)z * plane;

        __syncthreads();  // previous slice fully consumed before overwriting the tile
        // Only active threads write the tile: an inactive thread's slot may be the
        // halo slot of an active neighbour at the x/y edge of the interior.
        if (active) {
            tile[ty + 1][tx + 1] = center;
            if (loadL) tile[ty + 1][tx] = slab[(size_t)yc * nx + (xc - 1)];
            if (loadR) tile[ty + 1][tx + 2] = slab[(size_t)yc * nx + (xc + 1)];
            if (loadF) tile[ty][tx + 1] = slab[(size_t)(yc - 1) * nx + xc];
            if (loadB) tile[ty + 2][tx + 1] = slab[(size_t)(yc + 1) * nx + xc];
        }
        __syncthreads();

        if (active) {
            const Real left = tile[ty + 1][tx];
            const Real right = tile[ty + 1][tx + 2];
            const Real front = tile[ty][tx + 1];
            const Real back = tile[ty + 2][tx + 1];
            // Same evaluation order as the original scalar code (bitwise reproducible).
            ocol[(size_t)z * plane] = div7(center + left + right + front + back + bottom + top);
        }

        bottom = center;
        center = top;
    }
}

void initializeGrid(Real* d_grid, const size_t n) {
    const int threads = 256;
    const int blocks = (int)std::min<size_t>((n + threads - 1) / threads, 65535 * 16);
    initializeGridKernel<<<blocks, threads>>>(d_grid, n);
    CUDA_CHECK(cudaGetLastError());
}

// One stencil sweep: interior from input -> output.  Boundaries of `output`
// must already hold the (immutable) boundary values.
void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) return;  // no interior points
    const dim3 block(TILE_X, TILE_Y, 1);
    const dim3 grid((unsigned)((nx - 2 + TILE_X - 1) / TILE_X),
                    (unsigned)((ny - 2 + TILE_Y - 1) / TILE_Y),
                    (unsigned)((nz - 2 + Z_CHUNK - 1) / Z_CHUNK));
    stencilKernel<<<grid, block>>>(d_input, d_output, (int)nx, (int)ny, (int)nz);
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
    const size_t bytes = gridSize * sizeof(Real);
    
    // Allocate device grids (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, bytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, bytes));
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(d_grid1, gridSize);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Boundary cells are copied input->output every iteration in the reference
    // implementation and therefore never change.  Seeding grid2 with grid1 once
    // makes the boundaries of both buffers permanently correct, so each sweep
    // only has to update interior points.
    if (iterations > 0) {
        CUDA_CHECK(cudaMemcpyAsync(d_grid2, d_grid1, bytes, cudaMemcpyDeviceToDevice));
    }
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
    
    // Copy final grid back to host
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, bytes, cudaMemcpyDeviceToHost));
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
