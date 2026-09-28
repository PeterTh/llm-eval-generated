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

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                     \
        if (err_ != cudaSuccess) {                                                           \
            printf("CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,                \
                   cudaGetErrorString(err_));                                                \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Thread block / work distribution for the stencil kernel.
// Each thread owns one (x,y) column and marches over Z_TILE planes in z, keeping the
// z-neighbours in registers so only 5 of the 7 stencil points are re-loaded per plane.
static constexpr int BLOCK_X = 32;
static constexpr int BLOCK_Y = 8;
static constexpr int Z_TILE = 8;

// Division by 7 via reciprocal multiplication plus one Markstein FMA correction step.
// Since 1/7.0 is the correctly rounded reciprocal of 7, this yields the same
// round-to-nearest result as the hardware `x / 7.0` (bit-exact) but avoids the very slow
// FP64 divide sequence, which otherwise dominates the kernel on consumer GPUs.
// (Only finite operands are handled, which is all this stencil ever produces.)
__device__ __forceinline__ Real div7(const Real x) {
    constexpr Real r = 1.0 / 7.0;
    const Real q = x * r;
    const Real residual = __fma_rn(-7.0, q, x);  // exact
    return __fma_rn(residual, r, q);
}

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t gridSize) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < gridSize; idx += stride) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil computation over the interior points.
// Boundary points are invariant under the original algorithm (they are copied verbatim from
// the input every iteration), so both buffers are seeded with them once up front and the
// kernel only has to produce the interior.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = 1 + (size_t)blockIdx.x * BLOCK_X + threadIdx.x;
    const size_t y = 1 + (size_t)blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) return;

    const size_t zBegin = 1 + (size_t)blockIdx.z * Z_TILE;
    if (zBegin >= nz - 1) return;
    const size_t zEnd = min(zBegin + (size_t)Z_TILE, nz - 1);

    const size_t plane = nx * ny;
    size_t idx = idx3(x, y, zBegin, nx, ny);

    Real bottom = __ldg(&input[idx - plane]);
    Real center = __ldg(&input[idx]);

    for (size_t z = zBegin; z < zEnd; ++z, idx += plane) {
        const Real top = __ldg(&input[idx + plane]);
        const Real left = __ldg(&input[idx - 1]);
        const Real right = __ldg(&input[idx + 1]);
        const Real front = __ldg(&input[idx - nx]);
        const Real back = __ldg(&input[idx + nx]);

        // Simple averaging stencil
        output[idx] = div7(center + left + right + front + back + bottom + top);

        bottom = center;
        center = top;
    }
}

void stencilIteration(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) return;  // no interior points
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 grid((unsigned)((nx - 2 + BLOCK_X - 1) / BLOCK_X),
                    (unsigned)((ny - 2 + BLOCK_Y - 1) / BLOCK_Y),
                    (unsigned)((nz - 2 + Z_TILE - 1) / Z_TILE));
    stencilKernel<<<grid, block>>>(input, output, nx, ny, nz);
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
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));

    // Initialize
    printf("Initializing grid...\n");
    {
        const int threads = 256;
        const int blocks = (int)std::min<size_t>((gridSize + threads - 1) / threads, 65535);
        initializeGridKernel<<<blocks, threads>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
        // Seed the second buffer as well: the stencil leaves boundary points untouched, and
        // they never change value, so pre-copying makes both buffers boundary-correct.
        CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridSize * sizeof(Real), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

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
    const double seconds = std::chrono::duration<double>(end - start).count();

    CUDA_CHECK(cudaGetLastError());

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / seconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy the result back to the host for output / validation
    std::vector<Real> finalGrid(gridSize);
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
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
