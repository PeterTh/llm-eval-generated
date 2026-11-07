// 3D 7-point stencil benchmark - CUDA implementation.
//
// This file is compiled as CUDA (see CMakeLists.txt, which sets LANGUAGE CUDA
// on this source).  The whole grid lives on the GPU for the duration of the
// benchmark; only initialization results (for validation / result printing)
// are copied back to the host.

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

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, \
                    cudaGetErrorString(err__));                                       \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// grid[idx] = (idx % 19) * 1.0  for every cell (same as the original host loop)
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        grid[i] = (Real)(i % 19) * 1.0;
    }
}

// Correctly rounded division by 7 using one multiply and two FMAs instead of
// the generic IEEE double-division sequence.  Consumer GPUs have very low FP64
// throughput, and the generic a / 7.0 sequence (reciprocal + several Newton
// steps + fix-ups) dominated the kernel's run time.  By Markstein's theorem,
// for y = RN(1/b) and a faithful q0 = RN(a*y), the correction
//     r  = fma(-b, q0, a)      (exact residual)
//     q1 = fma(r, y, q0)
// yields exactly RN(a/b) for every finite a that neither overflows nor
// produces a subnormal quotient, as long as b's significand is not all ones
// (7 = 1.11b * 2^2).  The grid values here are bounded (|a| <= 126), so the
// result is bit-identical to the original `a / 7.0`.
__device__ __forceinline__ Real divideBy7(const Real a) {
    const Real inv7 = 1.0 / 7.0;  // correctly rounded reciprocal (compile-time constant)
    const Real q0 = a * inv7;
    const Real r = fma(-7.0, q0, a);
    return fma(r, inv7, q0);
}

// Tile configuration for the stencil kernel: each block owns a TX x TY tile in
// the x/y plane and marches along z over a chunk of planes, keeping the
// bottom/center/top values in registers and the current plane (plus a one-cell
// halo) in shared memory so every input element is read from global memory
// (almost) exactly once per iteration.
constexpr int TX = 32;
constexpr int TY = 8;

__global__ void __launch_bounds__(TX * TY)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz, const int zChunk) {
    __shared__ Real tile[TY + 2][TX + 2];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Interior coordinates handled by this thread (1 .. n-2)
    const int x = blockIdx.x * TX + tx + 1;
    const int y = blockIdx.y * TY + ty + 1;

    const int zBegin = blockIdx.z * zChunk + 1;
    const int zEnd = min(zBegin + zChunk, nz - 1);  // exclusive

    const bool active = (x < nx - 1) && (y < ny - 1);
    // Clamp so inactive threads still issue in-bounds loads (results unused)
    const int xc = active ? x : nx - 2;
    const int yc = active ? y : ny - 2;

    const size_t plane = (size_t)nx * ny;
    const size_t base = (size_t)yc * nx + xc;

    // Halo responsibilities (clamped similarly for safety)
    const bool loadLeft = (tx == 0);
    const bool loadRight = (tx == TX - 1) || (x == nx - 2);
    const bool loadFront = (ty == 0);
    const bool loadBack = (ty == TY - 1) || (y == ny - 2);

    Real bottom = input[(size_t)(zBegin - 1) * plane + base];
    Real center = input[(size_t)zBegin * plane + base];

    for (int z = zBegin; z < zEnd; ++z) {
        const Real* __restrict__ in = input + (size_t)z * plane;
        const Real top = input[(size_t)(z + 1) * plane + base];

        // Populate shared tile for the current plane
        // Only active threads write: an inactive (clamped) thread would
        // otherwise overwrite a halo slot owned by its active neighbour.
        if (active) {
            tile[ty + 1][tx + 1] = center;
            if (loadLeft)  tile[ty + 1][tx]      = in[base - 1];
            if (loadRight) tile[ty + 1][tx + 2]  = in[base + 1];
            if (loadFront) tile[ty][tx + 1]      = in[base - nx];
            if (loadBack)  tile[ty + 2][tx + 1]  = in[base + nx];
        }
        __syncthreads();

        if (active) {
            const Real left = tile[ty + 1][tx];
            const Real right = tile[ty + 1][tx + 2];
            const Real front = tile[ty][tx + 1];
            const Real back = tile[ty + 2][tx + 1];
            // Same operand order as the original code for bit-identical results
            output[(size_t)z * plane + base] =
                divideBy7(center + left + right + front + back + bottom + top);
        }
        __syncthreads();  // before the tile is overwritten for the next plane

        bottom = center;
        center = top;
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

static int g_numSMs = 1;

void initializeGrid(Real* d_grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t n = nx * ny * nz;
    if (n == 0) return;
    const int block = 256;
    const size_t want = (n + block - 1) / block;
    const int grid = (int)std::min<size_t>(want, (size_t)g_numSMs * 32);
    initializeGridKernel<<<grid, block>>>(d_grid, n);
    CUDA_CHECK(cudaGetLastError());
}

// Pick how many z-planes each block processes so that there are enough
// blocks to fill the GPU while still amortizing the register/shared-memory
// pipeline along z.
static int chooseZChunk(const size_t nx, const size_t ny, const size_t nz) {
    const size_t ix = nx - 2, iy = ny - 2, iz = nz - 2;
    const size_t blocksXY = ((ix + TX - 1) / TX) * ((iy + TY - 1) / TY);
    const size_t targetBlocks = (size_t)g_numSMs * 16;
    size_t chunk = 1;
    if (blocksXY > 0) {
        const size_t zBlocksWanted = std::max<size_t>(1, (targetBlocks + blocksXY - 1) / blocksXY);
        chunk = std::max<size_t>(1, (iz + zBlocksWanted - 1) / zBlocksWanted);
    }
    chunk = std::min<size_t>(chunk, 64);
    chunk = std::min<size_t>(chunk, std::max<size_t>(iz, 1));
    return (int)chunk;
}

// 7-point stencil computation on the interior.  Boundary cells are never
// written by the kernel; both device buffers already hold the (identical)
// boundary values, which is exactly what the original per-iteration boundary
// copy produced.
void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz, const int zChunk) {
    if (nx < 3 || ny < 3 || nz < 3) return;  // no interior points
    const size_t ix = nx - 2, iy = ny - 2, iz = nz - 2;
    dim3 block(TX, TY);
    dim3 grid((unsigned)((ix + TX - 1) / TX),
              (unsigned)((iy + TY - 1) / TY),
              (unsigned)((iz + zChunk - 1) / zChunk));
    stencilKernel<<<grid, block>>>(d_input, d_output, (int)nx, (int)ny, (int)nz, zChunk);
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

    // Select device and query SM count for launch sizing
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    CUDA_CHECK(cudaDeviceGetAttribute(&g_numSMs, cudaDevAttrMultiProcessorCount, device));
    
    // Allocate grids on the device (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, std::max<size_t>(gridSize, 1) * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, std::max<size_t>(gridSize, 1) * sizeof(Real)));
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(d_grid1, nx, ny, nz);
    // The boundary of the output is always a copy of the input boundary, which
    // never changes.  Seed the second buffer with the full initial grid so both
    // buffers carry correct boundary values and the kernels only touch the
    // interior.
    CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridSize * sizeof(Real), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int zChunk = chooseZChunk(nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz, zChunk);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz, zChunk);
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
    
    // Copy final grid back to host for result printing / validation
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
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
