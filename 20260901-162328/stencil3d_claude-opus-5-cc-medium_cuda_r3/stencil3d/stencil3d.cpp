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

#define CUDA_CHECK(call)                                                                             \
    do {                                                                                             \
        const cudaError_t err__ = (call);                                                            \
        if (err__ != cudaSuccess) {                                                                  \
            printf("CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE);                                                                      \
        }                                                                                            \
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

// ---------------------------------------------------------------------------
// CUDA 7-point stencil
//
// Each thread block owns a BX x BY tile in (x,y) and marches along z, keeping
// the z-1 / z+1 neighbours in registers and staging the current z-plane in
// shared memory so that the x/y neighbours are read only once from memory.
//
// The domain boundary is never written by the stencil: the original code copies
// the input boundary to the output on every iteration, and since the boundary
// of the initial grid never changes, seeding both device buffers with the
// initial grid once makes every later iteration produce identical results.
// ---------------------------------------------------------------------------

static constexpr int BX = 32;
static constexpr int BY = 4;
static constexpr int SH_X = BX + 2;
static constexpr int SH_Y = BY + 2;

// Correctly rounded x / 7.0 without an FP64 divide (which is emulated by a long
// instruction sequence on consumer GPUs and dominates this kernel).
//
// With c = RN(1/7): q0 = RN(x*c); r = fma(-7, q0, x) is exact, so x/7 = q0 + r/7
// exactly, and fma(r, c, q0) rounds q0 + (r/7)(1+d), |d| <= u, i.e. a value within
// a relative u^2 of x/7. Since 7 is odd, x/7 is never closer than ~2^-53 ulp to a
// rounding midpoint, so the single rounding of the fma yields RN(x/7) exactly.
__device__ __forceinline__ Real divideBy7(const Real x) {
    constexpr Real c = 1.0 / 7.0;  // RN(1/7)
    const Real q0 = __dmul_rn(x, c);
    const Real r = __fma_rn(-7.0, q0, x);
    return __fma_rn(r, c, q0);
}

__global__ __launch_bounds__(BX* BY) void stencilKernel(const Real* __restrict__ in, Real* __restrict__ out, const int nx,
                                                        const int ny, const int nz, const int zChunk) {
    __shared__ Real plane[SH_Y][SH_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    // Origin of the halo region staged in shared memory.
    const int x0 = static_cast<int>(blockIdx.x) * BX;
    const int y0 = static_cast<int>(blockIdx.y) * BY;

    // Interior point owned by this thread.
    const int x = x0 + tx + 1;
    const int y = y0 + ty + 1;
    const bool active = (x < nx - 1) && (y < ny - 1);

    const long long slice = static_cast<long long>(nx) * ny;

    const int zBegin = static_cast<int>(blockIdx.z) * zChunk + 1;
    int zEnd = zBegin + zChunk;
    if (zEnd > nz - 1) zEnd = nz - 1;
    if (zBegin >= zEnd) return;

    const long long xyBase = static_cast<long long>(y) * nx + x;

    // Sliding window in z.
    Real below = active ? in[(zBegin - 1) * slice + xyBase] : Real(0);
    Real center = active ? in[static_cast<long long>(zBegin) * slice + xyBase] : Real(0);

    // Cooperative halo load of one z-plane into shared memory.
    const int tid = ty * BX + tx;
    const int nThreads = BX * BY;

    for (int z = zBegin; z < zEnd; ++z) {
        const long long planeBase = static_cast<long long>(z) * slice;

        __syncthreads();
        for (int i = tid; i < SH_X * SH_Y; i += nThreads) {
            const int lx = i % SH_X;
            const int ly = i / SH_X;
            int gx = x0 + lx;
            int gy = y0 + ly;
            if (gx > nx - 1) gx = nx - 1;
            if (gy > ny - 1) gy = ny - 1;
            plane[ly][lx] = in[planeBase + static_cast<long long>(gy) * nx + gx];
        }
        __syncthreads();

        const Real above = active ? in[planeBase + slice + xyBase] : Real(0);

        if (active) {
            const Real left = plane[ty + 1][tx];
            const Real right = plane[ty + 1][tx + 2];
            const Real front = plane[ty][tx + 1];
            const Real back = plane[ty + 2][tx + 1];

            // Same summation order as the reference implementation.
            const Real sum = center + left + right + front + back + below + above;
            out[planeBase + xyBase] = divideBy7(sum);
        }

        below = center;
        center = above;
    }
}

// Host-side driver: runs `iterations` stencil sweeps on the device.
static void runStencilGPU(std::vector<Real>& grid1, std::vector<Real>& grid2, const size_t nx, const size_t ny, const size_t nz,
                          const int iterations, double& elapsedSeconds) {
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);

    Real* dA = nullptr;
    Real* dB = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));

    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);
    const bool hasInterior = (nx > 2) && (ny > 2) && (nz > 2);

    dim3 block(BX, BY, 1);
    dim3 grid(1, 1, 1);
    int zChunk = 1;
    if (hasInterior) {
        const int ix = static_cast<int>((nx - 2 + BX - 1) / BX);
        const int iy = static_cast<int>((ny - 2 + BY - 1) / BY);
        const int iz = static_cast<int>(nz - 2);
        // Enough blocks to saturate the device while keeping the z-reuse long.
        zChunk = 32;
        while (zChunk > 1 && static_cast<long long>(ix) * iy * ((iz + zChunk - 1) / zChunk) < 2048) {
            zChunk /= 2;
        }
        grid = dim3(ix, iy, static_cast<unsigned>((iz + zChunk - 1) / zChunk));
    }

    // Page-lock the host buffers so the PCIe transfers run at full speed.
    std::vector<Real>& hostResult = (iterations % 2 == 0) ? grid1 : grid2;
    const bool pinnedIn = (cudaHostRegister(grid1.data(), bytes, cudaHostRegisterDefault) == cudaSuccess);
    const bool pinnedOut = (&hostResult == &grid1) ? false
                                                   : (cudaHostRegister(hostResult.data(), bytes, cudaHostRegisterDefault) == cudaSuccess);
    cudaGetLastError();  // clear a potential registration failure

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::high_resolution_clock::now();

    if (iterations > 0) {
        // Seed both buffers so the (invariant) boundary is correct in either one.
        CUDA_CHECK(cudaMemcpy(dA, grid1.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, dA, bytes, cudaMemcpyDeviceToDevice));

        if (hasInterior) {
            for (int iter = 0; iter < iterations; ++iter) {
                const Real* in = (iter % 2 == 0) ? dA : dB;
                Real* out = (iter % 2 == 0) ? dB : dA;
                stencilKernel<<<grid, block>>>(in, out, inx, iny, inz, zChunk);
            }
            CUDA_CHECK(cudaGetLastError());
        }

        // Result lives in dA when an even number of sweeps ran, dB otherwise.
        const Real* result = (iterations % 2 == 0) ? dA : dB;
        CUDA_CHECK(cudaMemcpy(hostResult.data(), result, bytes, cudaMemcpyDeviceToHost));
    }

    const auto end = std::chrono::high_resolution_clock::now();
    elapsedSeconds = std::chrono::duration<double>(end - start).count();

    if (pinnedIn) CUDA_CHECK(cudaHostUnregister(grid1.data()));
    if (pinnedOut) CUDA_CHECK(cudaHostUnregister(hostResult.data()));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
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

    // Warm up the CUDA context so it is not charged to the measured region.
    CUDA_CHECK(cudaFree(nullptr));

    // Run stencil iterations
    printf("Running stencil computation...\n");
    double seconds = 0.0;
    runStencilGPU(grid1, grid2, nx, ny, nz, iterations, seconds);

    printf("Computation time: %ld ms\n", static_cast<long>(std::llround(seconds * 1000.0)));

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / seconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
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
