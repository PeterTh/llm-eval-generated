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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        const cudaError_t err_ = (call);                                                   \
        if (err_ != cudaSuccess) {                                                         \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__,       \
                   __LINE__, cudaGetErrorString(err_));                                    \
            exit(1);                                                                       \
        }                                                                                  \
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
// Exact division by 7.
//
// A hardware IEEE double division costs ~25 FP64 instructions and, on the FP64-
// throughput-limited consumer GPUs this benchmark targets, it dominates the
// runtime of the stencil.  The Markstein refinement below (reciprocal multiply
// + one exact FMA residual correction) yields the correctly rounded quotient -
// bit-for-bit identical to `x / 7.0` - in three FP64 instructions.  It relies
// on `1.0 / 7.0` being the correctly rounded reciprocal and on `fma` computing
// the residual exactly.
// ---------------------------------------------------------------------------
__device__ __forceinline__ Real div7(const Real x) {
    const Real inv7 = 1.0 / 7.0;
    const Real q = x * inv7;
    const Real r = fma(-7.0, q, x);  // exact residual
    return fma(r, inv7, q);
}

// ---------------------------------------------------------------------------
// CUDA 7-point stencil
//
// Each thread owns one interior (x, y) column and marches it along z, so the
// z-1 / z+1 neighbours stay in registers (2.5D blocking) and only one new value
// per cell has to be fetched; the x / y neighbours are served by the caches.
// The z range is split into chunks so that enough blocks are in flight to fill
// the device while keeping the concurrently active blocks close together in
// memory.
//
// Boundary cells are never written: the original algorithm copies them
// unchanged on every iteration, so both device buffers are seeded with the
// initial grid and their boundaries stay valid for the whole run.
// ---------------------------------------------------------------------------

static constexpr int TX = 32;
static constexpr int TY = 4;
static constexpr int Z_SPLIT = 32;  // target number of z chunks

__global__ __launch_bounds__(TX* TY) void stencilKernel(const Real* __restrict__ input,
                                                        Real* __restrict__ output,
                                                        const int nx, const int ny, const int nz,
                                                        const int zChunk) {
    const int x = 1 + blockIdx.x * TX + threadIdx.x;
    const int y = 1 + blockIdx.y * TY + threadIdx.y;

    const int z0 = 1 + blockIdx.z * zChunk;
    const int z1 = min(z0 + zChunk, nz - 1);
    if (x >= nx - 1 || y >= ny - 1 || z0 >= z1) return;

    const size_t slice = (size_t)nx * (size_t)ny;
    const size_t base = (size_t)y * (size_t)nx + (size_t)x;
    const Real* in = input + base + (size_t)z0 * slice;

    Real prev = in[-(ptrdiff_t)slice];  // z - 1
    Real cur = in[0];                   // z

    for (int z = z0; z < z1; ++z, in += slice) {
        const Real next = in[slice];  // z + 1

        const Real center = cur;
        const Real left = in[-1];
        const Real right = in[1];
        const Real front = in[-nx];
        const Real back = in[nx];
        const Real bottom = prev;
        const Real top = next;

        output[base + (size_t)z * slice] =
            div7(center + left + right + front + back + bottom + top);

        prev = cur;
        cur = next;
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

    // Host grid
    std::vector<Real> grid1(gridSize);

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    // Device grids (double buffering); both are seeded with the initial data so
    // that the boundary cells - which the stencil only ever copies - are already
    // in place in either buffer.
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMemcpy(d_grid1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid2, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));

    const int inx = (int)nx, iny = (int)ny, inz = (int)nz;
    const int nxi = inx - 2, nyi = iny - 2, nzi = inz - 2;  // interior extents

    dim3 block(TX, TY);
    dim3 grid(1, 1, 1);
    int zChunk = 1;
    if (nxi > 0 && nyi > 0 && nzi > 0) {
        zChunk = std::max(1, nzi / Z_SPLIT);
        grid = dim3((unsigned)((nxi + TX - 1) / TX), (unsigned)((nyi + TY - 1) / TY),
                    (unsigned)((nzi + zChunk - 1) / zChunk));
    }

    // Run stencil iterations
    printf("Running stencil computation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    if (nxi > 0 && nyi > 0 && nzi > 0) {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* in = (iter % 2 == 0) ? d_grid1 : d_grid2;
            Real* out = (iter % 2 == 0) ? d_grid2 : d_grid1;
            stencilKernel<<<grid, block>>>(in, out, inx, iny, inz, zChunk);
        }
        CUDA_CHECK(cudaGetLastError());
    }
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
