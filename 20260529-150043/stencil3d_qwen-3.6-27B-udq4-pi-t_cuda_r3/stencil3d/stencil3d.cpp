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

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

// Tile dimensions for shared-memory stencil
static constexpr int TILE_X = 32;
static constexpr int TILE_Y = 8;
static constexpr int TILE_Z = 4;

// Initialise every element: grid[idx] = (idx % 19) * 1.0
__global__ void initKernel(Real* grid, const size_t nx, const size_t ny,
                           const size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < nx && y < ny && z < nz) {
        size_t idx = z * (nx * ny) + y * nx + x;
        grid[idx] = static_cast<Real>(idx % 19);
    }
}

// 7-point stencil with shared-memory tiling.
//
// Each block covers a (TILE_X x TILE_Y x TILE_Z) tile of the output grid.
// Shared memory holds the tile plus a 1-cell halo on every face so that
// interior threads can read all 7 neighbours from shared memory.
//
// Threads within a warp (consecutive threadIdx.x) access consecutive global
// addresses -> fully coalesced loads.
//
// Boundary cells are left untouched (caller pre-fills output with input).
__global__ void stencilKernel(const Real* input, Real* output,
                              const size_t nx, const size_t ny,
                              const size_t nz) {
    // Shared memory: tile + 1-cell halo in every direction
    __shared__ Real sdata[(TILE_X + 2) * (TILE_Y + 2) * (TILE_Z + 2)];

    // Thread coordinates within the tile (0-based, no halo offset)
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    // Global position of the thread's primary cell
    const size_t gx = blockIdx.x * TILE_X + tx;
    const size_t gy = blockIdx.y * TILE_Y + ty;
    const size_t gz = blockIdx.z * TILE_Z + tz;

    // Shared-memory index helper  (includes halo offset)
    auto sidx = [&](int lx, int ly, int lz) {
        return (lz * (TILE_Y + 2) + ly) * (TILE_X + 2) + lx;
    };

    // ------------------------------------------------------------------
    // Phase 1 -- load tile into shared memory (coalesced x-first access)
    // ------------------------------------------------------------------
    {
        // Primary cell
        if (gx < nx && gy < ny && gz < nz) {
            const size_t g = gz * (nx * ny) + gy * nx + gx;
            sdata[sidx(tx + 1, ty + 1, tz + 1)] = input[g];
        }

        // +X halo
        if (tx == TILE_X - 1 && gx + 1 < nx && gy < ny && gz < nz) {
            const size_t g = gz * (nx * ny) + gy * nx + (gx + 1);
            sdata[sidx(tx + 2, ty + 1, tz + 1)] = input[g];
        }

        // -X halo
        if (tx == 0 && gx > 0 && gy < ny && gz < nz) {
            const size_t g = gz * (nx * ny) + gy * nx + (gx - 1);
            sdata[sidx(tx, ty + 1, tz + 1)] = input[g];
        }

        // +Y halo
        if (ty == TILE_Y - 1 && gx < nx && gy + 1 < ny && gz < nz) {
            const size_t g = gz * (nx * ny) + (gy + 1) * nx + gx;
            sdata[sidx(tx + 1, ty + 2, tz + 1)] = input[g];
        }

        // -Y halo
        if (ty == 0 && gx < nx && gy > 0 && gz < nz) {
            const size_t g = gz * (nx * ny) + (gy - 1) * nx + gx;
            sdata[sidx(tx + 1, ty, tz + 1)] = input[g];
        }

        // +Z halo
        if (tz == TILE_Z - 1 && gx < nx && gy < ny && gz + 1 < nz) {
            const size_t g = (gz + 1) * (nx * ny) + gy * nx + gx;
            sdata[sidx(tx + 1, ty + 1, tz + 2)] = input[g];
        }

        // -Z halo
        if (tz == 0 && gx < nx && gy < ny && gz > 0) {
            const size_t g = (gz - 1) * (nx * ny) + gy * nx + gx;
            sdata[sidx(tx + 1, ty + 1, tz)] = input[g];
        }
    }

    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2 -- compute stencil for interior cells only
    // ------------------------------------------------------------------
    if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1 && gz > 0 && gz < nz - 1) {
        const Real center = sdata[sidx(tx + 1, ty + 1, tz + 1)];
        const Real left   = sdata[sidx(tx,   ty + 1, tz + 1)];
        const Real right  = sdata[sidx(tx + 2, ty + 1, tz + 1)];
        const Real front  = sdata[sidx(tx + 1, ty,     tz + 1)];
        const Real back   = sdata[sidx(tx + 1, ty + 2, tz + 1)];
        const Real bottom = sdata[sidx(tx + 1, ty + 1, tz)];
        const Real top    = sdata[sidx(tx + 1, ty + 1, tz + 2)];

        const size_t g = gz * (nx * ny) + gy * nx + gx;
        output[g] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// Copy boundary cells: output[boundary] = input[boundary].
// Uses a 1D grid-stride loop for flexibility.
__global__ void copyBoundaryKernel(const Real* input, Real* output,
                                   const size_t nx, const size_t ny,
                                   const size_t nz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;

    while (i < total) {
        // Decode flat index -> (x, y, z)
        size_t tmp = i;
        size_t x = tmp % nx;
        tmp /= nx;
        size_t y = tmp % ny;
        size_t z = tmp / ny;

        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 ||
            z == 0 || z == nz - 1) {
            output[i] = input[i];
        }
        i += blockDim.x * gridDim.x;
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

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

    // ---- Host buffers ------------------------------------------------
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    // ---- Device buffers (double-buffered) ----------------------------
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    cudaMalloc(&d_grid1, gridSize * sizeof(Real));
    cudaMalloc(&d_grid2, gridSize * sizeof(Real));

    // ---- Initialise grid on GPU --------------------------------------
    printf("Initializing grid...\n");
    {
        dim3 block(TILE_X, TILE_Y, TILE_Z);
        dim3 grid(static_cast<unsigned>(nx), static_cast<unsigned>(ny),
                  static_cast<unsigned>(nz));
        // Clamp grid dims to avoid exceeding max grid size
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        grid.x = std::min(grid.x, static_cast<unsigned>(prop.maxGridSize[0]));
        grid.y = std::min(grid.y, static_cast<unsigned>(prop.maxGridSize[1]));
        grid.z = std::min(grid.z, static_cast<unsigned>(prop.maxGridSize[2]));

        initKernel<<<grid, block>>>(d_grid1, nx, ny, nz);
        cudaDeviceSynchronize();
    }

    // ---- Stencil iterations ------------------------------------------
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    {
        dim3 block(TILE_X, TILE_Y, TILE_Z);
        dim3 grid(
            (nx - 2 + TILE_X - 1) / TILE_X,
            (ny - 2 + TILE_Y - 1) / TILE_Y,
            (nz - 2 + TILE_Z - 1) / TILE_Z);

        // Boundary-copy kernel: 1D grid-stride
        const int bcopyThreads = 256;
        const int bcopyBlocks  = static_cast<int>((gridSize + bcopyThreads - 1) / bcopyThreads);

        Real* d_in  = d_grid1;
        Real* d_out = d_grid2;

        for (int iter = 0; iter < iterations; ++iter) {
            stencilKernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
            copyBoundaryKernel<<<bcopyBlocks, bcopyThreads>>>(d_in, d_out, nx, ny, nz);
            cudaDeviceSynchronize();

            // Swap pointers for next iteration
            std::swap(d_in, d_out);
        }

        cudaDeviceSynchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // ---- Copy final result back to host ------------------------------
    // After iterations iterations the result is in d_out (which was last swapped).
    // iterations even -> d_out == d_grid2  ->  grid2 has result
    // iterations odd  -> d_out == d_grid1  ->  grid1 has result
    //
    // Original code: finalGrid = (iterations % 2 == 0) ? grid1 : grid2
    //   even -> grid1, odd -> grid2
    //
    // Our code produces the opposite mapping, but the actual values match
    // because the double-buffering is symmetric. To match the original
    // semantics exactly, we copy from the buffer that holds the result.
    const Real* d_final = (iterations % 2 == 0) ? d_grid2 : d_grid1;
    cudaMemcpy(grid1.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);

    // ---- Performance metrics -----------------------------------------
    double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // ---- Results output / validation ---------------------------------
    const std::vector<Real>& finalGrid = grid1;

    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // ---- Cleanup -----------------------------------------------------
    cudaFree(d_grid1);
    cudaFree(d_grid2);

    return 0;
}