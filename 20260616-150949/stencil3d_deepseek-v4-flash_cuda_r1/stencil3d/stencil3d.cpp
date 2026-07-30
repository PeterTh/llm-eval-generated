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

// 3D index calculation (must be visible to both host and device code)
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ---------------------------------------------------------------------------
// CUDA kernel
// ---------------------------------------------------------------------------
// Single-pass stencil kernel using shared memory tiling.
//
// Each 3D block loads a tile (with 1-element halo) into shared memory so
// that the 7-point stencil for interior points can be served entirely from
// fast on-chip memory.  Boundary points are handled by direct copy within
// the same kernel.
// ---------------------------------------------------------------------------
template <int BLOCK_X, int BLOCK_Y, int BLOCK_Z>
__global__ void stencil_shared_kernel(
    const Real* __restrict__ input,
    Real* __restrict__ output,
    const size_t nx, const size_t ny, const size_t nz) {

    __shared__ Real tile[BLOCK_Z + 2][BLOCK_Y + 2][BLOCK_X + 2];

    // Global grid position of this thread
    const size_t gx = blockIdx.x * BLOCK_X + threadIdx.x;
    const size_t gy = blockIdx.y * BLOCK_Y + threadIdx.y;
    const size_t gz = blockIdx.z * BLOCK_Z + threadIdx.z;

    // Local index inside the shared-memory tile (offset by 1 for the halo)
    const size_t lx = threadIdx.x + 1;
    const size_t ly = threadIdx.y + 1;
    const size_t lz = threadIdx.z + 1;

    // ---- Load central point – every valid grid position is stored ----
    if (gx < nx && gy < ny && gz < nz) {
        tile[lz][ly][lx] = input[idx3(gx, gy, gz, nx, ny)];
    }

    // ---- Load the six halo faces (bounds-checked) ----
    // Clamp checking: only load a halo address when it falls inside the grid.
    // The left/front/bottom halos are needed by blocks that start at > 0
    // in that dimension; the right/back/top halos by blocks that end before n-1.

    if (threadIdx.x == 0) {
        if (gx > 0 && gy < ny && gz < nz) {
            tile[lz][ly][0] = input[idx3(gx - 1, gy, gz, nx, ny)];
        }
    }
    if (threadIdx.x == BLOCK_X - 1) {
        if (gx + 1 < nx && gy < ny && gz < nz) {
            tile[lz][ly][BLOCK_X + 1] = input[idx3(gx + 1, gy, gz, nx, ny)];
        }
    }
    if (threadIdx.y == 0) {
        if (gy > 0 && gx < nx && gz < nz) {
            tile[lz][0][lx] = input[idx3(gx, gy - 1, gz, nx, ny)];
        }
    }
    if (threadIdx.y == BLOCK_Y - 1) {
        if (gy + 1 < ny && gx < nx && gz < nz) {
            tile[lz][BLOCK_Y + 1][lx] = input[idx3(gx, gy + 1, gz, nx, ny)];
        }
    }
    if (threadIdx.z == 0) {
        if (gz > 0 && gx < nx && gy < ny) {
            tile[0][ly][lx] = input[idx3(gx, gy, gz - 1, nx, ny)];
        }
    }
    if (threadIdx.z == BLOCK_Z - 1) {
        if (gz + 1 < nz && gx < nx && gy < ny) {
            tile[BLOCK_Z + 1][ly][lx] = input[idx3(gx, gy, gz + 1, nx, ny)];
        }
    }

    __syncthreads();

    // ---- Compute result ----
    if (gx < nx && gy < ny && gz < nz) {
        if (gx == 0 || gx == nx - 1 || gy == 0 || gy == ny - 1 || gz == 0 || gz == nz - 1) {
            // Boundary point – copy directly (input was stored into shared memory)
            output[idx3(gx, gy, gz, nx, ny)] = tile[lz][ly][lx];
        } else {
            // Interior point – 7-point stencil from shared memory.
            // All seven neighbours are guaranteed to be in the tile because:
            //   - gx in [1, nx-2]  →  gx-1 ≥ 0  and gx+1 ≤ nx-1  (in grid)
            //   - Those positions were loaded by adjacent threads or halos.
            const Real center = tile[lz][ly][lx];
            const Real left   = tile[lz][ly][lx - 1];
            const Real right  = tile[lz][ly][lx + 1];
            const Real front  = tile[lz][ly - 1][lx];
            const Real back   = tile[lz][ly + 1][lx];
            const Real bottom = tile[lz - 1][ly][lx];
            const Real top    = tile[lz + 1][ly][lx];

            output[idx3(gx, gy, gz, nx, ny)] =
                (center + left + right + front + back + bottom + top) * Real(1.0 / 7.0);
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: launches one stencil iteration on the GPU
// ---------------------------------------------------------------------------
static void launchStencilIteration(
    const Real* d_input, Real* d_output,
    const size_t nx, const size_t ny, const size_t nz) {

    // Balanced 3D tile.  Occupancy: 6 blocks/SM × 256 threads = 1536 thr (100%).
    // 10×10×6 × 8 B = 4800 B per block < 48 KB shared memory.
    constexpr int BLOCK_X = 8;
    constexpr int BLOCK_Y = 8;
    constexpr int BLOCK_Z = 4;

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(
        (nx + BLOCK_X - 1) / BLOCK_X,
        (ny + BLOCK_Y - 1) / BLOCK_Y,
        (nz + BLOCK_Z - 1) / BLOCK_Z
    );

    stencil_shared_kernel<BLOCK_X, BLOCK_Y, BLOCK_Z><<<grid, block>>>(
        d_input, d_output, nx, ny, nz);

    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------
// Host-side helpers (unchanged semantics)
// ---------------------------------------------------------------------------

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
    
    const size_t gridSize = nx * ny * nz;
    
    // ---- Host memory for initialisation / I/O ----
    printf("Initializing grid...\n");
    std::vector<Real> grid1(gridSize);
    initializeGrid(grid1, nx, ny, nz);
    
    // ---- Device memory ----
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    
    // Copy initial grid to device
    CUDA_CHECK(cudaMemcpy(d_grid1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    
    // ---- Warm-up to initialise GPU state ----
    launchStencilIteration(d_grid1, d_grid2, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    // Restore d_grid2 to the initial data so both buffers match
    CUDA_CHECK(cudaMemcpy(d_grid2, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- Timed stencil iterations on GPU ----
    printf("Running stencil computation...\n");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            launchStencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            launchStencilIteration(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    
    float duration_ms;
    CUDA_CHECK(cudaEventElapsedTime(&duration_ms, start, stop));
    
    printf("Computation time: %.0f ms\n", duration_ms);
    
    // Determine which device buffer holds the final result
    const Real* d_result = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    
    // ---- Copy result back to host ----
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_result, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
        } else {
            printf("Validation: FAILED\n");
        }
        
        // ---- Cleanup ----
        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        return valid ? 0 : 1;
    }
    
    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    return 0;
}
