#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---- CUDA kernels ----

// Shared-memory tile dimensions (tunable; good defaults for most GPUs)
constexpr int BLOCK_X = 16;
constexpr int BLOCK_Y = 8;
constexpr int BLOCK_Z = 8;

// Shared memory tile includes 1-cell halo on each face
constexpr int TILE_X = BLOCK_X + 2;
constexpr int TILE_Y = BLOCK_Y + 2;
constexpr int TILE_Z = BLOCK_Z + 2;

__global__ void initGridKernel(Real* grid, const int nx, const int ny, const int nz) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t total = static_cast<size_t>(nx) * ny * nz;
    for (size_t i = idx; i < total; i += stride) {
        grid[i] = static_cast<Real>(static_cast<int>(i % 19)) * 1.0;
    }
}

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                               const int nx, const int ny, const int nz) {
    extern __shared__ char smem[];
    Real* s_tile = reinterpret_cast<Real*>(smem);

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    // Global coordinates of the thread's interior cell
    const int gx = static_cast<int>(blockIdx.x) * BLOCK_X + tx;
    const int gy = static_cast<int>(blockIdx.y) * BLOCK_Y + ty;
    const int gz = static_cast<int>(blockIdx.z) * BLOCK_Z + tz;

    // Each thread loads its own cell into the center of the tile
    // Shared memory position is offset by 1 in each dimension (halo)
    const int si = (tz + 1) * TILE_X * TILE_Y + (ty + 1) * TILE_X + (tx + 1);
    {
        const int ix = (gx >= 0 && gx < nx)   ? gx   : (gx < 0 ? 0 : nx - 1);
        const int iy = (gy >= 0 && gy < ny)   ? gy   : (gy < 0 ? 0 : ny - 1);
        const int iz = (gz >= 0 && gz < nz)   ? gz   : (gz < 0 ? 0 : nz - 1);
        s_tile[si] = input[static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx + ix];
    }

    // Load halo cells: each thread on the edge of the block loads the halo
    // X-dimension halo
    if (tx == 0) {
        // Left halo
        const int ix = (gx - 1 >= 0) ? gx - 1 : 0;
        const int iy = (gy >= 0 && gy < ny)   ? gy   : (gy < 0 ? 0 : ny - 1);
        const int iz = (gz >= 0 && gz < nz)   ? gz   : (gz < 0 ? 0 : nz - 1);
        s_tile[si - 1] = input[static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx + ix];
    }
    if (tx == BLOCK_X - 1) {
        // Right halo
        const int ix = (gx + 1 < nx) ? gx + 1 : nx - 1;
        const int iy = (gy >= 0 && gy < ny)   ? gy   : (gy < 0 ? 0 : ny - 1);
        const int iz = (gz >= 0 && gz < nz)   ? gz   : (gz < 0 ? 0 : nz - 1);
        s_tile[si + 1] = input[static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx + ix];
    }
    // Y-dimension halo
    if (ty == 0) {
        // Front halo
        const int ix = (gx >= 0 && gx < nx)   ? gx   : (gx < 0 ? 0 : nx - 1);
        const int iy = (gy - 1 >= 0) ? gy - 1 : 0;
        const int iz = (gz >= 0 && gz < nz)   ? gz   : (gz < 0 ? 0 : nz - 1);
        s_tile[si - TILE_X] = input[static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx + ix];
    }
    if (ty == BLOCK_Y - 1) {
        // Back halo
        const int ix = (gx >= 0 && gx < nx)   ? gx   : (gx < 0 ? 0 : nx - 1);
        const int iy = (gy + 1 < ny) ? gy + 1 : ny - 1;
        const int iz = (gz >= 0 && gz < nz)   ? gz   : (gz < 0 ? 0 : nz - 1);
        s_tile[si + TILE_X] = input[static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx + ix];
    }
    // Z-dimension halo
    if (tz == 0) {
        // Bottom halo
        const int ix = (gx >= 0 && gx < nx)   ? gx   : (gx < 0 ? 0 : nx - 1);
        const int iy = (gy >= 0 && gy < ny)   ? gy   : (gy < 0 ? 0 : ny - 1);
        const int iz = (gz - 1 >= 0) ? gz - 1 : 0;
        s_tile[si - TILE_X * TILE_Y] = input[static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx + ix];
    }
    if (tz == BLOCK_Z - 1) {
        // Top halo
        const int ix = (gx >= 0 && gx < nx)   ? gx   : (gx < 0 ? 0 : nx - 1);
        const int iy = (gy >= 0 && gy < ny)   ? gy   : (gy < 0 ? 0 : ny - 1);
        const int iz = (gz + 1 < nz) ? gz + 1 : nz - 1;
        s_tile[si + TILE_X * TILE_Y] = input[static_cast<size_t>(iz) * nx * ny + static_cast<size_t>(iy) * nx + ix];
    }

    __syncthreads();

    // --- Compute stencil (interior points only) ---
    if (gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1 && gz > 0 && gz < nz - 1) {
        const Real center = s_tile[si];
        const Real left   = s_tile[si - 1];
        const Real right  = s_tile[si + 1];
        const Real front  = s_tile[si - TILE_X];
        const Real back   = s_tile[si + TILE_X];
        const Real bottom = s_tile[si - TILE_X * TILE_Y];
        const Real top    = s_tile[si + TILE_X * TILE_Y];
        output[static_cast<size_t>(gz) * nx * ny + static_cast<size_t>(gy) * nx + gx] =
            (center + left + right + front + back + bottom + top) / 7.0;
    }
}

// Kernel to copy boundary values from input to output
__global__ void boundaryKernel(const Real* __restrict__ input, Real* __restrict__ output,
                                const int nx, const int ny, const int nz) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    const size_t total = static_cast<size_t>(nx) * ny * nz;
    for (size_t i = idx; i < total; i += stride) {
        const size_t x = i % nx;
        const size_t y = (i / nx) % ny;
        const size_t z = i / (nx * ny);
        if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
            output[i] = input[i];
        }
    }
}

// Host-side wrapper: launches stencil + boundary kernels
void stencilIteration(Real* d_input, Real* d_output, const int nx, const int ny, const int nz) {
    dim3 blockDim(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 gridDim(
        (nx  + BLOCK_X - 1) / BLOCK_X,
        (ny  + BLOCK_Y - 1) / BLOCK_Y,
        (nz  + BLOCK_Z - 1) / BLOCK_Z
    );

    size_t smemSize = 2 * TILE_X * TILE_Y * TILE_Z * sizeof(Real);
    stencilKernel<<<gridDim, blockDim, smemSize>>>(d_input, d_output, nx, ny, nz);

    // Grid-stride boundary kernel
    const int bsize = 256;
    const int gsize = (nx * ny * nz + bsize - 1) / bsize;
    boundaryKernel<<<gsize, bsize>>>(d_input, d_output, nx, ny, nz);
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
    
    // Allocate device memory (double buffering)
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    cudaMalloc(&d_grid1, gridSize * sizeof(Real));
    cudaMalloc(&d_grid2, gridSize * sizeof(Real));

    // Initialize on GPU
    printf("Initializing grid...\n");
    const int bsize = 256;
    const int gsize = (gridSize + bsize - 1) / bsize;
    initGridKernel<<<gsize, bsize>>>(d_grid1, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
    cudaDeviceSynchronize();
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
        } else {
            stencilIteration(d_grid2, d_grid1, static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
        }
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy final result back to host for validation/printing
    std::vector<Real> finalGrid(gridSize);
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
    
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

    cudaFree(d_grid1);
    cudaFree(d_grid2);
    
    return 0;
}
