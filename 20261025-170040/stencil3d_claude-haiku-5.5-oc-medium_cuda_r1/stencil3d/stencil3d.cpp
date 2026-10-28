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

// CUDA tiling configuration
constexpr unsigned TILE_X = 32;   // threads in x per block (coalesced dimension)
constexpr unsigned TILE_Y = 8;    // threads in y per block
constexpr size_t Z_CHUNK = 8;     // z-planes marched by each block (more blocks for occupancy)
constexpr unsigned BLOCK_THREADS = TILE_X * TILE_Y;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize grid: grid[idx] = (idx % 19) * 1.0
__global__ void initGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil iteration, including boundary copy.
// Each block owns a TILE_X x TILE_Y column of (x,y) and marches over a chunk of
// z-planes. The current plane is staged in shared memory (with a 1-cell halo) so
// the in-plane neighbours are reused across threads; the z-1 value is carried in a
// register and the z+1 value is read directly from global memory.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    __shared__ Real tile[TILE_Y + 2][TILE_X + 2];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tid = ty * TILE_X + tx;

    const size_t baseX = static_cast<size_t>(blockIdx.x) * TILE_X;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * TILE_Y;
    const size_t gx = baseX + tx;
    const size_t gy = baseY + ty;
    const bool valid = (gx < nx) && (gy < ny);

    const size_t z0 = static_cast<size_t>(blockIdx.z) * Z_CHUNK;
    const size_t z1 = min(z0 + Z_CHUNK, nz);

    const size_t planeStride = nx * ny;

    // Value at z-1 for this column (only needed for interior cells).
    Real below = 0.0;
    if (valid && z0 > 0) {
        below = input[idx3(gx, gy, z0 - 1, nx, ny)];
    }

    constexpr unsigned TILE_SIZE = (TILE_X + 2) * (TILE_Y + 2);

    for (size_t z = z0; z < z1; ++z) {
        // Cooperatively stage plane z (including halo) into shared memory.
        // Out-of-range halo positions are clamped; they are only read for
        // boundary cells, which never use the stencil result.
        for (unsigned i = tid; i < TILE_SIZE; i += BLOCK_THREADS) {
            const unsigned sy = i / (TILE_X + 2);
            const unsigned sx = i % (TILE_X + 2);
            long long px = static_cast<long long>(baseX) + static_cast<long long>(sx) - 1;
            long long py = static_cast<long long>(baseY) + static_cast<long long>(sy) - 1;
            px = min(max(px, 0LL), static_cast<long long>(nx) - 1);
            py = min(max(py, 0LL), static_cast<long long>(ny) - 1);
            tile[sy][sx] = input[z * planeStride + static_cast<size_t>(py) * nx + static_cast<size_t>(px)];
        }
        __syncthreads();

        if (valid) {
            const size_t idx = z * planeStride + gy * nx + gx;
            const Real center = tile[ty + 1][tx + 1];
            Real result;

            if (gx == 0 || gx == nx - 1 || gy == 0 || gy == ny - 1 || z == 0 || z == nz - 1) {
                result = center;  // boundary: copy input
            } else {
                const Real left   = tile[ty + 1][tx];
                const Real right  = tile[ty + 1][tx + 2];
                const Real front  = tile[ty][tx + 1];
                const Real back   = tile[ty + 2][tx + 1];
                const Real top    = input[idx + planeStride];
                // Same summation order as the reference implementation.
                result = (center + left + right + front + back + below + top) / 7.0;
            }

            output[idx] = result;
            below = center;
        }
        // Ensure all threads finished reading the tile before it is overwritten.
        __syncthreads();
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
    const size_t gridBytes = gridSize * sizeof(Real);
    
    // Device buffers (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes > 0 ? gridBytes : sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes > 0 ? gridBytes : sizeof(Real)));
    
    const unsigned initThreads = 256;
    const size_t initBlocks = (gridSize + initThreads - 1) / initThreads;
    
    // Initialize
    printf("Initializing grid...\n");
    if (gridSize > 0) {
        initGridKernel<<<static_cast<unsigned>(initBlocks), initThreads>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
        // Second buffer starts zeroed (matches prior zero-initialized host vector).
        CUDA_CHECK(cudaMemset(d_grid2, 0, gridBytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    const dim3 block(TILE_X, TILE_Y, 1);
    const dim3 grid(static_cast<unsigned>((nx + TILE_X - 1) / TILE_X),
                    static_cast<unsigned>((ny + TILE_Y - 1) / TILE_Y),
                    static_cast<unsigned>((nz + Z_CHUNK - 1) / Z_CHUNK));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (gridSize > 0) {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilKernel<<<grid, block>>>(d_grid1, d_grid2, nx, ny, nz);
            } else {
                stencilKernel<<<grid, block>>>(d_grid2, d_grid1, nx, ny, nz);
            }
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
    
    // Copy final result back to host
    std::vector<Real> finalGrid(gridSize);
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
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
