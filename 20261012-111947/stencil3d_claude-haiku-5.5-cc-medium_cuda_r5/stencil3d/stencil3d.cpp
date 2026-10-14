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

// Thread block shape for the stencil kernel (x is the fastest-varying dimension).
// Each block updates a TX x TY x TZ tile of cells and stages the tile plus a
// one-cell halo in shared memory, so every interior cell reads its six neighbours
// from shared memory instead of global memory.
constexpr int TX = 32;
constexpr int TY = 4;
constexpr int TZ = 4;

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

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t gridSize) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < gridSize) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil computation (one kernel launch per iteration).
// Interior cells are averaged; boundary cells are copied from the input.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const int nx, const int ny, const int nz) {
    __shared__ Real tile[TZ + 2][TY + 2][TX + 2];

    // Global coordinates of the tile's halo origin (one cell before the block's first cell)
    const int x0 = blockIdx.x * TX - 1;
    const int y0 = blockIdx.y * TY - 1;
    const int z0 = blockIdx.z * TZ - 1;

    // Cooperative load of the tile, including halo, using a flat thread index.
    // Halo cells outside the grid are never read: only interior cells use the halo,
    // and those always have all their neighbours inside the grid.
    constexpr int tileElems = (TX + 2) * (TY + 2) * (TZ + 2);
    const int nThreads = blockDim.x * blockDim.y * blockDim.z;
    const int tid = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    for (int t = tid; t < tileElems; t += nThreads) {
        const int lx = t % (TX + 2);
        const int ly = (t / (TX + 2)) % (TY + 2);
        const int lz = t / ((TX + 2) * (TY + 2));
        const int gx = x0 + lx;
        const int gy = y0 + ly;
        const int gz = z0 + lz;
        if (gx >= 0 && gx < nx && gy >= 0 && gy < ny && gz >= 0 && gz < nz) {
            tile[lz][ly][lx] = input[idx3(gx, gy, gz, nx, ny)];
        }
    }
    __syncthreads();

    const int x = blockIdx.x * TX + threadIdx.x;
    const int y = blockIdx.y * TY + threadIdx.y;
    const int z = blockIdx.z * TZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t idx = idx3(x, y, z, nx, ny);
    const bool interior = x > 0 && x < nx - 1 && y > 0 && y < ny - 1 && z > 0 && z < nz - 1;
    if (!interior) {
        output[idx] = input[idx];
        return;
    }

    // Local tile coordinates of this cell (offset by the halo)
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int lz = threadIdx.z + 1;

    // Simple averaging stencil (same summation order as the reference implementation)
    const Real center = tile[lz][ly][lx];
    const Real left = tile[lz][ly][lx - 1];
    const Real right = tile[lz][ly][lx + 1];
    const Real front = tile[lz][ly - 1][lx];
    const Real back = tile[lz][ly + 1][lx];
    const Real bottom = tile[lz - 1][ly][lx];
    const Real top = tile[lz + 1][ly][lx];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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

// Number of blocks needed to cover n cells with blocks of size b (at least one, so empty grids launch cleanly)
inline unsigned int blocksFor(const size_t n, const int b) {
    return static_cast<unsigned int>(std::max<size_t>(1, (n + b - 1) / b));
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

    const dim3 block(TX, TY, TZ);
    const dim3 gridDim(blocksFor(nx, TX), blocksFor(ny, TY), blocksFor(nz, TZ));
    const int inx = static_cast<int>(nx);
    const int iny = static_cast<int>(ny);
    const int inz = static_cast<int>(nz);

    // Initialize
    printf("Initializing grid...\n");
    initializeGridKernel<<<static_cast<unsigned int>(std::max<size_t>(1, (gridSize + 255) / 256)), 256>>>(d_grid1, gridSize);
    CUDA_CHECK(cudaGetLastError());
    // Finish initialization (and CUDA module loading) before starting the timer
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<gridDim, block>>>(d_grid1, d_grid2, inx, iny, inz);
        } else {
            stencilKernel<<<gridDim, block>>>(d_grid2, d_grid1, inx, iny, inz);
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

    // Copy the final grid back to the host for output and validation
    const Real* d_finalGrid = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_finalGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));

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
                exitCode = 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    return exitCode;
}
