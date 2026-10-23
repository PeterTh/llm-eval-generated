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

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Division by the constant 7 using a reciprocal and one Newton refinement.
// The result is the correctly rounded quotient for every finite input.
__device__ __forceinline__ Real div7(const Real x) {
    const Real r = 1.0 / 7.0;
    Real q = x * r;
    const Real rem = __fma_rn(-7.0, q, x);
    q = __fma_rn(rem, r, q);
    return q;
}

__global__ void initializeGridKernel(Real* __restrict__ grid,
                                     const size_t nx,
                                     const size_t ny,
                                     const size_t nz) {
    const size_t total = nx * ny * nz;
    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (; i < total; i += stride) {
        grid[i] = static_cast<Real>(i % 19);
    }
}

// Copy the six boundary faces from the initialized grid into the second
// buffer. Boundary values never change, so they only need to be seeded once.
__global__ void seedBoundaryKernel(const Real* __restrict__ src,
                                   Real* __restrict__ dst,
                                   const size_t nx,
                                   const size_t ny,
                                   const size_t nz) {
    const size_t plane = nx * ny;
    const size_t zin = (nz > 2) ? nz - 2 : 0;
    const size_t yin = (ny > 2) ? ny - 2 : 0;
    const size_t faceZ = 2 * plane;
    const size_t faceY = 2 * zin * nx;
    const size_t faceX = 2 * zin * yin;
    const size_t total = faceZ + faceY + faceX;

    size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;

    for (; i < total; i += stride) {
        size_t idx;
        if (i < faceZ) {
            size_t k = i;
            const size_t z = (k < plane) ? 0 : nz - 1;
            if (k >= plane) k -= plane;
            idx = z * plane + k;
        } else if (i < faceZ + faceY) {
            size_t k = i - faceZ;
            const size_t perZ = 2 * nx;
            const size_t z = 1 + k / perZ;
            const size_t r = k % perZ;
            const size_t y = (r < nx) ? 0 : ny - 1;
            const size_t x = (r < nx) ? r : r - nx;
            idx = idx3(x, y, z, nx, ny);
        } else {
            size_t k = i - faceZ - faceY;
            const size_t perZ = 2 * yin;
            const size_t z = 1 + k / perZ;
            const size_t r = k % perZ;
            const size_t y = (r < yin) ? 1 + r : 1 + (r - yin);
            const size_t x = (r < yin) ? 0 : nx - 1;
            idx = idx3(x, y, z, nx, ny);
        }
        dst[idx] = src[idx];
    }
}

// 7-point stencil over an interior tile staged in shared memory. Each thread
// produces ZPER consecutive output planes, reusing the values already read for
// the previous plane (its "top" becomes the next "center" and its "center"
// becomes the next "bottom").
template <int TX, int TY, int TZ, int BZ>
__global__ void stencilIterationKernel(const Real* __restrict__ input,
                                       Real* __restrict__ output,
                                       const size_t nx,
                                       const size_t ny,
                                       const size_t nz) {
    constexpr int SX = TX + 2;
    constexpr int SY = TY + 2;
    constexpr int SZ = TZ + 2;
    constexpr int ZPER = TZ / BZ;
    static_assert(TZ % BZ == 0, "TZ must be a multiple of BZ");

    __shared__ Real tile[SX * SY * SZ];

    const size_t x0 = static_cast<size_t>(blockIdx.x) * TX + 1;
    const size_t y0 = static_cast<size_t>(blockIdx.y) * TY + 1;
    const size_t z0 = static_cast<size_t>(blockIdx.z) * TZ + 1;

    // Stage the input tile, halo included.
    for (int lz = threadIdx.z; lz < SZ; lz += BZ) {
        const size_t gz = z0 + static_cast<size_t>(lz) - 1;
        for (int ly = threadIdx.y; ly < SY; ly += TY) {
            const size_t gy = y0 + static_cast<size_t>(ly) - 1;
            for (int lx = threadIdx.x; lx < SX; lx += TX) {
                const size_t gx = x0 + static_cast<size_t>(lx) - 1;
                Real value = static_cast<Real>(0);
                if (gx < nx && gy < ny && gz < nz) {
                    value = input[idx3(gx, gy, gz, nx, ny)];
                }
                tile[(lz * SY + ly) * SX + lx] = value;
            }
        }
    }

    __syncthreads();

    const size_t gx = x0 + threadIdx.x;
    const size_t gy = y0 + threadIdx.y;

    if (gx < nx - 1 && gy < ny - 1) {
        const int lz0 = threadIdx.z * ZPER;
        size_t gz = z0 + static_cast<size_t>(lz0);
        if (gz < nz - 1) {
            int s = ((lz0 + 1) * SY + (threadIdx.y + 1)) * SX +
                    (threadIdx.x + 1);

            Real center = tile[s];
            Real top = tile[s + SX * SY];
            const Real left0 = tile[s - 1];
            const Real right0 = tile[s + 1];
            const Real front0 = tile[s - SX];
            const Real back0 = tile[s + SX];
            const Real bottom0 = tile[s - SX * SY];

            output[idx3(gx, gy, gz, nx, ny)] =
                div7(center + left0 + right0 + front0 + back0 + bottom0 + top);

            #pragma unroll
            for (int q = 1; q < ZPER; ++q) {
                gz += 1;
                if (gz >= nz - 1) break;
                s += SX * SY;

                const Real bottom = center;
                center = top;
                const Real left = tile[s - 1];
                const Real right = tile[s + 1];
                const Real front = tile[s - SX];
                const Real back = tile[s + SX];
                top = tile[s + SX * SY];

                output[idx3(gx, gy, gz, nx, ny)] =
                    div7(center + left + right + front + back + bottom + top);
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
    const size_t allocSize = (gridSize != 0) ? gridSize : 1;

    CUDA_CHECK(cudaSetDevice(0));

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, allocSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, allocSize * sizeof(Real)));

    // Initialize
    printf("Initializing grid...\n");
    {
        const int blockSize = 256;
        const int maxBlocks = 65535;
        const size_t wantedBlocks = (gridSize + blockSize - 1) / blockSize;
        const int blocks = static_cast<int>(std::min<size_t>(wantedBlocks, maxBlocks));
        initializeGridKernel<<<blocks, blockSize>>>(d_grid1, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());

        const size_t boundaryCells =
            2 * nx * ny +
            ((nz > 2) ? 2 * (nz - 2) * nx : 0) +
            ((nz > 2 && ny > 2) ? 2 * (nz - 2) * (ny - 2) : 0);
        const size_t wantedBoundaryBlocks =
            (boundaryCells + blockSize - 1) / blockSize;
        const int boundaryBlocks = static_cast<int>(
            std::min<size_t>(wantedBoundaryBlocks, maxBlocks));
        if (gridSize != 0 && boundaryBlocks > 0) {
            seedBoundaryKernel<<<boundaryBlocks, blockSize>>>(
                d_grid1, d_grid2, nx, ny, nz);
            CUDA_CHECK(cudaGetLastError());
        }
    }

    // Run stencil iterations
    printf("Running stencil computation...\n");
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::high_resolution_clock::now();

    constexpr int TILE_X = 16;
    constexpr int TILE_Y = 8;
    constexpr int TILE_Z = 16;
    constexpr int BLOCK_Z = 4;
    const dim3 block(TILE_X, TILE_Y, BLOCK_Z);

    const size_t ix = (nx > 2) ? nx - 2 : 0;
    const size_t iy = (ny > 2) ? ny - 2 : 0;
    const size_t iz = (nz > 2) ? nz - 2 : 0;
    const dim3 grid(static_cast<unsigned>((ix + TILE_X - 1) / TILE_X),
                    static_cast<unsigned>((iy + TILE_Y - 1) / TILE_Y),
                    static_cast<unsigned>((iz + TILE_Z - 1) / TILE_Z));

    if (ix != 0 && iy != 0 && iz != 0) {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIterationKernel<TILE_X, TILE_Y, TILE_Z, BLOCK_Z>
                    <<<grid, block>>>(d_grid1, d_grid2, nx, ny, nz);
            } else {
                stencilIterationKernel<TILE_X, TILE_Y, TILE_Z, BLOCK_Z>
                    <<<grid, block>>>(d_grid2, d_grid1, nx, ny, nz);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    const double elapsedSeconds = duration.count() / 1e6;

    long long elapsedMs = static_cast<long long>(duration.count() / 1000);
    if (elapsedMs == 0 && duration.count() > 0) elapsedMs = 1;
    printf("Computation time: %lld ms\n", elapsedMs);

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / elapsedSeconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy the final grid back to the host for output/validation
    std::vector<Real> finalGrid(gridSize);
    if (gridSize != 0) {
        const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real),
                              cudaMemcpyDeviceToHost));
    }

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
            CUDA_CHECK(cudaFree(d_grid1));
            CUDA_CHECK(cudaFree(d_grid2));
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    return 0;
}
