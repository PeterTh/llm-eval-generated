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

// The x dimension spans a complete warp so global loads are coalesced.  The
// tile is small enough to allow two resident blocks on current GPUs while
// retaining all seven stencil neighbors in shared memory.
constexpr int BlockX = 32;
constexpr int BlockY = 4;
constexpr int BlockZ = 2;
constexpr int TileX = BlockX + 2;
constexpr int TileY = BlockY + 2;
constexpr int TileZ = BlockZ + 2;
constexpr int ThreadsPerBlock = BlockX * BlockY * BlockZ;

// 3D index calculation (x is the contiguous dimension).
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y,
                                                 const size_t z, const size_t nx,
                                                 const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error) noexcept {
    std::fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(const cudaError_t error, const char* operation) noexcept {
    if (error != cudaSuccess) {
        cudaFailure(operation, error);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t elementCount) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t linear = first; linear < elementCount; linear += stride) {
        grid[linear] = static_cast<Real>(linear % 19);
    }
}

// Each block loads a (BlockX+2) x (BlockY+2) x (BlockZ+2) tile.  Loading the
// halo once changes the seven global reads per output into shared-memory
// reads, while the grid-stride loops also support dimensions beyond CUDA's
// grid limits without changing the launch configuration.
__global__ __launch_bounds__(ThreadsPerBlock, 3)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output,
                   const size_t nx, const size_t ny, const size_t nz) {
    extern __shared__ Real tile[];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const size_t thread = static_cast<size_t>(tx) +
                          static_cast<size_t>(BlockX) *
                              (static_cast<size_t>(ty) + static_cast<size_t>(BlockY) * tz);
    const size_t blockThreads = static_cast<size_t>(ThreadsPerBlock);
    const size_t tilesX = (nx + BlockX - 1) / BlockX;
    const size_t tilesY = (ny + BlockY - 1) / BlockY;
    const size_t tilesZ = (nz + BlockZ - 1) / BlockZ;
    const size_t sharedTileSize = static_cast<size_t>(TileX) * TileY * TileZ;

    for (size_t tileIndexZ = blockIdx.z; tileIndexZ < tilesZ; tileIndexZ += gridDim.z) {
        for (size_t tileIndexY = blockIdx.y; tileIndexY < tilesY; tileIndexY += gridDim.y) {
            for (size_t tileIndexX = blockIdx.x; tileIndexX < tilesX; tileIndexX += gridDim.x) {
                const size_t originX = tileIndexX * BlockX;
                const size_t originY = tileIndexY * BlockY;
                const size_t originZ = tileIndexZ * BlockZ;

                for (size_t sharedLinear = thread; sharedLinear < sharedTileSize;
                     sharedLinear += blockThreads) {
                    const size_t sx = sharedLinear % TileX;
                    const size_t sy = (sharedLinear / TileX) % TileY;
                    const size_t sz = sharedLinear / (static_cast<size_t>(TileX) * TileY);

                    // Avoid unsigned underflow for the outermost halo while
                    // keeping the common, in-range path branch-free.
                    const bool xValid = sx == 0 ? originX != 0 : originX + sx - 1 < nx;
                    const bool yValid = sy == 0 ? originY != 0 : originY + sy - 1 < ny;
                    const bool zValid = sz == 0 ? originZ != 0 : originZ + sz - 1 < nz;
                    const size_t globalX = sx == 0 && originX == 0 ? 0 : originX + sx - 1;
                    const size_t globalY = sy == 0 && originY == 0 ? 0 : originY + sy - 1;
                    const size_t globalZ = sz == 0 && originZ == 0 ? 0 : originZ + sz - 1;

                    if (xValid && yValid && zValid) {
                        tile[sharedLinear] = input[idx3(globalX, globalY, globalZ, nx, ny)];
                    } else {
                        tile[sharedLinear] = 0.0;
                    }
                }
                __syncthreads();

                const size_t x = originX + static_cast<size_t>(tx);
                const size_t y = originY + static_cast<size_t>(ty);
                const size_t z = originZ + static_cast<size_t>(tz);
                if (x < nx && y < ny && z < nz) {
                    const size_t globalIndex = idx3(x, y, z, nx, ny);
                    if (x == 0 || x + 1 >= nx || y == 0 || y + 1 >= ny ||
                        z == 0 || z + 1 >= nz) {
                        output[globalIndex] = input[globalIndex];
                    } else {
                        const size_t center =
                            (static_cast<size_t>(tz + 1) * TileY + (ty + 1)) * TileX + (tx + 1);
                        const Real centerValue = tile[center];
                        const Real left = tile[center - 1];
                        const Real right = tile[center + 1];
                        const Real front = tile[center - TileX];
                        const Real back = tile[center + TileX];
                        const Real bottom = tile[center - static_cast<size_t>(TileX) * TileY];
                        const Real top = tile[center + static_cast<size_t>(TileX) * TileY];

                        output[globalIndex] =
                            (centerValue + left + right + front + back + bottom + top) / 7.0;
                    }
                }
                // Threads in a partial block may have no output, but all
                // threads reach this barrier before the tile is reused.
                __syncthreads();
            }
        }
    }
}

void initializeGrid(Real* deviceGrid, const size_t elementCount) {
    if (elementCount == 0) {
        return;
    }

    constexpr unsigned int threads = 256;
    constexpr unsigned int maxBlocks = 65535;
    const size_t requiredBlocks = (elementCount + threads - 1) / threads;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min(requiredBlocks, static_cast<size_t>(maxBlocks)));
    initializeGridKernel<<<blocks, threads>>>(deviceGrid, elementCount);
    CUDA_CHECK(cudaGetLastError());
}

void stencilIteration(const Real* input, Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const size_t tilesX = (nx + BlockX - 1) / BlockX;
    const size_t tilesY = (ny + BlockY - 1) / BlockY;
    const size_t tilesZ = (nz + BlockZ - 1) / BlockZ;

    // CUDA permits a much larger x grid than y/z.  The kernel's grid-stride
    // loops cover the uncommon case where any dimension exceeds its limit.
    constexpr unsigned int maxGridX = 2147483647u;
    constexpr unsigned int maxGridYZ = 65535u;
    const dim3 grid(
        static_cast<unsigned int>(std::min(tilesX, static_cast<size_t>(maxGridX))),
        static_cast<unsigned int>(std::min(tilesY, static_cast<size_t>(maxGridYZ))),
        static_cast<unsigned int>(std::min(tilesZ, static_cast<size_t>(maxGridYZ))));
    constexpr dim3 block(BlockX, BlockY, BlockZ);
    constexpr size_t sharedBytes = static_cast<size_t>(TileX) * TileY * TileZ * sizeof(Real);

    stencilKernel<<<grid, block, sharedBytes>>>(input, output, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                   [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

    if (nx == 0 || ny == 0 || nz == 0) {
        std::fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);

    // Keep both buffers resident on the GPU.  Only the final buffer is copied
    // back, so host/device transfers do not participate in the timed loop.
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), gridBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), gridBytes));
    // std::vector value-initializes both original buffers.  Clearing the
    // unused buffer preserves that behavior for zero or negative iteration
    // counts without affecting the timed stencil loop.
    CUDA_CHECK(cudaMemset(deviceGrid2, 0, gridBytes));

    printf("Initializing grid...\n");
    initializeGrid(deviceGrid1, gridSize);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilIteration(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* finalDeviceGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, gridBytes, cudaMemcpyDeviceToHost));

    // Print results for external validation
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));

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
