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

// 3D index calculation.  The x dimension is contiguous so adjacent CUDA
// threads access adjacent elements in global memory.
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
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

namespace {

constexpr unsigned kBlockX = 8;
constexpr unsigned kBlockY = 8;
constexpr unsigned kBlockZ = 8;
constexpr unsigned kTileX = kBlockX + 2;
constexpr unsigned kTileY = kBlockY + 2;
constexpr unsigned kTileZ = kBlockZ + 2;
constexpr unsigned kTileElements = kTileX * kTileY * kTileZ;

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

__device__ inline unsigned tileIndex(const unsigned x, const unsigned y, const unsigned z) noexcept {
    return (z * kTileY + y) * kTileX + x;
}

__device__ inline Real loadOrZero(const Real* __restrict__ input,
                                  const size_t x, const size_t y, const size_t z,
                                  const size_t nx, const size_t ny, const size_t nz) noexcept {
    return (x < nx && y < ny && z < nz) ? input[idx3(x, y, z, nx, ny)] : 0.0;
}

// One block computes an 8^3 output tile.  The 10^3 shared-memory tile holds
// the complete one-cell halo, reducing the seven global reads per output to
// a cooperative tile load while retaining coalesced x-direction accesses.
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    __shared__ Real tile[kTileElements];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned tz = threadIdx.z;

    const size_t blockX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t blockY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t blockZ = static_cast<size_t>(blockIdx.z) * kBlockZ;

    // Load the 8^3 core with all threads, then load each halo face, edge, and
    // corner exactly once.  This avoids integer division/modulo in the tile
    // load loop while keeping the dominant core loads coalesced.
    const unsigned sx = tx + 1;
    const unsigned sy = ty + 1;
    const unsigned sz = tz + 1;
    tile[tileIndex(sx, sy, sz)] = loadOrZero(input, blockX + sx, blockY + sy, blockZ + sz,
                                             nx, ny, nz);

    if (tx == 0) {
        tile[tileIndex(0, sy, sz)] = loadOrZero(input, blockX, blockY + sy, blockZ + sz,
                                                nx, ny, nz);
        tile[tileIndex(kTileX - 1, sy, sz)] = loadOrZero(
            input, blockX + kTileX - 1, blockY + sy, blockZ + sz, nx, ny, nz);
    }
    if (ty == 0) {
        tile[tileIndex(sx, 0, sz)] = loadOrZero(input, blockX + sx, blockY, blockZ + sz,
                                                nx, ny, nz);
        tile[tileIndex(sx, kTileY - 1, sz)] = loadOrZero(
            input, blockX + sx, blockY + kTileY - 1, blockZ + sz, nx, ny, nz);
    }
    if (tz == 0) {
        tile[tileIndex(sx, sy, 0)] = loadOrZero(input, blockX + sx, blockY + sy, blockZ,
                                                nx, ny, nz);
        tile[tileIndex(sx, sy, kTileZ - 1)] = loadOrZero(
            input, blockX + sx, blockY + sy, blockZ + kTileZ - 1, nx, ny, nz);
    }

    if (tx == 0 && ty == 0) {
        tile[tileIndex(0, 0, sz)] = loadOrZero(input, blockX, blockY, blockZ + sz,
                                               nx, ny, nz);
        tile[tileIndex(kTileX - 1, 0, sz)] = loadOrZero(
            input, blockX + kTileX - 1, blockY, blockZ + sz, nx, ny, nz);
        tile[tileIndex(0, kTileY - 1, sz)] = loadOrZero(
            input, blockX, blockY + kTileY - 1, blockZ + sz, nx, ny, nz);
        tile[tileIndex(kTileX - 1, kTileY - 1, sz)] = loadOrZero(
            input, blockX + kTileX - 1, blockY + kTileY - 1, blockZ + sz, nx, ny, nz);
    }
    if (tx == 0 && tz == 0) {
        tile[tileIndex(0, sy, 0)] = loadOrZero(input, blockX, blockY + sy, blockZ,
                                               nx, ny, nz);
        tile[tileIndex(kTileX - 1, sy, 0)] = loadOrZero(
            input, blockX + kTileX - 1, blockY + sy, blockZ, nx, ny, nz);
        tile[tileIndex(0, sy, kTileZ - 1)] = loadOrZero(
            input, blockX, blockY + sy, blockZ + kTileZ - 1, nx, ny, nz);
        tile[tileIndex(kTileX - 1, sy, kTileZ - 1)] = loadOrZero(
            input, blockX + kTileX - 1, blockY + sy, blockZ + kTileZ - 1, nx, ny, nz);
    }
    if (ty == 0 && tz == 0) {
        tile[tileIndex(sx, 0, 0)] = loadOrZero(input, blockX + sx, blockY, blockZ,
                                               nx, ny, nz);
        tile[tileIndex(sx, kTileY - 1, 0)] = loadOrZero(
            input, blockX + sx, blockY + kTileY - 1, blockZ, nx, ny, nz);
        tile[tileIndex(sx, 0, kTileZ - 1)] = loadOrZero(
            input, blockX + sx, blockY, blockZ + kTileZ - 1, nx, ny, nz);
        tile[tileIndex(sx, kTileY - 1, kTileZ - 1)] = loadOrZero(
            input, blockX + sx, blockY + kTileY - 1, blockZ + kTileZ - 1, nx, ny, nz);
    }
    if (tx == 0 && ty == 0 && tz == 0) {
        tile[tileIndex(0, 0, 0)] = loadOrZero(input, blockX, blockY, blockZ, nx, ny, nz);
        tile[tileIndex(kTileX - 1, 0, 0)] = loadOrZero(
            input, blockX + kTileX - 1, blockY, blockZ, nx, ny, nz);
        tile[tileIndex(0, kTileY - 1, 0)] = loadOrZero(
            input, blockX, blockY + kTileY - 1, blockZ, nx, ny, nz);
        tile[tileIndex(kTileX - 1, kTileY - 1, 0)] = loadOrZero(
            input, blockX + kTileX - 1, blockY + kTileY - 1, blockZ, nx, ny, nz);
        tile[tileIndex(0, 0, kTileZ - 1)] = loadOrZero(
            input, blockX, blockY, blockZ + kTileZ - 1, nx, ny, nz);
        tile[tileIndex(kTileX - 1, 0, kTileZ - 1)] = loadOrZero(
            input, blockX + kTileX - 1, blockY, blockZ + kTileZ - 1, nx, ny, nz);
        tile[tileIndex(0, kTileY - 1, kTileZ - 1)] = loadOrZero(
            input, blockX, blockY + kTileY - 1, blockZ + kTileZ - 1, nx, ny, nz);
        tile[tileIndex(kTileX - 1, kTileY - 1, kTileZ - 1)] = loadOrZero(
            input, blockX + kTileX - 1, blockY + kTileY - 1, blockZ + kTileZ - 1,
            nx, ny, nz);
    }
    __syncthreads();

    const size_t x = blockX + tx + 1;
    const size_t y = blockY + ty + 1;
    const size_t z = blockZ + tz + 1;
    if (x < nx - 1 && y < ny - 1 && z < nz - 1) {
        const unsigned center = tileIndex(tx + 1, ty + 1, tz + 1);
        output[idx3(x, y, z, nx, ny)] =
            (tile[center] +
             tile[center - 1] + tile[center + 1] +
             tile[center - kTileX] + tile[center + kTileX] +
             tile[center - kTileX * kTileY] + tile[center + kTileX * kTileY]) / 7.0;
    }
}

void stencilIteration(const Real* input, Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) {
        return;
    }

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(
        static_cast<unsigned>((nx - 2 + kBlockX - 1) / kBlockX),
        static_cast<unsigned>((ny - 2 + kBlockY - 1) / kBlockY),
        static_cast<unsigned>((nz - 2 + kBlockZ - 1) / kBlockZ));
    stencilKernel<<<grid, block>>>(input, output, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void validateDimensions(const size_t nx, const size_t ny, const size_t nz, const int iterations) {
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be positive and iterations non-negative.\n");
        std::exit(EXIT_FAILURE);
    }

    constexpr size_t maxSize = static_cast<size_t>(-1);
    if (nx > maxSize / ny || nx * ny > maxSize / nz) {
        std::fprintf(stderr, "Grid size is too large.\n");
        std::exit(EXIT_FAILURE);
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
    validateDimensions(nx, ny, nz, iterations);
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;
    if (gridSize > static_cast<size_t>(-1) / sizeof(Real)) {
        std::fprintf(stderr, "Grid allocation is too large.\n");
        return 1;
    }
    const size_t gridBytes = gridSize * sizeof(Real);
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    // Keep both buffers resident on the GPU for the complete iteration loop.
    // The second copy also preserves all boundary values before the first
    // stencil launch; every iteration writes only the interior.
    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, gridBytes));
    CUDA_CHECK(cudaMemcpy(deviceGrid1, grid1.data(), gridBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(deviceGrid2, deviceGrid1, gridBytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilIteration(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
    }

    // Kernel launches are asynchronous; the synchronization makes the
    // measured interval include all device work and catches runtime errors.
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    const double durationMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", durationMs);
    
    // Calculate performance metrics
    const size_t interiorX = nx > 2 ? nx - 2 : 0;
    const size_t interiorY = ny > 2 ? ny - 2 : 0;
    const size_t interiorZ = nz > 2 ? nz - 2 : 0;
    const double cellUpdates = static_cast<double>(interiorX) *
                               static_cast<double>(interiorY) *
                               static_cast<double>(interiorZ) * iterations;
    const double mcups = durationMs > 0.0
        ? cellUpdates / (durationMs / 1000.0) / 1e6
        : 0.0;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real>& finalGridStorage = (iterations % 2 == 0) ? grid1 : grid2;
    Real* finalDeviceGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
    CUDA_CHECK(cudaMemcpy(finalGridStorage.data(), finalDeviceGrid, gridBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));

    const std::vector<Real>& finalGrid = finalGridStorage;
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
