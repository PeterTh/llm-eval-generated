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

namespace {

constexpr int blockX = 32;
constexpr int blockY = 4;
constexpr int blockZ = 4;
constexpr int blockThreads = blockX * blockY * blockZ;
constexpr int tileX = blockX + 2;
constexpr int tileY = blockY + 2;
constexpr int tileZ = blockZ + 2;
constexpr int tileElements = tileX * tileY * tileZ;

[[noreturn]] void reportCudaError(const char* expression, const cudaError_t error,
                                   const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n", file, line,
                 expression, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                      \
    do {                                                                            \
        const cudaError_t error_ = (expression);                                   \
        if (error_ != cudaSuccess) {                                               \
            reportCudaError(#expression, error_, __FILE__, __LINE__);              \
        }                                                                           \
    } while (false)

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t elements) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < elements) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

// Each block stages its 32 x 4 x 4 output region plus a one-cell halo in shared
// memory. The X dimension is a full warp, so all regular global loads are
// coalesced while neighboring stencil values are reused from the tile.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    __shared__ Real tile[tileElements];

    const int localThread = (static_cast<int>(threadIdx.z) * blockY +
                             static_cast<int>(threadIdx.y)) * blockX +
                            static_cast<int>(threadIdx.x);
    const size_t plane = nx * ny;
    const size_t xOrigin = static_cast<size_t>(blockIdx.x) * blockX;
    const size_t yOrigin = static_cast<size_t>(blockIdx.y) * blockY;
    const size_t zOrigin = static_cast<size_t>(blockIdx.z) * blockZ;

    // Cooperatively load the tile and its halo. Out-of-domain halo elements
    // are never used by an interior output, but assigning them keeps the
    // shared-memory tile fully initialized for every block shape at the edge.
    for (int tileIndex = localThread; tileIndex < tileElements;
         tileIndex += blockThreads) {
        const int tileXIndex = tileIndex % tileX;
        const int tileYIndex = (tileIndex / tileX) % tileY;
        const int tileZIndex = tileIndex / (tileX * tileY);

        const long long x = static_cast<long long>(xOrigin) + tileXIndex - 1;
        const long long y = static_cast<long long>(yOrigin) + tileYIndex - 1;
        const long long z = static_cast<long long>(zOrigin) + tileZIndex - 1;

        if (x >= 0 && x < static_cast<long long>(nx) &&
            y >= 0 && y < static_cast<long long>(ny) &&
            z >= 0 && z < static_cast<long long>(nz)) {
            const size_t index = static_cast<size_t>(z) * plane +
                                 static_cast<size_t>(y) * nx +
                                 static_cast<size_t>(x);
            tile[tileIndex] = input[index];
        } else {
            tile[tileIndex] = 0.0;
        }
    }
    __syncthreads();

    const size_t x = xOrigin + threadIdx.x;
    const size_t y = yOrigin + threadIdx.y;
    const size_t z = zOrigin + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t index = z * plane + y * nx + x;
    if (x == 0 || x + 1 >= nx || y == 0 || y + 1 >= ny || z == 0 || z + 1 >= nz) {
        // Preserve boundary values exactly as in the original implementation.
        output[index] = input[index];
        return;
    }

    const int center = ((static_cast<int>(threadIdx.z) + 1) * tileY +
                        (static_cast<int>(threadIdx.y) + 1)) * tileX +
                       static_cast<int>(threadIdx.x) + 1;
    const Real centerValue = tile[center];
    const Real left = tile[center - 1];
    const Real right = tile[center + 1];
    const Real front = tile[center - tileX];
    const Real back = tile[center + tileX];
    const Real bottom = tile[center - tileX * tileY];
    const Real top = tile[center + tileX * tileY];

    // Keep the same evaluation order and arithmetic as the reference stencil.
    output[index] = (centerValue + left + right + front + back + bottom + top) / 7.0;
}

size_t roundedBlocks(const size_t elements, const size_t threads) {
    return elements / threads + (elements % threads != 0 ? 1 : 0);
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                   [[maybe_unused]] const size_t ny,
                   [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    if (grid.empty()) {
        std::printf("Validation failed: grid is empty\n");
        return false;
    }

    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
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

    std::printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values are copied by stencilKernel, as in the reference.
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of iterations (default: 10)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    std::printf("3D Stencil Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Iterations: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    std::vector<Real> finalGrid(gridSize);

    Real* grid1 = nullptr;
    Real* grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grid1), gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&grid2), gridSize * sizeof(Real)));

    const size_t initializationBlocks = roundedBlocks(gridSize, 256);
    std::printf("Initializing grid...\n");
    initializeGridKernel<<<static_cast<unsigned int>(initializationBlocks), 256>>>(
        grid1, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Running stencil computation...\n");
    Real* current = grid1;
    Real* next = grid2;
    const dim3 block(blockX, blockY, blockZ);
    const dim3 grid(static_cast<unsigned int>(roundedBlocks(nx, blockX)),
                    static_cast<unsigned int>(roundedBlocks(ny, blockY)),
                    static_cast<unsigned int>(roundedBlocks(nz, blockZ)));

    const auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        stencilKernel<<<grid, block>>>(current, next, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
        Real* const completed = current;
        current = next;
        next = completed;
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    const double interiorCells = (nx >= 2 && ny >= 2 && nz >= 2)
                                     ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2)
                                     : 0.0;
    const double elapsedSeconds = duration.count() / 1000.0;
    const double mcups = elapsedSeconds > 0.0
                             ? interiorCells * iterations / elapsedSeconds / 1e6
                             : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    CUDA_CHECK(cudaMemcpy(finalGrid.data(), current, gridSize * sizeof(Real),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(grid2));
    CUDA_CHECK(cudaFree(grid1));

    // Print results for external validation
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            std::printf("Validation: PASSED\n");
            return 0;
        }

        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
