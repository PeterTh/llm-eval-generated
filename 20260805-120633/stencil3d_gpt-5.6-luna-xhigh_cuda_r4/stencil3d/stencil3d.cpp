#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

namespace {

constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 4;
constexpr int kTileX = kBlockX + 2;
constexpr int kTileY = kBlockY + 2;
constexpr int kTileZ = kBlockZ + 2;
constexpr int kThreadsPerBlock = kBlockX * kBlockY * kBlockZ;

inline void checkCuda(const cudaError_t error, const char* const operation,
                      const char* const file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n",
                     operation, file, line, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

__device__ __forceinline__ int tileIndex(const int x, const int y, const int z) {
    return (z * kTileY + y) * kTileX + x;
}

// Initialize in device memory so the benchmark uses the GPU for the complete grid setup.
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t elementCount) {
    const size_t threadId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t index = threadId; index < elementCount; index += stride) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

// 7-point stencil computation. Each block loads its output tile and one-cell halo into
// shared memory, reducing the seven global reads per output cell to the tile surface cost.
__global__ void stencilIterationKernel(const Real* __restrict__ input,
                                       Real* __restrict__ output,
                                       const size_t nx, const size_t ny, const size_t nz) {
    __shared__ Real tile[kTileX * kTileY * kTileZ];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const int threadId = (tz * kBlockY + ty) * kBlockX + tx;

    const long long baseX = static_cast<long long>(blockIdx.x) * kBlockX;
    const long long baseY = static_cast<long long>(blockIdx.y) * kBlockY;
    const long long baseZ = static_cast<long long>(blockIdx.z) * kBlockZ;

    // There are more tile elements than threads, so each thread loads one or two
    // elements. The signed coordinates make edge blocks safe for arbitrary dimensions.
    constexpr int tileElementCount = kTileX * kTileY * kTileZ;
    for (int linear = threadId; linear < tileElementCount; linear += kThreadsPerBlock) {
        const int sx = linear % kTileX;
        const int sy = (linear / kTileX) % kTileY;
        const int sz = linear / (kTileX * kTileY);
        const long long gx = baseX + sx - 1;
        const long long gy = baseY + sy - 1;
        const long long gz = baseZ + sz - 1;

        if (gx >= 0 && gy >= 0 && gz >= 0 &&
            static_cast<size_t>(gx) < nx && static_cast<size_t>(gy) < ny && static_cast<size_t>(gz) < nz) {
            tile[tileIndex(sx, sy, sz)] = input[idx3(static_cast<size_t>(gx), static_cast<size_t>(gy),
                                                    static_cast<size_t>(gz), nx, ny)];
        } else {
            tile[tileIndex(sx, sy, sz)] = 0.0;
        }
    }

    __syncthreads();

    const size_t x = static_cast<size_t>(baseX + tx);
    const size_t y = static_cast<size_t>(baseY + ty);
    const size_t z = static_cast<size_t>(baseZ + tz);
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t index = idx3(x, y, z, nx, ny);
    const int sx = tx + 1;
    const int sy = ty + 1;
    const int sz = tz + 1;

    // Boundaries are copied exactly as in the scalar implementation.
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[index] = input[index];
        return;
    }

    const Real center = tile[tileIndex(sx, sy, sz)];
    const Real left = tile[tileIndex(sx - 1, sy, sz)];
    const Real right = tile[tileIndex(sx + 1, sy, sz)];
    const Real front = tile[tileIndex(sx, sy - 1, sz)];
    const Real back = tile[tileIndex(sx, sy + 1, sz)];
    const Real bottom = tile[tileIndex(sx, sy, sz - 1)];
    const Real top = tile[tileIndex(sx, sy, sz + 1)];

    output[index] = (center + left + right + front + back + bottom + top) / 7.0;
}

void launchStencilIteration(const Real* input, Real* output,
                            const size_t nx, const size_t ny, const size_t nz) {
    const auto ceilDiv = [](const size_t value, const size_t divisor) {
        return (value + divisor - 1) / divisor;
    };

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(static_cast<unsigned int>(std::max<size_t>(1, ceilDiv(nx, kBlockX))),
                    static_cast<unsigned int>(std::max<size_t>(1, ceilDiv(ny, kBlockY))),
                    static_cast<unsigned int>(std::max<size_t>(1, ceilDiv(nz, kBlockZ))));
    stencilIterationKernel<<<grid, block>>>(input, output, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    if (grid.empty()) {
        printf("Value range: empty grid\n");
        return true;
    }
    
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

} // namespace

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
    
    // Allocate host grids (double buffering) and the corresponding device grids.
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    if (gridSize != 0) {
        const size_t gridBytes = gridSize * sizeof(Real);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), gridBytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), gridBytes));
        // Preserve the value-initialized state of the original second host buffer
        // for the zero/negative-iteration edge case.
        CUDA_CHECK(cudaMemset(deviceGrid2, 0, gridBytes));
    }
    
    // Initialize
    printf("Initializing grid...\n");
    if (gridSize != 0) {
        constexpr unsigned int initThreads = 256;
        const size_t requestedBlocks = (gridSize + initThreads - 1) / initThreads;
        const unsigned int initBlocks = static_cast<unsigned int>(std::max<size_t>(1, std::min<size_t>(requestedBlocks, 65535)));
        initializeGridKernel<<<initBlocks, initThreads>>>(deviceGrid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            launchStencilIteration(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            launchStencilIteration(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double interiorCells = (nx > 2 && ny > 2 && nz > 2)
        ? static_cast<double>(nx - 2) * static_cast<double>(ny - 2) * static_cast<double>(nz - 2)
        : 0.0;
    const double elapsedSeconds = duration.count() / 1000.0;
    const double mcups = elapsedSeconds > 0.0
        ? interiorCells * static_cast<double>(std::max(iterations, 0)) / elapsedSeconds / 1e6
        : 0.0;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy only the final device buffer back to the host for reporting/validation.
    if (gridSize != 0) {
        const Real* deviceFinalGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
        Real* hostFinalGrid = (iterations % 2 == 0) ? grid1.data() : grid2.data();
        CUDA_CHECK(cudaMemcpy(hostFinalGrid, deviceFinalGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }

    if (deviceGrid1 != nullptr) {
        CUDA_CHECK(cudaFree(deviceGrid1));
        CUDA_CHECK(cudaFree(deviceGrid2));
    }

    // Print results for external validation
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;

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
