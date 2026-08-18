#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                  const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* const operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call) \
    do { \
        const cudaError_t cudaStatus = (call); \
        if (cudaStatus != cudaSuccess) { \
            cudaFailure(cudaStatus, #call); \
        } \
    } while (false)

// The x dimension is contiguous in memory.  A 32 x 4 x 2 tile therefore gives
// coalesced loads while sharing the values reused by neighboring stencil
// points.  The two z planes keep the shared-memory footprint small enough for
// high occupancy on both current and older NVIDIA GPUs.
constexpr int kBlockX = 32;
constexpr int kBlockY = 4;
constexpr int kBlockZ = 2;

__global__ void initializeGridKernel(Real* const grid, const size_t total) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t index = first; index < total; index += stride) {
        grid[index] = static_cast<Real>(index % 19);
    }
}

__global__ void stencilIterationKernel(const Real* __restrict__ input,
                                       Real* __restrict__ output,
                                       const size_t nx,
                                       const size_t ny,
                                       const size_t nz) {
    constexpr int sharedX = kBlockX + 2;
    constexpr int sharedY = kBlockY + 2;
    constexpr int sharedPlane = sharedX * sharedY;
    __shared__ Real tile[sharedPlane * (kBlockZ + 2)];

    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);
    const int tz = static_cast<int>(threadIdx.z);
    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + tx;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + ty;
    const size_t z = static_cast<size_t>(blockIdx.z) * kBlockZ + tz;
    const bool inBounds = x < nx && y < ny && z < nz;

    const int tileIndex = (tz + 1) * sharedPlane + (ty + 1) * sharedX + tx + 1;
    if (inBounds) {
        tile[tileIndex] = input[idx3(x, y, z, nx, ny)];
    }

    // Load the six halo faces.  Only faces are needed for a 7-point stencil;
    // edges and corners never participate in the calculation.
    if (tx == 0 && inBounds && x > 0) {
        tile[tileIndex - 1] = input[idx3(x - 1, y, z, nx, ny)];
    }
    if (tx == kBlockX - 1 && inBounds && x + 1 < nx) {
        tile[tileIndex + 1] = input[idx3(x + 1, y, z, nx, ny)];
    }
    if (ty == 0 && inBounds && y > 0) {
        tile[tileIndex - sharedX] = input[idx3(x, y - 1, z, nx, ny)];
    }
    if (ty == kBlockY - 1 && inBounds && y + 1 < ny) {
        tile[tileIndex + sharedX] = input[idx3(x, y + 1, z, nx, ny)];
    }
    if (tz == 0 && inBounds && z > 0) {
        tile[tileIndex - sharedPlane] = input[idx3(x, y, z - 1, nx, ny)];
    }
    if (tz == kBlockZ - 1 && inBounds && z + 1 < nz) {
        tile[tileIndex + sharedPlane] = input[idx3(x, y, z + 1, nx, ny)];
    }

    __syncthreads();

    if (!inBounds) {
        return;
    }

    const size_t index = idx3(x, y, z, nx, ny);
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[index] = input[index];
    } else {
        const Real center = tile[tileIndex];
        const Real left = tile[tileIndex - 1];
        const Real right = tile[tileIndex + 1];
        const Real front = tile[tileIndex - sharedX];
        const Real back = tile[tileIndex + sharedX];
        const Real bottom = tile[tileIndex - sharedPlane];
        const Real top = tile[tileIndex + sharedPlane];
        output[index] = (center + left + right + front + back + bottom + top) / 7.0;
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
    
    size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);

    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    // All grid storage used by the stencil lives on the GPU.  The host
    // vectors remain only for the final result and the existing reporting
    // interface.
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp deviceProperties{};
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, 0));

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), gridBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), gridBytes));

    // Initialize
    printf("Initializing grid...\n");
    constexpr int kInitializationThreads = 256;
    const size_t requestedInitializationBlocks =
        (gridSize + kInitializationThreads - 1) / kInitializationThreads;
    const size_t availableInitializationBlocks =
        std::max<size_t>(1, static_cast<size_t>(deviceProperties.multiProcessorCount) * 32);
    const unsigned int initializationBlocks = static_cast<unsigned int>(
        std::max<size_t>(1, std::min(requestedInitializationBlocks, availableInitializationBlocks)));
    initializeGridKernel<<<initializationBlocks, kInitializationThreads>>>(deviceGrid1, gridSize);
    CUDA_CHECK(cudaGetLastError());

    // Run stencil iterations
    printf("Running stencil computation...\n");
    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(static_cast<unsigned int>((nx + kBlockX - 1) / kBlockX),
                    static_cast<unsigned int>((ny + kBlockY - 1) / kBlockY),
                    static_cast<unsigned int>((nz + kBlockZ - 1) / kBlockZ));

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIterationKernel<<<grid, block>>>(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilIterationKernel<<<grid, block>>>(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    const long durationMilliseconds = static_cast<long>(elapsedMilliseconds);
    
    printf("Computation time: %ld ms\n", durationMilliseconds);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    const double elapsedSeconds = std::max(static_cast<double>(elapsedMilliseconds) / 1000.0, 1.0e-9);
    double mcups = cellUpdates / elapsedSeconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    Real* finalDeviceGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, gridBytes, cudaMemcpyDeviceToHost));
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    // Validation
    int result = 0;
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));
    return result;
}
