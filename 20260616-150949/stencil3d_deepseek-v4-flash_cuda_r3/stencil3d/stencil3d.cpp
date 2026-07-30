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

// ========== CUDA Error Checking ==========
#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                \
                __FILE__, __LINE__, cudaGetErrorString(err));        \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// ========== CUDA Kernels ==========

// Grid initialization kernel (1D grid)
__global__ void initGridKernel(Real* grid, const size_t N) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil kernel with shared memory tiling
// Each block covers a (blockDim.x, blockDim.y, blockDim.z) tile plus a 1-element
// halo in each dimension. Halo regions are loaded cooperatively by boundary
// threads to reduce global memory traffic.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    extern __shared__ Real s_data[];

    constexpr int HALO = 1;
    const int TILE_X = blockDim.x + 2 * HALO;
    const int TILE_Y = blockDim.y + 2 * HALO;

    const int tx = threadIdx.x + HALO;  // 1-based in tile
    const int ty = threadIdx.y + HALO;
    const int tz = threadIdx.z + HALO;

    const size_t gx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t gy = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t gz = blockIdx.z * blockDim.z + threadIdx.z;
    const size_t nx_ny = nx * ny;

    // ---- Load interior point into shared memory ----
    if (gx < nx && gy < ny && gz < nz) {
        const size_t gidx = gz * nx_ny + gy * nx + gx;
        s_data[tz * TILE_Y * TILE_X + ty * TILE_X + tx] = input[gidx];
    }

    // ---- Load halos cooperatively (avoid out-of-bounds reads) ----
    // Left halo  (x = 0 in tile)
    if (threadIdx.x == 0 && gx > 0 && gx - 1 < nx && gy < ny && gz < nz) {
        const size_t gidx = gz * nx_ny + gy * nx + (gx - 1);
        s_data[tz * TILE_Y * TILE_X + ty * TILE_X + 0] = input[gidx];
    }
    // Right halo (x = blockDim.x + 1 in tile)
    if (threadIdx.x == blockDim.x - 1 && gx + 1 < nx && gy < ny && gz < nz) {
        const size_t gidx = gz * nx_ny + gy * nx + (gx + 1);
        s_data[tz * TILE_Y * TILE_X + ty * TILE_X + (blockDim.x + 1)] = input[gidx];
    }
    // Front halo (y = 0 in tile)
    if (threadIdx.y == 0 && gy > 0 && gx < nx && gz < nz) {
        const size_t gidx = gz * nx_ny + (gy - 1) * nx + gx;
        s_data[tz * TILE_Y * TILE_X + 0 * TILE_X + tx] = input[gidx];
    }
    // Back halo  (y = blockDim.y + 1 in tile)
    if (threadIdx.y == blockDim.y - 1 && gy + 1 < ny && gx < nx && gz < nz) {
        const size_t gidx = gz * nx_ny + (gy + 1) * nx + gx;
        s_data[tz * TILE_Y * TILE_X + (blockDim.y + 1) * TILE_X + tx] = input[gidx];
    }
    // Bottom halo (z = 0 in tile)
    if (threadIdx.z == 0 && gz > 0 && gx < nx && gy < ny) {
        const size_t gidx = (gz - 1) * nx_ny + gy * nx + gx;
        s_data[0 * TILE_Y * TILE_X + ty * TILE_X + tx] = input[gidx];
    }
    // Top halo  (z = blockDim.z + 1 in tile)
    if (threadIdx.z == blockDim.z - 1 && gz + 1 < nz && gx < nx && gy < ny) {
        const size_t gidx = (gz + 1) * nx_ny + gy * nx + gx;
        s_data[(blockDim.z + 1) * TILE_Y * TILE_X + ty * TILE_X + tx] = input[gidx];
    }

    __syncthreads();

    // Out-of-bounds guard
    if (gx >= nx || gy >= ny || gz >= nz) return;

    const size_t idx = gz * nx_ny + gy * nx + gx;
    const int sidx = tz * TILE_Y * TILE_X + ty * TILE_X + tx;

    // Boundary points: copy input value to output
    if (gx == 0 || gx == nx - 1 || gy == 0 || gy == ny - 1 ||
        gz == 0 || gz == nz - 1) {
        output[idx] = s_data[sidx];
        return;
    }

    // Interior: 7-point averaging stencil (all reads from shared memory)
    const Real center = s_data[sidx];
    const Real left   = s_data[sidx - 1];
    const Real right  = s_data[sidx + 1];
    const Real front  = s_data[sidx - TILE_X];
    const Real back   = s_data[sidx + TILE_X];
    const Real bottom = s_data[sidx - TILE_Y * TILE_X];
    const Real top    = s_data[sidx + TILE_Y * TILE_X];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

// ========== Host Helper Functions ==========

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

    // ---- Allocate device memory ----
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));

    // ---- Initialize grid on device ----
    printf("Initializing grid...\n");
    {
        constexpr int initBlockSize = 256;
        const int initGridSize = static_cast<int>((gridSize + initBlockSize - 1) / initBlockSize);
        initGridKernel<<<initGridSize, initBlockSize>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Configure stencil kernel dimensions ----
    // 16 x 16 x 4 = 1024 threads/block, good occupancy on modern GPUs
    constexpr dim3 block3d(16, 16, 4);
    const dim3 grid3d(
        static_cast<unsigned int>((nx + block3d.x - 1) / block3d.x),
        static_cast<unsigned int>((ny + block3d.y - 1) / block3d.y),
        static_cast<unsigned int>((nz + block3d.z - 1) / block3d.z)
    );

    // Shared memory: (blockDim.x + 2) * (blockDim.y + 2) * (blockDim.z + 2) doubles
    const size_t sharedMemSize =
        static_cast<size_t>(block3d.x + 2) *
        static_cast<size_t>(block3d.y + 2) *
        static_cast<size_t>(block3d.z + 2) * sizeof(Real);

    // ---- Run stencil iterations (GPU-timed) ----
    printf("Running stencil computation...\n");

    cudaEvent_t startEvent = nullptr, stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<grid3d, block3d, sharedMemSize>>>(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilKernel<<<grid3d, block3d, sharedMemSize>>>(d_grid2, d_grid1, nx, ny, nz);
        }
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float gpuTimeMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&gpuTimeMs, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    // Check for kernel launch errors
    CUDA_CHECK(cudaGetLastError());

    printf("Computation time: %f ms\n", static_cast<double>(gpuTimeMs));

    // Calculate performance metrics
    const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    const double mcups = cellUpdates / (gpuTimeMs / 1000.0) / 1.0e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // ---- Copy final result back to host ----
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));

    // ---- Print results for external validation ----
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    // ---- Validation ----
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(finalGrid, nx, ny, nz);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d_grid1));
            CUDA_CHECK(cudaFree(d_grid2));
            return 1;
        }
    }

    // ---- Cleanup ----
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    return 0;
}
