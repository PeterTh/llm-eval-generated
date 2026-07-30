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

// Block dimensions for the 3D stencil kernel
#define BX 32
#define BY 4
#define BZ 4

// Shared memory dimensions (tile + 1-cell halo on each side)
#define SX (BX + 2)
#define SY (BY + 2)
#define SZ (BZ + 2)
#define SMEM_SIZE (SX * SY * SZ)
#define BLOCK_THREADS (BX * BY * BZ)

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Initialize grid on device: grid[idx] = (idx % 19) * 1.0
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t total) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        grid[idx] = (Real)(idx % 19);
    }
}

// 7-point stencil kernel with shared memory tiling
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    __shared__ Real smem[SMEM_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;
    const int tid = tz * BY * BX + ty * BX + tx;

    const size_t plane = nx * ny;

    // Load shared memory tile + halo using flat iteration
    for (int i = tid; i < SMEM_SIZE; i += BLOCK_THREADS) {
        int sz = i / (SY * SX);
        int rem = i % (SY * SX);
        int sy = rem / SX;
        int sx = rem % SX;

        int gix = blockIdx.x * BX + sx;
        int giy = blockIdx.y * BY + sy;
        int giz = blockIdx.z * BZ + sz;

        if (gix < (int)nx && giy < (int)ny && giz < (int)nz) {
            smem[i] = input[(size_t)giz * plane + (size_t)giy * nx + (size_t)gix];
        }
    }

    __syncthreads();

    // Compute global coordinates of the point this thread processes
    const int gx = blockIdx.x * BX + tx + 1;
    const int gy = blockIdx.y * BY + ty + 1;
    const int gz = blockIdx.z * BZ + tz + 1;

    // Only compute interior points
    if (gx < (int)(nx - 1) && gy < (int)(ny - 1) && gz < (int)(nz - 1)) {
        // Shared memory indices for this thread's point
        const int sx = tx + 1;
        const int sy = ty + 1;
        const int sz = tz + 1;
        const int s_idx = sz * SY * SX + sy * SX + sx;

        const Real center = smem[s_idx];
        const Real left   = smem[s_idx - 1];
        const Real right  = smem[s_idx + 1];
        const Real front  = smem[s_idx - SX];
        const Real back   = smem[s_idx + SX];
        const Real bottom = smem[s_idx - SY * SX];
        const Real top    = smem[s_idx + SY * SX];

        output[(size_t)gz * plane + (size_t)gy * nx + (size_t)gx] =
            (center + left + right + front + back + bottom + top) * (1.0 / 7.0);
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
    size_t gridBytes = gridSize * sizeof(Real);

    // Allocate device memory (double buffering)
    Real *d_grid1, *d_grid2;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes));

    // Initialize grid on device
    printf("Initializing grid...\n");
    {
        int blockSize = 256;
        int numBlocks = (int)((gridSize + blockSize - 1) / blockSize);
        initializeGridKernel<<<numBlocks, blockSize>>>(d_grid1, gridSize);
    }

    // Copy initial values to second buffer so boundary values are correct in both
    CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridBytes, cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Set up kernel launch parameters for interior points
    dim3 block(BX, BY, BZ);
    dim3 grid_dim(
        (unsigned int)((nx - 2 + BX - 1) / BX),
        (unsigned int)((ny - 2 + BY - 1) / BY),
        (unsigned int)((nz - 2 + BZ - 1) / BZ)
    );

    // Run stencil iterations
    printf("Running stencil computation...\n");

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<grid_dim, block>>>(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilKernel<<<grid_dim, block>>>(d_grid2, d_grid1, nx, ny, nz);
        }
    }

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float duration_ms;
    CUDA_CHECK(cudaEventElapsedTime(&duration_ms, start, stop));

    printf("Computation time: %ld ms\n", (long)duration_ms);

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration_ms / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy result back to host
    std::vector<Real> finalGrid(gridSize);
    const Real* d_finalGrid = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_finalGrid, gridBytes, cudaMemcpyDeviceToHost));

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
            cudaFree(d_grid1);
            cudaFree(d_grid2);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            cudaFree(d_grid1);
            cudaFree(d_grid2);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return 1;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
