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

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Correctly rounded s / 7.0 (bit-identical to IEEE division) using integer
// arithmetic, which is much faster than FP64 division on GPUs with reduced
// double-precision throughput. Since 7 is odd, a tie can never occur, so the
// quotient is rounded up exactly when the remainder is >= 4.
// Zero, subnormal, Inf and NaN inputs fall back to hardware division.
__device__ __forceinline__ Real divideBy7(const Real s) {
    const unsigned long long bits = (unsigned long long)__double_as_longlong(s);
    const unsigned long long expo = (bits >> 52) & 0x7FFull;
    if (expo < 4 || expo == 0x7FFull) return s / 7.0;
    const unsigned long long mant = (bits & 0xFFFFFFFFFFFFFull) | (1ull << 52);
    // Choose the shift so that the quotient lands in [2^52, 2^53)
    const unsigned shift = (mant < (7ull << 50)) ? 3u : 2u;
    const unsigned long long num = mant << shift;
    unsigned long long q = num / 7ull;
    const unsigned long long rem = num - q * 7ull;
    q += (rem >= 4ull) ? 1ull : 0ull;
    const unsigned long long res = (bits & 0x8000000000000000ull) |
                                   ((expo - shift) << 52) |
                                   (q & 0xFFFFFFFFFFFFFull);
    return __longlong_as_double((long long)res);
}

// 7-point stencil computation (interior averaging + boundary copy fused).
// Each thread owns one (x, y) column and marches along a chunk of z,
// keeping the z-1 / z / z+1 values in registers.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz, const int zChunk) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const int zBegin = blockIdx.z * zChunk;
    const int zEnd = min(zBegin + zChunk, nz);
    if (zBegin >= zEnd) return;

    const size_t plane = (size_t)nx * ny;
    const size_t col = (size_t)y * nx + x;

    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
    if (xyBoundary) {
        for (int z = zBegin; z < zEnd; ++z) {
            const size_t idx = z * plane + col;
            output[idx] = __ldg(&input[idx]);
        }
        return;
    }

    size_t idx = zBegin * plane + col;
    Real below = (zBegin > 0) ? __ldg(&input[idx - plane]) : 0.0;
    Real center = __ldg(&input[idx]);
    for (int z = zBegin; z < zEnd; ++z, idx += plane) {
        const Real top = (z + 1 < nz) ? __ldg(&input[idx + plane]) : 0.0;
        if (z == 0 || z == nz - 1) {
            output[idx] = center;
        } else {
            const Real left = __ldg(&input[idx - 1]);
            const Real right = __ldg(&input[idx + 1]);
            const Real front = __ldg(&input[idx - nx]);
            const Real back = __ldg(&input[idx + nx]);
            const Real bottom = below;
            // Simple averaging stencil (same summation order as the reference)
            output[idx] = divideBy7(center + left + right + front + back + bottom + top);
        }
        below = center;
        center = top;
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
    
    // Allocate grids (double buffering) on the device; host copy holds the final result
    std::vector<Real> hostGrid(gridSize);
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    const size_t bytes = std::max<size_t>(gridSize, 1) * sizeof(Real);
    CUDA_CHECK(cudaMalloc(&d_grid1, bytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, bytes));
    
    int device = 0, numSMs = 1;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
    
    // Initialize
    printf("Initializing grid...\n");
    if (gridSize > 0) {
        const int threads = 256;
        const size_t blocksNeeded = (gridSize + threads - 1) / threads;
        const int blocks = (int)std::min<size_t>(blocksNeeded, (size_t)numSMs * 32);
        initializeGridKernel<<<blocks, threads>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Launch configuration: 2D tiles in (x, y), z split into chunks so that
    // enough blocks exist to fill the GPU while keeping long z-marches.
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const unsigned gx = (unsigned)((nx + BLOCK_X - 1) / BLOCK_X);
    const unsigned gy = (unsigned)((ny + BLOCK_Y - 1) / BLOCK_Y);
    const size_t xyBlocks = std::max<size_t>((size_t)gx * gy, 1);
    const size_t targetBlocks = (size_t)numSMs * 256;
    size_t zParts = (targetBlocks + xyBlocks - 1) / xyBlocks;
    zParts = std::max<size_t>(1, std::min<size_t>(zParts, std::max<size_t>(nz / 32, 1)));
    const int zChunk = (int)((nz + zParts - 1) / std::max<size_t>(zParts, 1));
    const unsigned gz = zChunk > 0 ? (unsigned)((nz + zChunk - 1) / zChunk) : 0;
    const dim3 grid(gx, gy, gz);
    const bool launch = gridSize > 0;
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (!launch) break;
        if (iter % 2 == 0) {
            stencilKernel<<<grid, block>>>(d_grid1, d_grid2, (int)nx, (int)ny, (int)nz, zChunk);
        } else {
            stencilKernel<<<grid, block>>>(d_grid2, d_grid1, (int)nx, (int)ny, (int)nz, zChunk);
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
    
    // Retrieve final grid
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(hostGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
    // Print results for external validation
    const std::vector<Real>& finalGrid = hostGrid;
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
