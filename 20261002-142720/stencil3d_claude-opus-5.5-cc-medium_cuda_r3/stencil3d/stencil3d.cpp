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
constexpr int BLOCK_Y = 16;

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Correctly rounded x / 7.0, bit-identical to IEEE division.
// FP64 throughput is very low on many GPUs, while integer throughput is high,
// so the quotient is computed exactly on the integer pipeline from the IEEE-754
// bit pattern (mantissa / 7 with round-to-nearest-even). Zeros, subnormals,
// tiny values and Inf/NaN take the regular division path.
__device__ __forceinline__ Real divideBy7(const Real x) {
    const unsigned long long b = (unsigned long long)__double_as_longlong(x);
    const unsigned E = (unsigned)(b >> 52) & 0x7ffu;
    if (E >= 16u && E <= 2046u) {
        const unsigned long long M = (b & 0x000FFFFFFFFFFFFFull) | 0x0010000000000000ull;  // 53-bit significand
        const unsigned long long T = M << 10;
        const unsigned long long Q = T / 7ull;            // 60 or 61 significant bits
        const unsigned long long R = T - Q * 7ull;        // nonzero => inexact (sticky)
        const unsigned sh = 7u + (unsigned)(Q >> 60);     // bits to drop to keep 53
        const unsigned long long m = Q >> sh;
        const unsigned long long rb = Q & ((1ull << sh) - 1ull);
        const unsigned long long half = 1ull << (sh - 1u);
        const unsigned long long up = (rb > half || (rb == half && (R != 0ull || (m & 1ull)))) ? 1ull : 0ull;
        // A rounding carry out of the significand correctly bumps the exponent.
        const unsigned long long bits = (b & 0x8000000000000000ull) +
            ((unsigned long long)(E + sh - 10u) << 52) + (m + up - 0x0010000000000000ull);
        return __longlong_as_double((long long)bits);
    }
    return x / 7.0;
}

// 7-point stencil computation (interior averaging + boundary copy in one pass).
// Each thread owns one (x, y) column and marches through a chunk of z planes,
// keeping the z-1 / z / z+1 values in registers.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const size_t nx, const size_t ny, const size_t nz, const size_t zChunk) {
    const size_t x = blockIdx.x * (size_t)BLOCK_X + threadIdx.x;
    const size_t y = blockIdx.y * (size_t)BLOCK_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t z0 = blockIdx.z * zChunk;
    if (z0 >= nz) return;
    const size_t z1 = (z0 + zChunk < nz) ? z0 + zChunk : nz;

    const size_t plane = nx * ny;
    size_t idx = idx3(x, y, z0, nx, ny);

    const bool xyBoundary = (x == 0 || x == nx - 1 || y == 0 || y == ny - 1);
    if (xyBoundary) {
        for (size_t z = z0; z < z1; ++z, idx += plane) {
            output[idx] = input[idx];
        }
        return;
    }

    Real bottom = (z0 > 0) ? input[idx - plane] : Real(0);
    Real center = input[idx];
    for (size_t z = z0; z < z1; ++z, idx += plane) {
        if (z == 0 || z == nz - 1) {
            output[idx] = center;
            bottom = center;
            if (z + 1 < nz) center = input[idx + plane];
            continue;
        }
        const Real top = input[idx + plane];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];

        // Simple averaging stencil (same summation order as the reference)
        output[idx] = divideBy7(center + left + right + front + back + bottom + top);

        bottom = center;
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
    
    // Allocate device grids (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
    // grid2 starts zero-initialized, matching std::vector semantics
    CUDA_CHECK(cudaMemset(d_grid2, 0, gridSize * sizeof(Real)));

    int device = 0;
    int numSMs = 1;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    // Initialize
    printf("Initializing grid...\n");
    if (gridSize > 0) {
        initializeGridKernel<<<numSMs * 8, 256>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Launch configuration: 2D tiles in x/y, z split into chunks so that
    // enough blocks exist to fill the GPU.
    const size_t blocksX = (nx + BLOCK_X - 1) / BLOCK_X;
    const size_t blocksY = (ny + BLOCK_Y - 1) / BLOCK_Y;
    const size_t xyBlocks = std::max<size_t>(1, blocksX * blocksY);
    const size_t targetBlocks = (size_t)numSMs * 64;
    size_t zChunks = (targetBlocks + xyBlocks - 1) / xyBlocks;
    zChunks = std::max<size_t>(1, std::min(zChunks, (nz + 7) / 8));
    zChunks = std::min<size_t>(zChunks, 65535);
    const size_t zChunk = std::max<size_t>(1, (nz + zChunks - 1) / zChunks);
    zChunks = (nz + zChunk - 1) / zChunk;
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 grid((unsigned)blocksX, (unsigned)blocksY, (unsigned)zChunks);
    const bool doLaunch = gridSize > 0;

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations && doLaunch; ++iter) {
        if (iter % 2 == 0) {
            stencilKernel<<<grid, block>>>(d_grid1, d_grid2, nx, ny, nz, zChunk);
        } else {
            stencilKernel<<<grid, block>>>(d_grid2, d_grid1, nx, ny, nz, zChunk);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Copy final grid back to host
    std::vector<Real> finalGrid(gridSize);
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
