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
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Thread block shape for the stencil: each thread marches along z over a chunk
constexpr int BX = 64;
constexpr int BY = 4;
constexpr int ZCHUNK = 32;

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Copy boundary values (all cells with x/y/z on a face) from input to output
__global__ void copyBoundaryKernel(const Real* __restrict__ input, Real* __restrict__ output,
                                   const size_t nx, const size_t ny, const size_t nz) {
    const size_t n = nx * ny * nz;
    for (size_t idx = blockIdx.x * (size_t)blockDim.x + threadIdx.x; idx < n;
         idx += (size_t)gridDim.x * blockDim.x) {
        const size_t x = idx % nx;
        const size_t t = idx / nx;
        const size_t y = t % ny;
        const size_t z = t / ny;
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            output[idx] = input[idx];
        }
    }
}

// Correctly rounded x / 7.0 (bit-identical to IEEE division, round-to-nearest-even)
// computed with integer arithmetic. GeForce GPUs have very low FP64 throughput and
// a double division costs ~10 FP64 instructions, which would make the stencil
// FP64-bound instead of memory-bound. Zero, subnormal, inf/nan and extreme
// exponents fall back to the hardware division path.
__device__ __forceinline__ Real divideBy7(const Real x) {
    const unsigned long long bits = __double_as_longlong(x);
    const unsigned int e = (unsigned int)(bits >> 52) & 0x7ffu;
    if (e - 16u > 2030u) return x / 7.0;
    const unsigned long long m = (bits & 0xfffffffffffffull) | 0x10000000000000ull;
    const unsigned long long n = m << 11;           // m in [2^52, 2^53) -> n < 2^64
    const unsigned long long q = n / 7ull;          // q in [2^60.2, 2^61.2)
    const unsigned long long r = n - q * 7ull;      // remainder -> sticky bit
    const int sh = 11 - __clzll(q);                 // drop 8 or 9 bits to keep 53
    const unsigned long long mant = q >> sh;
    const unsigned long long dropped = q & ((1ull << sh) - 1);
    const unsigned long long half = 1ull << (sh - 1);
    const bool roundUp = dropped > half || (dropped == half && (r != 0 || (mant & 1)));
    // Adding the 53-bit mantissa (with implicit bit) to (exponent - 1) also handles
    // a rounding carry into the exponent.
    const unsigned long long res = (bits & 0x8000000000000000ull) +
                                   ((unsigned long long)(e + sh - 12) << 52) + mant + roundUp;
    return __longlong_as_double(res);
}

// 7-point stencil on interior points. Each thread owns one (x,y) column and
// walks a chunk of z, keeping the z-neighbors in registers.
__global__ void __launch_bounds__(BX * BY)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = 1 + blockIdx.x * (size_t)BX + threadIdx.x;
    const size_t y = 1 + blockIdx.y * (size_t)BY + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) return;

    const size_t z0 = 1 + blockIdx.z * (size_t)ZCHUNK;
    size_t zEnd = z0 + ZCHUNK;
    if (zEnd > nz - 1) zEnd = nz - 1;

    const size_t plane = nx * ny;
    size_t idx = idx3(x, y, z0, nx, ny);

    Real bottom = __ldg(&input[idx - plane]);
    Real center = __ldg(&input[idx]);
    for (size_t z = z0; z < zEnd; ++z, idx += plane) {
        const Real top = __ldg(&input[idx + plane]);
        const Real left = __ldg(&input[idx - 1]);
        const Real right = __ldg(&input[idx + 1]);
        const Real front = __ldg(&input[idx - nx]);
        const Real back = __ldg(&input[idx + nx]);

        // Simple averaging stencil (same summation order as the reference)
        output[idx] = divideBy7(center + left + right + front + back + bottom + top);

        bottom = center;
        center = top;
    }
}

static unsigned int gridStrideBlocks(const size_t n, const int threads) {
    size_t blocks = (n + threads - 1) / threads;
    const size_t maxBlocks = 65535 * 16;
    if (blocks > maxBlocks) blocks = maxBlocks;
    if (blocks == 0) blocks = 1;
    return (unsigned int)blocks;
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
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    if (gridSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_grid1, bytes));
        CUDA_CHECK(cudaMalloc(&d_grid2, bytes));
        CUDA_CHECK(cudaMemset(d_grid2, 0, bytes));
    }

    // Initialize
    printf("Initializing grid...\n");
    if (gridSize > 0) {
        initializeGridKernel<<<gridStrideBlocks(gridSize, 256), 256>>>(d_grid1, gridSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (gridSize > 0 && iterations > 0) {
        // Boundary values are only ever copied, so they are identical in both
        // buffers after the first iteration: copy them once into grid2.
        copyBoundaryKernel<<<gridStrideBlocks(gridSize, 256), 256>>>(d_grid1, d_grid2, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());

        if (nx > 2 && ny > 2 && nz > 2) {
            const dim3 block(BX, BY, 1);
            const dim3 grid((unsigned int)((nx - 2 + BX - 1) / BX),
                            (unsigned int)((ny - 2 + BY - 1) / BY),
                            (unsigned int)((nz - 2 + ZCHUNK - 1) / ZCHUNK));
            for (int iter = 0; iter < iterations; ++iter) {
                if (iter % 2 == 0) {
                    stencilKernel<<<grid, block>>>(d_grid1, d_grid2, nx, ny, nz);
                } else {
                    stencilKernel<<<grid, block>>>(d_grid2, d_grid1, nx, ny, nz);
                }
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(grid1.data(), d_grid1, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(grid2.data(), d_grid2, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
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
