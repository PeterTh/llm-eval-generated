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
            exit(1);                                                              \
        }                                                                         \
    } while (0)

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;
constexpr int Z_CHUNK = 16;

__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Correctly rounded s / 7.0 (bit-identical to IEEE division) using the
// reciprocal plus one FMA-based Markstein correction step. FP64 throughput is the
// bottleneck on consumer GPUs and this is much cheaper than the generic division.
__device__ __forceinline__ Real div7(const Real s) {
    constexpr Real inv7 = 1.0 / 7.0;
    const Real q = s * inv7;
    const Real r = fma(-q, 7.0, s);
    return fma(r, inv7, q);
}

// 7-point stencil computation over interior points.
// Each thread owns one (x, y) column and marches along a chunk of z, keeping the
// z-neighbours in registers. Boundary values never change (they are copied from
// input to output every iteration), so both buffers are initialised with the same
// boundary once and only the interior is written here.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilIterationKernel(const Real* __restrict__ input,
                       Real* __restrict__ output,
                       const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = (size_t)blockIdx.x * BLOCK_X + threadIdx.x + 1;
    const size_t y = (size_t)blockIdx.y * BLOCK_Y + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;

    const size_t zBegin = (size_t)blockIdx.z * Z_CHUNK + 1;
    if (zBegin >= nz - 1) return;
    const size_t zEnd = zBegin + Z_CHUNK < nz - 1 ? zBegin + Z_CHUNK : nz - 1;

    const size_t plane = nx * ny;
    size_t idx = idx3(x, y, zBegin, nx, ny);

    Real bottom = __ldg(&input[idx - plane]);
    Real center = __ldg(&input[idx]);
    for (size_t z = zBegin; z < zEnd; ++z) {
        const Real top = __ldg(&input[idx + plane]);
        const Real left = __ldg(&input[idx - 1]);
        const Real right = __ldg(&input[idx + 1]);
        const Real front = __ldg(&input[idx - nx]);
        const Real back = __ldg(&input[idx + nx]);

        // Simple averaging stencil (same summation order as the reference)
        output[idx] = div7((center + left + right + front + back + bottom + top));

        bottom = center;
        center = top;
        idx += plane;
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
    
    // Device grids (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));

    // Initialize
    printf("Initializing grid...\n");
    {
        int device = 0, numSMs = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));
        const int threads = 256;
        size_t blocks = (gridSize + threads - 1) / threads;
        blocks = std::min<size_t>(blocks, (size_t)numSMs * 32);
        if (blocks > 0) {
            initializeGridKernel<<<(unsigned)blocks, threads>>>(d_grid1, gridSize);
            CUDA_CHECK(cudaGetLastError());
            if (iterations >= 0) {
                // Both buffers share the (never changing) boundary values.
                CUDA_CHECK(cudaMemcpy(d_grid2, d_grid1, gridSize * sizeof(Real), cudaMemcpyDeviceToDevice));
            } else {
                // No iterations run; keep the second buffer zeroed like the reference.
                CUDA_CHECK(cudaMemset(d_grid2, 0, gridSize * sizeof(Real)));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const dim3 grid(hasInterior ? (unsigned)((nx - 2 + BLOCK_X - 1) / BLOCK_X) : 0,
                    hasInterior ? (unsigned)((ny - 2 + BLOCK_Y - 1) / BLOCK_Y) : 0,
                    hasInterior ? (unsigned)((nz - 2 + Z_CHUNK - 1) / Z_CHUNK) : 0);

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (hasInterior) {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIterationKernel<<<grid, block>>>(d_grid1, d_grid2, nx, ny, nz);
            } else {
                stencilIterationKernel<<<grid, block>>>(d_grid2, d_grid1, nx, ny, nz);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), (iterations % 2 == 0) ? d_grid1 : d_grid2,
                          gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
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
