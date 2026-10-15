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
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t err_ = (call);                                               \
        if (err_ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                         \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

constexpr int BX = 32;
constexpr int BY = 8;
constexpr int ZCHUNK = 32;

__global__ void initializeKernel(Real* a, Real* b, const size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        const Real v = (Real)(i % 19) * 1.0;
        a[i] = v;
        b[i] = v;  // boundaries are constant, so both buffers carry them
    }
}

// 7-point stencil; each thread marches along z keeping the z-column in registers.
// Only interior points are written; boundaries were initialised in both buffers.
__global__ void __launch_bounds__(BX * BY)
stencilKernel(const Real* __restrict__ in, Real* __restrict__ out,
              const int nx, const int ny, const int nz) {
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x < 1 || x >= nx - 1 || y < 1 || y >= ny - 1) return;

    const int zBeg = 1 + blockIdx.z * ZCHUNK;
    const int zEnd = min(zBeg + ZCHUNK, nz - 1);
    const size_t plane = (size_t)nx * ny;
    size_t idx = (size_t)zBeg * plane + (size_t)y * nx + x;

    Real bottom = __ldg(in + idx - plane);
    Real center = __ldg(in + idx);
    for (int z = zBeg; z < zEnd; ++z, idx += plane) {
        const Real top = __ldg(in + idx + plane);
        const Real left = __ldg(in + idx - 1);
        const Real right = __ldg(in + idx + 1);
        const Real front = __ldg(in + idx - nx);
        const Real back = __ldg(in + idx + nx);
        out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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
    std::vector<Real> hostGrid(gridSize);
    Real *d1 = nullptr, *d2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d2, gridSize * sizeof(Real)));

    // Initialize
    printf("Initializing grid...\n");
    initializeKernel<<<1024, 256>>>(d1, d2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    Real* cur = d1;
    Real* nxt = d2;
    if (nx > 2 && ny > 2 && nz > 2) {
        dim3 block(BX, BY, 1);
        dim3 grid((unsigned)((nx + BX - 1) / BX), (unsigned)((ny + BY - 1) / BY),
                  (unsigned)((nz - 2 + ZCHUNK - 1) / ZCHUNK));
        for (int iter = 0; iter < iterations; ++iter) {
            stencilKernel<<<grid, block>>>(cur, nxt, (int)nx, (int)ny, (int)nz);
            std::swap(cur, nxt);
        }
    } else {
        for (int iter = 0; iter < iterations; ++iter) std::swap(cur, nxt);
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
    
    // Print results for external validation
    CUDA_CHECK(cudaMemcpy(hostGrid.data(), cur, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d1));
    CUDA_CHECK(cudaFree(d2));
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
