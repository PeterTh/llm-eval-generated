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

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

constexpr int BX = 32;
constexpr int BY = 8;
constexpr int ZCHUNK = 16;

__global__ void initializeGridKernel(Real* grid, const size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        grid[i] = (i % 19) * 1.0;
    }
}

// 7-point stencil over interior points; each thread marches along z keeping
// the z-neighbours in registers. Boundaries are identical in both buffers
// (copied once up front), so they need no per-iteration copy.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = blockIdx.x * (size_t)BX + threadIdx.x + 1;
    const size_t y = blockIdx.y * (size_t)BY + threadIdx.y + 1;
    if (x >= nx - 1 || y >= ny - 1) return;
    const size_t z0 = 1 + (size_t)blockIdx.z * ZCHUNK;
    size_t z1 = z0 + ZCHUNK;
    if (z1 > nz - 1) z1 = nz - 1;
    const size_t plane = nx * ny;
    size_t idx = z0 * plane + y * nx + x;

    Real bottom = input[idx - plane];
    Real center = input[idx];
    for (size_t z = z0; z < z1; ++z, idx += plane) {
        const Real top = input[idx + plane];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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
    Real *d1 = nullptr, *d2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d2, gridSize * sizeof(Real)));

    // Initialize
    printf("Initializing grid...\n");
    initializeGridKernel<<<(unsigned)std::min<size_t>((gridSize + 255) / 256, 65535), 256>>>(d1, gridSize);
    CUDA_CHECK(cudaGetLastError());
    // Boundaries never change, so seed the second buffer with them once
    CUDA_CHECK(cudaMemcpy(d2, d1, gridSize * sizeof(Real), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (nx > 2 && ny > 2 && nz > 2) {
        dim3 block(BX, BY);
        dim3 grid((unsigned)((nx - 2 + BX - 1) / BX), (unsigned)((ny - 2 + BY - 1) / BY),
                  (unsigned)((nz - 2 + ZCHUNK - 1) / ZCHUNK));
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilKernel<<<grid, block>>>(d1, d2, nx, ny, nz);
            } else {
                stencilKernel<<<grid, block>>>(d2, d1, nx, ny, nz);
            }
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), (iterations % 2 == 0) ? d1 : d2, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    cudaFree(d1);
    cudaFree(d2);

    printf("Computation time: %ld ms\n", duration.count());
    
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
