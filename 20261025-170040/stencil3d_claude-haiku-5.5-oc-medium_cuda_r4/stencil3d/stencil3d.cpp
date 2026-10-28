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
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err__));                                   \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

// Thread block shape: x is the fastest-varying dimension (coalesced accesses).
constexpr unsigned kBlockX = 32;
constexpr unsigned kBlockY = 4;
constexpr unsigned kBlockZ = 4;
constexpr unsigned kInitBlock = 256;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initialize grid: grid[i] = (i % 19) * 1.0, one thread per element.
__global__ void initGridKernel(Real* __restrict__ grid, const size_t total) {
    const size_t i = static_cast<size_t>(blockIdx.x) * kInitBlock + threadIdx.x;
    if (i < total) {
        grid[i] = (i % 19) * 1.0;
    }
}

// One 7-point stencil sweep: one thread per cell.
// Interior cells are averaged; boundary cells are copied from the input.
__global__ void __launch_bounds__(kBlockX * kBlockY * kBlockZ)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * kBlockZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const size_t idx = idx3(x, y, z, nx, ny);

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[idx] = input[idx];
        return;
    }

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - nx];
    const Real back   = input[idx + nx];
    const Real bottom = input[idx - nx * ny];
    const Real top    = input[idx + nx * ny];

    // Simple averaging stencil (same operation order as the reference CPU code)
    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

static void launchInit(Real* d_grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t total = nx * ny * nz;
    if (total == 0) return;
    const unsigned blocks = static_cast<unsigned>((total + kInitBlock - 1) / kInitBlock);
    initGridKernel<<<blocks, kInitBlock>>>(d_grid, total);
    CUDA_CHECK(cudaGetLastError());
}

static void launchStencil(const Real* d_in, Real* d_out, const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) return;
    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(static_cast<unsigned>((nx + kBlockX - 1) / kBlockX),
                    static_cast<unsigned>((ny + kBlockY - 1) / kBlockY),
                    static_cast<unsigned>((nz + kBlockZ - 1) / kBlockZ));
    stencilKernel<<<grid, block>>>(d_in, d_out, nx, ny, nz);
    CUDA_CHECK(cudaGetLastError());
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
    CUDA_CHECK(cudaMemset(d_grid2, 0, gridSize * sizeof(Real)));
    
    // Initialize
    printf("Initializing grid...\n");
    launchInit(d_grid1, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            launchStencil(d_grid1, d_grid2, nx, ny, nz);
        } else {
            launchStencil(d_grid2, d_grid1, nx, ny, nz);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy final grid back to host for output / validation
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    int exitCode = 0;
    if (printResults || validate) {
        std::vector<Real> finalGrid(gridSize);
        if (gridSize > 0) {
            CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
        }
        
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
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    return exitCode;
}
