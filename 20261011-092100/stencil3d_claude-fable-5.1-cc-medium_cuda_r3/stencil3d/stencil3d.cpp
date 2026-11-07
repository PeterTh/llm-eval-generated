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

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            exit(1);                                                                     \
        }                                                                                \
    } while (0)

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                 const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// GPU kernels
// ---------------------------------------------------------------------------

// Grid initialisation: grid[idx] = (idx % 19) * 1.0, grid-stride loop.
__global__ void initializeGridKernel(Real* __restrict__ grid, const size_t n) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; idx < n; idx += stride) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil, one thread per (x, y) column, marching along z.
// The three z-neighbours (bottom / center / top) are kept in registers and
// slid along the column so every input element is loaded from global memory
// (or L1/L2) once per thread. Reads along x within a warp are fully coalesced.
// Boundary cells are copied straight through (output = input), exactly as in
// the reference implementation. Summation order and the division by 7.0 are
// preserved so results are bit-identical to the CPU code.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 8;

__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
              const int nx, const int ny, const int nz) {
    const int x = blockIdx.x * BLOCK_X + threadIdx.x;
    const int y = blockIdx.y * BLOCK_Y + threadIdx.y;
    if (x >= nx || y >= ny) return;

    const size_t plane = static_cast<size_t>(nx) * ny;
    const bool xyBoundary = (x == 0) || (x == nx - 1) || (y == 0) || (y == ny - 1);

    size_t idx = static_cast<size_t>(y) * nx + x;  // z == 0

    if (xyBoundary) {
        // Whole column lies on the boundary: straight copy.
        for (int z = 0; z < nz; ++z, idx += plane) {
            output[idx] = input[idx];
        }
        return;
    }

    // z == 0 plane is boundary.
    Real bottom = input[idx];
    output[idx] = bottom;
    if (nz == 1) return;

    idx += plane;
    Real center = input[idx];
    Real top;

    for (int z = 1; z < nz - 1; ++z, idx += plane) {
        top = input[idx + plane];
        const Real left  = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back  = input[idx + nx];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        bottom = center;
        center = top;
    }

    // z == nz - 1 plane is boundary.
    output[idx] = center;
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------

void initializeGrid(Real* d_grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t n = nx * ny * nz;
    if (n == 0) return;
    const int threads = 256;
    const size_t blocksNeeded = (n + threads - 1) / threads;
    const int blocks = static_cast<int>(std::min<size_t>(blocksNeeded, 65535 * 4));
    initializeGridKernel<<<blocks, threads>>>(d_grid, n);
    CUDA_CHECK(cudaGetLastError());
}

// 7-point stencil computation (one iteration), launched asynchronously.
void stencilIteration(const Real* d_input, Real* d_output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) return;
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid(static_cast<unsigned>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned>((ny + BLOCK_Y - 1) / BLOCK_Y));
    stencilKernel<<<grid, block>>>(d_input, d_output,
                                   static_cast<int>(nx), static_cast<int>(ny), static_cast<int>(nz));
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
    CUDA_CHECK(cudaFree(0));  // Initialise the CUDA context up front.
    CUDA_CHECK(cudaMalloc(&d_grid1, std::max<size_t>(gridSize, 1) * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, std::max<size_t>(gridSize, 1) * sizeof(Real)));
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(d_grid1, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz);
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
    
    // Copy the final grid back to the host
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));
    
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
