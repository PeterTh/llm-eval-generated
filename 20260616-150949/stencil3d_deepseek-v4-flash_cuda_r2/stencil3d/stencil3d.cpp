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

// CUDA error checking macro
#define CUDA_CHECK(call) do {                                          \
    cudaError_t _err = call;                                           \
    if (_err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA error at %s:%d: '%s' (%d)\n",           \
                __FILE__, __LINE__, cudaGetErrorString(_err), _err);   \
        exit(EXIT_FAILURE);                                            \
    }                                                                  \
} while (0)

//------------------------------------------------------------------------------
// CUDA kernel: 7-point stencil computation + boundary copy
// Each thread processes one grid cell. Interior points compute the stencil
// average; boundary points are copied from input to output.
//------------------------------------------------------------------------------
__global__ void stencil_kernel(const Real* __restrict__ input,
                                Real* __restrict__ output,
                                const int nx, const int ny, const int nz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = (size_t)z * (size_t)nx * (size_t)ny +
                       (size_t)y * (size_t)nx + (size_t)x;

    if (x >= 1 && x < nx - 1 && y >= 1 && y < ny - 1 && z >= 1 && z < nz - 1) {
        // Interior: 7-point stencil
        const Real sum = input[idx] +
                         input[idx - 1]          + input[idx + 1] +
                         input[idx - nx]         + input[idx + nx] +
                         input[idx - (size_t)nx * (size_t)ny] +
                         input[idx + (size_t)nx * (size_t)ny];
        output[idx] = sum * (Real)(1.0 / 7.0);
    } else {
        // Boundary: copy input to output
        output[idx] = input[idx];
    }
}

//------------------------------------------------------------------------------
// Multi-step GPU-accelerated stencil iteration (double buffering)
// Device memory is managed by the caller so CUDA init cost is excluded
// from the timed section.
//------------------------------------------------------------------------------
void runStencilOnGPU(Real* d_grid1, Real* d_grid2,
                     const int nx, const int ny, const int nz,
                     const int iterations) {
    // Kernel launch configuration
    constexpr int BLOCK_X = 16;
    constexpr int BLOCK_Y = 16;
    constexpr int BLOCK_Z = 4;
    const dim3 block_size(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid_size((nx + BLOCK_X - 1) / BLOCK_X,
                          (ny + BLOCK_Y - 1) / BLOCK_Y,
                          (nz + BLOCK_Z - 1) / BLOCK_Z);

    // Run iterations with double buffering on device
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencil_kernel<<<grid_size, block_size>>>(d_grid1, d_grid2,
                                                       nx, ny, nz);
        } else {
            stencil_kernel<<<grid_size, block_size>>>(d_grid2, d_grid1,
                                                       nx, ny, nz);
        }
    }

    // Synchronize and check for errors
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
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
    int nx = 128;
    int ny = 0;  // Will be set to nx if not specified
    int nz = 0;  // Will be set to nx if not specified
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

    printf("3D Stencil Benchmark (CUDA)\n");
    printf("Grid size: %d x %d x %d\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = (size_t)nx * ny * nz;

    // Allocate host grids (double buffering)
    std::vector<Real> h_grid1(gridSize);
    std::vector<Real> h_grid2(gridSize);

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(h_grid1, (size_t)nx, (size_t)ny, (size_t)nz);

    // Allocate device memory (also triggers CUDA driver initialization)
    printf("Initializing CUDA...\n");
    Real *d_grid1 = nullptr, *d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(d_grid1, h_grid1.data(), gridSize * sizeof(Real),
                          cudaMemcpyHostToDevice));

    // Run stencil iterations on GPU (timed)
    printf("Running stencil computation on GPU...\n");
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<Real>& h_result = (iterations % 2 == 0) ? h_grid1 : h_grid2;
    runStencilOnGPU(d_grid1, d_grid2, nx, ny, nz, iterations);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy result back to host
    const Real* d_result = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    CUDA_CHECK(cudaMemcpy(h_result.data(), d_result, gridSize * sizeof(Real),
                          cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_grid1));
    CUDA_CHECK(cudaFree(d_grid2));

    double totalMs = (double)duration.count();
    if (totalMs < 1.0) {
        // Use microseconds for precise reporting
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
        printf("Computation time: %ld us\n", us);
        totalMs = us / 1000.0;
    } else {
        printf("Computation time: %ld ms\n", duration.count());
    }

    // Calculate performance metrics
    double cellUpdates = (double)((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    double sec = totalMs / 1000.0;
    double mcups = cellUpdates / sec / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(h_result, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(h_result, (size_t)nx, (size_t)ny, (size_t)nz);

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
