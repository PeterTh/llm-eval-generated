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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

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

// 7-point stencil computation, one GPU thread per interior grid point.
//
// Only the interior is written: the boundary values never change across
// iterations, so both device buffers are pre-initialized with the same
// boundary values and their boundary regions remain valid forever. This is
// exactly equivalent to copying the boundaries every iteration.
__global__ __launch_bounds__(256)
void stencilKernel(const Real* __restrict__ input,
                   Real* __restrict__ output,
                   const int nx, const int ny, const int nz) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const int y = blockIdx.y * blockDim.y + threadIdx.y + 1;
    const int z = blockIdx.z * blockDim.z + threadIdx.z + 1;

    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) return;

    const size_t plane = static_cast<size_t>(nx) * static_cast<size_t>(ny);
    const size_t idx = static_cast<size_t>(z) * plane +
                       static_cast<size_t>(y) * static_cast<size_t>(nx) +
                       static_cast<size_t>(x);

    const Real center = input[idx];
    const Real left   = input[idx - 1];
    const Real right  = input[idx + 1];
    const Real front  = input[idx - nx];
    const Real back   = input[idx + nx];
    const Real bottom = input[idx - plane];
    const Real top    = input[idx + plane];

    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
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

    // Host grids (double buffering); both start with identical boundary values.
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);

    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    grid2 = grid1;

    // Device grids (double buffering)
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    if (gridSize > 0) {
        CUDA_CHECK(cudaMalloc(&d_grid1, gridSize * sizeof(Real)));
        CUDA_CHECK(cudaMalloc(&d_grid2, gridSize * sizeof(Real)));
        CUDA_CHECK(cudaMemcpy(d_grid1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grid2, grid2.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice));
    }

    const bool hasInterior = (nx >= 3 && ny >= 3 && nz >= 3);

    // Thread block: 32 x 8 x 1 (256 threads), coalesced along x.
    const dim3 block(32, 8, 1);
    dim3 grid(1, 1, 1);
    if (hasInterior) {
        grid.x = static_cast<unsigned>((nx - 2 + block.x - 1) / block.x);
        grid.y = static_cast<unsigned>((ny - 2 + block.y - 1) / block.y);
        grid.z = static_cast<unsigned>((nz - 2 + block.z - 1) / block.z);
    }

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (!hasInterior) break;
        if (iter % 2 == 0) {
            stencilKernel<<<grid, block>>>(d_grid1, d_grid2,
                                           static_cast<int>(nx),
                                           static_cast<int>(ny),
                                           static_cast<int>(nz));
        } else {
            stencilKernel<<<grid, block>>>(d_grid2, d_grid1,
                                           static_cast<int>(nx),
                                           static_cast<int>(ny),
                                           static_cast<int>(nz));
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

    // Fetch the final grid back to the host for printing / validation.
    const Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    if (gridSize > 0) {
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_grid1));
        CUDA_CHECK(cudaFree(d_grid2));
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
