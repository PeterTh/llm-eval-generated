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
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Thread block tile dimensions used for the shared-memory XY tiling.
static constexpr int TX = 32;
static constexpr int TY = 8;

// 3D index calculation
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
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

// 7-point stencil computation. Each thread owns a fixed (x, y) column and
// walks the z dimension, reusing an XY shared-memory tile (with halo) for
// the left/right/front/back neighbors across the whole thread block, while
// the top/bottom neighbors are read directly (they are not reused by other
// threads in the block).
__global__ void stencilKernel(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    extern __shared__ Real tile[];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + tx;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + ty;
    const bool valid = (x < nx) && (y < ny);

    const size_t planeStride = nx * ny;
    const int sw = blockDim.x + 2;

    for (size_t z = 0; z < nz; ++z) {
        size_t idx = 0;
        if (valid) {
            idx = z * planeStride + y * nx + x;

            tile[(ty + 1) * sw + (tx + 1)] = input[idx];
            if (tx == 0 && x > 0) {
                tile[(ty + 1) * sw + 0] = input[idx - 1];
            }
            if (tx == static_cast<int>(blockDim.x) - 1 && x < nx - 1) {
                tile[(ty + 1) * sw + (tx + 2)] = input[idx + 1];
            }
            if (ty == 0 && y > 0) {
                tile[0 * sw + (tx + 1)] = input[idx - nx];
            }
            if (ty == static_cast<int>(blockDim.y) - 1 && y < ny - 1) {
                tile[(ty + 2) * sw + (tx + 1)] = input[idx + nx];
            }
        }
        __syncthreads();

        if (valid) {
            if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
                output[idx] = input[idx];
            } else {
                const Real center = tile[(ty + 1) * sw + (tx + 1)];
                const Real left = tile[(ty + 1) * sw + tx];
                const Real right = tile[(ty + 1) * sw + (tx + 2)];
                const Real front = tile[ty * sw + (tx + 1)];
                const Real back = tile[(ty + 2) * sw + (tx + 1)];
                const Real bottom = input[idx - planeStride];
                const Real top = input[idx + planeStride];

                // Simple averaging stencil
                output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
        __syncthreads();
    }
}

void stencilIteration(const Real* d_input, Real* d_output,
                       const size_t nx, const size_t ny, const size_t nz) {
    dim3 block(TX, TY);
    dim3 grid(static_cast<unsigned int>((nx + TX - 1) / TX),
              static_cast<unsigned int>((ny + TY - 1) / TY));
    const size_t sharedBytes = static_cast<size_t>(TX + 2) * (TY + 2) * sizeof(Real);

    stencilKernel<<<grid, block, sharedBytes>>>(d_input, d_output, nx, ny, nz);
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
    const size_t gridBytes = gridSize * sizeof(Real);

    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    // Allocate device buffers and copy initial data
    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&d_grid1, gridBytes));
    CUDA_CHECK(cudaMalloc(&d_grid2, gridBytes));
    CUDA_CHECK(cudaMemcpy(d_grid1, grid1.data(), gridBytes, cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy result back to host
    Real* d_final = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    std::vector<Real> finalGrid(gridSize);
    CUDA_CHECK(cudaMemcpy(finalGrid.data(), d_final, gridBytes, cudaMemcpyDeviceToHost));

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
