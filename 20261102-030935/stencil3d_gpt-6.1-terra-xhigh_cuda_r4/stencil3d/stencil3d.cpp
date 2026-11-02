#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
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

// Each warp processes consecutive x coordinates, keeping accesses coalesced.
// Boundary cells are initialized in both buffers and remain unchanged.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t rows) {
    const size_t x = 1 + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    const size_t rowStride = static_cast<size_t>(gridDim.y) * blockDim.y;
    for (size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
         row < rows; row += rowStride) {
        const size_t y = row % (ny - 2) + 1;
        const size_t z = row / (ny - 2) + 1;
        const size_t idx = z * plane + y * nx + x;
        const Real center = input[idx];
        const Real left = input[idx - 1];
        const Real right = input[idx + 1];
        const Real front = input[idx - nx];
        const Real back = input[idx + nx];
        const Real bottom = input[idx - plane];
        const Real top = input[idx + plane];
        output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

static void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
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
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    std::vector<Real> grid(gridSize);
    printf("Initializing grid...\n");
    initializeGrid(grid, nx, ny, nz);

    const size_t bytes = gridSize * sizeof(Real);
    Real* input = nullptr;
    Real* output = nullptr;
    checkCuda(cudaMalloc(&input, bytes), "allocating input grid");
    checkCuda(cudaMalloc(&output, bytes), "allocating output grid");
    checkCuda(cudaMemcpy(input, grid.data(), bytes, cudaMemcpyHostToDevice), "copying input grid");
    checkCuda(cudaMemcpy(output, input, bytes, cudaMemcpyDeviceToDevice), "initializing output grid");

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (nx > 2 && ny > 2 && nz > 2) {
        const size_t rows = (ny - 2) * (nz - 2);
        const dim3 block(32, 4);
        const dim3 launch(static_cast<unsigned int>((nx - 3) / block.x + 1),
                          static_cast<unsigned int>(std::min<size_t>((rows - 1) / block.y + 1, 65535)));
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<launch, block>>>(input, output, nx, ny, rows);
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "launching stencil kernel");
    }
    checkCuda(cudaDeviceSynchronize(), "running stencil kernel");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                         static_cast<double>(ny > 2 ? ny - 2 : 0) *
                         static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
    double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = cellUpdates / seconds / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(grid.data(), input, bytes, cudaMemcpyDeviceToHost), "copying result grid");
    }
    checkCuda(cudaFree(input), "freeing input grid");
    checkCuda(cudaFree(output), "freeing output grid");
    const std::vector<Real>& finalGrid = grid;
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
