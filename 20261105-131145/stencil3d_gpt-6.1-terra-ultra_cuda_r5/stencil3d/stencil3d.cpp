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

// Both buffers start with the same fixed boundary values.  The stencil only
// writes interior cells, so those boundaries remain valid for every iteration.
__global__ void initializeGrid(Real* grid1, Real* grid2, size_t count,
                               bool zeroSecond) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        grid1[i] = value;
        grid2[i] = zeroSecond ? 0.0 : value;
    }
}

// Each warp spans X, and each thread walks a small Z slab, retaining the
// previous and next planes in registers.  Grid-stride loops handle dimensions
// larger than the hardware launch limits without changing the indexing.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    constexpr size_t zTile = 4;
    const size_t plane = nx * ny;
    for (size_t y = 1 + size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny - 1; y += size_t(gridDim.y) * blockDim.y) {
        for (size_t x = 1 + size_t(blockIdx.x) * blockDim.x + threadIdx.x;
             x < nx - 1; x += size_t(gridDim.x) * blockDim.x) {
            for (size_t z0 = 1 + size_t(blockIdx.z) * zTile;
                 z0 < nz - 1; z0 += size_t(gridDim.z) * zTile) {
                size_t index = z0 * plane + y * nx + x;
                Real bottom = input[index - plane];
                Real center = input[index];
                const size_t end = min(z0 + zTile, nz - 1);
                for (size_t z = z0; z < end; ++z, index += plane) {
                    const Real top = input[index + plane];
                    const Real left = input[index - 1];
                    const Real right = input[index + 1];
                    const Real front = input[index - nx];
                    const Real back = input[index + nx];
                    output[index] = (center + left + right + front + back + bottom + top) / 7.0;
                    bottom = center;
                    center = top;
                }
            }
        }
    }
}

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(error));
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid or oversized grid dimensions\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);
    Real *deviceGrid1 = nullptr, *deviceGrid2 = nullptr;
    checkCuda(cudaMalloc(&deviceGrid1, gridBytes), "allocating first grid");
    checkCuda(cudaMalloc(&deviceGrid2, gridBytes), "allocating second grid");

    printf("Initializing grid...\n");
    const unsigned initBlocks = static_cast<unsigned>(
        std::min<size_t>((gridSize - 1) / 256 + 1, 65535));
    initializeGrid<<<initBlocks, 256>>>(deviceGrid1, deviceGrid2, gridSize,
                                         iterations < 0 && iterations % 2 != 0);
    checkCuda(cudaGetLastError(), "launching initialization");
    checkCuda(cudaDeviceSynchronize(), "initializing grids");

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    Real* input = deviceGrid1;
    Real* output = deviceGrid2;
    if (nx > 2 && ny > 2 && nz > 2) {
        const dim3 block(32, 8);
        const dim3 grid(
            static_cast<unsigned>(std::min<size_t>((nx - 3) / block.x + 1, 2147483647)),
            static_cast<unsigned>(std::min<size_t>((ny - 3) / block.y + 1, 65535)),
            static_cast<unsigned>(std::min<size_t>((nz - 3) / 4 + 1, 65535)));
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<grid, block>>>(input, output, nx, ny, nz);
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "launching stencil");
        checkCuda(cudaDeviceSynchronize(), "running stencil");
    }
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));
    const double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
    const double mcups = elapsedMs > 0 ? cellUpdates / (elapsedMs * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        // For an empty interior, both buffers are already identical except for
        // the original program's negative odd iteration case.
        const Real* finalDeviceGrid = iterations % 2 == 0 ? deviceGrid1 : deviceGrid2;
        checkCuda(cudaMemcpy(finalGrid.data(), finalDeviceGrid, gridBytes,
                             cudaMemcpyDeviceToHost), "copying result to host");
    }
    checkCuda(cudaFree(deviceGrid1), "freeing first grid");
    checkCuda(cudaFree(deviceGrid2), "freeing second grid");

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
