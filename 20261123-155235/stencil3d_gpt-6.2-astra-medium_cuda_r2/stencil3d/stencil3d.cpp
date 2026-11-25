#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrids(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

constexpr unsigned planesPerTile = 4;

__global__ void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x == 0 || x >= nx - 1) return;

    const size_t plane = nx * ny;
    for (size_t y = 1 + size_t(blockIdx.y) * blockDim.y + threadIdx.y; y < ny - 1;
         y += size_t(gridDim.y) * blockDim.y) {
        for (size_t z = 1 + size_t(blockIdx.z) * planesPerTile; z < nz - 1;
             z += size_t(gridDim.z) * planesPerTile) {
            size_t i = z * plane + y * nx + x;
            Real bottom = input[i - plane];
            Real center = input[i];
            // A warp spans adjacent x values. Carry two planes in registers.
            #pragma unroll
            for (unsigned k = 0; k < planesPerTile && z + k < nz - 1; ++k) {
                const Real top = input[i + plane];
                const Real sum = center + input[i - 1] + input[i + 1] +
                                 input[i - nx] + input[i + nx] + bottom + top;
                // Split 1/7 into its leading and trailing 53 bits. For these
                // normal-range sums, sum/7 is at least 1/14 ulp from a rounding
                // midpoint, so the fused final rounding agrees with division.
                constexpr Real reciprocalHigh = 0x1.2492492492492p-3;
                constexpr Real reciprocalLow = 0x1.2492492492492p-57;
                output[i] = __fma_rn(sum, reciprocalHigh, sum * reciprocalLow);
                bottom = center;
                center = top;
                i += plane;
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
    
    const size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || nx > maxElements / ny || nx * ny > maxElements / nz) {
        std::fprintf(stderr, "Invalid grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* input = nullptr;
    Real* output = nullptr;
    checkCuda(cudaMalloc(&input, bytes), "Allocate input grid");
    checkCuda(cudaMalloc(&output, bytes), "Allocate output grid");

    printf("Initializing grid...\n");
    const unsigned initBlocks = std::min<size_t>((gridSize + 255) / 256, 65535);
    initializeGrids<<<initBlocks, 256>>>(input, output, gridSize);
    checkCuda(cudaGetLastError(), "Launch initialization");
    checkCuda(cudaDeviceSynchronize(), "Initialize grids");

    printf("Running stencil computation...\n");
    auto start = std::chrono::steady_clock::now();
    double cellUpdates = 0;
    if (nx > 2 && ny > 2 && nz > 2 && iterations > 0) {
        const dim3 block(32, 4);
        const dim3 grid((nx - 1 + block.x - 1) / block.x,
                        std::min<size_t>((ny - 2 + block.y - 1) / block.y, 65535),
                        std::min<size_t>((nz - 2 + planesPerTile - 1) / planesPerTile, 65535));
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<grid, block>>>(input, output, nx, ny, nz);
            checkCuda(cudaGetLastError(), "Launch stencil iteration");
            std::swap(input, output);
        }
        cellUpdates = double(nx - 2) * (ny - 2) * (nz - 2) * iterations;
    } else if (iterations < 0 && iterations % 2 != 0) {
        // The original selects its zero-filled second buffer in this case.
        checkCuda(cudaMemset(input, 0, bytes), "Clear result grid");
    }
    checkCuda(cudaDeviceSynchronize(), "Complete stencil computation");
    const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
    printf("Computation time: %.3f ms\n", duration.count() * 1000.0);
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / duration.count() / 1e6);

    checkCuda(cudaFree(output), "Free scratch grid");
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost),
                  "Copy result grid");
    }
    checkCuda(cudaFree(input), "Free result grid");
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
