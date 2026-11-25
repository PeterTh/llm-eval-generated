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

__global__ void initializeGrid(Real* first, Real* second, size_t count) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += size_t(gridDim.x) * blockDim.x) {
        Real value = static_cast<Real>(i % 19);
        first[i] = value;
        second[i] = value;
    }
}

// Adjacent threads traverse adjacent columns. Slide a three-value register
// window along z so each new plane supplies just one new vertical neighbor.
constexpr unsigned planesPerTile = 8;
__global__ void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                                 size_t nx, size_t plane, size_t nz) {
    const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column < nx || column >= plane - nx) return;
    const size_t x = column % nx;
    if (x == 0 || x == nx - 1) return;

    for (size_t z = 1 + size_t(blockIdx.y) * planesPerTile; z < nz - 1;
         z += size_t(gridDim.y) * planesPerTile) {
        size_t i = column + z * plane;
        Real bottom = input[i - plane];
        Real center = input[i];
        const unsigned depth = nz - 1 - z < planesPerTile ? nz - 1 - z : planesPerTile;
        for (unsigned k = 0; k < depth; ++k, i += plane) {
            const Real top = input[i + plane];
            output[i] = (center + input[i - 1] + input[i + 1] + input[i - nx]
                         + input[i + nx] + bottom + top) / 7.0;
            bottom = center;
            center = top;
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
    
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxCells / ny || nx * ny > maxCells / nz) {
        std::fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real* current = nullptr;
    Real* next = nullptr;
    checkCuda(cudaMalloc(&current, bytes), "Allocating input grid");
    checkCuda(cudaMalloc(&next, bytes), "Allocating output grid");

    printf("Initializing grid...\n");
    constexpr unsigned threads = 128;
    const unsigned initBlocks = std::min<size_t>((gridSize + threads - 1) / threads, 65535);
    initializeGrid<<<initBlocks, threads>>>(current, next, gridSize);
    checkCuda(cudaGetLastError(), "Launching grid initialization");
    checkCuda(cudaDeviceSynchronize(), "Initializing grids");

    printf("Running stencil computation...\n");
    auto start = std::chrono::steady_clock::now();
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    if (hasInterior && iterations > 0) {
        const dim3 blocks((plane + threads - 1) / threads,
                          std::min<size_t>((nz - 2 + planesPerTile - 1) / planesPerTile, 65535));
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<blocks, threads>>>(current, next, nx, plane, nz);
            std::swap(current, next);
        }
        checkCuda(cudaGetLastError(), "Launching stencil");
        checkCuda(cudaDeviceSynchronize(), "Computing stencil");
    }
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    const auto millis = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %ld ms\n", millis.count());
    const double updates = hasInterior ? double(nx - 2) * (ny - 2) * (nz - 2) * std::max(iterations, 0) : 0;
    printf("Performance: %.3f MCellUpdates/s\n", updates / elapsed.count() / 1e6);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        // The serial loop selects its zero-initialized second buffer for odd
        // negative iteration counts, even though no updates were performed.
        if (!(iterations < 0 && iterations % 2 != 0)) {
            checkCuda(cudaMemcpy(finalGrid.data(), current, bytes, cudaMemcpyDeviceToHost),
                      "Reading final grid");
        }
    }
    checkCuda(cudaFree(next), "Freeing output grid");
    checkCuda(cudaFree(current), "Freeing input grid");
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
