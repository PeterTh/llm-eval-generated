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

// Check every CUDA operation, including asynchronous execution errors.
#define CUDA_CHECK(call) do { \
    cudaError_t error = (call); \
    if (error != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__global__ void initializeGrid(Real* grid1, Real* grid2, size_t count) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const Real value = static_cast<Real>(i % 19);
        grid1[i] = value;
        grid2[i] = value;
    }
}

// Each thread walks a short column in z, keeping the adjacent planes in
// registers. Threads in a warp access consecutive x coordinates.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz) {
    constexpr size_t zChunk = 4;
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    if (x >= nx - 1) return;
    const size_t plane = nx * ny;
    for (size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += static_cast<size_t>(gridDim.y) * blockDim.y) {
        for (size_t zStart = static_cast<size_t>(blockIdx.z) * zChunk + 1;
             zStart < nz - 1; zStart += static_cast<size_t>(gridDim.z) * zChunk) {
            size_t index = zStart * plane + y * nx + x;
            Real bottom = input[index - plane];
            Real center = input[index];
            const size_t zEnd = min(zStart + zChunk, nz - 1);
            for (size_t z = zStart; z < zEnd; ++z, index += plane) {
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
        fprintf(stderr, "Invalid or too large grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    Real *grid1 = nullptr, *grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1, bytes));
    CUDA_CHECK(cudaMalloc(&grid2, bytes));

    printf("Initializing grid...\n");
    constexpr unsigned int initThreads = 256;
    const unsigned int initBlocks = static_cast<unsigned int>(
        std::min<size_t>((gridSize - 1) / initThreads + 1, 65535));
    initializeGrid<<<initBlocks, initThreads>>>(grid1, grid2, gridSize);
    CUDA_CHECK(cudaGetLastError());
    // The original second buffer is all zero if a negative odd iteration
    // count selects it without executing any iterations.
    if (iterations < 0 && iterations % 2 != 0) {
        CUDA_CHECK(cudaMemset(grid2, 0, bytes));
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (nx > 2 && ny > 2 && nz > 2) {
        const dim3 block(32, 4);
        const dim3 grid(static_cast<unsigned int>((nx - 3) / block.x + 1),
                        static_cast<unsigned int>(std::min<size_t>((ny - 3) / block.y + 1, 65535)),
                        static_cast<unsigned int>(std::min<size_t>((nz - 3) / 4 + 1, 65535)));
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* input = (iter % 2 == 0) ? grid1 : grid2;
            Real* output = (iter % 2 == 0) ? grid2 : grid1;
            stencilIteration<<<grid, block>>>(input, output, nx, ny, nz);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                         static_cast<double>(ny > 2 ? ny - 2 : 0) *
                         static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        const Real* finalDeviceGrid = (iterations % 2 == 0) ? grid1 : grid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));
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
