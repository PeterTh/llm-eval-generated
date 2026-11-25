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

namespace {
constexpr unsigned blockX = 32;
constexpr unsigned blockY = 2;
constexpr unsigned planesPerBlock = 8;
constexpr unsigned maxGridYZ = 65535;
constexpr unsigned maxLinearBlocks = 65535;

void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrid(Real* first, Real* second, size_t size, bool zero) {
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < size; i += stride) {
        const Real value = zero ? 0.0 : static_cast<Real>(i % 19);
        first[i] = value;
        if (second) {
            second[i] = value;
        }
    }
}

__device__ __noinline__ Real divideSmallSum(Real sum) {
    return sum / 7.0;
}

__device__ __forceinline__ Real stencilAverage(Real sum) {
    if (sum < 0x1p-960) {
        return divideSmallSum(sum);
    }
    // Split 1/7 across two doubles. The fused product's error is much smaller
    // than the distance from sum/7 to a rounding midpoint.
    constexpr Real reciprocal = 1.0 / 7.0;
    constexpr Real reciprocalTail = reciprocal * 0x1p-54;
    return __fma_rn(sum, reciprocal, __dmul_rn(sum, reciprocalTail));
}

template <typename Index>
__device__ __forceinline__ Real stencilValue(const Real* input, Index i, Index nx,
                                             Real bottom, Real center, Real top) {
    return stencilAverage(center + input[i - 1] + input[i + 1] + input[i - nx] +
                          input[i + nx] + bottom + top);
}

template <typename Index>
__global__ void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                                 Index nx, Index ny, Index nz) {
    // Map each warp to consecutive X coordinates and reuse the Z neighbors in registers.
    const Index x = blockIdx.x * blockX + threadIdx.x;
    if (x == 0 || x >= nx - 1) {
        return;
    }

    const Index plane = nx * ny;
    for (Index y = 1 + blockIdx.y * blockY + threadIdx.y; y < ny - 1;
         y += gridDim.y * blockY) {
        for (Index z = 1 + blockIdx.z * planesPerBlock; z < nz - 1;
             z += gridDim.z * planesPerBlock) {
            const Index count = min(static_cast<Index>(planesPerBlock), nz - 1 - z);
            Index i = z * plane + y * nx + x;
            Real bottom = input[i - plane];
            Real center = input[i];
            for (Index k = 0; k < count; ++k, i += plane) {
                const Real top = input[i + plane];
                output[i] = stencilValue(input, i, nx, bottom, center, top);
                bottom = center;
                center = top;
            }
        }
    }
}

template <typename Index>
__global__ void stencilNarrow(const Real* __restrict__ input, Real* __restrict__ output,
                              Index nx, Index ny, Index count) {
    // Pack interior rows into a warp when the X dimension is narrow.
    const Index width = nx - 2;
    const Index interiorPlane = width * (ny - 2);
    const Index plane = nx * ny;
    const size_t stride = gridDim.x * blockDim.x;
    for (size_t j = blockIdx.x * blockDim.x + threadIdx.x; j < count; j += stride) {
        const Index cell = static_cast<Index>(j);
        const Index z = cell / interiorPlane;
        const Index rowCell = cell % interiorPlane;
        const Index y = rowCell / width;
        const Index x = rowCell % width;
        const Index i = (z + 1) * plane + (y + 1) * nx + x + 1;
        output[i] = stencilValue(input, i, nx, input[i - plane], input[i], input[i + plane]);
    }
}

template <typename Index>
void runStencil(Real*& input, Real*& output, Index nx, Index ny, Index nz, int iterations) {
    const dim3 threads(blockX, blockY);
    const dim3 blocks((nx - 1 + blockX - 1) / blockX,
                      std::min((ny - 2 + blockY - 1) / blockY, Index{maxGridYZ}),
                      std::min((nz - 2 + planesPerBlock - 1) / planesPerBlock, Index{maxGridYZ}));
    constexpr unsigned narrowThreads = 128;
    const Index count = (nx - 2) * (ny - 2) * (nz - 2);
    const unsigned narrowBlocks = std::min(size_t{maxLinearBlocks},
                                          (size_t{count} + narrowThreads - 1) / narrowThreads);

    for (int iter = 0; iter < iterations; ++iter) {
        if (nx <= blockX / 2) {
            stencilNarrow<<<narrowBlocks, narrowThreads>>>(input, output, nx, ny, count);
        } else {
            stencilIteration<<<blocks, threads>>>(input, output, nx, ny, nz);
        }
        std::swap(input, output);
    }
    checkCuda(cudaGetLastError(), "Launch stencil kernel");
}
}  // namespace

bool validateResult(const std::vector<Real>& grid) {
    if (grid.empty()) {
        return false;
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const Real val : grid) {
        if (!std::isfinite(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
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
    
    constexpr size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx && (ny > maxElements / nx || nz > maxElements / (nx * ny))) {
        fprintf(stderr, "Grid size is too large\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t gridBytes = gridSize * sizeof(Real);
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const bool runIterations = hasInterior && iterations > 0;

    Real* input = nullptr;
    Real* output = nullptr;
    if (gridSize) {
        checkCuda(cudaMalloc(&input, gridBytes), "Allocate input grid");
        if (runIterations) {
            checkCuda(cudaMalloc(&output, gridBytes), "Allocate output grid");
        }
    }

    // Both buffers start with the same boundary values, so each iteration writes
    // only interior points. A negative odd iteration count selects the original zero buffer.
    printf("Initializing grid...\n");
    constexpr unsigned initThreads = 256;
    const unsigned initBlocks = std::clamp((gridSize + initThreads - 1) / initThreads,
                                           size_t{1}, size_t{maxLinearBlocks});
    initializeGrid<<<initBlocks, initThreads>>>(input, output, gridSize,
                                                iterations < 0 && iterations % 2 != 0);
    checkCuda(cudaGetLastError(), "Launch initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "Initialize grids");

    printf("Running stencil computation...\n");
    const auto start = std::chrono::steady_clock::now();
    if (runIterations) {
        if (gridSize <= std::numeric_limits<unsigned>::max()) {
            runStencil(input, output, static_cast<unsigned>(nx), static_cast<unsigned>(ny),
                       static_cast<unsigned>(nz), iterations);
        } else {
            runStencil(input, output, nx, ny, nz, iterations);
        }
        checkCuda(cudaDeviceSynchronize(), "Run stencil iterations");
    }
    const auto end = std::chrono::steady_clock::now();
    const auto elapsed = end - start;
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %ld ms\n", duration.count());

    const size_t interiorSize = hasInterior ? (nx - 2) * (ny - 2) * (nz - 2) : 0;
    const double cellUpdates = static_cast<double>(interiorSize) * iterations;
    const double mcups = cellUpdates / std::chrono::duration<double>(elapsed).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if ((printResults || validate) && gridSize) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), input, gridBytes, cudaMemcpyDeviceToHost), "Read grid");
    }
    if (output) {
        checkCuda(cudaFree(output), "Free output grid");
    }
    if (input) {
        checkCuda(cudaFree(input), "Free input grid");
    }

    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid);
        
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
