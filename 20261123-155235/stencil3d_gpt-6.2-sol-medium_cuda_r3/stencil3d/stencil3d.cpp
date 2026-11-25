#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

constexpr unsigned int blockSize = 128;
constexpr unsigned int planesPerTile = 8;
constexpr unsigned int maxGridYZ = 65535;
constexpr unsigned int initializationBlockSize = 256;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

constexpr size_t divideRoundUp(size_t size, size_t divisor) {
    return size / divisor + (size % divisor != 0);
}

__device__ __forceinline__ Real divideBySeven(Real sum) {
    // The initialized grid stays in [0, 18], so its sums cannot overflow.
    // The residual is exactly representable. The quotient is at least 1/14
    // ulp from any rounding midpoint, so the rounded reciprocal also suffices
    // for the correction, including when the quotient is subnormal.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = __dmul_rn(sum, reciprocal);
    const Real remainder = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(remainder, reciprocal, quotient);
}

__global__ void initializeGrid(Real* first, Real* second, size_t size, bool zero) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < size; i += stride) {
        const Real value = zero ? 0.0 : static_cast<Real>(i % 19);
        first[i] = value;
        if (second) {
            second[i] = value;
        }
    }
}

template <typename Index>
__global__ void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                                 Index nx, Index ny, Index nz) {
    const Index x = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x == 0 || x >= nx - 1) {
        return;
    }

    const Index plane = nx * ny;
    // Stride over tiles if either grid dimension reached CUDA's launch limit.
    for (Index y = 1 + static_cast<Index>(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny - 1; y += static_cast<Index>(gridDim.y) * blockDim.y) {
        for (Index z = 1 + static_cast<Index>(blockIdx.z) * planesPerTile;
             z < nz - 1; z += static_cast<Index>(gridDim.z) * planesPerTile) {
            Index i = z * plane + y * nx + x;
            Real bottom = input[i - plane];
            Real center = input[i];
            const Index count = min(static_cast<Index>(planesPerTile), nz - 1 - z);

            // Adjacent threads traverse x, and each thread reuses its z neighbors.
            #pragma unroll
            for (Index k = 0; k < count; ++k) {
                const Real left = input[i - 1];
                const Real right = input[i + 1];
                const Real front = input[i - nx];
                const Real back = input[i + nx];
                const Real top = input[i + plane];
                const Real sum = center + left + right + front + back + bottom + top;
                output[i] = divideBySeven(sum);
                bottom = center;
                center = top;
                i += plane;
            }
        }
    }
}

template <typename Index>
void runIterations(Real*& current, Real*& next, size_t nx, size_t ny, size_t nz, int iterations) {
    // Group narrow rows together so they still occupy a useful fraction of a warp.
    const unsigned int width = nx <= 4 ? 4 : nx <= 8 ? 8 : nx <= 16 ? 16 : 32;
    const unsigned int threads = ny - 2 <= 32 / width ? 32 : blockSize;
    const dim3 block(width, threads / width);
    const size_t tilesY = divideRoundUp(ny - 2, block.y);
    const size_t tilesZ = divideRoundUp(nz - 2, planesPerTile);
    const dim3 grid(divideRoundUp(nx - 1, block.x), std::min<size_t>(tilesY, maxGridYZ),
                    std::min<size_t>(tilesZ, maxGridYZ));
    for (int iter = 0; iter < iterations; ++iter) {
        stencilIteration<Index><<<grid, block>>>(current, next, nx, ny, nz);
        // The kernels only update the interior. Both buffers have the same boundary.
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError(), "Launching CUDA stencil");
}

} // namespace

bool validateResult(const std::vector<Real>& grid) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    if (grid.empty()) {
        return true;
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
    if (nx > maxCells || (nx != 0 && ny > maxCells / nx) ||
        (nx != 0 && ny != 0 && nz > maxCells / (nx * ny))) {
        fprintf(stderr, "Grid dimensions exceed the addressable size\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    const bool updateInterior = iterations > 0 && nx > 2 && ny > 2 && nz > 2;
    Real* current = nullptr;
    Real* next = nullptr;

    printf("Initializing grid...\n");
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Creating CUDA start event");
    checkCuda(cudaEventCreate(&stop), "Creating CUDA stop event");
    if (gridSize != 0) {
        checkCuda(cudaMalloc(&current, bytes), "Allocating CUDA grid");
        if (updateInterior) {
            checkCuda(cudaMalloc(&next, bytes), "Allocating CUDA stencil buffer");
        }
        const unsigned int blocks = std::min<size_t>(divideRoundUp(gridSize, initializationBlockSize), maxGridYZ);
        // Negative odd counts originally selected the zero-initialized second buffer.
        const bool zero = iterations < 0 && iterations % 2 != 0;
        initializeGrid<<<blocks, initializationBlockSize>>>(current, next, gridSize, zero);
        checkCuda(cudaGetLastError(), "Launching CUDA grid initialization");
    }

    printf("Running stencil computation...\n");
    checkCuda(cudaEventRecord(start), "Recording CUDA start event");
    if (updateInterior) {
        if (gridSize <= std::numeric_limits<uint32_t>::max()) {
            runIterations<uint32_t>(current, next, nx, ny, nz, iterations);
        } else {
            runIterations<size_t>(current, next, nx, ny, nz, iterations);
        }
    }
    checkCuda(cudaEventRecord(stop), "Recording CUDA stop event");
    checkCuda(cudaEventSynchronize(stop), "Synchronizing CUDA stencil");
    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Measuring CUDA stencil time");
    checkCuda(cudaEventDestroy(start), "Destroying CUDA start event");
    checkCuda(cudaEventDestroy(stop), "Destroying CUDA stop event");

    printf("Computation time: %.3f ms\n", milliseconds);
    const double cellUpdates = updateInterior
        ? static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations : 0.0;
    const double mcups = milliseconds > 0.0f ? cellUpdates / (milliseconds * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if ((printResults || validate) && gridSize != 0) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), current, bytes, cudaMemcpyDeviceToHost),
                  "Copying CUDA stencil result");
    }
    checkCuda(cudaFree(current), "Freeing CUDA grid");
    checkCuda(cudaFree(next), "Freeing CUDA stencil buffer");

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
