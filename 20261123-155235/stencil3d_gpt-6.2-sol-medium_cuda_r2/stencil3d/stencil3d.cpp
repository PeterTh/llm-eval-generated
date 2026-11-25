#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

namespace {

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

template <typename Index>
__global__ void initializeGrid(Real* first, Real* second, Index count,
                               Index nx, Index ny, Index nz, bool zeroSecond) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += stride) {
        const Index cell = static_cast<Index>(index);
        const Real value = static_cast<Real>(cell % 19);
        first[index] = value;
        if (zeroSecond) {
            second[index] = 0.0;
        } else {
            const Index row = cell / nx;
            const Index x = cell % nx;
            const Index y = row % ny;
            const Index z = row / ny;
            // Initialize the fixed faces in both buffers so kernels only write interior cells.
            if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
                second[index] = value;
            }
        }
    }
}

constexpr unsigned planesPerThread = 4;
constexpr unsigned maxGridYZ = 65535;

__device__ __forceinline__ Real divideBySeven(Real sum) {
    // For these finite sums the fused residual is exact. Since seven is odd,
    // the correction error is too small to cross a binary rounding midpoint.
    constexpr Real inverse = 1.0 / 7.0;
    const Real estimate = sum * inverse;
    const Real residual = __fma_rn(-7.0, estimate, sum);
    return __fma_rn(residual, inverse, estimate);
}

template <typename Index>
__global__ void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                                 Index nx, Index ny, Index nz) {
    const Index x = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x == 0 || x >= nx - 1) {
        return;
    }

    const Index plane = nx * ny;
    for (Index y = static_cast<Index>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
         y < ny - 1; y += static_cast<Index>(gridDim.y) * blockDim.y) {
        for (Index z = static_cast<Index>(blockIdx.z) * planesPerThread + 1;
             z < nz - 1; z += static_cast<Index>(gridDim.z) * planesPerThread) {
            Index index = z * plane + y * nx + x;
            Real bottom = input[index - plane];
            Real center = input[index];

            // Reuse the three values along Z as the thread advances through its planes.
            #pragma unroll
            for (unsigned offset = 0; offset < planesPerThread; ++offset) {
                if (z + offset >= nz - 1) {
                    break;
                }

                const Real top = input[index + plane];
                const Real sum = center + input[index - 1] + input[index + 1] +
                                 input[index - nx] + input[index + nx] + bottom + top;
                output[index] = divideBySeven(sum);

                bottom = center;
                center = top;
                index += plane;
            }
        }
    }
}

}  // namespace

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    if (grid.empty()) {
        printf("Validation failed: grid is empty\n");
        return false;
    }

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
    
    constexpr size_t maxElements = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx != 0 && ny > maxElements / nx / nz) {
        fprintf(stderr, "Grid dimensions exceed the addressable grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    const bool use32BitIndices = gridSize <= std::numeric_limits<unsigned>::max();
    const bool zeroSecond = iterations < 0 && iterations % 2 != 0;
    Real* input = nullptr;
    Real* output = nullptr;

    printf("Initializing grid...\n");
    if (gridSize != 0) {
        checkCuda(cudaMalloc(&input, bytes), "Allocating first device grid");
        checkCuda(cudaMalloc(&output, bytes), "Allocating second device grid");
        constexpr size_t initThreads = 256;
        const unsigned initBlocks = std::min<size_t>((gridSize + initThreads - 1) / initThreads, maxGridYZ);
        if (use32BitIndices) {
            initializeGrid<unsigned><<<initBlocks, initThreads>>>(input, output, gridSize, nx, ny, nz, zeroSecond);
        } else {
            initializeGrid<size_t><<<initBlocks, initThreads>>>(input, output, gridSize, nx, ny, nz, zeroSecond);
        }
        checkCuda(cudaGetLastError(), "Launching grid initialization");
    }

    printf("Running stencil computation...\n");
    cudaEvent_t start;
    cudaEvent_t stop;
    checkCuda(cudaEventCreate(&start), "Creating start event");
    checkCuda(cudaEventCreate(&stop), "Creating stop event");
    checkCuda(cudaEventRecord(start), "Recording start event");

    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    if (hasInterior && iterations > 0) {
        // A warp spans adjacent rows when a row is narrower than a warp.
        const unsigned blockX = nx <= 8 ? 8 : nx <= 16 ? 16 : 32;
        const dim3 threads(blockX, 128 / blockX);
        const size_t yTiles = (ny - 2 + threads.y - 1) / threads.y;
        const size_t zTiles = (nz - 2 + planesPerThread - 1) / planesPerThread;
        const dim3 blocks((nx - 2) / threads.x + 1,
                          std::min<size_t>(yTiles, maxGridYZ),
                          std::min<size_t>(zTiles, maxGridYZ));

        for (int iter = 0; iter < iterations; ++iter) {
            if (use32BitIndices) {
                stencilIteration<unsigned><<<blocks, threads>>>(input, output, nx, ny, nz);
            } else {
                stencilIteration<size_t><<<blocks, threads>>>(input, output, nx, ny, nz);
            }
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "Launching stencil iterations");
    } else if (zeroSecond) {
        std::swap(input, output);
    }

    checkCuda(cudaEventRecord(stop), "Recording stop event");
    checkCuda(cudaEventSynchronize(stop), "Waiting for GPU computation");
    float elapsedMs = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, stop), "Measuring GPU computation");
    checkCuda(cudaEventDestroy(start), "Destroying start event");
    checkCuda(cudaEventDestroy(stop), "Destroying stop event");

    printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));
    const double cellUpdates = hasInterior && iterations > 0
                               ? static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations
                               : 0.0;
    const double mcups = elapsedMs > 0.0f ? cellUpdates / (elapsedMs * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (gridSize != 0) {
        if (printResults || validate) {
            finalGrid.resize(gridSize);
            checkCuda(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost),
                      "Copying final grid to host");
        }
        checkCuda(cudaFree(input), "Freeing first device grid");
        checkCuda(cudaFree(output), "Freeing second device grid");
    }

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
