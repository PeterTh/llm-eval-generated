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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrid(Real* first, Real* second, size_t count, bool zero) {
    const size_t stride = size_t{blockDim.x} * gridDim.x;
    for (size_t i = size_t{blockIdx.x} * blockDim.x + threadIdx.x; i < count; i += stride) {
        const Real value = zero ? 0.0 : static_cast<Real>(i % 19);
        first[i] = value;
        if (second) {
            second[i] = value;
        }
    }
}

__device__ inline Real average(Real sum) {
    // For the stencil's bounded, normal sums, the FMA gives an exact residual.
    // The correction rounds like division because sum / 7 is at least 1/14
    // ulp from a rounding midpoint. Zero also passes through exactly.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = __dmul_rn(sum, reciprocal);
    const Real residual = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(residual, reciprocal, quotient);
}

template <unsigned Planes>
__global__ void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                                 unsigned nx, unsigned ny, unsigned nz, unsigned yTiles) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x + 1;
    const size_t plane = size_t{nx} * ny;

    // Grid-stride loops keep the full domain addressable when Y or Z exceeds
    // the CUDA grid dimension limit.
    for (unsigned z = blockIdx.z * Planes + 1; z < nz - 1; z += gridDim.z * Planes) {
        for (unsigned tile = blockIdx.y; tile < yTiles; tile += gridDim.y) {
            const unsigned y = tile * blockDim.y + threadIdx.y + 1;
            if (x >= nx - 1 || y >= ny - 1) {
                continue;
            }

            size_t i = z * plane + size_t{y} * nx + x;
            Real below = input[i - plane];
            Real center = input[i];

            // Keep adjacent Z planes in registers across successive updates.
#pragma unroll
            for (unsigned p = 0; p < Planes; ++p) {
                if (z + p >= nz - 1) {
                    break;
                }
                const Real above = input[i + plane];
                const Real sum = center + input[i - 1] + input[i + 1]
                                        + input[i - nx] + input[i + nx] + below + above;
                output[i] = average(sum);
                below = center;
                center = above;
                i += plane;
            }
        }
    }
}

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
    constexpr size_t maxDimension = std::numeric_limits<int>::max();
    if (nx > maxDimension || ny > maxDimension || nz > maxDimension ||
        (nx != 0 && ny > maxElements / nx) ||
        (nx * ny != 0 && nz > maxElements / (nx * ny))) {
        fprintf(stderr, "Grid dimensions are too large\n");
        return EXIT_FAILURE;
    }

    const size_t gridSize = nx * ny * nz;
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const bool runStencil = hasInterior && iterations > 0;
    Real* current = nullptr;
    Real* next = nullptr;
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "Create CUDA start event");
    checkCuda(cudaEventCreate(&stop), "Create CUDA stop event");

    printf("Initializing grid...\n");
    if (gridSize != 0) {
        const size_t bytes = gridSize * sizeof(Real);
        checkCuda(cudaMalloc(&current, bytes), "Allocate CUDA grid");
        if (runStencil) {
            checkCuda(cudaMalloc(&next, bytes), "Allocate second CUDA grid");
        }

        // Both buffers start with identical boundaries, which no stencil launch writes.
        constexpr unsigned threads = 256;
        const unsigned blocks = std::min<size_t>((gridSize + threads - 1) / threads, 4096);
        // With a negative odd iteration count, the original selects the zeroed buffer.
        const bool zero = iterations < 0 && iterations % 2 != 0;
        initializeGrid<<<blocks, threads>>>(current, next, gridSize, zero);
        checkCuda(cudaGetLastError(), "Launch grid initialization");
    }

    printf("Running stencil computation...\n");
    checkCuda(cudaEventRecord(start), "Record CUDA start event");
    if (runStencil) {
        constexpr unsigned planes = 4;
        const unsigned blockWidth = nx <= 10 ? 8 : nx <= 18 ? 16 : 32;
        const dim3 threads(blockWidth, 128 / blockWidth);
        const unsigned xTiles = (nx - 2 + threads.x - 1) / threads.x;
        const unsigned yTiles = (ny - 2 + threads.y - 1) / threads.y;
        const unsigned zTiles = (nz - 2 + planes - 1) / planes;
        const dim3 blocks(xTiles, std::min(yTiles, 65535u), std::min(zTiles, 65535u));

        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<planes><<<blocks, threads>>>(current, next, nx, ny, nz, yTiles);
            std::swap(current, next);
        }
        checkCuda(cudaGetLastError(), "Launch CUDA stencil");
    }
    checkCuda(cudaEventRecord(stop), "Record CUDA stop event");
    checkCuda(cudaEventSynchronize(stop), "Wait for CUDA stencil");

    float milliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Measure CUDA stencil time");
    checkCuda(cudaEventDestroy(start), "Destroy CUDA start event");
    checkCuda(cudaEventDestroy(stop), "Destroy CUDA stop event");

    printf("Computation time: %.3f ms\n", milliseconds);
    const double cellUpdates = runStencil ? double((nx - 2) * (ny - 2) * (nz - 2)) * iterations : 0;
    const double mcups = milliseconds > 0 ? cellUpdates / (milliseconds * 1000.0) : 0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if ((printResults || validate) && gridSize != 0) {
        finalGrid.resize(gridSize);
        checkCuda(cudaMemcpy(finalGrid.data(), current, gridSize * sizeof(Real),
                             cudaMemcpyDeviceToHost), "Copy CUDA result to host");
    }
    checkCuda(cudaFree(current), "Free CUDA grid");
    checkCuda(cudaFree(next), "Free second CUDA grid");

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
