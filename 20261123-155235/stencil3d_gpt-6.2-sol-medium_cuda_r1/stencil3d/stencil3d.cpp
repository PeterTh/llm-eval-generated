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

constexpr unsigned planesPerTile = 4;
constexpr unsigned threadsPerBlock = 256;
constexpr unsigned maxGridYZ = 65535;

size_t ceilDiv(size_t count, size_t tile) {
    return count / tile + (count % tile != 0);
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeGrids(Real* first, Real* second, size_t count, bool zero) {
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += stride) {
        const Real value = zero ? 0.0 : static_cast<Real>(i % 19);
        first[i] = value;
        if (second) {
            second[i] = value;
        }
    }
}

__device__ __forceinline__ Real divideBySeven(Real sum) {
    // Correct the reciprocal product with its exact residual. For binary inputs
    // divided by 7, the quotient is at least 1/14 ulp away from a rounding midpoint.
    constexpr Real reciprocal = 1.0 / 7.0;
    const Real quotient = sum * reciprocal;
    const Real residual = __fma_rn(-7.0, quotient, sum);
    return __fma_rn(reciprocal, residual, quotient);
}

__global__ void stencilIteration(const Real* __restrict__ input, Real* __restrict__ output,
                                 unsigned nx, unsigned ny, unsigned nz) {
    // Adjacent lanes work on adjacent X cells. Each thread reuses its Z neighbors.
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x == 0 || x >= nx - 1) {
        return;
    }

    const size_t plane = static_cast<size_t>(nx) * ny;
    const unsigned tileStride = gridDim.z * planesPerTile;
    const unsigned rowStride = gridDim.y * blockDim.y;
    for (unsigned y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
         y < ny - 1; y += rowStride) {
        for (unsigned z = 1 + blockIdx.z * planesPerTile;
             z < nz - 1; z += tileStride) {
            size_t i = z * plane + static_cast<size_t>(y) * nx + x;
            Real bottom = input[i - plane];
            Real center = input[i];
            const unsigned planes = min(planesPerTile, nz - 1 - z);
#pragma unroll
            for (unsigned offset = 0; offset < planesPerTile; ++offset) {
                if (offset >= planes) {
                    break;
                }
                const Real top = input[i + plane];
                const Real sum = center + input[i - 1] + input[i + 1] + input[i - nx]
                               + input[i + nx] + bottom + top;
                output[i] = divideBySeven(sum);
                bottom = center;
                center = top;
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
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid.empty() ? 0.0 : grid.front();
    Real maxVal = minVal;
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
    
    constexpr size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(Real);
    if (nx > maxCells || (nx && ny > maxCells / nx) ||
        (nx && ny && nz > maxCells / (nx * ny))) {
        fprintf(stderr, "Grid size is too large\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(Real);
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    const bool runStencil = iterations > 0 && hasInterior;

    Real* input = nullptr;
    Real* output = nullptr;
    printf("Initializing grid...\n");
    if (gridSize) {
        checkCuda(cudaMalloc(&input, bytes), "Allocate CUDA input grid");
        if (runStencil) {
            checkCuda(cudaMalloc(&output, bytes), "Allocate CUDA output grid");
        }

        // Both buffers start with the fixed boundary values. Kernels only write the interior.
        const unsigned blocks = std::min<size_t>(ceilDiv(gridSize, threadsPerBlock), 65535);
        // The original selects the zero-initialized buffer for negative odd counts.
        const bool zero = iterations < 0 && iterations % 2 != 0;
        initializeGrids<<<blocks, threadsPerBlock>>>(input, output, gridSize, zero);
        checkCuda(cudaGetLastError(), "Launch CUDA grid initialization");
    }

    printf("Running stencil computation...\n");
    float milliseconds = 0.0f;
    if (runStencil) {
        // Pack multiple rows into a warp for narrow grids.
        const unsigned width = nx >= 32 ? 32 : nx >= 16 ? 16 : nx >= 8 ? 8 : 4;
        const dim3 threads(width, std::min<size_t>(threadsPerBlock / width, ny - 2));
        const dim3 blocks(ceilDiv(nx - 1, threads.x),
                          std::min<size_t>(ceilDiv(ny - 2, threads.y), maxGridYZ),
                          std::min<size_t>(ceilDiv(nz - 2, planesPerTile), maxGridYZ));

        cudaEvent_t start, stop;
        checkCuda(cudaEventCreate(&start), "Create CUDA start event");
        checkCuda(cudaEventCreate(&stop), "Create CUDA stop event");
        checkCuda(cudaEventRecord(start), "Record CUDA start event");
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<blocks, threads>>>(input, output, nx, ny, nz);
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "Launch CUDA stencil");
        checkCuda(cudaEventRecord(stop), "Record CUDA stop event");
        checkCuda(cudaEventSynchronize(stop), "Complete CUDA stencil");
        checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "Measure CUDA stencil time");
        checkCuda(cudaEventDestroy(start), "Destroy CUDA start event");
        checkCuda(cudaEventDestroy(stop), "Destroy CUDA stop event");
    } else if (gridSize) {
        checkCuda(cudaDeviceSynchronize(), "Complete CUDA grid initialization");
    }

    printf("Computation time: %.3f ms\n", milliseconds);
    const double cellUpdates = runStencil ? static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations : 0.0;
    const double mcups = milliseconds > 0 ? cellUpdates / (milliseconds * 1000.0) : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        if (gridSize) {
            checkCuda(cudaMemcpy(finalGrid.data(), input, bytes, cudaMemcpyDeviceToHost),
                      "Read CUDA result");
        }
    }
    checkCuda(cudaFree(output), "Free CUDA output grid");
    checkCuda(cudaFree(input), "Free CUDA input grid");

    // Print results for external validation
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
