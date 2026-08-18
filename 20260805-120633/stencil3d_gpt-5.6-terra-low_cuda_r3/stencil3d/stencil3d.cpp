#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

[[noreturn]] void cudaCheck(const cudaError_t status, const char* operation) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call) do { const cudaError_t status_ = (call); if (status_ != cudaSuccess) cudaCheck(status_, #call); } while (false)

__global__ void initializeGrid(Real* const grid, const size_t gridSize) {
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < gridSize;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        grid[idx] = static_cast<Real>(idx % 19);
    }
}

// A linear launch keeps neighboring X cells in a warp contiguous and processes
// both interior and boundary cells in one bandwidth-efficient pass.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 const size_t nx, const size_t ny,
                                 const size_t nz, const size_t gridSize) {
    const size_t plane = nx * ny;
    for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < gridSize;
         idx += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t z = idx / plane;
        const size_t inPlane = idx - z * plane;
        const size_t y = inPlane / nx;
        const size_t x = inPlane - y * nx;

        if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
            output[idx] = input[idx];
        } else {
            output[idx] = (input[idx] + input[idx - 1] + input[idx + 1] +
                           input[idx - nx] + input[idx + nx] +
                           input[idx - plane] + input[idx + plane]) / 7.0;
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
    
    size_t gridSize = nx * ny * nz;
    
    if (nx == 0 || ny == 0 || nz == 0 || gridSize / nx / ny != nz) {
        fprintf(stderr, "Grid dimensions must be non-zero and their product must fit in size_t\n");
        return 1;
    }

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceGrid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&deviceGrid2, gridSize * sizeof(Real)));

    int blockSize = 0;
    int minimumGridSize = 0;
    CUDA_CHECK(cudaOccupancyMaxPotentialBlockSize(
        &minimumGridSize, &blockSize, stencilIteration, 0, 0));
    // Launch enough blocks to saturate all SMs while letting each thread process
    // multiple cells for very large domains without a grid-size limit.
    int multiprocessors = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, 0));
    const int gridBlocks = std::max(minimumGridSize, multiprocessors * 32);

    // Initialize directly on the GPU; initialization is deliberately excluded
    // from the measured stencil-computation interval, as in the original code.
    printf("Initializing grid...\n");
    initializeGrid<<<gridBlocks, blockSize>>>(deviceGrid1, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    cudaEvent_t startEvent;
    cudaEvent_t endEvent;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&endEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));
    
    for (int iter = 0; iter < iterations; ++iter) {
        const Real* input = (iter % 2 == 0) ? deviceGrid1 : deviceGrid2;
        Real* output = (iter % 2 == 0) ? deviceGrid2 : deviceGrid1;
        stencilIteration<<<gridBlocks, blockSize>>>(input, output, nx, ny, nz, gridSize);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(endEvent));
    CUDA_CHECK(cudaEventSynchronize(endEvent));
    float milliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, startEvent, endEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(endEvent));
    
    printf("Computation time: %.3f ms\n", milliseconds);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (milliseconds / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real> finalGrid;
    if (printResults || validate) {
        finalGrid.resize(gridSize);
        const Real* deviceFinalGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), deviceFinalGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(deviceGrid1));
    CUDA_CHECK(cudaFree(deviceGrid2));

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
