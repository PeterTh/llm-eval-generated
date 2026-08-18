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

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        const cudaError_t error = (call);                                                   \
        if (error != cudaSuccess) {                                                         \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(error));                                        \
            std::exit(EXIT_FAILURE);                                                        \
        }                                                                                   \
    } while (0)

// Initializing on the device avoids an otherwise unnecessary host-to-device copy.
__global__ void initializeGridKernel(Real* grid, const size_t count) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) grid[index] = static_cast<Real>(index % 19);
}

// Each thread owns one output cell.  This keeps x-neighbours contiguous between adjacent
// threads, producing coalesced loads and stores while retaining the exact 7-point semantics.
__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t plane = nx * ny;
    const size_t index = z * plane + y * nx + x;
    if (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz) {
        output[index] = input[index];
    } else {
        output[index] = (input[index] + input[index - 1] + input[index + 1] +
                         input[index - nx] + input[index + nx] + input[index - plane] +
                         input[index + plane]) / 7.0;
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128, ny = 0, nz = 0;
    int iterations = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = atoi(argv[++i]);
        else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = atoi(argv[++i]);
        else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx < 2 || ny < 2 || nz < 2 || iterations < 0 || nx > std::numeric_limits<size_t>::max() / ny / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }

    printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",
           nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    const size_t gridSize = nx * ny * nz;
    std::vector<Real> finalGrid;
    Real *grid1 = nullptr, *grid2 = nullptr;
    CUDA_CHECK(cudaMalloc(&grid1, gridSize * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&grid2, gridSize * sizeof(Real)));

    printf("Initializing grid...\n");
    constexpr unsigned int initBlockSize = 256;
    const unsigned int initBlocks = static_cast<unsigned int>((gridSize + initBlockSize - 1) / initBlockSize);
    initializeGridKernel<<<initBlocks, initBlockSize>>>(grid1, gridSize);
    CUDA_CHECK(cudaGetLastError());

    printf("Running stencil computation...\n");
    constexpr dim3 block(32, 4, 2);
    const dim3 grid(static_cast<unsigned int>((nx + block.x - 1) / block.x),
                    static_cast<unsigned int>((ny + block.y - 1) / block.y),
                    static_cast<unsigned int>((nz + block.z - 1) / block.z));
    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iterations; ++iter) {
        Real* input = (iter & 1) ? grid2 : grid1;
        Real* output = (iter & 1) ? grid1 : grid2;
        stencilKernel<<<grid, block>>>(input, output, nx, ny, nz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    printf("Computation time: %ld ms\n", static_cast<long>(milliseconds));
    const double cellUpdates = static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations;
    const double mcups = milliseconds > 0.0f ? cellUpdates / milliseconds / 1e3 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        finalGrid.resize(gridSize);
        const Real* finalDeviceGrid = (iterations % 2 == 0) ? grid1 : grid2;
        CUDA_CHECK(cudaMemcpy(finalGrid.data(), finalDeviceGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(grid1));
    CUDA_CHECK(cudaFree(grid2));
    if (printResults) print_results(finalGrid, "Grid");
    if (validate) {
        printf("Validating result...\n");
        if (validateResult(finalGrid, nx, ny, nz)) { printf("Validation: PASSED\n"); return 0; }
        printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
