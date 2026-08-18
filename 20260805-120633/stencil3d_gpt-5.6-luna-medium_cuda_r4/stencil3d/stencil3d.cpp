#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation) {
    fprintf(stderr, "CUDA error in %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        cudaFailure(error, operation);
    }
}

__global__ void stencilInterior(const Real* __restrict__ input,
                               Real* __restrict__ output,
                               const size_t nx, const size_t ny, const size_t nz) {
    // The x dimension is contiguous in memory, so mapping it to threadIdx.x
    // gives coalesced loads and stores for every z/y plane.
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x + 1;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y + 1;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z + 1;

    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) {
        return;
    }

    const size_t plane = nx * ny;
    const size_t index = z * plane + y * nx + x;
    const Real center = input[index];
    const Real left = input[index - 1];
    const Real right = input[index + 1];
    const Real front = input[index - nx];
    const Real back = input[index + nx];
    const Real bottom = input[index - plane];
    const Real top = input[index + plane];
    output[index] = (center + left + right + front + back + bottom + top) / 7.0;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

void stencilIteration(const Real* input, Real* output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) {
        return;
    }
    constexpr unsigned int tile = 8;
    const dim3 block(tile, tile, tile);
    const dim3 grid(static_cast<unsigned int>((nx - 2 + tile - 1) / tile),
                    static_cast<unsigned int>((ny - 2 + tile - 1) / tile),
                    static_cast<unsigned int>((nz - 2 + tile - 1) / tile));
    stencilInterior<<<grid, block>>>(input, output, nx, ny, nz);
    checkCuda(cudaGetLastError(), "stencilInterior launch");
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
    
    // Host grids are retained for initialization, final copy-back, and the
    // existing result/validation interfaces. The actual iteration data stays
    // resident on the GPU.
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    Real* deviceGrid1 = nullptr;
    Real* deviceGrid2 = nullptr;
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceGrid1), gridSize * sizeof(Real)),
              "cudaMalloc deviceGrid1");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&deviceGrid2), gridSize * sizeof(Real)),
              "cudaMalloc deviceGrid2");
    checkCuda(cudaMemcpy(deviceGrid1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice),
              "copy initial grid");
    // Every iteration copies boundaries from its input. Since boundaries are
    // never modified, initializing the second buffer once is equivalent and
    // removes a full boundary kernel/copy from every iteration.
    checkCuda(cudaMemcpy(deviceGrid2, deviceGrid1, gridSize * sizeof(Real), cudaMemcpyDeviceToDevice),
              "initialize second grid");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&end), "cudaEventCreate end");
    checkCuda(cudaEventRecord(start), "cudaEventRecord start");
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(deviceGrid1, deviceGrid2, nx, ny, nz);
        } else {
            stencilIteration(deviceGrid2, deviceGrid1, nx, ny, nz);
        }
    }

    checkCuda(cudaEventRecord(end), "cudaEventRecord end");
    checkCuda(cudaEventSynchronize(end), "cudaEventSynchronize end");
    float elapsedMilliseconds = 0.0f;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, end), "cudaEventElapsedTime");
    printf("Computation time: %.3f ms\n", elapsedMilliseconds);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = elapsedMilliseconds > 0.0f
        ? cellUpdates / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e6
        : 0.0;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    std::vector<Real>& finalGridStorage = (iterations % 2 == 0) ? grid1 : grid2;
    const Real* deviceFinalGrid = (iterations % 2 == 0) ? deviceGrid1 : deviceGrid2;
    checkCuda(cudaMemcpy(finalGridStorage.data(), deviceFinalGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost),
              "copy final grid");
    const std::vector<Real>& finalGrid = finalGridStorage;
    if (printResults) {
        print_results(finalGrid, "Grid");
    }

    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(end), "cudaEventDestroy end");
    checkCuda(cudaFree(deviceGrid1), "cudaFree deviceGrid1");
    checkCuda(cudaFree(deviceGrid2), "cudaFree deviceGrid2");
    
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
