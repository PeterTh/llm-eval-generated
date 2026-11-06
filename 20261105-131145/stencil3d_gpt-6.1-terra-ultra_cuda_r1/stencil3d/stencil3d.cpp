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

// Each thread updates a short column in z.  Consecutive threads access
// consecutive x coordinates, and the center and z neighbors are reused as the
// thread advances through its column.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesY, size_t tileCount) {
    constexpr size_t zChunk = 4;
    const size_t x = 1 + static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t plane = nx * ny;
    for (size_t tile = blockIdx.y; tile < tileCount; tile += gridDim.y) {
        const size_t y = 1 + (tile % tilesY) * blockDim.y + threadIdx.y;
        const size_t zStart = 1 + (tile / tilesY) * zChunk;
        if (x >= nx - 1 || y >= ny - 1) continue;

        size_t idx = zStart * plane + y * nx + x;
        Real bottom = input[idx - plane];
        Real center = input[idx];
        for (size_t z = zStart; z < nz - 1 && z < zStart + zChunk; ++z) {
            const Real top = input[idx + plane];
            const Real left = input[idx - 1];
            const Real right = input[idx + 1];
            const Real front = input[idx - nx];
            const Real back = input[idx + nx];
            output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            bottom = center;
            center = top;
            idx += plane;
        }
    }
}

static void checkCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
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
    
    // The boundary never changes, so initialize it in both device buffers once.
    std::vector<Real> grid1(gridSize);
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    Real* device1 = nullptr;
    Real* device2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    checkCuda(cudaMalloc(&device1, bytes), "allocating the first grid");
    checkCuda(cudaMalloc(&device2, bytes), "allocating the second grid");
    checkCuda(cudaMemcpy(device1, grid1.data(), bytes, cudaMemcpyHostToDevice),
              "copying the initial grid to the device");
    checkCuda(cudaMemcpy(device2, device1, bytes, cudaMemcpyDeviceToDevice),
              "initializing the second device grid");

    // Prepare the launch and warm up the kernel before timing, so CUDA's
    // first-launch setup and any device-code JIT are excluded from the measurement.
    constexpr unsigned int blockX = 32;
    constexpr unsigned int blockY = 4;
    constexpr size_t zChunk = 4;
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    size_t tilesY = 0;
    size_t tileCount = 0;
    dim3 grid;
    const dim3 block(blockX, blockY);
    if (hasInterior) {
        tilesY = (ny - 2 + blockY - 1) / blockY;
        const size_t tilesZ = (nz - 2 + zChunk - 1) / zChunk;
        tileCount = tilesY * tilesZ;
        grid = dim3(static_cast<unsigned int>((nx - 2 + blockX - 1) / blockX),
                    static_cast<unsigned int>(std::min<size_t>(tileCount, 65535)));
        if (iterations > 0) {
            stencilIteration<<<grid, block>>>(device1, device2, nx, ny, nz, tilesY, tileCount);
            checkCuda(cudaGetLastError(), "launching the warm-up kernel");
            checkCuda(cudaDeviceSynchronize(), "running the warm-up kernel");
        }
    }

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    if (hasInterior) {
        Real* input = device1;
        Real* output = device2;
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<grid, block>>>(input, output, nx, ny, nz, tilesY, tileCount);
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "launching the stencil kernel");
    }
    checkCuda(cudaDeviceSynchronize(), "running the stencil kernels");
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();

    printf("Computation time: %.3f ms\n", elapsedMs);

    const double cellUpdates = hasInterior ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * std::max(iterations, 0) : 0.0;
    const double mcups = cellUpdates / elapsedMs / 1000.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        // The original program selects its zero-initialized second grid for a
        // negative odd iteration count, even though it runs no iterations.
        if (iterations < 0 && iterations % 2 != 0) {
            std::fill(grid1.begin(), grid1.end(), 0.0);
        } else {
            const Real* finalDevice = (iterations % 2 == 0) ? device1 : device2;
            checkCuda(cudaMemcpy(grid1.data(), finalDevice, bytes, cudaMemcpyDeviceToHost),
                      "copying the result to the host");
        }
    }
    checkCuda(cudaFree(device1), "freeing the first grid");
    checkCuda(cudaFree(device2), "freeing the second grid");
    const std::vector<Real>& finalGrid = grid1;
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
