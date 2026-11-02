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

// Each thread walks a short run in Z, keeping its three Z neighbors in
// registers. Adjacent threads access adjacent X values for coalesced loads.
__global__ void stencilIteration(const Real* __restrict__ input,
                                 Real* __restrict__ output,
                                 size_t nx, size_t ny, size_t nz,
                                 size_t tilesX, size_t tilesY) {
    constexpr size_t zPerBlock = 4;
    size_t tile = blockIdx.x;
    const size_t tileX = tile % tilesX;
    tile /= tilesX;
    const size_t tileY = tile % tilesY;
    const size_t tileZ = tile / tilesY;
    const size_t x = tileX * blockDim.x + threadIdx.x + 1;
    const size_t y = tileY * blockDim.y + threadIdx.y + 1;
    const size_t z = tileZ * zPerBlock + 1;
    if (x >= nx - 1 || y >= ny - 1 || z >= nz - 1) return;

    const size_t plane = nx * ny;
    size_t index = z * plane + y * nx + x;
    Real bottom = input[index - plane];
    Real center = input[index];
    #pragma unroll
    for (size_t dz = 0; dz < zPerBlock && z + dz < nz - 1; ++dz) {
        const Real top = input[index + plane];
        const Real left = input[index - 1];
        const Real right = input[index + 1];
        const Real front = input[index - nx];
        const Real back = input[index + nx];
        output[index] = (center + left + right + front + back + bottom + top) / 7.0;
        bottom = center;
        center = top;
        index += plane;
    }
}

static void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
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
    
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Boundaries never change, so initialize both device buffers once.
    std::vector<Real> grid1(gridSize);
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    Real *device1 = nullptr, *device2 = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    checkCuda(cudaMalloc(&device1, bytes), "allocating first grid");
    checkCuda(cudaMalloc(&device2, bytes), "allocating second grid");
    checkCuda(cudaMemcpy(device1, grid1.data(), bytes, cudaMemcpyHostToDevice), "copying initial grid");
    checkCuda(cudaMemcpy(device2, device1, bytes, cudaMemcpyDeviceToDevice), "initializing second grid");

    const dim3 block(32, 4);
    size_t tilesX = 0, tilesY = 0, tiles = 0;
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    if (hasInterior && iterations > 0) {
        tilesX = (nx - 2 + block.x - 1) / block.x;
        tilesY = (ny - 2 + block.y - 1) / block.y;
        const size_t tilesZ = (nz - 2 + 3) / 4;
        tiles = tilesX * tilesY * tilesZ;
        int device = 0;
        cudaDeviceProp properties{};
        checkCuda(cudaGetDevice(&device), "getting device");
        checkCuda(cudaGetDeviceProperties(&properties, device), "getting device properties");
        if (tiles > static_cast<size_t>(properties.maxGridSize[0])) {
            fprintf(stderr, "Grid is too large for CUDA launch\n");
            return 1;
        }
    }

    // The device-to-device initialization may be asynchronous on the host.
    checkCuda(cudaDeviceSynchronize(), "finishing grid initialization");
    printf("Running stencil computation...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    Real* input = device1;
    Real* output = device2;
    if (hasInterior) {
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration<<<static_cast<unsigned int>(tiles), block>>>(
                input, output, nx, ny, nz, tilesX, tilesY);
            std::swap(input, output);
        }
        checkCuda(cudaGetLastError(), "launching stencil kernel");
    }
    checkCuda(cudaDeviceSynchronize(), "running stencil kernel");
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Computation time: %ld ms\n", duration.count());
    const double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0) *
                               static_cast<double>(ny > 2 ? ny - 2 : 0) *
                               static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);

    if (printResults || validate) {
        checkCuda(cudaMemcpy(grid1.data(), input, bytes, cudaMemcpyDeviceToHost), "copying result grid");
    }
    checkCuda(cudaFree(device1), "freeing first grid");
    checkCuda(cudaFree(device2), "freeing second grid");
    const std::vector<Real>& finalGrid = grid1;

    // Print results for external validation
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
