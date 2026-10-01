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

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

// Each thread advances through a short Z segment, reusing its neighboring
// planes in registers. Both device grids start with the same boundary values.
__global__ void stencilKernel(const Real* __restrict__ input,
                              Real* __restrict__ output,
                              size_t nx, size_t ny, size_t nz) {
    const size_t x = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t y = 1 + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx - 1 || y >= ny - 1) return;

    const size_t plane = nx * ny;
    const size_t end = min(static_cast<size_t>(1 + (blockIdx.z + 1) * 8), nz - 1);
    const size_t first = 1 + blockIdx.z * 8;
    size_t pos = first * plane + y * nx + x;
    Real bottom = input[pos - plane];
    Real center = input[pos];
    for (size_t z = first; z < end; ++z, pos += plane) {
        const Real top = input[pos + plane];
        const Real left = input[pos - 1];
        const Real right = input[pos + 1];
        const Real front = input[pos - nx];
        const Real back = input[pos + nx];
        output[pos] = (center + left + right + front + back + bottom + top) / 7.0;
        bottom = center;
        center = top;
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

    if (nx == 0 || ny == 0 || nz == 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz ||
        nx * ny * nz > std::numeric_limits<size_t>::max() / sizeof(Real)) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate and initialize the host grid.
    std::vector<Real> grid1(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    const size_t bytes = gridSize * sizeof(Real);
    Real* device1 = nullptr;
    Real* device2 = nullptr;
    checkCuda(cudaMalloc(&device1, bytes), "cudaMalloc grid 1");
    checkCuda(cudaMalloc(&device2, bytes), "cudaMalloc grid 2");
    checkCuda(cudaMemcpy(device1, grid1.data(), bytes, cudaMemcpyHostToDevice), "copy initial grid");
    checkCuda(cudaMemcpy(device2, device1, bytes, cudaMemcpyDeviceToDevice), "initialize second grid");
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    Real* input = device1;
    Real* output = device2;
    if (nx > 2 && ny > 2 && nz > 2) {
        const dim3 block(32, 8);
        const dim3 grid((nx - 2 + block.x - 1) / block.x,
                        (ny - 2 + block.y - 1) / block.y,
                        (nz - 2 + 7) / 8);
        for (int iter = 0; iter < iterations; ++iter) {
            stencilKernel<<<grid, block>>>(input, output, nx, ny, nz);
            std::swap(input, output);
        }
    }
    checkCuda(cudaGetLastError(), "stencil launch");
    checkCuda(cudaDeviceSynchronize(), "stencil computation");
    
    auto end = std::chrono::high_resolution_clock::now();
    const double elapsedMs = std::chrono::duration<double, std::milli>(end - start).count();
    
    printf("Computation time: %.3f ms\n", elapsedMs);
    
    // Calculate performance metrics
    double cellUpdates = (double)(nx > 2 ? nx - 2 : 0) *
                         (ny > 2 ? ny - 2 : 0) * (nz > 2 ? nz - 2 : 0) * iterations;
    double mcups = cellUpdates / (elapsedMs * 1000.0);
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults || validate) {
        checkCuda(cudaMemcpy(grid1.data(), input, bytes, cudaMemcpyDeviceToHost), "copy result grid");
    }
    checkCuda(cudaFree(device1), "free grid 1");
    checkCuda(cudaFree(device2), "free grid 2");
    if (printResults) {
        print_results(grid1, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(grid1, nx, ny, nz);
        
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
