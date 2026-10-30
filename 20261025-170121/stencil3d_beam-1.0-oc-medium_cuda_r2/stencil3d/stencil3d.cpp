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

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__global__ void stencilKernel(const Real* __restrict__ input, Real* __restrict__ output,
                             size_t nx, size_t ny, size_t nz) {
    size_t x = blockIdx.x * blockDim.x + threadIdx.x;
    size_t y = blockIdx.y * blockDim.y + threadIdx.y;
    size_t z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;
    size_t idx = z * (nx * ny) + y * nx + x;
    if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
        output[idx] = input[idx];
        return;
    }
    Real center = input[idx];
    Real left = input[idx - 1];
    Real right = input[idx + 1];
    Real front = input[idx - nx];
    Real back = input[idx + nx];
    Real bottom = input[idx - nx*ny];
    Real top = input[idx + nx*ny];
    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    for (size_t z = 0; z < nz; ++z)
        for (size_t y = 0; y < ny; ++y)
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    size_t gridSize = nx * ny * nz;
    std::vector<Real> grid1(gridSize), grid2(gridSize);
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    Real *d1, *d2;
    cudaMalloc(&d1, gridSize * sizeof(Real));
    cudaMalloc(&d2, gridSize * sizeof(Real));
    cudaMemcpy(d1, grid1.data(), gridSize * sizeof(Real), cudaMemcpyHostToDevice);

    dim3 block(8,8,8);
    dim3 grid((nx+block.x-1)/block.x, (ny+block.y-1)/block.y, (nz+block.z-1)/block.z);

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) stencilKernel<<<grid, block>>>(d1, d2, nx, ny, nz);
        else stencilKernel<<<grid, block>>>(d2, d1, nx, ny, nz);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());
    double cellUpdates = (double)((nx-2)*(ny-2)*(nz-2))*iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* finalDev = (iterations % 2 == 0) ? d2 : d1;
    std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid2 : grid1;
    cudaMemcpy(finalGrid.data(), finalDev, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);

    if (printResults) print_results(finalGrid, "Grid");

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        if (valid) { printf("Validation: PASSED\n"); }
        else { printf("Validation: FAILED\n"); }
        cudaFree(d1); cudaFree(d2);
        return valid ? 0 : 1;
    }
    cudaFree(d1); cudaFree(d2);
    return 0;
}
