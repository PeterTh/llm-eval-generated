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

__global__ void initializeGridKernel(Real* grid, size_t nx, size_t ny, size_t nz) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    while (idx < total) {
        grid[idx] = (idx % 19) * 1.0;
        idx += (size_t)blockDim.x * gridDim.x;
    }
}

void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    size_t total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    initializeGridKernel<<<blocks, threads>>>(grid, nx, ny, nz);
    cudaDeviceSynchronize();
}

__global__ void stencilKernel(const Real* input, Real* output, size_t nx, size_t ny, size_t nz) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = nx * ny * nz;
    while (idx < total) {
        size_t z = idx / (nx * ny);
        size_t rem = idx % (nx * ny);
        size_t y = rem / nx;
        size_t x = rem % nx;
        if (x == 0 || x == nx-1 || y == 0 || y == ny-1 || z == 0 || z == nz-1) {
            output[idx] = input[idx];
        } else {
            const Real center = input[idx];
            const Real left = input[idx - 1];
            const Real right = input[idx + 1];
            const Real front = input[idx - nx];
            const Real back = input[idx + nx];
            const Real bottom = input[idx - nx*ny];
            const Real top = input[idx + nx*ny];
            output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
        }
        idx += (size_t)blockDim.x * gridDim.x;
    }
}

void stencilIteration(const Real* input, Real* output, const size_t nx, const size_t ny, const size_t nz) {
    size_t total = nx * ny * nz;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 65535) blocks = 65535;
    stencilKernel<<<blocks, threads>>>(input, output, nx, ny, nz);
    cudaDeviceSynchronize();
}

bool validateResult(const Real* grid, size_t nx, size_t ny, size_t nz) {
    size_t total = nx * ny * nz;
    std::vector<Real> h(total);
    cudaMemcpy(h.data(), grid, total * sizeof(Real), cudaMemcpyDeviceToHost);
    for (const auto& val : h) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    Real minVal = h[0];
    Real maxVal = h[0];
    for (const auto& val : h) {
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
    size_t nx = 128;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 10;
    bool validate = false;
    bool printResults = false;

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

    Real* d_grid1 = nullptr;
    Real* d_grid2 = nullptr;
    cudaMalloc(&d_grid1, gridSize * sizeof(Real));
    cudaMalloc(&d_grid2, gridSize * sizeof(Real));

    printf("Initializing grid...\n");
    initializeGrid(d_grid1, nx, ny, nz);

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(d_grid1, d_grid2, nx, ny, nz);
        } else {
            stencilIteration(d_grid2, d_grid1, nx, ny, nz);
        }
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    const Real* finalGrid = (iterations % 2 == 0) ? d_grid1 : d_grid2;
    if (printResults) {
        std::vector<Real> h(gridSize);
        cudaMemcpy(h.data(), finalGrid, gridSize * sizeof(Real), cudaMemcpyDeviceToHost);
        print_results(h, "Grid");
    }

    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
        cudaFree(d_grid1);
        cudaFree(d_grid2);
        return valid ? 0 : 1;
    }

    cudaFree(d_grid1);
    cudaFree(d_grid2);
    return 0;
}
