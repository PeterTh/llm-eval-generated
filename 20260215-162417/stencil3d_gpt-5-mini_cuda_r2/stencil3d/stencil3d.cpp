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

// CUDA error checking
static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
}

// Kernel: initialize grid
__global__ void initKernel(Real* grid, size_t nx, size_t ny, size_t nz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    if (idx >= total) return;
    grid[idx] = static_cast<Real>(idx % 19) * 1.0;
}

// Kernel: interior 7-point stencil
__global__ void stencilKernel(const Real* input, Real* output, size_t nx, size_t ny, size_t nz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    if (idx >= total) return;

    const size_t nxy = nx * ny;
    const size_t z = idx / nxy;
    const size_t rem = idx % nxy;
    const size_t y = rem / nx;
    const size_t x = rem % nx;

    // Skip boundaries
    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) return;

    const Real center = input[idx];
    const Real left = input[idx - 1];
    const Real right = input[idx + 1];
    const Real front = input[idx - nx];
    const Real back = input[idx + nx];
    const Real bottom = input[idx - nxy];
    const Real top = input[idx + nxy];

    output[idx] = (center + left + right + front + back + bottom + top) / static_cast<Real>(7.0);
}

// Kernel: copy boundary values
__global__ void copyBoundaryKernel(const Real* input, Real* output, size_t nx, size_t ny, size_t nz) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = nx * ny * nz;
    if (idx >= total) return;

    const size_t nxy = nx * ny;
    const size_t z = idx / nxy;
    const size_t rem = idx % nxy;
    const size_t y = rem / nx;
    const size_t x = rem % nx;

    if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        output[idx] = input[idx];
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
    
    // Allocate host grids for final copy / validation
    std::vector<Real> hostGridA(gridSize);
    std::vector<Real> hostGridB(gridSize);

    // Allocate device grids
    Real* d_a = nullptr;
    Real* d_b = nullptr;
    const size_t bytes = gridSize * sizeof(Real);
    checkCuda(cudaMalloc(&d_a, bytes), "cudaMalloc d_a");
    checkCuda(cudaMalloc(&d_b, bytes), "cudaMalloc d_b");

    // Initialize on device
    const int threads = 256;
    const long long total = (long long)gridSize;
    const int blocks = (int)((total + threads - 1) / threads);
    initKernel<<<blocks, threads>>>(d_a, nx, ny, nz);
    checkCuda(cudaGetLastError(), "initKernel launch");
    checkCuda(cudaDeviceSynchronize(), "initKernel sync");

    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Iterations: ping-pong between d_a and d_b
    for (int iter = 0; iter < iterations; ++iter) {
        // compute interior
        stencilKernel<<<blocks, threads>>>(d_a, d_b, nx, ny, nz);
        checkCuda(cudaGetLastError(), "stencilKernel launch");
        // copy boundaries
        copyBoundaryKernel<<<blocks, threads>>>(d_a, d_b, nx, ny, nz);
        checkCuda(cudaGetLastError(), "copyBoundaryKernel launch");
        // Ensure kernels complete before next iteration
        checkCuda(cudaDeviceSynchronize(), "iteration sync");

        // swap
        std::swap(d_a, d_b);
    }

    // Ensure all work done
    checkCuda(cudaDeviceSynchronize(), "final sync");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Copy final result back to host
    Real* d_final = d_a; // after last swap d_a holds latest
    std::vector<Real>& finalHost = (iterations % 2 == 0) ? hostGridA : hostGridB;
    checkCuda(cudaMemcpy(finalHost.data(), d_final, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    if (printResults) {
        print_results(finalHost, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalHost, nx, ny, nz);
        if (valid) {
            printf("Validation: PASSED\n");
            // free device memory
            cudaFree(d_a);
            cudaFree(d_b);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            cudaFree(d_a);
            cudaFree(d_b);
            return 1;
        }
    }

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);

    return 0;
}
