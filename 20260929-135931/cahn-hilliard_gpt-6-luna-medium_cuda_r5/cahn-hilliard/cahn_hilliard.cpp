#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ double laplacian(const double* a, size_t i, size_t x, size_t y, size_t z,
                                             size_t nx, size_t ny, size_t nz,
                                             double dx, double dy, double dz) {
    const size_t plane = nx * ny;
    const size_t xp = x + (x + 1 < nx), xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny), yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz), zn = z - (z > 0);
    const double center = a[i];
    const double cxx = (a[i + (xp - x)] + a[i - (x - xn)] - 2.0 * center) / (dx * dx);
    const double cyy = (a[i + (yp - y) * nx] + a[i - (y - yn) * nx] - 2.0 * center) / (dy * dy);
    const double czz = (a[i + (zp - z) * plane] + a[i - (z - zn) * plane] - 2.0 * center) / (dz * dz);
    return cxx + cyy + czz;
}

__global__ void initializeKernel(double* c, size_t n, size_t vol) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i < n) {
        const double pseudo = (((i + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                         size_t nx, size_t ny, size_t nz, size_t n,
                                         double dx, double dy, double dz, double gamma,
                                         double e_AA, double e_BB, double e_AB) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
          + 3.0 * cv + cv * cv * cv
          - gamma * laplacian(c, i, x, y, z, nx, ny, nz, dx, dy, dz);
}

__global__ void updateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                              const double* __restrict__ mu, size_t nx, size_t ny, size_t nz,
                              size_t n, double D, double dt, double dx, double dy, double dz) {
    const size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    const size_t x = i % nx, y = (i / nx) % ny, z = i / (nx * ny);
    cnew[i] = cold[i] + dt * D * laplacian(mu, i, x, y, z, nx, ny, nz, dx, dy, dz);
}

bool validateResult(const std::vector<double>& c) {
    for (const auto val : c) {
        if (!std::isfinite(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minVal = c[0], maxVal = c[0];
    for (const auto val : c) { minVal = std::min(minVal, val); maxVal = std::max(maxVal, val); }
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
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
    if (nx == 0 || ny == 0 || nz == 0) { fprintf(stderr, "Grid dimensions must be positive\n"); return 1; }

    printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");
    const double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    const double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    const double gamma = 0.5, D = 1.0;
    const size_t gridSize = nx * ny * nz;
    std::vector<double> result(gridSize);
    double *d_cold = nullptr, *d_cnew = nullptr, *d_mu = nullptr;
    cudaCheck(cudaMalloc(&d_cold, gridSize * sizeof(double)), "allocating concentration buffers");
    cudaCheck(cudaMalloc(&d_cnew, gridSize * sizeof(double)), "allocating concentration buffers");
    cudaCheck(cudaMalloc(&d_mu, gridSize * sizeof(double)), "allocating chemical potential buffer");
    constexpr int blockSize = 256;
    const int blocks = static_cast<int>((gridSize + blockSize - 1) / blockSize);
    printf("Initializing concentration field...\n");
    initializeKernel<<<blocks, blockSize>>>(d_cold, gridSize, gridSize);
    cudaCheck(cudaGetLastError(), "launching initialization kernel");
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, blockSize>>>(d_cold, d_mu, nx, ny, nz, gridSize, dx, dy, dz,
                                                       gamma, e_AA, e_BB, e_AB);
        cudaCheck(cudaGetLastError(), "launching chemical potential kernel");
        updateKernel<<<blocks, blockSize>>>(d_cnew, d_cold, d_mu, nx, ny, nz, gridSize, D, dt, dx, dy, dz);
        cudaCheck(cudaGetLastError(), "launching update kernel");
        std::swap(d_cold, d_cnew);
    }
    cudaCheck(cudaMemcpy(result.data(), d_cold, gridSize * sizeof(double), cudaMemcpyDeviceToHost), "copying result");
    auto end = std::chrono::high_resolution_clock::now();
    cudaCheck(cudaFree(d_cold), "freeing device memory");
    cudaCheck(cudaFree(d_cnew), "freeing device memory");
    cudaCheck(cudaFree(d_mu), "freeing device memory");
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    printf("Computation time: %ld ms\n", duration.count());
    const double seconds = std::chrono::duration<double>(end - start).count();
    printf("Performance: %.3f MCellUpdates/s\n", seconds > 0 ? gridSize * static_cast<double>(iterations) / seconds / 1e6 : 0.0);
    if (printResults) print_results(result, "Concentration");
    if (validate) {
        printf("Validating result...\n");
        const bool valid = validateResult(result);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
