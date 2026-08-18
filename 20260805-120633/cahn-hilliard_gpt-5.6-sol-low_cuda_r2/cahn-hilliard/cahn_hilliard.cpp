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

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                     cudaGetErrorString(error_));                               \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
} while (false)

// The x-major layout makes consecutive threads access consecutive doubles.
__device__ __forceinline__ double laplacian(const double* __restrict__ a,
                                             size_t i, size_t x, size_t y,
                                             size_t z, size_t nx, size_t ny,
                                             size_t nz) {
    const size_t plane = nx * ny;
    const double center = a[i];
    const double xm = a[x ? i - 1 : i];
    const double xp = a[x + 1 < nx ? i + 1 : i];
    const double ym = a[y ? i - nx : i];
    const double yp = a[y + 1 < ny ? i + nx : i];
    const double zm = a[z ? i - plane : i];
    const double zp = a[z + 1 < nz ? i + plane : i];
    // dx, dy and dz are all one in this benchmark.
    return (xp + xm - 2.0 * center) + (yp + ym - 2.0 * center) +
           (zp + zm - 2.0 * center);
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c,
                                         double* __restrict__ mu, size_t n,
                                         size_t nx, size_t ny, size_t nz,
                                         double gamma, double eAA,
                                         double eBB, double eAB) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t plane = nx * ny;
    const size_t z = i / plane;
    const size_t rem = i - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    const double cv = c[i];
    mu[i] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB -
                   2.0 * cv * eAB) +
            3.0 * cv + cv * cv * cv -
            gamma * laplacian(c, i, x, y, z, nx, ny, nz);
}

__global__ void updateKernel(double* __restrict__ cnew,
                             const double* __restrict__ cold,
                             const double* __restrict__ mu, size_t n,
                             size_t nx, size_t ny, size_t nz,
                             double dtD) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const size_t plane = nx * ny;
    const size_t z = i / plane;
    const size_t rem = i - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    cnew[i] = cold[i] + dtD * laplacian(mu, i, x, y, z, nx, ny, nz);
}

void initializeConcentration(std::vector<double>& c) {
    const size_t vol = c.size();
    for (size_t i = 0; i < vol; ++i) {
        const double pseudo = (((i + 1) * size_t{1299709}) % vol) /
                              static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c) {
    if (c.empty()) return false;
    double minVal = c[0], maxVal = c[0];
    for (double val : c) {
        if (!std::isfinite(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    std::printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 10.0 || minVal < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n", p);
    std::printf("Options:\n  -x <num>     Grid size in X dimension (default: 64)\n"
                "  -y <num>     Grid size in Y dimension (default: same as X)\n"
                "  -z <num>     Grid size in Z dimension (default: same as X)\n"
                "  -i <num>     Number of time steps (default: 20)\n"
                "  -v           Enable validation\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-x") && i + 1 < argc) nx = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-y") && i + 1 < argc) ny = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-z") && i + 1 < argc) nz = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-i") && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (!ny) ny = nx;
    if (!nz) nz = nx;
    if (!nx || !ny || !nz || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n"
                "Time steps: %d\nValidation: %s\n", nx, ny, nz, iterations,
                validate ? "enabled" : "disabled");

    std::vector<double> result(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(result);

    double *dCold = nullptr, *dNew = nullptr, *dMu = nullptr;
    CUDA_CHECK(cudaMalloc(&dCold, bytes));
    CUDA_CHECK(cudaMalloc(&dNew, bytes));
    CUDA_CHECK(cudaMalloc(&dMu, bytes));
    CUDA_CHECK(cudaMemcpy(dCold, result.data(), bytes, cudaMemcpyHostToDevice));

    constexpr unsigned blockSize = 256;
    const size_t blockCount = (gridSize + blockSize - 1) / blockSize;
    if (blockCount > std::numeric_limits<unsigned>::max()) {
        std::fprintf(stderr, "Grid is too large for CUDA launch geometry\n");
        return 1;
    }
    const dim3 grid(static_cast<unsigned>(blockCount));
    constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0, gamma = 0.5, dtD = 0.01;

    std::printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<grid, blockSize>>>(dCold, dMu, gridSize, nx, ny, nz,
                                                     gamma, eAA, eBB, eAB);
        updateKernel<<<grid, blockSize>>>(dNew, dCold, dMu, gridSize, nx, ny, nz, dtD);
        std::swap(dCold, dNew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(result.data(), dCold, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dMu));
    CUDA_CHECK(cudaFree(dNew));
    CUDA_CHECK(cudaFree(dCold));

    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    const long ms = static_cast<long>(us / 1000);
    std::printf("Computation time: %ld ms\n", ms);
    const double mcups = us ? (static_cast<double>(gridSize) * iterations / us) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    if (printResults) print_results(result, "Concentration");
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(result);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
