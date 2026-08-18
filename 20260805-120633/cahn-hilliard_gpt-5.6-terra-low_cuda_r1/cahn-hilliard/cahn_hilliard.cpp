#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int kThreadsPerBlock = 256;

void checkCuda(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d for %s: %s\n", file, line, expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expr) checkCuda((expr), #expr, __FILE__, __LINE__)

__device__ __forceinline__ double laplacian(const double* __restrict__ field, const size_t index,
                                              const size_t x, const size_t y, const size_t z,
                                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t plane = nx * ny;
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);

    // dx, dy and dz are all one in this benchmark, so reciprocal squared spacings
    // are folded away while retaining the original stencil and clamped boundaries.
    return field[index + (xp - x)] + field[index + (xn - x)] +
           field[index + (yp - y) * nx] + field[index + (yn - y) * nx] +
           field[index + (zp - z) * plane] + field[index + (zn - z) * plane] -
           6.0 * field[index];
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t count, const size_t nx, const size_t ny,
                                        const size_t nz, const double gamma, const double eAA,
                                        const double eBB, const double eAB) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const size_t plane = nx * ny;
    const size_t z = index / plane;
    const size_t rem = index - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    const double cv = c[index];
    mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                3.0 * cv + cv * cv * cv - gamma * laplacian(c, index, x, y, z, nx, ny, nz);
}

__global__ void updateKernel(const double* __restrict__ cold, const double* __restrict__ mu,
                             double* __restrict__ cnew, const size_t count, const size_t nx,
                             const size_t ny, const size_t nz, const double dtD) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const size_t plane = nx * ny;
    const size_t z = index / plane;
    const size_t rem = index - z * plane;
    const size_t y = rem / nx;
    const size_t x = rem - y * nx;
    cnew[index] = cold[index] + dtD * laplacian(mu, index, x, y, z, nx, ny, nz);
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t index = 0; index < volume; ++index) {
        const double pseudo = (((index + 1) * 1299709 % volume) / static_cast<double>(volume));
        c[index] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c) {
    for (const double value : c) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    double minValue = c[0];
    double maxValue = c[0];
    for (const double value : c) {
        minValue = std::min(minValue, value);
        maxValue = std::max(maxValue, value);
    }
    std::printf("Concentration range: [%.6f, %.6f]\n", minValue, maxValue);
    if (maxValue > 10.0 || minValue < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    size_t nx = 64, ny = 0, nz = 0;
    int iterations = 20;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) nx = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) ny = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) nz = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) iterations = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny || nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    if (gridSize == 0 || (gridSize + kThreadsPerBlock - 1) / kThreadsPerBlock > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Grid is too large for this CUDA launch configuration\n");
        return 1;
    }
    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, nx, ny, nz);

    double *dCold = nullptr, *dCnew = nullptr, *dMu = nullptr;
    CUDA_CHECK(cudaMalloc(&dCold, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dCnew, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dMu, gridSize * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(dCold, concentration.data(), gridSize * sizeof(double), cudaMemcpyHostToDevice));

    const unsigned int blocks = static_cast<unsigned int>((gridSize + kThreadsPerBlock - 1) / kThreadsPerBlock);
    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, kThreadsPerBlock>>>(dCold, dMu, gridSize, nx, ny, nz,
                                                                0.5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        CUDA_CHECK(cudaGetLastError());
        updateKernel<<<blocks, kThreadsPerBlock>>>(dCold, dMu, dCnew, gridSize, nx, ny, nz, 0.01);
        CUDA_CHECK(cudaGetLastError());
        std::swap(dCold, dCnew);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMs = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double mcups = elapsedMs > 0.0F ? static_cast<double>(gridSize) * iterations / (elapsedMs * 1000.0) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) CUDA_CHECK(cudaMemcpy(concentration.data(), dCold, gridSize * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dMu));
    CUDA_CHECK(cudaFree(dCnew));
    CUDA_CHECK(cudaFree(dCold));

    if (printResults) print_results(concentration, "Concentration");
    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(concentration)) { std::printf("Validation: PASSED\n"); return 0; }
        std::printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
