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

#define CUDA_CHECK(call) do { \
    const cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (0)

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             const size_t index, const size_t x,
                                             const size_t y, const size_t z,
                                             const size_t nx, const size_t ny,
                                             const size_t nz, const size_t plane) {
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);
    const double center = field[index];
    // Preserve the operation grouping of the scalar reference implementation.
    const double cxx = field[z * plane + y * nx + xp] + field[z * plane + y * nx + xn] - 2.0 * center;
    const double cyy = field[z * plane + yp * nx + x] + field[z * plane + yn * nx + x] - 2.0 * center;
    const double czz = field[zp * plane + y * nx + x] + field[zn * plane + y * nx + x] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void initializeConcentration(double* c, const size_t count) {
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        const size_t pseudoInt = ((id + 1) * static_cast<size_t>(1299709)) % count;
        c[id] = -1.0 + 2.0 * (static_cast<double>(pseudoInt) / static_cast<double>(count));
    }
}

__global__ void chemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                        const size_t count, const size_t nx, const size_t ny,
                                        const size_t nz, const double gamma, const double eAA,
                                        const double eBB, const double eAB) {
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= count) return;
    const size_t plane = nx * ny;
    const size_t z = id / plane;
    const size_t remainder = id - z * plane;
    const size_t y = remainder / nx;
    const size_t x = remainder - y * nx;
    const double cv = c[id];
    mu[id] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB)
           + 3.0 * cv + cv * cv * cv
           - gamma * laplacian(c, id, x, y, z, nx, ny, nz, plane);
}

__global__ void updateKernel(const double* __restrict__ cold, const double* __restrict__ mu,
                             double* __restrict__ cnew, const size_t count, const size_t nx,
                             const size_t ny, const size_t nz, const double dtD) {
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= count) return;
    const size_t plane = nx * ny;
    const size_t z = id / plane;
    const size_t remainder = id - z * plane;
    const size_t y = remainder / nx;
    const size_t x = remainder - y * nx;
    cnew[id] = cold[id] + dtD * laplacian(mu, id, x, y, z, nx, ny, nz, plane);
}

bool validateResult(const std::vector<double>& c) {
    for (const double value : c) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto [minIt, maxIt] = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *minIt, *maxIt);
    if (*maxIt > 10.0 || *minIt < -10.0) {
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
    if (gridSize == 0) { std::fprintf(stderr, "Grid dimensions must be positive\n"); return 1; }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    CUDA_CHECK(cudaMalloc(&cold, gridSize * sizeof(*cold)));
    CUDA_CHECK(cudaMalloc(&cnew, gridSize * sizeof(*cnew)));
    CUDA_CHECK(cudaMalloc(&mu, gridSize * sizeof(*mu)));
    constexpr int threads = 256;
    const int blocks = static_cast<int>((gridSize + threads - 1) / threads);
    std::printf("Initializing concentration field...\n");
    initializeConcentration<<<blocks, threads>>>(cold, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, threads>>>(cold, mu, gridSize, nx, ny, nz, 0.5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        updateKernel<<<blocks, threads>>>(cold, mu, cnew, gridSize, nx, ny, nz, 0.01);
        std::swap(cold, cnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    std::printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));
    const double seconds = std::max(static_cast<double>(elapsedMs) / 1000.0, 1.0e-9);
    std::printf("Performance: %.3f MCellUpdates/s\n", static_cast<double>(gridSize) * iterations / seconds / 1.0e6);

    if (printResults || validate) {
        std::vector<double> result(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), cold, gridSize * sizeof(*cold), cudaMemcpyDeviceToHost));
        if (printResults) print_results(result, "Concentration");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(result);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
            return valid ? 0 : 1;
        }
    }
    CUDA_CHECK(cudaFree(cold)); CUDA_CHECK(cudaFree(cnew)); CUDA_CHECK(cudaFree(mu));
    return 0;
}

