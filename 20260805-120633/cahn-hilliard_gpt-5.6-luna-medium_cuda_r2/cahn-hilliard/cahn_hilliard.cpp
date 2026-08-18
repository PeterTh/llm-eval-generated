#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

[[noreturn]] void cudaFailure(const char* operation, const cudaError_t error) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(operation, error);
}

__device__ __forceinline__ size_t index3(const size_t x, const size_t y, const size_t z,
                                         const size_t nx, const size_t plane) {
    return z * plane + y * nx + x;
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                           const size_t x, const size_t y, const size_t z,
                                           const size_t nx, const size_t ny, const size_t nz,
                                           const double dx2, const double dy2, const double dz2) {
    const size_t plane = nx * ny;
    const size_t center = index3(x, y, z, nx, plane);
    const size_t xp = index3(x + (x + 1 < nx), y, z, nx, plane);
    const size_t xn = index3(x - (x > 0), y, z, nx, plane);
    const size_t yp = index3(x, y + (y + 1 < ny), z, nx, plane);
    const size_t yn = index3(x, y - (y > 0), z, nx, plane);
    const size_t zp = index3(x, y, z + (z + 1 < nz), nx, plane);
    const size_t zn = index3(x, y, z - (z > 0), nx, plane);
    const double value = field[center];
    return (field[xp] + field[xn] - 2.0 * value) / dx2
         + (field[yp] + field[yn] - 2.0 * value) / dy2
         + (field[zp] + field[zn] - 2.0 * value) / dz2;
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration,
                                        double* __restrict__ chemicalPotential,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double dx2, const double dy2, const double dz2,
                                        const double gamma, const double eAA,
                                        const double eBB, const double eAB) {
    const size_t plane = nx * ny;
    const size_t count = plane * nz;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t z = i / plane;
        const size_t rem = i - z * plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        const double c = concentration[i];
        chemicalPotential[i] = 4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB)
                             + 3.0 * c + c * c * c
                             - gamma * laplacian(concentration, x, y, z, nx, ny, nz,
                                                 dx2, dy2, dz2);
    }
}

__global__ void updateKernel(double* __restrict__ next, const double* __restrict__ current,
                             const double* __restrict__ chemicalPotential,
                             const size_t nx, const size_t ny, const size_t nz,
                             const double coefficient, const double dx2,
                             const double dy2, const double dz2) {
    const size_t plane = nx * ny;
    const size_t count = plane * nz;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t z = i / plane;
        const size_t rem = i - z * plane;
        const size_t y = rem / nx;
        const size_t x = rem - y * nx;
        next[i] = current[i] + coefficient * laplacian(chemicalPotential, x, y, z,
                                                       nx, ny, nz, dx2, dy2, dz2);
    }
}

void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny,
                             const size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t i = 0; i < volume; ++i) {
        const double pseudo = ((((i + 1) * 1299709) % volume) / static_cast<double>(volume));
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c) {
    for (const double value : c) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minValue = c[0], maxValue = c[0];
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

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    std::printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    std::printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    std::printf("  -i <num>     Number of time steps (default: 20)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
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
    const size_t gridSize = nx * ny * nz;
    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\n",
                nx, ny, nz);
    std::printf("Time steps: %d\nValidation: %s\n", iterations, validate ? "enabled" : "disabled");

    constexpr double dx2 = 1.0, dy2 = 1.0, dz2 = 1.0;
    constexpr double dt = 0.01, gamma = 0.5, diffusivity = 1.0;
    constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    std::vector<double> hostCurrent(gridSize), hostNext(gridSize);
    initializeConcentration(hostCurrent, nx, ny, nz);
    std::printf("Initializing concentration field...\nRunning Cahn-Hilliard simulation...\n");

    double *current = nullptr, *next = nullptr, *chemicalPotential = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&current), bytes), "cudaMalloc(current)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&next), bytes), "cudaMalloc(next)");
    checkCuda(cudaMalloc(reinterpret_cast<void**>(&chemicalPotential), bytes), "cudaMalloc(mu)");
    checkCuda(cudaMemcpy(current, hostCurrent.data(), bytes, cudaMemcpyHostToDevice), "initial copy");

    const int blocks = std::min<size_t>((gridSize + 255) / 256, 65535);
    const dim3 grid(static_cast<unsigned int>(std::max(1, blocks)));
    const dim3 block(256);
    checkCuda(cudaDeviceSynchronize(), "initial synchronization");
    const auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<grid, block>>>(current, chemicalPotential, nx, ny, nz,
                                                  dx2, dy2, dz2, gamma, eAA, eBB, eAB);
        updateKernel<<<grid, block>>>(next, current, chemicalPotential, nx, ny, nz,
                                      dt * diffusivity, dx2, dy2, dz2);
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError(), "kernel launch");
    checkCuda(cudaDeviceSynchronize(), "simulation");
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    checkCuda(cudaMemcpy(hostCurrent.data(), current, bytes, cudaMemcpyDeviceToHost), "result copy");
    cudaFree(current); cudaFree(next); cudaFree(chemicalPotential);

    std::printf("Computation time: %ld ms\n", duration.count());
    const double mcups = static_cast<double>(gridSize) * iterations /
                         (duration.count() / 1000.0) / 1e6;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);
    if (printResults) print_results(hostCurrent, "Concentration");
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(hostCurrent);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
