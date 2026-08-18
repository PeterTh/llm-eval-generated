#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// One-dimensional launch order makes adjacent x cells adjacent threads, which
// keeps all seven-point stencil loads coalesced.  The stride form also permits
// grids larger than the maximum practical launch size.
__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             const size_t index, const size_t x,
                                             const size_t y, const size_t z,
                                             const size_t nx, const size_t ny,
                                             const size_t nz) {
    const size_t plane = nx * ny;
    const double center = field[index];
    const size_t xp = (x + 1 < nx) ? index + 1 : index;
    const size_t xn = (x != 0) ? index - 1 : index;
    const size_t yp = (y + 1 < ny) ? index + nx : index;
    const size_t yn = (y != 0) ? index - nx : index;
    const size_t zp = (z + 1 < nz) ? index + plane : index;
    const size_t zn = (z != 0) ? index - plane : index;
    // Retain the three-term evaluation order of the reference implementation.
    const double cxx = field[xp] + field[xn] - 2.0 * center;
    const double cyy = field[yp] + field[yn] - 2.0 * center;
    const double czz = field[zp] + field[zn] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* __restrict__ concentration, const size_t volume) {
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = start; index < volume; index += stride) {
        const double pseudo = ((static_cast<double>(((index + 1) * 1299709ULL) % volume)) /
                               static_cast<double>(volume));
        concentration[index] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration,
                                        double* __restrict__ potential,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double gamma, const double eAA,
                                        const double eBB, const double eAB) {
    const size_t volume = nx * ny * nz;
    const size_t plane = nx * ny;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = start; index < volume; index += stride) {
        const size_t z = index / plane;
        const size_t remainder = index - z * plane;
        const size_t y = remainder / nx;
        const size_t x = remainder - y * nx;
        const double value = concentration[index];
        potential[index] = 4.5 * ((value + 1.0) * eAA + (value - 1.0) * eBB - 2.0 * value * eAB)
                         + 3.0 * value + value * value * value
                         - gamma * laplacian(concentration, index, x, y, z, nx, ny, nz);
    }
}

__global__ void updateKernel(const double* __restrict__ oldConcentration,
                             const double* __restrict__ potential,
                             double* __restrict__ newConcentration,
                             const size_t nx, const size_t ny, const size_t nz,
                             const double dtD) {
    const size_t volume = nx * ny * nz;
    const size_t plane = nx * ny;
    const size_t start = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = start; index < volume; index += stride) {
        const size_t z = index / plane;
        const size_t remainder = index - z * plane;
        const size_t y = remainder / nx;
        const size_t x = remainder - y * nx;
        newConcentration[index] = oldConcentration[index]
                                + dtD * laplacian(potential, index, x, y, z, nx, ny, nz);
    }
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    double minValue = concentration[0];
    double maxValue = concentration[0];
    for (const double value : concentration) {
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
    if (nx == 0 || ny == 0 || nz == 0 || nx > std::numeric_limits<size_t>::max() / ny / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    const size_t volume = nx * ny * nz;
    const size_t bytes = volume * sizeof(double);
    constexpr int threads = 256;
    const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>((volume + threads - 1) / threads, 65535));

    double *oldConcentration = nullptr, *newConcentration = nullptr, *potential = nullptr;
    checkCuda(cudaMalloc(&oldConcentration, bytes), "allocating concentration buffer");
    checkCuda(cudaMalloc(&newConcentration, bytes), "allocating update buffer");
    checkCuda(cudaMalloc(&potential, bytes), "allocating potential buffer");

    std::printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<blocks, threads>>>(oldConcentration, volume);
    checkCuda(cudaGetLastError(), "launching initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration field");

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&stop), "creating stop event");
    std::printf("Running Cahn-Hilliard simulation...\n");
    checkCuda(cudaEventRecord(start), "recording start event");
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, threads>>>(oldConcentration, potential, nx, ny, nz,
                                                      0.5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        updateKernel<<<blocks, threads>>>(oldConcentration, potential, newConcentration, nx, ny, nz, 0.01);
        std::swap(oldConcentration, newConcentration);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaEventRecord(stop), "recording stop event");
    checkCuda(cudaEventSynchronize(stop), "synchronizing simulation");
    float elapsedMilliseconds = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMilliseconds, start, stop), "measuring simulation time");
    const long durationMilliseconds = static_cast<long>(elapsedMilliseconds);
    std::printf("Computation time: %ld ms\n", durationMilliseconds);
    const double mcups = static_cast<double>(volume) * iterations / (elapsedMilliseconds / 1000.0) / 1e6;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        std::vector<double> result(volume);
        checkCuda(cudaMemcpy(result.data(), oldConcentration, bytes, cudaMemcpyDeviceToHost), "copying final concentration");
        if (printResults) print_results(result, "Concentration");
        if (validate) {
            std::printf("Validating result...\n");
            if (!validateResult(result)) {
                std::printf("Validation: FAILED\n");
                cudaEventDestroy(start); cudaEventDestroy(stop);
                cudaFree(oldConcentration); cudaFree(newConcentration); cudaFree(potential);
                return 1;
            }
            std::printf("Validation: PASSED\n");
        }
    }
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(stop), "destroying stop event");
    checkCuda(cudaFree(oldConcentration), "freeing concentration buffer");
    checkCuda(cudaFree(newConcentration), "freeing update buffer");
    checkCuda(cudaFree(potential), "freeing potential buffer");
    return 0;
}
