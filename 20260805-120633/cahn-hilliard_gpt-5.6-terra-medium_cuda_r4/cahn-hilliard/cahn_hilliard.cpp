#include <algorithm>
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

constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 4;
constexpr int BLOCK_Z = 2;

inline void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ size_t index3(const size_t x, const size_t y, const size_t z,
                                         const size_t nx, const size_t plane) {
    return z * plane + y * nx + x;
}

__device__ __forceinline__ double laplacian(const double* __restrict__ field,
                                             const size_t x, const size_t y, const size_t z,
                                             const size_t nx, const size_t ny, const size_t nz,
                                             const double invDx2, const double invDy2,
                                             const double invDz2) {
    const size_t plane = nx * ny;
    const size_t center = index3(x, y, z, nx, plane);
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);
    const double value = field[center];
    return (field[index3(xp, y, z, nx, plane)] + field[index3(xn, y, z, nx, plane)] - 2.0 * value) * invDx2
         + (field[index3(x, yp, z, nx, plane)] + field[index3(x, yn, z, nx, plane)] - 2.0 * value) * invDy2
         + (field[index3(x, y, zp, nx, plane)] + field[index3(x, y, zn, nx, plane)] - 2.0 * value) * invDz2;
}

__global__ void initializeConcentrationKernel(double* __restrict__ concentration,
                                               const size_t nx, const size_t ny, const size_t nz) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t plane = nx * ny;
    const size_t linearId = index3(x, y, z, nx, plane);
    const size_t volume = plane * nz;
    const unsigned long long mixed = (static_cast<unsigned long long>(linearId) + 1ULL) * 1299709ULL;
    concentration[linearId] = -1.0 + 2.0 * (static_cast<double>(mixed % volume) / static_cast<double>(volume));
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration,
                                        double* __restrict__ chemicalPotential,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double gamma, const double eAA,
                                        const double eBB, const double eAB,
                                        const double invDx2, const double invDy2,
                                        const double invDz2) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = index3(x, y, z, nx, nx * ny);
    const double c = concentration[idx];
    chemicalPotential[idx] = 4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB)
                           + 3.0 * c + c * c * c
                           - gamma * laplacian(concentration, x, y, z, nx, ny, nz, invDx2, invDy2, invDz2);
}

__global__ void updateKernel(const double* __restrict__ oldConcentration,
                             const double* __restrict__ chemicalPotential,
                             double* __restrict__ newConcentration,
                             const size_t nx, const size_t ny, const size_t nz,
                             const double scale, const double invDx2,
                             const double invDy2, const double invDz2) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const size_t idx = index3(x, y, z, nx, nx * ny);
    newConcentration[idx] = oldConcentration[idx]
                          + scale * laplacian(chemicalPotential, x, y, z, nx, ny, nz, invDx2, invDy2, invDz2);
}

dim3 gridFor(const size_t nx, const size_t ny, const size_t nz) {
    return dim3(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                static_cast<unsigned int>((nz + BLOCK_Z - 1) / BLOCK_Z));
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

    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");

    constexpr double dx = 1.0, dy = 1.0, dz = 1.0, dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0), eBB = -(2.0 / 9.0), eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5, diffusivity = 1.0;
    const double invDx2 = 1.0 / (dx * dx), invDy2 = 1.0 / (dy * dy), invDz2 = 1.0 / (dz * dz);
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid = gridFor(nx, ny, nz);

    double *cold = nullptr, *cnew = nullptr, *mu = nullptr;
    checkCuda(cudaMalloc(&cold, bytes), "allocating concentration buffer");
    checkCuda(cudaMalloc(&cnew, bytes), "allocating update buffer");
    checkCuda(cudaMalloc(&mu, bytes), "allocating chemical potential buffer");
    std::printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<grid, block>>>(cold, nx, ny, nz);
    checkCuda(cudaGetLastError(), "launching initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration field");

    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    std::printf("Running Cahn-Hilliard simulation...\n");
    checkCuda(cudaEventRecord(start), "recording start event");
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<grid, block>>>(cold, mu, nx, ny, nz, gamma, eAA, eBB, eAB,
                                                  invDx2, invDy2, invDz2);
        updateKernel<<<grid, block>>>(cold, mu, cnew, nx, ny, nz, diffusivity * dt,
                                      invDx2, invDy2, invDz2);
        std::swap(cold, cnew);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "synchronizing simulation");
    float elapsedMs = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "calculating elapsed time");
    std::printf("Computation time: %ld ms\n", static_cast<long>(elapsedMs));
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    std::printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / (elapsedMs / 1000.0) / 1.0e6);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "copying result to host");
    }
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    checkCuda(cudaFree(mu), "freeing chemical potential buffer");
    checkCuda(cudaFree(cnew), "freeing update buffer");
    checkCuda(cudaFree(cold), "freeing concentration buffer");

    if (printResults) print_results(result, "Concentration");
    if (validate) {
        std::printf("Validating result...\n");
        if (!validateResult(result)) { std::printf("Validation: FAILED\n"); return 1; }
        std::printf("Validation: PASSED\n");
    }
    return 0;
}
