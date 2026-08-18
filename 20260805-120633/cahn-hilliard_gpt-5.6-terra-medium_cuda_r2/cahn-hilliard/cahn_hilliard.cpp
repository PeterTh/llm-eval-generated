#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

inline void checkCuda(const cudaError_t status, const char* const action) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", action, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__device__ __forceinline__ size_t idx3(const size_t x, const size_t y, const size_t z,
                                       const size_t nx, const size_t ny) {
    return z * nx * ny + y * nx + x;
}

__device__ __forceinline__ double laplacian(const double* const field, const size_t x,
                                            const size_t y, const size_t z, const size_t nx,
                                            const size_t ny, const size_t nz) {
    const size_t center = idx3(x, y, z, nx, ny);
    const size_t xp = x + (x + 1 < nx);
    const size_t xn = x - (x > 0);
    const size_t yp = y + (y + 1 < ny);
    const size_t yn = y - (y > 0);
    const size_t zp = z + (z + 1 < nz);
    const size_t zn = z - (z > 0);

    // dx, dy and dz are all one in this benchmark.  Preserve the original
    // operation order so its floating-point semantics are retained.
    const double c = field[center];
    const double cxx = field[idx3(xp, y, z, nx, ny)] + field[idx3(xn, y, z, nx, ny)] - 2.0 * c;
    const double cyy = field[idx3(x, yp, z, nx, ny)] + field[idx3(x, yn, z, nx, ny)] - 2.0 * c;
    const double czz = field[idx3(x, y, zp, nx, ny)] + field[idx3(x, y, zn, nx, ny)] - 2.0 * c;
    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* const concentration, const size_t count) {
    const size_t id = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id < count) {
        const size_t pseudo = ((id + 1) * static_cast<size_t>(1299709)) % count;
        concentration[id] = -1.0 + 2.0 * (static_cast<double>(pseudo) / static_cast<double>(count));
    }
}

__global__ void chemicalPotentialKernel(const double* __restrict__ concentration,
                                        double* __restrict__ chemicalPotential, const size_t nx,
                                        const size_t ny, const size_t nz, const double gamma,
                                        const double eAA, const double eBB, const double eAB) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t id = idx3(x, y, z, nx, ny);
        const double c = concentration[id];
        chemicalPotential[id] = 4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB) +
                                3.0 * c + c * c * c -
                                gamma * laplacian(concentration, x, y, z, nx, ny, nz);
    }
}

__global__ void updateKernel(double* __restrict__ next, const double* __restrict__ current,
                             const double* __restrict__ chemicalPotential, const size_t nx,
                             const size_t ny, const size_t nz, const double scale) {
    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t id = idx3(x, y, z, nx, ny);
        next[id] = current[id] + scale * laplacian(chemicalPotential, x, y, z, nx, ny, nz);
    }
}

bool validateResult(const std::vector<double>& concentration) {
    for (const double value : concentration) {
        if (std::isnan(value) || std::isinf(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto [minimum, maximum] = std::minmax_element(concentration.begin(), concentration.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *minimum, *maximum);
    if (*maximum > 10.0 || *minimum < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* const progName) {
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
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

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
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 || nx > std::numeric_limits<size_t>::max() / ny / nz) {
        std::fprintf(stderr, "Invalid grid dimensions or iteration count\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid is too large\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\nGrid size: %zu x %zu x %zu\nTime steps: %d\nValidation: %s\n",
                nx, ny, nz, iterations, validate ? "enabled" : "disabled");
    std::printf("Initializing concentration field...\n");

    constexpr unsigned int initThreads = 256;
    const unsigned int initBlocks = static_cast<unsigned int>((gridSize + initThreads - 1) / initThreads);
    const dim3 threads(32, 4, 2);
    const dim3 blocks(static_cast<unsigned int>((nx + threads.x - 1) / threads.x),
                      static_cast<unsigned int>((ny + threads.y - 1) / threads.y),
                      static_cast<unsigned int>((nz + threads.z - 1) / threads.z));
    if (static_cast<size_t>(initBlocks) * initThreads < gridSize || blocks.x == 0 || blocks.y == 0 || blocks.z == 0) {
        std::fprintf(stderr, "Grid exceeds CUDA launch capacity\n");
        return 1;
    }

    double *current = nullptr, *next = nullptr, *chemicalPotential = nullptr;
    checkCuda(cudaMalloc(&current, gridSize * sizeof(double)), "allocating concentration buffer");
    checkCuda(cudaMalloc(&next, gridSize * sizeof(double)), "allocating update buffer");
    checkCuda(cudaMalloc(&chemicalPotential, gridSize * sizeof(double)), "allocating chemical-potential buffer");
    initializeConcentrationKernel<<<initBlocks, initThreads>>>(current, gridSize);
    checkCuda(cudaGetLastError(), "launching initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "initializing concentration field");

    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start, end;
    checkCuda(cudaEventCreate(&start), "creating start event");
    checkCuda(cudaEventCreate(&end), "creating end event");
    checkCuda(cudaEventRecord(start), "recording start event");
    for (int t = 0; t < iterations; ++t) {
        chemicalPotentialKernel<<<blocks, threads>>>(current, chemicalPotential, nx, ny, nz,
                                                              0.5, -(2.0 / 9.0), -(2.0 / 9.0), 2.0 / 9.0);
        updateKernel<<<blocks, threads>>>(next, current, chemicalPotential, nx, ny, nz, 0.01);
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaEventRecord(end), "recording end event");
    checkCuda(cudaEventSynchronize(end), "waiting for simulation");

    float elapsedMs = 0.0F;
    checkCuda(cudaEventElapsedTime(&elapsedMs, start, end), "measuring simulation time");
    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double mcups = elapsedMs > 0.0F ? static_cast<double>(gridSize) * iterations / elapsedMs / 1.0e3 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        std::vector<double> result(gridSize);
        checkCuda(cudaMemcpy(result.data(), current, gridSize * sizeof(double), cudaMemcpyDeviceToHost),
                  "copying result to host");
        if (printResults) print_results(result, "Concentration");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(result);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            checkCuda(cudaEventDestroy(start), "destroying start event");
            checkCuda(cudaEventDestroy(end), "destroying end event");
            checkCuda(cudaFree(current), "freeing concentration buffer");
            checkCuda(cudaFree(next), "freeing update buffer");
            checkCuda(cudaFree(chemicalPotential), "freeing chemical-potential buffer");
            return valid ? 0 : 1;
        }
    }
    checkCuda(cudaEventDestroy(start), "destroying start event");
    checkCuda(cudaEventDestroy(end), "destroying end event");
    checkCuda(cudaFree(current), "freeing concentration buffer");
    checkCuda(cudaFree(next), "freeing update buffer");
    checkCuda(cudaFree(chemicalPotential), "freeing chemical-potential buffer");
    return 0;
}
