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

constexpr unsigned BLOCK_X = 32;
constexpr unsigned BLOCK_Y = 4;
constexpr unsigned BLOCK_Z = 2;
constexpr unsigned TILE_X = BLOCK_X + 2;
constexpr unsigned TILE_Y = BLOCK_Y + 2;
constexpr unsigned TILE_Z = BLOCK_Z + 2;
constexpr unsigned TILE_SIZE = TILE_X * TILE_Y * TILE_Z;

[[noreturn]] void cudaFailure(const char* operation, cudaError_t error) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        cudaFailure(operation, error);
    }
}

__device__ __forceinline__ size_t deviceIndex(size_t x, size_t y, size_t z,
                                               size_t nx, size_t plane) {
    return z * plane + y * nx + x;
}

// Cooperatively stage a block and its one-cell clamped halo.  Both stencil
// kernels use this routine so that each field value is normally fetched from
// global memory only once per block instead of once per stencil arm.
__device__ __forceinline__ void loadTile(const double* __restrict__ field,
                                         double* tile, size_t nx, size_t ny,
                                         size_t nz) {
    const unsigned thread = threadIdx.x + BLOCK_X *
                            (threadIdx.y + BLOCK_Y * threadIdx.z);
    const unsigned threads = BLOCK_X * BLOCK_Y * BLOCK_Z;
    const long long originX = static_cast<long long>(blockIdx.x) * BLOCK_X - 1;
    const long long originY = static_cast<long long>(blockIdx.y) * BLOCK_Y - 1;
    const long long originZ = static_cast<long long>(blockIdx.z) * BLOCK_Z - 1;
    const size_t plane = nx * ny;

    for (unsigned local = thread; local < TILE_SIZE; local += threads) {
        const unsigned sx = local % TILE_X;
        const unsigned sy = (local / TILE_X) % TILE_Y;
        const unsigned sz = local / (TILE_X * TILE_Y);

        const long long rawX = originX + sx;
        const long long rawY = originY + sy;
        const long long rawZ = originZ + sz;
        const size_t gx = static_cast<size_t>(max(0LL, min(rawX, static_cast<long long>(nx) - 1)));
        const size_t gy = static_cast<size_t>(max(0LL, min(rawY, static_cast<long long>(ny) - 1)));
        const size_t gz = static_cast<size_t>(max(0LL, min(rawZ, static_cast<long long>(nz) - 1)));
        tile[local] = field[deviceIndex(gx, gy, gz, nx, plane)];
    }
}

__device__ __forceinline__ double tileLaplacian(const double* tile,
                                                unsigned sx, unsigned sy,
                                                unsigned sz) {
    const unsigned center = sx + TILE_X * (sy + TILE_Y * sz);
    const double cxx = tile[center + 1] + tile[center - 1]
                     - 2.0 * tile[center];
    const double cyy = tile[center + TILE_X] + tile[center - TILE_X]
                     - 2.0 * tile[center];
    const double czz = tile[center + TILE_X * TILE_Y]
                     + tile[center - TILE_X * TILE_Y]
                     - 2.0 * tile[center];
    return cxx + cyy + czz;
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void chemicalPotentialKernel(const double* __restrict__ c,
                             double* __restrict__ mu,
                             size_t nx, size_t ny, size_t nz) {
    __shared__ double tile[TILE_SIZE];
    loadTile(c, tile, nx, ny, nz);
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const unsigned sx = threadIdx.x + 1;
    const unsigned sy = threadIdx.y + 1;
    const unsigned sz = threadIdx.z + 1;
    const unsigned center = sx + TILE_X * (sy + TILE_Y * sz);
    const double cv = tile[center];

    // dx, dy and dz are all 1.0 in this benchmark.  Keep the original
    // expression (rather than algebraically reducing it) to retain its
    // floating-point evaluation semantics.
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB =  (2.0 / 9.0);
    const double bulk = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB
                              - 2.0 * cv * eAB)
                      + 3.0 * cv + cv * cv * cv;
    const size_t index = deviceIndex(x, y, z, nx, nx * ny);
    mu[index] = bulk - 0.5 * tileLaplacian(tile, sx, sy, sz);
}

__global__ __launch_bounds__(BLOCK_X * BLOCK_Y * BLOCK_Z)
void updateKernel(double* __restrict__ cnew,
                  const double* __restrict__ cold,
                  const double* __restrict__ mu,
                  size_t nx, size_t ny, size_t nz) {
    __shared__ double tile[TILE_SIZE];
    loadTile(mu, tile, nx, ny, nz);
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) return;

    const unsigned sx = threadIdx.x + 1;
    const unsigned sy = threadIdx.y + 1;
    const unsigned sz = threadIdx.z + 1;
    const size_t index = deviceIndex(x, y, z, nx, nx * ny);
    cnew[index] = cold[index] + 0.01 * tileLaplacian(tile, sx, sy, sz);
}

void initializeConcentration(std::vector<double>& c, size_t nx, size_t ny,
                             size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t i = 0; i < volume; ++i) {
        const double pseudo = (((i + 1) * size_t{1299709}) % volume)
                            / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

bool validateResult(const std::vector<double>& c) {
    for (double value : c) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    const auto bounds = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *bounds.first,
                *bounds.second);
    if (*bounds.second > 10.0 || *bounds.first < -10.0) {
        std::printf("Validation failed: values out of expected range\n");
        return false;
    }
    return true;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
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
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0
        || nx > static_cast<size_t>(std::numeric_limits<long long>::max())
        || ny > static_cast<size_t>(std::numeric_limits<long long>::max())
        || nz > static_cast<size_t>(std::numeric_limits<long long>::max())
        || nx > std::numeric_limits<size_t>::max() / ny
        || nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iteration count must be valid positive values.\n");
        return 1;
    }

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Requested grid is too large.\n");
        return 1;
    }
    const size_t bytes = gridSize * sizeof(double);

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, nx, ny, nz);

    double* deviceCold = nullptr;
    double* deviceNew = nullptr;
    double* deviceMu = nullptr;
    checkCuda(cudaMalloc(&deviceCold, bytes), "allocating the current field");
    checkCuda(cudaMalloc(&deviceNew, bytes), "allocating the next field");
    checkCuda(cudaMalloc(&deviceMu, bytes), "allocating the chemical potential");
    checkCuda(cudaMemcpy(deviceCold, concentration.data(), bytes,
                         cudaMemcpyHostToDevice), "copying the initial field");

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned>((nz + BLOCK_Z - 1) / BLOCK_Z));

    std::printf("Running Cahn-Hilliard simulation on CUDA...\n");
    checkCuda(cudaDeviceSynchronize(), "preparing the simulation");
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < iterations; ++step) {
        chemicalPotentialKernel<<<grid, block>>>(deviceCold, deviceMu, nx, ny, nz);
        updateKernel<<<grid, block>>>(deviceNew, deviceCold, deviceMu, nx, ny, nz);
        std::swap(deviceCold, deviceNew);
    }
    checkCuda(cudaGetLastError(), "launching simulation kernels");
    checkCuda(cudaDeviceSynchronize(), "executing simulation kernels");
    const auto end = std::chrono::steady_clock::now();

    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    const auto elapsedMilliseconds =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::printf("Computation time: %lld ms\n",
                static_cast<long long>(elapsedMilliseconds));
    const double cellUpdates = static_cast<double>(gridSize)
                             * static_cast<double>(iterations);
    const double mcups = elapsedSeconds > 0.0
                       ? cellUpdates / elapsedSeconds / 1.0e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Preserve the original output behavior while avoiding a device-to-host
    // transfer when neither result printing nor validation was requested.
    if (printResults || validate) {
        checkCuda(cudaMemcpy(concentration.data(), deviceCold, bytes,
                             cudaMemcpyDeviceToHost), "copying the final field");
    }

    checkCuda(cudaFree(deviceMu), "freeing the chemical potential");
    checkCuda(cudaFree(deviceNew), "freeing the next field");
    checkCuda(cudaFree(deviceCold), "freeing the current field");

    if (printResults) {
        print_results(concentration, "Concentration");
    }
    if (validate) {
        std::printf("Validating result...\n");
        if (!validateResult(concentration)) {
            std::printf("Validation: FAILED\n");
            return 1;
        }
        std::printf("Validation: PASSED\n");
    }
    return 0;
}
