#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// One full warp spans X, making all interior stencil loads coalesced.  The
// remaining dimensions provide enough blocks and warps to saturate large GPUs.
constexpr int BLOCK_X = 32;
constexpr int BLOCK_Y = 4;
constexpr int BLOCK_Z = 2;
constexpr int THREADS = BLOCK_X * BLOCK_Y * BLOCK_Z;

[[noreturn]] void cudaFailure(const char* operation, cudaError_t error) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t cuda_check_error = (call);                            \
        if (cuda_check_error != cudaSuccess)                                    \
            cudaFailure(#call, cuda_check_error);                               \
    } while (false)

__device__ __forceinline__ int clampIndex(int value, int extent) {
    return max(0, min(value, extent - 1));
}

__global__ __launch_bounds__(THREADS)
void computeChemicalPotential(const double* __restrict__ c,
                              double* __restrict__ mu, int nx, int ny, int nz,
                              size_t plane) {
    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;

    const size_t index = static_cast<size_t>(z) * plane +
                         static_cast<size_t>(y) * nx + x;
    const double cv = c[index];
    const double cxx =
        c[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
          clampIndex(x + 1, nx)] +
        c[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
          clampIndex(x - 1, nx)] - 2.0 * cv;
    const double cyy =
        c[static_cast<size_t>(z) * plane +
          static_cast<size_t>(clampIndex(y + 1, ny)) * nx + x] +
        c[static_cast<size_t>(z) * plane +
          static_cast<size_t>(clampIndex(y - 1, ny)) * nx + x] - 2.0 * cv;
    const double czz =
        c[static_cast<size_t>(clampIndex(z + 1, nz)) * plane +
          static_cast<size_t>(y) * nx + x] +
        c[static_cast<size_t>(clampIndex(z - 1, nz)) * plane +
          static_cast<size_t>(y) * nx + x] - 2.0 * cv;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB -
                       2.0 * cv * eAB) +
                3.0 * cv + cv * cv * cv - 0.5 * (cxx + cyy + czz);
}

__global__ __launch_bounds__(THREADS)
void updateConcentration(const double* __restrict__ cold,
                         double* __restrict__ cnew,
                         const double* __restrict__ mu, int nx, int ny, int nz,
                         size_t plane) {
    const int x = static_cast<int>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz)
        return;

    const size_t index = static_cast<size_t>(z) * plane +
                         static_cast<size_t>(y) * nx + x;
    const double center = mu[index];
    const double mux =
        mu[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
           clampIndex(x + 1, nx)] +
        mu[static_cast<size_t>(z) * plane + static_cast<size_t>(y) * nx +
           clampIndex(x - 1, nx)] - 2.0 * center;
    const double muy =
        mu[static_cast<size_t>(z) * plane +
           static_cast<size_t>(clampIndex(y + 1, ny)) * nx + x] +
        mu[static_cast<size_t>(z) * plane +
           static_cast<size_t>(clampIndex(y - 1, ny)) * nx + x] - 2.0 * center;
    const double muz =
        mu[static_cast<size_t>(clampIndex(z + 1, nz)) * plane +
           static_cast<size_t>(y) * nx + x] +
        mu[static_cast<size_t>(clampIndex(z - 1, nz)) * plane +
           static_cast<size_t>(y) * nx + x] - 2.0 * center;
    cnew[index] = cold[index] + 0.01 * (mux + muy + muz);
}

void initializeConcentration(std::vector<double>& c, size_t nx, size_t ny,
                             size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t index = z * (nx * ny) + y * nx + x;
                const double pseudo =
                    (((index + 1) * 1299709) % volume) /
                    static_cast<double>(volume);
                c[index] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c) {
    for (double value : c) {
        if (!std::isfinite(value)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    const auto limits = std::minmax_element(c.begin(), c.end());
    std::printf("Concentration range: [%.6f, %.6f]\n", *limits.first,
                *limits.second);
    if (*limits.second > 10.0 || *limits.first < -10.0) {
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

bool parseNumber(const char* text, size_t& value, bool allowZero) {
    errno = 0;
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(text, &end, 10);
    if (errno || end == text || *end != '\0' || (!allowZero && parsed == 0) ||
        parsed > static_cast<unsigned long long>(std::numeric_limits<int>::max()))
        return false;
    value = static_cast<size_t>(parsed);
    return true;
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
        size_t parsed = 0;
        if ((std::strcmp(argv[i], "-x") == 0 ||
             std::strcmp(argv[i], "-y") == 0 ||
             std::strcmp(argv[i], "-z") == 0) &&
            i + 1 < argc) {
            const char option = argv[i][1];
            if (!parseNumber(argv[++i], parsed, false)) {
                std::fprintf(stderr, "Invalid grid size: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            if (option == 'x') nx = parsed;
            else if (option == 'y') ny = parsed;
            else nz = parsed;
        } else if (std::strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            size_t count = 0;
            if (!parseNumber(argv[++i], count, true)) {
                std::fprintf(stderr, "Invalid iteration count: %s\n", argv[i]);
                return EXIT_FAILURE;
            }
            iterations = static_cast<int>(count);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return EXIT_SUCCESS;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions are too large\n");
        return EXIT_FAILURE;
    }
    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid allocation is too large\n");
        return EXIT_FAILURE;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, nx, ny, nz);

    // Device allocation and the initial upload are setup costs, as allocation
    // and initialization were outside the timed region in the CPU benchmark.
    double* deviceCold = nullptr;
    double* deviceNew = nullptr;
    double* deviceMu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&deviceCold, bytes));
    CUDA_CHECK(cudaMalloc(&deviceNew, bytes));
    CUDA_CHECK(cudaMalloc(&deviceMu, bytes));
    CUDA_CHECK(cudaMemcpy(deviceCold, concentration.data(), bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFuncSetCacheConfig(computeChemicalPotential,
                                      cudaFuncCachePreferL1));
    CUDA_CHECK(cudaFuncSetCacheConfig(updateConcentration,
                                      cudaFuncCachePreferL1));

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid((nx + BLOCK_X - 1) / BLOCK_X,
                    (ny + BLOCK_Y - 1) / BLOCK_Y,
                    (nz + BLOCK_Z - 1) / BLOCK_Z);
    std::printf("Running Cahn-Hilliard simulation...\n");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int step = 0; step < iterations; ++step) {
        computeChemicalPotential<<<grid, block>>>(
            deviceCold, deviceMu, static_cast<int>(nx), static_cast<int>(ny),
            static_cast<int>(nz), nx * ny);
        updateConcentration<<<grid, block>>>(
            deviceCold, deviceNew, deviceMu, static_cast<int>(nx),
            static_cast<int>(ny), static_cast<int>(nz), nx * ny);
        std::swap(deviceCold, deviceNew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, stop));

    std::printf("Computation time: %.3f ms\n", elapsedMs);
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / (static_cast<double>(elapsedMs) * 1000.0);
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(concentration.data(), deviceCold, bytes,
                              cudaMemcpyDeviceToHost));
    }
    if (printResults)
        print_results(concentration, "Concentration");

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(concentration);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(deviceCold));
    CUDA_CHECK(cudaFree(deviceNew));
    CUDA_CHECK(cudaFree(deviceMu));
    return valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
