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

constexpr int kBlockX = 8;
constexpr int kBlockY = 8;
constexpr int kBlockZ = 4;
constexpr int kSharedX = kBlockX + 2;
constexpr int kSharedY = kBlockY + 2;
constexpr int kSharedZ = kBlockZ + 2;
constexpr double kTimeStep = 0.01;
constexpr double kGamma = 0.5;
constexpr double kEAA = -(2.0 / 9.0);
constexpr double kEBB = -(2.0 / 9.0);
constexpr double kEAB = 2.0 / 9.0;

[[noreturn]] void cudaFail(const cudaError_t status, const char* expression,
                           const char* file, const int line) {
    std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line,
                 expression, cudaGetErrorString(status));
    std::exit(EXIT_FAILURE);
}

#define CUDA_CHECK(expression)                                                    \
    do {                                                                          \
        const cudaError_t cudaStatus = (expression);                              \
        if (cudaStatus != cudaSuccess) {                                          \
            cudaFail(cudaStatus, #expression, __FILE__, __LINE__);                \
        }                                                                         \
    } while (false)

__device__ __forceinline__ size_t deviceIdx3(const size_t x, const size_t y,
                                              const size_t z, const size_t nx,
                                              const size_t plane) {
    return z * plane + y * nx + x;
}

// The tile holds the block and a one-cell halo.  Only face halo cells are needed
// by a 7-point stencil, so the edges and corners need not be loaded.
__device__ void loadStencilTile(const double* __restrict__ field,
                                double tile[kSharedZ][kSharedY][kSharedX],
                                const size_t nx, const size_t ny, const size_t nz,
                                const size_t plane) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tz = threadIdx.z;

    const size_t baseX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t baseZ = static_cast<size_t>(blockIdx.z) * kBlockZ;
    const size_t x = min(baseX + static_cast<size_t>(tx), nx - 1);
    const size_t y = min(baseY + static_cast<size_t>(ty), ny - 1);
    const size_t z = min(baseZ + static_cast<size_t>(tz), nz - 1);

    tile[tz + 1][ty + 1][tx + 1] = field[deviceIdx3(x, y, z, nx, plane)];

    if (tx == 0) {
        const size_t xm = x == 0 ? 0 : x - 1;
        tile[tz + 1][ty + 1][0] = field[deviceIdx3(xm, y, z, nx, plane)];
    }
    if (tx == kBlockX - 1) {
        const size_t xp = x + 1 < nx ? x + 1 : x;
        tile[tz + 1][ty + 1][kBlockX + 1] = field[deviceIdx3(xp, y, z, nx, plane)];
    }
    if (ty == 0) {
        const size_t ym = y == 0 ? 0 : y - 1;
        tile[tz + 1][0][tx + 1] = field[deviceIdx3(x, ym, z, nx, plane)];
    }
    if (ty == kBlockY - 1) {
        const size_t yp = y + 1 < ny ? y + 1 : y;
        tile[tz + 1][kBlockY + 1][tx + 1] = field[deviceIdx3(x, yp, z, nx, plane)];
    }
    if (tz == 0) {
        const size_t zm = z == 0 ? 0 : z - 1;
        tile[0][ty + 1][tx + 1] = field[deviceIdx3(x, y, zm, nx, plane)];
    }
    if (tz == kBlockZ - 1) {
        const size_t zp = z + 1 < nz ? z + 1 : z;
        tile[kBlockZ + 1][ty + 1][tx + 1] = field[deviceIdx3(x, y, zp, nx, plane)];
    }
    __syncthreads();
}

__device__ __forceinline__ double tileLaplacian(
    const double tile[kSharedZ][kSharedY][kSharedX], const int tx, const int ty,
    const int tz) {
    const int x = tx + 1;
    const int y = ty + 1;
    const int z = tz + 1;
    const double center = tile[z][y][x];
    // The benchmark's fixed grid spacing is one in every dimension, so the
    // divisions in the reference stencil are exact identity operations.
    const double cxx = tile[z][y][x + 1] + tile[z][y][x - 1] - 2.0 * center;
    const double cyy = tile[z][y + 1][x] + tile[z][y - 1][x] - 2.0 * center;
    const double czz = tile[z + 1][y][x] + tile[z - 1][y][x] - 2.0 * center;
    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                               const size_t volume) {
    const size_t linearId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (linearId < volume) {
        const double pseudo =
            (((linearId + 1) * static_cast<size_t>(1299709)) % volume) /
            static_cast<double>(volume);
        c[linearId] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu, const size_t nx,
    const size_t ny, const size_t nz) {
    __shared__ double tile[kSharedZ][kSharedY][kSharedX];
    const size_t plane = nx * ny;
    loadStencilTile(c, tile, nx, ny, nz, plane);

    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * kBlockZ + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const double cv = tile[threadIdx.z + 1][threadIdx.y + 1][threadIdx.x + 1];
        mu[deviceIdx3(x, y, z, nx, plane)] =
            4.5 * ((cv + 1.0) * kEAA + (cv - 1.0) * kEBB - 2.0 * cv * kEAB) +
            3.0 * cv + cv * cv * cv -
            kGamma * tileLaplacian(tile, threadIdx.x, threadIdx.y, threadIdx.z);
    }
}

__global__ void cahnHilliardUpdateKernel(
    const double* __restrict__ cold, const double* __restrict__ mu,
    double* __restrict__ cnew, const size_t nx, const size_t ny, const size_t nz) {
    __shared__ double muTile[kSharedZ][kSharedY][kSharedX];
    const size_t plane = nx * ny;
    loadStencilTile(mu, muTile, nx, ny, nz, plane);

    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * kBlockZ + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const size_t index = deviceIdx3(x, y, z, nx, plane);
        cnew[index] = cold[index] + kTimeStep *
                      tileLaplacian(muTile, threadIdx.x, threadIdx.y, threadIdx.z);
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

bool validGridSize(const size_t nx, const size_t ny, const size_t nz, size_t* volume) {
    if (nx == 0 || ny == 0 || nz == 0 || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        return false;
    }
    *volume = nx * ny * nz;
    return true;
}

size_t ceilDiv(const size_t value, const size_t divisor) {
    return 1 + (value - 1) / divisor;
}

bool launchGridIsSupported(const size_t nx, const size_t ny, const size_t nz,
                           const cudaDeviceProp& properties) {
    return ceilDiv(nx, kBlockX) <= static_cast<size_t>(properties.maxGridSize[0]) &&
           ceilDiv(ny, kBlockY) <= static_cast<size_t>(properties.maxGridSize[1]) &&
           ceilDiv(nz, kBlockZ) <= static_cast<size_t>(properties.maxGridSize[2]);
}

}  // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = std::atoi(argv[++i]);
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

    size_t gridSize = 0;
    if (!validGridSize(nx, ny, nz, &gridSize) || iterations < 0) {
        std::fprintf(stderr, "Grid dimensions must be positive without overflow, and time steps must be non-negative.\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    if (!launchGridIsSupported(nx, ny, nz, properties) ||
        ceilDiv(gridSize, static_cast<size_t>(kBlockX * kBlockY * kBlockZ)) >
            static_cast<size_t>(properties.maxGridSize[0])) {
        std::fprintf(stderr, "Grid dimensions exceed CUDA launch limits for the selected device.\n");
        return 1;
    }

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(static_cast<unsigned int>(ceilDiv(nx, kBlockX)),
                    static_cast<unsigned int>(ceilDiv(ny, kBlockY)),
                    static_cast<unsigned int>(ceilDiv(nz, kBlockZ)));
    const unsigned int initBlocks = static_cast<unsigned int>(
        ceilDiv(gridSize, static_cast<size_t>(kBlockX * kBlockY * kBlockZ)));

    double* deviceCold = nullptr;
    double* deviceCnew = nullptr;
    double* deviceMu = nullptr;
    const size_t bytes = gridSize * sizeof(double);
    CUDA_CHECK(cudaMalloc(&deviceCold, bytes));
    CUDA_CHECK(cudaMalloc(&deviceCnew, bytes));
    CUDA_CHECK(cudaMalloc(&deviceMu, bytes));

    std::printf("Initializing concentration field...\n");
    initializeConcentrationKernel<<<initBlocks, kBlockX * kBlockY * kBlockZ>>>(deviceCold, gridSize);
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start{};
    cudaEvent_t stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(start));
    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<grid, block>>>(deviceCold, deviceMu, nx, ny, nz);
        cahnHilliardUpdateKernel<<<grid, block>>>(deviceCold, deviceMu, deviceCnew, nx, ny, nz);
        std::swap(deviceCold, deviceCnew);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMilliseconds));
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedMilliseconds > 0.0F
                             ? cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1.0e3)
                             : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(cold.data(), deviceCold, bytes, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(deviceMu));
    CUDA_CHECK(cudaFree(deviceCnew));
    CUDA_CHECK(cudaFree(deviceCold));

    if (printResults) {
        print_results(cold, "Concentration");
    }
    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(cold)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }
    return 0;
}
