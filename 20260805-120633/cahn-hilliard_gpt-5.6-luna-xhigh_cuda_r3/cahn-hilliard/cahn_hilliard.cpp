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

namespace {

// The tile dimensions are deliberately powers of two.  The x dimension is
// contiguous in memory, which gives coalesced global loads for every tile.
constexpr unsigned int kBlockX = 8;
constexpr unsigned int kBlockY = 8;
constexpr unsigned int kBlockZ = 8;
constexpr unsigned int kTileX = kBlockX + 2;
constexpr unsigned int kTileY = kBlockY + 2;
constexpr unsigned int kTileZ = kBlockZ + 2;
constexpr size_t kTileVolume = static_cast<size_t>(kTileX) * kTileY * kTileZ;

void checkCuda(const cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

__device__ __forceinline__ size_t clampedCoordinate(const unsigned int local,
                                                    const size_t origin,
                                                    const size_t extent) {
    if (local == 0) {
        return origin == 0 ? 0 : origin - 1;
    }

    const size_t coordinate = origin + static_cast<size_t>(local) - 1;
    return coordinate < extent ? coordinate : extent - 1;
}

__device__ __forceinline__ size_t tileIndex(const unsigned int x,
                                            const unsigned int y,
                                            const unsigned int z) {
    return (static_cast<size_t>(z) * kTileY + y) * kTileX + x;
}

__device__ __forceinline__ size_t threadIndexInBlock() {
    return (static_cast<size_t>(threadIdx.z) * blockDim.y + threadIdx.y) *
               blockDim.x +
           threadIdx.x;
}

__device__ __forceinline__ size_t blockThreadCount() {
    return static_cast<size_t>(blockDim.x) * blockDim.y * blockDim.z;
}

__global__ void initializeConcentrationKernel(double* __restrict__ concentration,
                                               const size_t gridSize,
                                               const size_t volume) {
    const size_t thread = static_cast<size_t>(blockIdx.x) * blockDim.x +
                           threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t linearId = thread; linearId < gridSize; linearId += stride) {
        const size_t pseudo = ((linearId + 1) * static_cast<size_t>(1299709)) % volume;
        concentration[linearId] = -1.0 +
                                   2.0 * (static_cast<double>(pseudo) /
                                          static_cast<double>(volume));
    }
}

// Compute the chemical potential.  Each block loads one 8^3 output tile and
// its one-cell halo into shared memory.  The clamped coordinates exactly
// reproduce the original boundary condition at all six faces.
__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
    const size_t nx, const size_t ny, const size_t nz,
    const double invDx2, const double invDy2, const double invDz2,
    const double gamma, const double eAA, const double eBB, const double eAB,
    const size_t tileCountZ) {
    __shared__ double tile[kTileVolume];

    const size_t originX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t originY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t planeSize = nx * ny;
    const size_t thread = threadIndexInBlock();
    const size_t threadCount = blockThreadCount();

    for (size_t blockZ = blockIdx.z; blockZ < tileCountZ; blockZ += gridDim.z) {
        const size_t originZ = blockZ * kBlockZ;

        for (size_t linear = thread; linear < kTileVolume; linear += threadCount) {
            const unsigned int localX = static_cast<unsigned int>(linear % kTileX);
            const unsigned int localY = static_cast<unsigned int>((linear / kTileX) % kTileY);
            const unsigned int localZ = static_cast<unsigned int>(linear / (kTileX * kTileY));

            const size_t globalX = clampedCoordinate(localX, originX, nx);
            const size_t globalY = clampedCoordinate(localY, originY, ny);
            const size_t globalZ = clampedCoordinate(localZ, originZ, nz);
            tile[linear] = concentration[globalZ * planeSize + globalY * nx + globalX];
        }
        __syncthreads();

        const size_t x = originX + threadIdx.x;
        const size_t y = originY + threadIdx.y;
        const size_t z = originZ + threadIdx.z;
        if (x < nx && y < ny && z < nz) {
            const size_t center = tileIndex(threadIdx.x + 1, threadIdx.y + 1,
                                            threadIdx.z + 1);
            const double cv = tile[center];

            const double laplacian =
                (tile[center + 1] + tile[center - 1] - 2.0 * cv) * invDx2 +
                (tile[center + kTileX] + tile[center - kTileX] - 2.0 * cv) * invDy2 +
                (tile[center + kTileX * kTileY] +
                 tile[center - kTileX * kTileY] - 2.0 * cv) * invDz2;

            const size_t output = z * planeSize + y * nx + x;
            chemicalPotential[output] =
                4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                3.0 * cv + cv * cv * cv - gamma * laplacian;
        }
        __syncthreads();
    }
}

// Compute the conserved Cahn-Hilliard update from the chemical-potential
// field.  This is a separate kernel because every update uses the complete
// chemical-potential field produced by the preceding phase.
__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ concentrationNew,
    const double* __restrict__ concentrationOld,
    const double* __restrict__ chemicalPotential,
    const size_t nx, const size_t ny, const size_t nz,
    const double invDx2, const double invDy2, const double invDz2,
    const double dtD, const size_t tileCountZ) {
    __shared__ double tile[kTileVolume];

    const size_t originX = static_cast<size_t>(blockIdx.x) * kBlockX;
    const size_t originY = static_cast<size_t>(blockIdx.y) * kBlockY;
    const size_t planeSize = nx * ny;
    const size_t thread = threadIndexInBlock();
    const size_t threadCount = blockThreadCount();

    for (size_t blockZ = blockIdx.z; blockZ < tileCountZ; blockZ += gridDim.z) {
        const size_t originZ = blockZ * kBlockZ;

        for (size_t linear = thread; linear < kTileVolume; linear += threadCount) {
            const unsigned int localX = static_cast<unsigned int>(linear % kTileX);
            const unsigned int localY = static_cast<unsigned int>((linear / kTileX) % kTileY);
            const unsigned int localZ = static_cast<unsigned int>(linear / (kTileX * kTileY));

            const size_t globalX = clampedCoordinate(localX, originX, nx);
            const size_t globalY = clampedCoordinate(localY, originY, ny);
            const size_t globalZ = clampedCoordinate(localZ, originZ, nz);
            tile[linear] = chemicalPotential[globalZ * planeSize + globalY * nx + globalX];
        }
        __syncthreads();

        const size_t x = originX + threadIdx.x;
        const size_t y = originY + threadIdx.y;
        const size_t z = originZ + threadIdx.z;
        if (x < nx && y < ny && z < nz) {
            const size_t center = tileIndex(threadIdx.x + 1, threadIdx.y + 1,
                                            threadIdx.z + 1);
            const double laplacian =
                (tile[center + 1] + tile[center - 1] -
                 2.0 * tile[center]) * invDx2 +
                (tile[center + kTileX] + tile[center - kTileX] -
                 2.0 * tile[center]) * invDy2 +
                (tile[center + kTileX * kTileY] +
                 tile[center - kTileX * kTileY] -
                 2.0 * tile[center]) * invDz2;

            const size_t output = z * planeSize + y * nx + x;
            concentrationNew[output] = concentrationOld[output] + dtD * laplacian;
        }
        __syncthreads();
    }
}

size_t ceilDivide(const size_t value, const size_t divisor) {
    return value == 0 ? 0 : (value - 1) / divisor + 1;
}

dim3 simulationGrid(const size_t nx, const size_t ny, const size_t nz,
                    size_t& tileCountZ) {
    tileCountZ = ceilDivide(nz, kBlockZ);
    const size_t tileCountX = ceilDivide(nx, kBlockX);
    const size_t tileCountY = ceilDivide(ny, kBlockY);

    constexpr size_t maxGridDimension =
        static_cast<size_t>(std::numeric_limits<unsigned int>::max());
    constexpr size_t maxGridZ = 65535;
    if (tileCountX > maxGridDimension || tileCountY > maxGridDimension ||
        tileCountZ == 0) {
        std::fprintf(stderr, "Grid dimensions are too large for CUDA\n");
        std::exit(EXIT_FAILURE);
    }

    return dim3(static_cast<unsigned int>(tileCountX),
                static_cast<unsigned int>(tileCountY),
                static_cast<unsigned int>(std::min(tileCountZ, maxGridZ)));
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
            nx = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = static_cast<size_t>(std::atoi(argv[++i]));
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

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (nx == 0 || ny == 0 || nz == 0) {
        std::fprintf(stderr, "Grid dimensions must be positive\n");
        return 1;
    }

    if (nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid size overflows host index range\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid allocation size overflows host index range\n");
        return 1;
    }
    const size_t allocationBytes = gridSize * sizeof(double);

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Physical parameters.  They match the original implementation exactly.
    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = (2.0 / 9.0);
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;

    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);

    double* deviceCold = nullptr;
    double* deviceCnew = nullptr;
    double* deviceChemicalPotential = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCold), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCnew), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceChemicalPotential),
                          allocationBytes));

    std::printf("Initializing concentration field...\n");
    constexpr unsigned int initializationThreads = 256;
    const size_t initializationBlocks = std::min(
        ceilDivide(gridSize, initializationThreads), static_cast<size_t>(65535));
    initializeConcentrationKernel<<<static_cast<unsigned int>(initializationBlocks),
                                    initializationThreads>>>(
        deviceCold, gridSize, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    size_t tileCountZ = 0;
    const dim3 grid = simulationGrid(nx, ny, nz, tileCountZ);
    const dim3 block(kBlockX, kBlockY, kBlockZ);

    std::printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<grid, block>>>(
            deviceCold, deviceChemicalPotential, nx, ny, nz,
            1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz),
            gamma, eAA, eBB, eAB, tileCountZ);
        cahnHilliardUpdateKernel<<<grid, block>>>(
            deviceCnew, deviceCold, deviceChemicalPotential, nx, ny, nz,
            1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz),
            dt * diffusion, tileCountZ);

        std::swap(deviceCold, deviceCnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds =
        std::chrono::duration<double>(end - start).count();

    std::printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedSeconds > 0.0
                             ? cellUpdates / elapsedSeconds / 1e6
                             : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    CUDA_CHECK(cudaMemcpy(cold.data(), deviceCold, allocationBytes,
                          cudaMemcpyDeviceToHost));

    if (printResults) {
        print_results(cold, "Concentration");
    }

    CUDA_CHECK(cudaFree(deviceChemicalPotential));
    CUDA_CHECK(cudaFree(deviceCnew));
    CUDA_CHECK(cudaFree(deviceCold));

    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(cold);

        if (valid) {
            std::printf("Validation: PASSED\n");
            return 0;
        }

        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
