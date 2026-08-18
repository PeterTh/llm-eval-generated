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
constexpr int kTileX = kBlockX + 2;
constexpr int kTileY = kBlockY + 2;
constexpr int kTileZ = kBlockZ + 2;
constexpr int kTileSize = kTileX * kTileY * kTileZ;

inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                             const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void checkCuda(const cudaError_t status, const char* operation, const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s at %s:%d: %s\n", operation, file, line,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

__device__ __forceinline__ int clampCoordinate(const int value, const int upper) {
    return value < 0 ? 0 : (value >= upper ? upper - 1 : value);
}

template <typename T>
__device__ __forceinline__ void loadTile(const T* __restrict__ source, T* tile,
                                         const int nx, const int ny, const int nz) {
    const int threadId = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
    const int threadCount = blockDim.x * blockDim.y * blockDim.z;
    const int baseX = static_cast<int>(blockIdx.x) * kBlockX;
    const int baseY = static_cast<int>(blockIdx.y) * kBlockY;
    const int baseZ = static_cast<int>(blockIdx.z) * kBlockZ;
    const size_t plane = static_cast<size_t>(nx) * ny;

    // The one-cell halo implements the original clamped boundary condition.  Each
    // block loads the complete tile cooperatively, reducing the 7 global loads per
    // output point of a naive stencil to roughly 2.34 loads for full blocks.
    for (int tileIndex = threadId; tileIndex < kTileSize; tileIndex += threadCount) {
        const int tileX = tileIndex % kTileX;
        const int tileY = (tileIndex / kTileX) % kTileY;
        const int tileZ = tileIndex / (kTileX * kTileY);
        const int globalX = clampCoordinate(baseX + tileX - 1, nx);
        const int globalY = clampCoordinate(baseY + tileY - 1, ny);
        const int globalZ = clampCoordinate(baseZ + tileZ - 1, nz);
        tile[tileIndex] = source[static_cast<size_t>(globalZ) * plane +
                                 static_cast<size_t>(globalY) * nx + globalX];
    }
}

__device__ __forceinline__ double tileLaplacian(const double* tile, const int tileIndex,
                                                const double invDx2, const double invDy2,
                                                const double invDz2) {
    const double center = tile[tileIndex];
    const double cxx = (tile[tileIndex + 1] + tile[tileIndex - 1] - 2.0 * center) * invDx2;
    const double cyy = (tile[tileIndex + kTileX] + tile[tileIndex - kTileX] - 2.0 * center) * invDy2;
    const int zStride = kTileX * kTileY;
    const double czz = (tile[tileIndex + zStride] + tile[tileIndex - zStride] - 2.0 * center) * invDz2;
    return cxx + cyy + czz;
}

__global__ __launch_bounds__(kBlockX * kBlockY * kBlockZ, 2)
void chemicalPotentialKernel(const double* __restrict__ concentration, double* __restrict__ chemicalPotential,
                             const int nx, const int ny, const int nz,
                             const double invDx2, const double invDy2, const double invDz2,
                             const double gamma, const double eAA, const double eBB, const double eAB) {
    __shared__ double tile[kTileSize];
    loadTile(concentration, tile, nx, ny, nz);
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x) * kBlockX + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * kBlockY + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * kBlockZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int tileIndex = (threadIdx.z + 1) * (kTileX * kTileY) +
                          (threadIdx.y + 1) * kTileX + threadIdx.x + 1;
    const double concentrationValue = tile[tileIndex];
    const double laplacian = tileLaplacian(tile, tileIndex, invDx2, invDy2, invDz2);
    const size_t outputIndex = (static_cast<size_t>(z) * ny + y) * nx + x;

    chemicalPotential[outputIndex] =
        4.5 * ((concentrationValue + 1.0) * eAA + (concentrationValue - 1.0) * eBB -
               2.0 * concentrationValue * eAB) +
        3.0 * concentrationValue + concentrationValue * concentrationValue * concentrationValue -
        gamma * laplacian;
}

__global__ __launch_bounds__(kBlockX * kBlockY * kBlockZ, 2)
void cahnHilliardUpdateKernel(double* __restrict__ nextConcentration,
                              const double* __restrict__ concentration,
                              const double* __restrict__ chemicalPotential,
                              const int nx, const int ny, const int nz,
                              const double dtD, const double invDx2, const double invDy2,
                              const double invDz2) {
    __shared__ double muTile[kTileSize];
    loadTile(chemicalPotential, muTile, nx, ny, nz);
    __syncthreads();

    const int x = static_cast<int>(blockIdx.x) * kBlockX + threadIdx.x;
    const int y = static_cast<int>(blockIdx.y) * kBlockY + threadIdx.y;
    const int z = static_cast<int>(blockIdx.z) * kBlockZ + threadIdx.z;
    if (x >= nx || y >= ny || z >= nz) {
        return;
    }

    const int tileIndex = (threadIdx.z + 1) * (kTileX * kTileY) +
                          (threadIdx.y + 1) * kTileX + threadIdx.x + 1;
    const double laplacian = tileLaplacian(muTile, tileIndex, invDx2, invDy2, invDz2);
    const size_t outputIndex = (static_cast<size_t>(z) * ny + y) * nx + x;
    nextConcentration[outputIndex] = concentration[outputIndex] + dtD * laplacian;
}

void initializeConcentration(std::vector<double>& concentration, const size_t nx, const size_t ny,
                             const size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = idx3(x, y, z, nx, ny);
                const double pseudo = (((linearId + 1) * 1299709) % volume) /
                                      static_cast<double>(volume);
                concentration[linearId] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& concentration, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

    if (nx == 0 || ny == 0 || nz == 0 || nx > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        ny > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        nz > static_cast<size_t>(std::numeric_limits<int>::max())) {
        std::fprintf(stderr, "Grid dimensions must be positive 32-bit integers for the CUDA implementation.\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    constexpr double dx = 1.0;
    constexpr double dy = 1.0;
    constexpr double dz = 1.0;
    constexpr double dt = 0.01;
    constexpr double eAA = -(2.0 / 9.0);
    constexpr double eBB = -(2.0 / 9.0);
    constexpr double eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    constexpr double diffusion = 1.0;

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid is too large.\n");
        return 1;
    }

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, nx, ny, nz);

    double* deviceConcentration = nullptr;
    double* deviceNextConcentration = nullptr;
    double* deviceChemicalPotential = nullptr;
    CUDA_CHECK(cudaMalloc(&deviceConcentration, gridSize * sizeof(*deviceConcentration)));
    CUDA_CHECK(cudaMalloc(&deviceNextConcentration, gridSize * sizeof(*deviceNextConcentration)));
    CUDA_CHECK(cudaMalloc(&deviceChemicalPotential, gridSize * sizeof(*deviceChemicalPotential)));
    CUDA_CHECK(cudaMemcpy(deviceConcentration, concentration.data(), gridSize * sizeof(*deviceConcentration),
                          cudaMemcpyHostToDevice));

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid((static_cast<unsigned int>(nx) + kBlockX - 1) / kBlockX,
                    (static_cast<unsigned int>(ny) + kBlockY - 1) / kBlockY,
                    (static_cast<unsigned int>(nz) + kBlockZ - 1) / kBlockZ);
    const int deviceNx = static_cast<int>(nx);
    const int deviceNy = static_cast<int>(ny);
    const int deviceNz = static_cast<int>(nz);
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    for (int step = 0; step < iterations; ++step) {
        chemicalPotentialKernel<<<grid, block>>>(deviceConcentration, deviceChemicalPotential, deviceNx, deviceNy,
                                                  deviceNz, invDx2, invDy2, invDz2, gamma, eAA, eBB, eAB);
        CUDA_CHECK(cudaGetLastError());
        cahnHilliardUpdateKernel<<<grid, block>>>(deviceNextConcentration, deviceConcentration,
                                                   deviceChemicalPotential, deviceNx, deviceNy, deviceNz,
                                                   dt * diffusion, invDx2, invDy2, invDz2);
        CUDA_CHECK(cudaGetLastError());
        std::swap(deviceConcentration, deviceNextConcentration);
    }
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    const auto wholeMilliseconds = static_cast<long>(elapsedMilliseconds);
    std::printf("Computation time: %ld ms\n", wholeMilliseconds);
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double seconds = static_cast<double>(elapsedMilliseconds) / 1000.0;
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1.0e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        CUDA_CHECK(cudaMemcpy(concentration.data(), deviceConcentration, gridSize * sizeof(*deviceConcentration),
                              cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFree(deviceChemicalPotential));
    CUDA_CHECK(cudaFree(deviceNextConcentration));
    CUDA_CHECK(cudaFree(deviceConcentration));

    if (printResults) {
        print_results(concentration, "Concentration");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(concentration, nx, ny, nz)) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
