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

// These dimensions give 512 threads per block while keeping the input tile and
// the chemical-potential tile small enough for several resident blocks per SM.
constexpr int kBlockX = 8;
constexpr int kBlockY = 8;
constexpr int kBlockZ = 8;
constexpr int kBlockThreads = kBlockX * kBlockY * kBlockZ;
constexpr int kHalo = 2;
constexpr int kCTileX = kBlockX + 2 * kHalo;
constexpr int kCTileY = kBlockY + 2 * kHalo;
constexpr int kCTileZ = kBlockZ + 2 * kHalo;
constexpr int kCTileXY = kCTileX * kCTileY;
constexpr int kCTileSize = kCTileXY * kCTileZ;

// The update needs chemical potential one cell beyond the output block.  Each
// of those values in turn needs the two-cell concentration halo above.
constexpr int kMuTileX = kBlockX + 2;
constexpr int kMuTileY = kBlockY + 2;
constexpr int kMuTileZ = kBlockZ + 2;
constexpr int kMuTileXY = kMuTileX * kMuTileY;
constexpr int kMuTileSize = kMuTileXY * kMuTileZ;

constexpr double kEAA = -(2.0 / 9.0);
constexpr double kEBB = -(2.0 / 9.0);
constexpr double kEAB = 2.0 / 9.0;
constexpr double kGamma = 0.5;
constexpr double kTimeStepDiffusivity = 0.01; // dt * D; both spacings are 1.

inline void initializeConcentration(std::vector<double>& c, const size_t nx,
                                    const size_t ny, const size_t nz) {
    const size_t volume = nx * ny * nz;
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t linearId = z * (nx * ny) + y * nx + x;
                const double pseudo =
                    (((linearId + 1) * 1299709) % volume) / static_cast<double>(volume);
                c[linearId] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

__device__ __forceinline__ size_t clampCoordinate(const long long coordinate,
                                                  const size_t extent) {
    if (coordinate < 0) {
        return 0;
    }
    if (coordinate >= static_cast<long long>(extent)) {
        return extent - 1;
    }
    return static_cast<size_t>(coordinate);
}

__device__ __forceinline__ double chemicalPotential(const double concentration,
                                                    const double laplacian) {
    // Kept in the same form as the original free-energy derivative.
    return 4.5 * ((concentration + 1.0) * kEAA + (concentration - 1.0) * kEBB -
                  2.0 * concentration * kEAB) +
           3.0 * concentration + concentration * concentration * concentration -
           kGamma * laplacian;
}

// A fused, shared-memory implementation of one complete Cahn-Hilliard step.
// It preserves the original two-stencil formulation while avoiding global
// stores and reloads of mu.  All threads participate in the tile construction,
// including boundary threads whose output falls outside a partial final block.
__global__ void cahnHilliardStep(const double* __restrict__ current,
                                 double* __restrict__ next, const size_t nx,
                                 const size_t ny, const size_t nz) {
    __shared__ double concentrationTile[kCTileSize];
    __shared__ double muTile[kMuTileSize];

    const int threadLinear =
        (static_cast<int>(threadIdx.z) * kBlockY + static_cast<int>(threadIdx.y)) * kBlockX +
        static_cast<int>(threadIdx.x);
    const size_t plane = nx * ny;

    // Load a two-cell clamped halo.  The one-dimensional cooperative mapping
    // has coalesced accesses within each x row and keeps the load balanced.
    for (int load = threadLinear; load < kCTileSize; load += kBlockThreads) {
        const int tileX = load % kCTileX;
        const int tileY = (load / kCTileX) % kCTileY;
        const int tileZ = load / kCTileXY;
        const long long rawX = static_cast<long long>(blockIdx.x) * kBlockX + tileX - kHalo;
        const long long rawY = static_cast<long long>(blockIdx.y) * kBlockY + tileY - kHalo;
        const long long rawZ = static_cast<long long>(blockIdx.z) * kBlockZ + tileZ - kHalo;
        const size_t x = clampCoordinate(rawX, nx);
        const size_t y = clampCoordinate(rawY, ny);
        const size_t z = clampCoordinate(rawZ, nz);
        concentrationTile[load] = current[z * plane + y * nx + x];
    }
    __syncthreads();

    // Form mu for the output block plus a one-cell halo.
    for (int item = threadLinear; item < kMuTileSize; item += kBlockThreads) {
        const int muX = item % kMuTileX;
        const int muY = (item / kMuTileX) % kMuTileY;
        const int muZ = item / kMuTileXY;
        const long long baseX = static_cast<long long>(blockIdx.x) * kBlockX;
        const long long baseY = static_cast<long long>(blockIdx.y) * kBlockY;
        const long long baseZ = static_cast<long long>(blockIdx.z) * kBlockZ;
        // Clamp the mu coordinate before selecting its concentration center.
        // Evaluating mu at a virtual clamped concentration coordinate would
        // use a different stencil than the physical boundary cell.
        const int centerX = static_cast<int>(static_cast<long long>(
                                                  clampCoordinate(baseX + muX - 1, nx)) -
                                              (baseX - kHalo));
        const int centerY = static_cast<int>(static_cast<long long>(
                                                  clampCoordinate(baseY + muY - 1, ny)) -
                                              (baseY - kHalo));
        const int centerZ = static_cast<int>(static_cast<long long>(
                                                  clampCoordinate(baseZ + muZ - 1, nz)) -
                                              (baseZ - kHalo));
        const int center = centerZ * kCTileXY + centerY * kCTileX + centerX;
        const double value = concentrationTile[center];
        const double cxx = concentrationTile[center + 1] + concentrationTile[center - 1] -
                           2.0 * value;
        const double cyy = concentrationTile[center + kCTileX] +
                           concentrationTile[center - kCTileX] -
                           2.0 * value;
        const double czz = concentrationTile[center + kCTileXY] +
                           concentrationTile[center - kCTileXY] -
                           2.0 * value;
        muTile[item] = chemicalPotential(value, cxx + cyy + czz);
    }
    __syncthreads();

    const size_t x = static_cast<size_t>(blockIdx.x) * kBlockX + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * kBlockY + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * kBlockZ + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const int muCenter = (static_cast<int>(threadIdx.z) + 1) * kMuTileXY +
                             (static_cast<int>(threadIdx.y) + 1) * kMuTileX +
                             static_cast<int>(threadIdx.x) + 1;
        const double muValue = muTile[muCenter];
        const double mxx = muTile[muCenter + 1] + muTile[muCenter - 1] - 2.0 * muValue;
        const double myy = muTile[muCenter + kMuTileX] + muTile[muCenter - kMuTileX] -
                           2.0 * muValue;
        const double mzz = muTile[muCenter + kMuTileXY] + muTile[muCenter - kMuTileXY] -
                           2.0 * muValue;
        const int concentrationCenter =
            (static_cast<int>(threadIdx.z) + kHalo) * kCTileXY +
            (static_cast<int>(threadIdx.y) + kHalo) * kCTileX +
            static_cast<int>(threadIdx.x) + kHalo;
        next[z * plane + y * nx + x] =
            concentrationTile[concentrationCenter] + kTimeStepDiffusivity * (mxx + myy + mzz);
    }
}

#define CUDA_CHECK(call)                                                                      \
    do {                                                                                      \
        const cudaError_t cudaStatus = (call);                                                \
        if (cudaStatus != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\\n", __FILE__, __LINE__,          \
                         cudaGetErrorString(cudaStatus));                                     \
            return EXIT_FAILURE;                                                              \
        }                                                                                     \
    } while (false)

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
            return EXIT_SUCCESS;
        } else {
            std::printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (ny == 0) {
        ny = nx;
    }
    if (nz == 0) {
        nz = nx;
    }
    if (nx == 0 || ny == 0 || nz == 0 || nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions must form a non-empty addressable volume\n");
        return EXIT_FAILURE;
    }

    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    const size_t blocksX = (nx + kBlockX - 1) / kBlockX;
    const size_t blocksY = (ny + kBlockY - 1) / kBlockY;
    const size_t blocksZ = (nz + kBlockZ - 1) / kBlockZ;
    if (blocksX > std::numeric_limits<unsigned int>::max() ||
        blocksY > std::numeric_limits<unsigned int>::max() ||
        blocksZ > std::numeric_limits<unsigned int>::max()) {
        std::fprintf(stderr, "Grid dimensions exceed CUDA launch limits\n");
        return EXIT_FAILURE;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<double> concentration(gridSize);
    std::printf("Initializing concentration field...\n");
    initializeConcentration(concentration, nx, ny, nz);

    double* deviceCurrent = nullptr;
    double* deviceNext = nullptr;
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc(&deviceCurrent, bytes));
    CUDA_CHECK(cudaMalloc(&deviceNext, bytes));
    CUDA_CHECK(cudaMemcpy(deviceCurrent, concentration.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFuncSetCacheConfig(cahnHilliardStep, cudaFuncCachePreferShared));

    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid(static_cast<unsigned int>(blocksX), static_cast<unsigned int>(blocksY),
                    static_cast<unsigned int>(blocksZ));
    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    for (int t = 0; t < iterations; ++t) {
        cahnHilliardStep<<<grid, block>>>(deviceCurrent, deviceNext, nx, ny, nz);
        std::swap(deviceCurrent, deviceNext);
    }
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));
    CUDA_CHECK(cudaGetLastError());

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    std::printf("Computation time: %.3f ms\n", elapsedMilliseconds);

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1.0e6;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    CUDA_CHECK(cudaMemcpy(concentration.data(), deviceCurrent, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));
    CUDA_CHECK(cudaFree(deviceCurrent));
    CUDA_CHECK(cudaFree(deviceNext));

    if (printResults) {
        print_results(concentration, "Concentration");
    }

    if (validate) {
        std::printf("Validating result...\n");
        if (validateResult(concentration, nx, ny, nz)) {
            std::printf("Validation: PASSED\n");
            return EXIT_SUCCESS;
        }
        std::printf("Validation: FAILED\n");
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
