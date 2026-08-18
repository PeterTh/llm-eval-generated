#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr int blockX = 8;
constexpr int blockY = 8;
constexpr int blockZ = 8;
constexpr int threadsPerBlock = blockX * blockY * blockZ;

// The fused stencil needs c two cells beyond the output block and mu one cell
// beyond it.  Keeping both tiles in shared memory avoids storing mu to global
// memory between the two original stencil stages.
constexpr int cTileX = blockX + 4;
constexpr int cTileY = blockY + 4;
constexpr int cTileZ = blockZ + 4;
constexpr int cTileElements = cTileX * cTileY * cTileZ;
constexpr int muTileX = blockX + 2;
constexpr int muTileY = blockY + 2;
constexpr int muTileZ = blockZ + 2;
constexpr int muTileElements = muTileX * muTileY * muTileZ;

inline void checkCuda(const cudaError_t error, const char* expression, const char* file, const int line) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d while executing %s: %s\n", file, line, expression,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ int cTileIndex(const int x, const int y, const int z) {
    return (z * cTileY + y) * cTileX + x;
}

__device__ __forceinline__ int muTileIndex(const int x, const int y, const int z) {
    return (z * muTileY + y) * muTileX + x;
}

__device__ __forceinline__ size_t clampCoordinate(const long long coordinate, const size_t extent) {
    return coordinate < 0 ? 0 : (coordinate >= static_cast<long long>(extent) ? extent - 1
                                                                                : static_cast<size_t>(coordinate));
}

__global__ void initializeConcentration(double* const concentration, const size_t volume) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < volume) {
        const size_t pseudoInteger = ((index + 1) * static_cast<size_t>(1299709)) % volume;
        const double pseudo = static_cast<double>(pseudoInteger) / static_cast<double>(volume);
        concentration[index] = -1.0 + 2.0 * pseudo;
    }
}

// One block advances an 8 x 8 x 8 output tile.  The first synchronization
// makes the c halo available for the mu stencil; the second makes the mu halo
// available for the concentration update stencil.
__global__ __launch_bounds__(threadsPerBlock, 2)
void cahnHilliardStep(double* __restrict__ next, const double* __restrict__ current, const size_t nx,
                      const size_t ny, const size_t nz) {
    __shared__ double cTile[cTileElements];
    __shared__ double muTile[muTileElements];

    const int thread = (threadIdx.z * blockY + threadIdx.y) * blockX + threadIdx.x;
    const long long blockBaseX = static_cast<long long>(blockIdx.x) * blockX;
    const long long blockBaseY = static_cast<long long>(blockIdx.y) * blockY;
    const long long blockBaseZ = static_cast<long long>(blockIdx.z) * blockZ;

    // Cooperative, x-major loads give coalesced global reads for the shared c
    // tile.  Coordinates outside the domain are clamped exactly as before.
    for (int element = thread; element < cTileElements; element += threadsPerBlock) {
        const int localX = element % cTileX;
        const int localY = (element / cTileX) % cTileY;
        const int localZ = element / (cTileX * cTileY);
        const size_t x = clampCoordinate(blockBaseX + localX - 2, nx);
        const size_t y = clampCoordinate(blockBaseY + localY - 2, ny);
        const size_t z = clampCoordinate(blockBaseZ + localZ - 2, nz);
        cTile[element] = current[(z * ny + y) * nx + x];
    }
    __syncthreads();

    // Calculate mu for the output tile and its one-cell halo.  The arithmetic
    // intentionally follows the original expression order.
    for (int element = thread; element < muTileElements; element += threadsPerBlock) {
        const int localX = element % muTileX;
        const int localY = (element / muTileX) % muTileY;
        const int localZ = element / (muTileX * muTileY);
        // A mu halo coordinate is a *clamped mu value*, not a chemical
        // potential evaluated at a fictitious coordinate outside the domain.
        // Map its center to the physical cell before taking its c stencil.
        int cIndex;
        if (localX > 0 && localX <= blockX && localY > 0 && localY <= blockY && localZ > 0 && localZ <= blockZ &&
            blockBaseX + localX - 1 < static_cast<long long>(nx) &&
            blockBaseY + localY - 1 < static_cast<long long>(ny) &&
            blockBaseZ + localZ - 1 < static_cast<long long>(nz)) {
            // The common case is a real output point, whose c center has a
            // fixed offset in the tile.  This skips boundary-coordinate work
            // for the 8 x 8 x 8 core of every block.
            cIndex = cTileIndex(localX + 1, localY + 1, localZ + 1);
        } else {
            const size_t physicalX = clampCoordinate(blockBaseX + localX - 1, nx);
            const size_t physicalY = clampCoordinate(blockBaseY + localY - 1, ny);
            const size_t physicalZ = clampCoordinate(blockBaseZ + localZ - 1, nz);
            const int cX = static_cast<int>(static_cast<long long>(physicalX) - blockBaseX + 2);
            const int cY = static_cast<int>(static_cast<long long>(physicalY) - blockBaseY + 2);
            const int cZ = static_cast<int>(static_cast<long long>(physicalZ) - blockBaseZ + 2);
            cIndex = cTileIndex(cX, cY, cZ);
        }
        const double c = cTile[cIndex];
        const double laplacianX = cTile[cIndex + 1] + cTile[cIndex - 1] - 2.0 * c;
        const double laplacianY = cTile[cIndex + cTileX] + cTile[cIndex - cTileX] - 2.0 * c;
        const double laplacianZ = cTile[cIndex + cTileX * cTileY] + cTile[cIndex - cTileX * cTileY] - 2.0 * c;
        const double laplacian = laplacianX + laplacianY + laplacianZ;

        constexpr double eAA = -(2.0 / 9.0);
        constexpr double eBB = -(2.0 / 9.0);
        constexpr double eAB = 2.0 / 9.0;
        muTile[element] = 4.5 * ((c + 1.0) * eAA + (c - 1.0) * eBB - 2.0 * c * eAB) + 3.0 * c + c * c * c -
                          0.5 * laplacian;
    }
    __syncthreads();

    const size_t x = static_cast<size_t>(blockBaseX + threadIdx.x);
    const size_t y = static_cast<size_t>(blockBaseY + threadIdx.y);
    const size_t z = static_cast<size_t>(blockBaseZ + threadIdx.z);
    if (x < nx && y < ny && z < nz) {
        const int mIndex = muTileIndex(threadIdx.x + 1, threadIdx.y + 1, threadIdx.z + 1);
        const double mu = muTile[mIndex];
        const double laplacianX = muTile[mIndex + 1] + muTile[mIndex - 1] - 2.0 * mu;
        const double laplacianY = muTile[mIndex + muTileX] + muTile[mIndex - muTileX] - 2.0 * mu;
        const double laplacianZ = muTile[mIndex + muTileX * muTileY] + muTile[mIndex - muTileX * muTileY] - 2.0 * mu;
        const double laplacian = laplacianX + laplacianY + laplacianZ;
        next[(z * ny + y) * nx + x] = cTile[cTileIndex(threadIdx.x + 2, threadIdx.y + 2, threadIdx.z + 2)] +
                                       0.01 * laplacian;
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

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const size_t gridSize = nx * ny * nz;
    if (gridSize == 0) {
        std::fprintf(stderr, "Grid dimensions must be positive.\n");
        return 1;
    }

    std::printf("Initializing concentration field...\n");
    double* current = nullptr;
    double* next = nullptr;
    CUDA_CHECK(cudaMalloc(&current, gridSize * sizeof(*current)));
    CUDA_CHECK(cudaMalloc(&next, gridSize * sizeof(*next)));

    constexpr int initializationThreads = 256;
    const dim3 initializationGrid(static_cast<unsigned int>((gridSize + initializationThreads - 1) / initializationThreads));
    initializeConcentration<<<initializationGrid, initializationThreads>>>(current, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 block(blockX, blockY, blockZ);
    const dim3 grid(static_cast<unsigned int>((nx + blockX - 1) / blockX),
                    static_cast<unsigned int>((ny + blockY - 1) / blockY),
                    static_cast<unsigned int>((nz + blockZ - 1) / blockZ));

    cudaEvent_t startEvent{};
    cudaEvent_t stopEvent{};
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));

    std::printf("Running Cahn-Hilliard simulation...\n");
    CUDA_CHECK(cudaEventRecord(startEvent));
    for (int timestep = 0; timestep < iterations; ++timestep) {
        cahnHilliardStep<<<grid, block>>>(next, current, nx, ny, nz);
        std::swap(current, next);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0F;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    std::printf("Computation time: %.3f ms\n", static_cast<double>(elapsedMilliseconds));
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedMilliseconds > 0.0F ? cellUpdates / (static_cast<double>(elapsedMilliseconds) * 1.0e3) : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults || validate) {
        std::vector<double> concentration(gridSize);
        CUDA_CHECK(cudaMemcpy(concentration.data(), current, gridSize * sizeof(*current), cudaMemcpyDeviceToHost));

        if (printResults) {
            print_results(concentration, "Concentration");
        }
        if (validate) {
            std::printf("Validating result...\n");
            if (validateResult(concentration)) {
                std::printf("Validation: PASSED\n");
            } else {
                std::printf("Validation: FAILED\n");
                CUDA_CHECK(cudaFree(current));
                CUDA_CHECK(cudaFree(next));
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(current));
    CUDA_CHECK(cudaFree(next));
    return 0;
}
