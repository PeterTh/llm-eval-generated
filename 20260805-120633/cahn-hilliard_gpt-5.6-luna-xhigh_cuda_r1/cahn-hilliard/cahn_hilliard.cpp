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

constexpr unsigned int BLOCK_X = 8;
constexpr unsigned int BLOCK_Y = 8;
constexpr unsigned int BLOCK_Z = 4;
constexpr unsigned int BLOCK_THREADS = BLOCK_X * BLOCK_Y * BLOCK_Z;
constexpr unsigned int TILE_X = BLOCK_X + 2;
constexpr unsigned int TILE_Y = BLOCK_Y + 2;
constexpr unsigned int TILE_Z = BLOCK_Z + 2;
constexpr unsigned int TILE_ELEMENTS = TILE_X * TILE_Y * TILE_Z;

// 3D index calculation. The host and device use the same row-major layout.
__host__ __device__ __forceinline__ constexpr size_t idx3(const size_t x, const size_t y,
                                                            const size_t z, const size_t nx,
                                                            const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void checkCuda(const cudaError_t status, const char* expression, const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, expression,
                     cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression, __FILE__, __LINE__)

__device__ __forceinline__ size_t clampTileCoordinate(const size_t origin,
                                                       const unsigned int local,
                                                       const size_t extent) {
    size_t coordinate;
    if (local == 0) {
        coordinate = (origin == 0) ? 0 : origin - 1;
    } else {
        coordinate = origin + static_cast<size_t>(local) - 1;
    }
    return (coordinate < extent) ? coordinate : extent - 1;
}

__device__ __forceinline__ size_t tileIndex(const unsigned int x, const unsigned int y,
                                            const unsigned int z) {
    return (static_cast<size_t>(z) * TILE_Y + y) * TILE_X + x;
}

// Load one clamped 3D stencil tile. The two-cell halo is reused by every
// output in the block, reducing the seven global reads per stencil to roughly
// 2.34 global reads per output (including the halo).
__device__ __forceinline__ void loadStencilTile(double* __restrict__ tile,
                                                const double* __restrict__ field,
                                                const size_t nx, const size_t ny,
                                                const size_t nz) {
    const size_t originX = static_cast<size_t>(blockIdx.x) * BLOCK_X;
    const size_t originY = static_cast<size_t>(blockIdx.y) * BLOCK_Y;
    const size_t originZ = static_cast<size_t>(blockIdx.z) * BLOCK_Z;
    const unsigned int threadId =
        (static_cast<unsigned int>(threadIdx.z) * BLOCK_Y + threadIdx.y) * BLOCK_X + threadIdx.x;

    for (unsigned int linear = threadId; linear < TILE_ELEMENTS; linear += BLOCK_THREADS) {
        unsigned int remainder = linear;
        const unsigned int localX = remainder % TILE_X;
        remainder /= TILE_X;
        const unsigned int localY = remainder % TILE_Y;
        const unsigned int localZ = remainder / TILE_Y;

        const size_t x = clampTileCoordinate(originX, localX, nx);
        const size_t y = clampTileCoordinate(originY, localY, ny);
        const size_t z = clampTileCoordinate(originZ, localZ, nz);
        tile[tileIndex(localX, localY, localZ)] = field[idx3(x, y, z, nx, ny)];
    }
    __syncthreads();
}

__device__ __forceinline__ double tileLaplacian(const double* __restrict__ tile,
                                                const unsigned int localX,
                                                const unsigned int localY,
                                                const unsigned int localZ,
                                                const double dxSquared,
                                                const double dySquared,
                                                const double dzSquared) {
    const unsigned int x = localX + 1;
    const unsigned int y = localY + 1;
    const unsigned int z = localZ + 1;
    const double center = tile[tileIndex(x, y, z)];

    // Keep the operation grouping of the scalar implementation: each second
    // derivative is formed before the three directions are added.
    const double cxx = (tile[tileIndex(x + 1, y, z)] + tile[tileIndex(x - 1, y, z)] -
                        2.0 * center) /
                       dxSquared;
    const double cyy = (tile[tileIndex(x, y + 1, z)] + tile[tileIndex(x, y - 1, z)] -
                        2.0 * center) /
                       dySquared;
    const double czz = (tile[tileIndex(x, y, z + 1)] + tile[tileIndex(x, y, z - 1)] -
                        2.0 * center) /
                       dzSquared;
    return cxx + cyy + czz;
}

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void initializeConcentrationKernel(double* __restrict__ c, const size_t n, const size_t volume) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t linearId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linearId < n; linearId += stride) {
        // Match the original size_t arithmetic and deterministic initialization exactly.
        const size_t pseudoInteger = ((linearId + 1) * static_cast<size_t>(1299709)) % volume;
        const double pseudo = pseudoInteger / static_cast<double>(volume);
        c[linearId] = -1.0 + 2.0 * pseudo;
    }
}

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const double dxSquared, const double dySquared,
                                    const double dzSquared, const double gamma,
                                    const double eAA, const double eBB, const double eAB) {
    extern __shared__ double tile[];
    loadStencilTile(tile, c, nx, ny, nz);

    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const unsigned int localX = threadIdx.x;
        const unsigned int localY = threadIdx.y;
        const unsigned int localZ = threadIdx.z;
        const size_t index = idx3(x, y, z, nx, ny);
        const double cv = tile[tileIndex(localX + 1, localY + 1, localZ + 1)];
        const double laplacian =
            tileLaplacian(tile, localX, localY, localZ, dxSquared, dySquared, dzSquared);

        mu[index] = 4.5 * ((cv + 1.0) * eAA + (cv - 1.0) * eBB - 2.0 * cv * eAB) +
                    3.0 * cv + cv * cv * cv - gamma * laplacian;
    }
}

__global__ __launch_bounds__(BLOCK_THREADS, 2)
void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                              const double* __restrict__ mu, const size_t nx, const size_t ny,
                              const size_t nz, const double D, const double dt,
                              const double dxSquared, const double dySquared,
                              const double dzSquared) {
    extern __shared__ double tile[];
    loadStencilTile(tile, mu, nx, ny, nz);

    const size_t x = static_cast<size_t>(blockIdx.x) * BLOCK_X + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * BLOCK_Y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * BLOCK_Z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const double laplacian =
            tileLaplacian(tile, threadIdx.x, threadIdx.y, threadIdx.z, dxSquared, dySquared,
                          dzSquared);
        const size_t index = idx3(x, y, z, nx, ny);
        cnew[index] = cold[index] + dt * D * laplacian;
    }
}

size_t launchBlocksFor(const size_t elements, const cudaDeviceProp& properties) {
    constexpr size_t threads = 256;
    const size_t requiredBlocks = (elements + threads - 1) / threads;
    const size_t residentBlocks = static_cast<size_t>(properties.multiProcessorCount) * 8;
    return std::max<size_t>(1, std::min(requiredBlocks, residentBlocks));
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            std::printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    std::printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
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

    // Parse command line arguments
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

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;

    if (nx == 0 || ny == 0 || nz == 0 || iterations < 0 ||
        nx > std::numeric_limits<size_t>::max() / ny ||
        nx * ny > std::numeric_limits<size_t>::max() / nz) {
        std::fprintf(stderr, "Grid dimensions and iterations must describe a non-empty, valid grid.\n");
        return 1;
    }

    std::printf("Cahn-Hilliard Phase Separation Benchmark\n");
    std::printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    std::printf("Time steps: %d\n", iterations);
    std::printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    const size_t gridSize = nx * ny * nz;
    if (gridSize > std::numeric_limits<size_t>::max() / sizeof(double)) {
        std::fprintf(stderr, "Grid is too large to allocate.\n");
        return 1;
    }
    const size_t bytes = gridSize * sizeof(double);

    int device = 0;
    cudaDeviceProp properties{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    std::printf("Initializing concentration field...\n");
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), bytes));

    const size_t initializationBlocks = launchBlocksFor(gridSize, properties);
    initializeConcentrationKernel<<<static_cast<unsigned int>(initializationBlocks), 256>>>(
        d_cold, gridSize, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const dim3 block(BLOCK_X, BLOCK_Y, BLOCK_Z);
    const dim3 grid(static_cast<unsigned int>((nx + BLOCK_X - 1) / BLOCK_X),
                    static_cast<unsigned int>((ny + BLOCK_Y - 1) / BLOCK_Y),
                    static_cast<unsigned int>((nz + BLOCK_Z - 1) / BLOCK_Z));
    const size_t sharedBytes = static_cast<size_t>(TILE_ELEMENTS) * sizeof(double);
    const double dxSquared = dx * dx;
    const double dySquared = dy * dy;
    const double dzSquared = dz * dz;

    std::printf("Running Cahn-Hilliard simulation...\n");

    cudaEvent_t startEvent = nullptr;
    cudaEvent_t stopEvent = nullptr;
    CUDA_CHECK(cudaEventCreate(&startEvent));
    CUDA_CHECK(cudaEventCreate(&stopEvent));
    CUDA_CHECK(cudaEventRecord(startEvent));

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotentialKernel<<<grid, block, sharedBytes>>>(
            d_cold, d_mu, nx, ny, nz, dxSquared, dySquared, dzSquared, gamma, e_AA, e_BB, e_AB);
        CUDA_CHECK(cudaGetLastError());

        cahnHilliardUpdateKernel<<<grid, block, sharedBytes>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dxSquared, dySquared, dzSquared);
        CUDA_CHECK(cudaGetLastError());

        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaEventRecord(stopEvent));
    CUDA_CHECK(cudaEventSynchronize(stopEvent));

    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, startEvent, stopEvent));
    CUDA_CHECK(cudaEventDestroy(startEvent));
    CUDA_CHECK(cudaEventDestroy(stopEvent));

    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_cold));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_mu));

    // Retain the original millisecond-oriented report while using the
    // unrounded event duration for the throughput calculation.
    const long durationMilliseconds =
        std::max<long>(1, static_cast<long>(std::llround(elapsedMilliseconds)));
    std::printf("Computation time: %ld ms\n", durationMilliseconds);

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double seconds = static_cast<double>(elapsedMilliseconds) / 1000.0;
    const double mcups = (seconds > 0.0) ? cellUpdates / seconds / 1e6 : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        std::printf("Validating result...\n");
        const bool valid = validateResult(cold, nx, ny, nz);

        if (valid) {
            std::printf("Validation: PASSED\n");
            return 0;
        }
        std::printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
