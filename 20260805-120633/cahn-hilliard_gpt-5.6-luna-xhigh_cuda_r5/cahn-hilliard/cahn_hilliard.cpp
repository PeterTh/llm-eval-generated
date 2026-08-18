#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int kBlockX = 32;
constexpr unsigned int kBlockY = 4;
constexpr unsigned int kBlockZ = 2;
constexpr unsigned int kThreadsPerBlock = kBlockX * kBlockY * kBlockZ;
constexpr unsigned int kMaxGridZ = 65535;

inline void checkCuda(const cudaError_t status, const char* const operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

// 3D index calculation. The linear layout is identical to the original CPU code,
// with X as the contiguous dimension.
__host__ __device__ inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z,
                                                  const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

__device__ __forceinline__ size_t clampCoordinate(const long long coordinate, const size_t extent) {
    if (coordinate < 0) {
        return 0;
    }
    const size_t unsignedCoordinate = static_cast<size_t>(coordinate);
    return unsignedCoordinate < extent ? unsignedCoordinate : extent - 1;
}

__device__ __forceinline__ size_t tileIndex(const size_t x, const size_t y, const size_t z,
                                             const size_t tileX, const size_t tileY) {
    return (z * tileY + y) * tileX + x;
}

// Load a clamped 3D tile, including the one-cell halo needed by a 7-point
// Laplacian. Every thread participates in the load so that the tile is filled
// with coalesced global accesses along X.
__device__ __forceinline__ void loadStencilTile(const double* __restrict__ source,
                                                 double* __restrict__ tile,
                                                 const size_t nx, const size_t ny, const size_t nz,
                                                 const size_t baseX, const size_t baseY,
                                                 const size_t baseZ) {
    const size_t tileX = static_cast<size_t>(blockDim.x) + 2;
    const size_t tileY = static_cast<size_t>(blockDim.y) + 2;
    const size_t tileZ = static_cast<size_t>(blockDim.z) + 2;
    const size_t tileVolume = tileX * tileY * tileZ;
    const size_t threadId =
        (static_cast<size_t>(threadIdx.z) * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const size_t threadCount = static_cast<size_t>(blockDim.x) * blockDim.y * blockDim.z;

    for (size_t local = threadId; local < tileVolume; local += threadCount) {
        const size_t localX = local % tileX;
        const size_t localY = (local / tileX) % tileY;
        const size_t localZ = local / (tileX * tileY);

        const long long globalX = static_cast<long long>(baseX) + static_cast<long long>(localX) - 1;
        const long long globalY = static_cast<long long>(baseY) + static_cast<long long>(localY) - 1;
        const long long globalZ = static_cast<long long>(baseZ) + static_cast<long long>(localZ) - 1;

        const size_t x = clampCoordinate(globalX, nx);
        const size_t y = clampCoordinate(globalY, ny);
        const size_t z = clampCoordinate(globalZ, nz);
        tile[local] = source[idx3(x, y, z, nx, ny)];
    }
}

__global__ __launch_bounds__(kThreadsPerBlock, 2)
void initializeConcentrationKernel(double* __restrict__ c, const size_t gridSize, const size_t volume) {
    for (size_t linearId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linearId < gridSize;
         linearId += static_cast<size_t>(gridDim.x) * blockDim.x) {
        // Keep the integer arithmetic and deterministic sequence of the CPU
        // implementation exactly unchanged.
        const size_t pseudo = (((linearId + 1) * static_cast<size_t>(1299709)) % volume);
        c[linearId] = -1.0 + 2.0 * (pseudo / static_cast<double>(volume));
    }
}

__global__ __launch_bounds__(kThreadsPerBlock, 2)
void computeChemicalPotentialKernel(const double* __restrict__ c, double* __restrict__ mu,
                                    const size_t nx, const size_t ny, const size_t nz,
                                    const double dx, const double dy, const double dz,
                                    const double gamma, const double e_AA, const double e_BB,
                                    const double e_AB) {
    extern __shared__ double tile[];

    const size_t tileX = static_cast<size_t>(blockDim.x) + 2;
    const size_t tileY = static_cast<size_t>(blockDim.y) + 2;
    const size_t tileXY = tileX * tileY;
    const size_t centerX = static_cast<size_t>(threadIdx.x) + 1;
    const size_t centerY = static_cast<size_t>(threadIdx.y) + 1;
    const size_t centerZ = static_cast<size_t>(threadIdx.z) + 1;
    const size_t center = tileIndex(centerX, centerY, centerZ, tileX, tileY);

    const size_t baseX = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * blockDim.y;
    const size_t zStride = static_cast<size_t>(gridDim.z) * blockDim.z;

    for (size_t baseZ = static_cast<size_t>(blockIdx.z) * blockDim.z;
         baseZ < nz;
         baseZ += zStride) {
        loadStencilTile(c, tile, nx, ny, nz, baseX, baseY, baseZ);
        __syncthreads();

        const size_t x = baseX + threadIdx.x;
        const size_t y = baseY + threadIdx.y;
        const size_t z = baseZ + threadIdx.z;
        if (x < nx && y < ny && z < nz) {
            const size_t idx = idx3(x, y, z, nx, ny);
            const double cv = tile[center];

            const double cxx = (tile[center + 1] + tile[center - 1] - 2.0 * cv) / (dx * dx);
            const double cyy = (tile[center + tileX] + tile[center - tileX] - 2.0 * cv) / (dy * dy);
            const double czz = (tile[center + tileXY] + tile[center - tileXY] - 2.0 * cv) / (dz * dz);
            const double laplacian = cxx + cyy + czz;

            mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                    + 3.0 * cv + cv * cv * cv
                    - gamma * laplacian;
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kThreadsPerBlock, 2)
void cahnHilliardUpdateKernel(double* __restrict__ cnew, const double* __restrict__ cold,
                              const double* __restrict__ mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double D, const double dt,
                              const double dx, const double dy, const double dz) {
    extern __shared__ double tile[];

    const size_t tileX = static_cast<size_t>(blockDim.x) + 2;
    const size_t tileY = static_cast<size_t>(blockDim.y) + 2;
    const size_t tileXY = tileX * tileY;
    const size_t centerX = static_cast<size_t>(threadIdx.x) + 1;
    const size_t centerY = static_cast<size_t>(threadIdx.y) + 1;
    const size_t centerZ = static_cast<size_t>(threadIdx.z) + 1;
    const size_t center = tileIndex(centerX, centerY, centerZ, tileX, tileY);

    const size_t baseX = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * blockDim.y;
    const size_t zStride = static_cast<size_t>(gridDim.z) * blockDim.z;

    for (size_t baseZ = static_cast<size_t>(blockIdx.z) * blockDim.z;
         baseZ < nz;
         baseZ += zStride) {
        loadStencilTile(mu, tile, nx, ny, nz, baseX, baseY, baseZ);
        __syncthreads();

        const size_t x = baseX + threadIdx.x;
        const size_t y = baseY + threadIdx.y;
        const size_t z = baseZ + threadIdx.z;
        if (x < nx && y < ny && z < nz) {
            const size_t idx = idx3(x, y, z, nx, ny);
            const double laplacian =
                (tile[center + 1] + tile[center - 1] - 2.0 * tile[center]) / (dx * dx)
                + (tile[center + tileX] + tile[center - tileX] - 2.0 * tile[center]) / (dy * dy)
                + (tile[center + tileXY] + tile[center - tileXY] - 2.0 * tile[center]) / (dz * dz);
            cnew[idx] = cold[idx] + dt * D * laplacian;
        }
        __syncthreads();
    }
}

dim3 makeStencilGrid(const size_t nx, const size_t ny, const size_t nz) {
    const size_t blocksX = (nx + kBlockX - 1) / kBlockX;
    const size_t blocksY = (ny + kBlockY - 1) / kBlockY;
    const size_t blocksZ = (nz + kBlockZ - 1) / kBlockZ;
    const size_t maxGridX = static_cast<size_t>(std::numeric_limits<unsigned int>::max());

    if (blocksX > maxGridX || blocksY > maxGridX) {
        fprintf(stderr, "Grid dimensions are too large for a CUDA launch\n");
        std::exit(EXIT_FAILURE);
    }

    return dim3(static_cast<unsigned int>(blocksX),
                static_cast<unsigned int>(blocksY),
                static_cast<unsigned int>(std::min(blocksZ, static_cast<size_t>(kMaxGridZ))));
}

constexpr size_t stencilSharedBytes() {
    return static_cast<size_t>(kBlockX + 2) * (kBlockY + 2) * (kBlockZ + 2) * sizeof(double);
}

void initializeConcentration(double* const d_c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t volume = nx * ny * nz;
    const size_t initializationThreads = 256;
    const size_t requestedBlocks = (volume + initializationThreads - 1) / initializationThreads;
    const size_t blocks = std::min(requestedBlocks, static_cast<size_t>(2147483647));

    initializeConcentrationKernel<<<dim3(static_cast<unsigned int>(blocks)),
                                    dim3(static_cast<unsigned int>(initializationThreads))>>>
        (d_c, volume, volume);
    CUDA_CHECK(cudaGetLastError());
}

void computeChemicalPotential(double* const d_c, double* const d_mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB,
                              const double e_AB) {
    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid = makeStencilGrid(nx, ny, nz);
    computeChemicalPotentialKernel<<<grid, block, stencilSharedBytes()>>>
        (d_c, d_mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

void cahnHilliardUpdate(double* const d_cnew, const double* const d_cold,
                        const double* const d_mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt,
                        const double dx, const double dy, const double dz) {
    const dim3 block(kBlockX, kBlockY, kBlockZ);
    const dim3 grid = makeStencilGrid(nx, ny, nz);
    cahnHilliardUpdateKernel<<<grid, block, stencilSharedBytes()>>>
        (d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<double>& c) {
    // Check for NaN or Inf.
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }

    // Check if values are in reasonable range for concentration field.
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments.
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-x") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-y") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-z") == 0 && i + 1 < argc) {
            nz = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            iterations = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    if (ny == 0) ny = nx;
    if (nz == 0) nz = nx;
    if (nx == 0 || ny == 0 || nz == 0) {
        fprintf(stderr, "Grid dimensions must be greater than zero\n");
        return 1;
    }

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Physical parameters.
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
    const size_t allocationBytes = gridSize * sizeof(double);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), allocationBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), allocationBytes));

    printf("Initializing concentration field...\n");
    initializeConcentration(d_cold, nx, ny, nz);
    CUDA_CHECK(cudaDeviceSynchronize());

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential(d_cold, d_mu, nx, ny, nz, dx, dy, dz,
                                 gamma, e_AA, e_BB, e_AB);
        cahnHilliardUpdate(d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, dx, dy, dz);
        std::swap(d_cold, d_cnew);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    long durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    if (durationMs < 1) durationMs = 1;

    printf("Computation time: %ld ms\n", durationMs);

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = cellUpdates / (durationMs / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        CUDA_CHECK(cudaMemcpy(result.data(), d_cold, allocationBytes, cudaMemcpyDeviceToHost));
    }

    if (printResults) {
        print_results(result, "Concentration");
    }

    bool valid = true;
    if (validate) {
        printf("Validating result...\n");
        valid = validateResult(result);
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));

    return validate && !valid ? 1 : 0;
}
