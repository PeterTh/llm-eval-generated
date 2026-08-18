#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

// An 8^3 block gives the stencil kernels enough work to amortize the halo
// loads while keeping the shared-memory tile small.
constexpr unsigned int kTileX = 8;
constexpr unsigned int kTileY = 8;
constexpr unsigned int kTileZ = 8;
constexpr unsigned int kBlockThreads = kTileX * kTileY * kTileZ;
constexpr unsigned int kSharedX = kTileX + 2;
constexpr unsigned int kSharedY = kTileY + 2;
constexpr unsigned int kSharedZ = kTileZ + 2;
constexpr unsigned int kSharedElements = kSharedX * kSharedY * kSharedZ;
constexpr unsigned int kLinearBlockThreads = 256;

inline void checkCuda(const cudaError_t status, const char* operation,
                      const char* file, const int line) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s at %s:%d: %s\n", operation,
                     file, line, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation, __FILE__, __LINE__)

struct StencilLaunch {
    dim3 grid;
    dim3 block;
};

constexpr size_t ceilDiv(const size_t numerator, const size_t denominator) {
    return numerator / denominator + (numerator % denominator != 0 ? 1 : 0);
}

StencilLaunch makeStencilLaunch(const size_t nx, const size_t ny, const size_t nz) {
    cudaDeviceProp properties{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const size_t tilesX = ceilDiv(nx, kTileX);
    const size_t tilesY = ceilDiv(ny, kTileY);
    const size_t tilesZ = ceilDiv(nz, kTileZ);
    const size_t maxGridX = static_cast<size_t>(properties.maxGridSize[0]);
    const size_t maxGridY = static_cast<size_t>(properties.maxGridSize[1]);
    const size_t maxGridZ = static_cast<size_t>(properties.maxGridSize[2]);

    // The kernels use grid-stride loops over tiles, so dimensions larger than
    // the device grid limits remain supported.
    const auto gridX = static_cast<unsigned int>(std::min(tilesX, maxGridX));
    const auto gridY = static_cast<unsigned int>(std::min(tilesY, maxGridY));
    const auto gridZ = static_cast<unsigned int>(std::min(tilesZ, maxGridZ));
    return {{gridX, gridY, gridZ}, {kTileX, kTileY, kTileZ}};
}

dim3 makeInitializationGrid(const size_t elementCount) {
    cudaDeviceProp properties{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

    const size_t requestedBlocks = ceilDiv(elementCount, kLinearBlockThreads);
    const size_t maxGridX = static_cast<size_t>(properties.maxGridSize[0]);
    return {static_cast<unsigned int>(std::min(requestedBlocks, maxGridX)), 1, 1};
}

// 3D index calculation. The storage layout is identical to the original
// implementation: x is contiguous, followed by y and then z.
__device__ __forceinline__ size_t idx3(const size_t x, const size_t y,
                                       const size_t z, const size_t nx,
                                       const size_t ny) {
    return z * (nx * ny) + y * nx + x;
}

__device__ __forceinline__ unsigned int sharedIdx(const unsigned int x,
                                                  const unsigned int y,
                                                  const unsigned int z) {
    return z * (kSharedY * kSharedX) + y * kSharedX + x;
}

// Map a shared-memory tile coordinate (including one-cell halo) to the
// clamped global coordinate required by the original boundary conditions.
__device__ __forceinline__ size_t clampedCoordinate(const size_t base,
                                                    const unsigned int halo,
                                                    const size_t extent) {
    size_t coordinate;
    if (halo == 0) {
        coordinate = (base == 0) ? 0 : base - 1;
    } else {
        coordinate = base + static_cast<size_t>(halo - 1);
    }
    return coordinate < extent ? coordinate : extent - 1;
}

__device__ __forceinline__ double laplacianFromTile(
    const double* __restrict__ tile, const unsigned int x,
    const unsigned int y, const unsigned int z, const double dx,
    const double dy, const double dz) {
    const unsigned int center = sharedIdx(x, y, z);
    const double value = tile[center];
    const double cxx = (tile[sharedIdx(x + 1, y, z)] +
                        tile[sharedIdx(x - 1, y, z)] - 2.0 * value) /
                       (dx * dx);
    const double cyy = (tile[sharedIdx(x, y + 1, z)] +
                        tile[sharedIdx(x, y - 1, z)] - 2.0 * value) /
                       (dy * dy);
    const double czz = (tile[sharedIdx(x, y, z + 1)] +
                        tile[sharedIdx(x, y, z - 1)] - 2.0 * value) /
                       (dz * dz);
    return cxx + cyy + czz;
}

__device__ __forceinline__ double chemicalPotentialFromTile(
    const double* __restrict__ tile, const unsigned int x,
    const unsigned int y, const unsigned int z, const double dx,
    const double dy, const double dz, const double gamma, const double e_AA,
    const double e_BB, const double e_AB) {
    const double value = tile[sharedIdx(x, y, z)];
    return 4.5 * ((value + 1.0) * e_AA + (value - 1.0) * e_BB -
                  2.0 * value * e_AB) +
           3.0 * value + value * value * value -
           gamma * laplacianFromTile(tile, x, y, z, dx, dy, dz);
}

// Load one clamped shared-memory tile. Each block covers one tile in x/y and
// advances over z (and, for unusually large domains, x/y) with grid strides.
template <typename KernelBody>
__device__ __forceinline__ void forEachStencilCell(
    const double* __restrict__ input, const size_t nx, const size_t ny,
    const size_t nz, KernelBody body) {
    __shared__ double tile[kSharedElements];

    const size_t tilesX = (nx + kTileX - 1) / kTileX;
    const size_t tilesY = (ny + kTileY - 1) / kTileY;
    const size_t tilesZ = (nz + kTileZ - 1) / kTileZ;

    const unsigned int threadLinear =
        threadIdx.x + kTileX * (threadIdx.y + kTileY * threadIdx.z);

    for (size_t tileZ = blockIdx.z; tileZ < tilesZ; tileZ += gridDim.z) {
        for (size_t tileY = blockIdx.y; tileY < tilesY; tileY += gridDim.y) {
            for (size_t tileX = blockIdx.x; tileX < tilesX;
                 tileX += gridDim.x) {
                const size_t baseX = tileX * kTileX;
                const size_t baseY = tileY * kTileY;
                const size_t baseZ = tileZ * kTileZ;

                for (unsigned int linear = threadLinear;
                     linear < kSharedElements; linear += kBlockThreads) {
                    const unsigned int sx = linear % kSharedX;
                    const unsigned int sy = (linear / kSharedX) % kSharedY;
                    const unsigned int sz = linear / (kSharedX * kSharedY);
                    const size_t gx = clampedCoordinate(baseX, sx, nx);
                    const size_t gy = clampedCoordinate(baseY, sy, ny);
                    const size_t gz = clampedCoordinate(baseZ, sz, nz);
                    tile[linear] = input[idx3(gx, gy, gz, nx, ny)];
                }
                __syncthreads();

                const size_t x = baseX + threadIdx.x;
                const size_t y = baseY + threadIdx.y;
                const size_t z = baseZ + threadIdx.z;
                if (x < nx && y < ny && z < nz) {
                    body(tile, x, y, z, threadIdx.x + 1, threadIdx.y + 1,
                         threadIdx.z + 1);
                }
                __syncthreads();
            }
        }
    }
}

__global__ void initializeConcentrationKernel(double* __restrict__ c,
                                               const size_t elementCount,
                                               const size_t volume) {
    const size_t first = static_cast<size_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t linearId = first; linearId < elementCount;
         linearId += stride) {
        // Generate the same deterministic pseudo-random value in [-1, 1] as
        // the scalar implementation.
        const double pseudo =
            (((linearId + 1) * static_cast<size_t>(1299709)) % volume) /
            static_cast<double>(volume);
        c[linearId] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void computeChemicalPotentialKernel(
    const double* __restrict__ c, double* __restrict__ mu, const size_t nx,
    const size_t ny, const size_t nz, const double dx, const double dy,
    const double dz, const double gamma, const double e_AA, const double e_BB,
    const double e_AB) {
    forEachStencilCell(c, nx, ny, nz,
                       [=] __device__(const double* tile, const size_t x,
                                      const size_t y, const size_t z,
                                      const unsigned int localX,
                                      const unsigned int localY,
                                      const unsigned int localZ) {
                           mu[idx3(x, y, z, nx, ny)] =
                               chemicalPotentialFromTile(
                                   tile, localX, localY, localZ, dx, dy, dz,
                                   gamma, e_AA, e_BB, e_AB);
                       });
}

__global__ void cahnHilliardUpdateKernel(
    double* __restrict__ cnew, const double* __restrict__ cold,
    const double* __restrict__ mu, const size_t nx, const size_t ny,
    const size_t nz, const double D, const double dt, const double dx,
    const double dy, const double dz) {
    forEachStencilCell(mu, nx, ny, nz,
                       [=] __device__(const double* tile, const size_t x,
                                      const size_t y, const size_t z,
                                      const unsigned int localX,
                                      const unsigned int localY,
                                      const unsigned int localZ) {
                           const size_t index = idx3(x, y, z, nx, ny);
                           cnew[index] =
                               cold[index] +
                               dt * D * laplacianFromTile(
                                             tile, localX, localY, localZ, dx,
                                             dy, dz);
                       });
}

void initializeConcentration(double* deviceC, const size_t nx, const size_t ny,
                             const size_t nz) {
    const size_t volume = nx * ny * nz;
    const dim3 block(kLinearBlockThreads, 1, 1);
    const dim3 grid = makeInitializationGrid(volume);
    initializeConcentrationKernel<<<grid, block>>>(deviceC, volume, volume);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeChemicalPotential(const double* deviceC, double* deviceMu,
                              const size_t nx, const size_t ny,
                              const size_t nz, const double dx, const double dy,
                              const double dz, const double gamma,
                              const double e_AA, const double e_BB,
                              const double e_AB, const StencilLaunch& launch) {
    computeChemicalPotentialKernel<<<launch.grid, launch.block>>>(
        deviceC, deviceMu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
    CUDA_CHECK(cudaGetLastError());
}

void cahnHilliardUpdate(double* deviceCnew, const double* deviceCold,
                        const double* deviceMu, const size_t nx,
                        const size_t ny, const size_t nz, const double D,
                        const double dt, const double dx, const double dy,
                        const double dz, const StencilLaunch& launch) {
    cahnHilliardUpdateKernel<<<launch.grid, launch.block>>>(
        deviceCnew, deviceCold, deviceMu, nx, ny, nz, D, dt, dx, dy, dz);
    CUDA_CHECK(cudaGetLastError());
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
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
    if (nx == 0 || ny == 0 || nz == 0) {
        std::fprintf(stderr, "Grid dimensions must be positive\n");
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
    const size_t bytes = gridSize * sizeof(double);
    double* deviceCold = nullptr;
    double* deviceCnew = nullptr;
    double* deviceMu = nullptr;

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCold), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceCnew), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceMu), bytes));

    // All field initialization and evolution is performed on the GPU. The
    // host vector is only populated after the simulation for reporting and
    // validation, matching the original externally visible behavior.
    std::printf("Initializing concentration field...\n");
    initializeConcentration(deviceCold, nx, ny, nz);

    const StencilLaunch launch = makeStencilLaunch(nx, ny, nz);

    std::printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::high_resolution_clock::now();

    for (int t = 0; t < iterations; ++t) {
        computeChemicalPotential(deviceCold, deviceMu, nx, ny, nz, dx, dy,
                                 dz, gamma, e_AA, e_BB, e_AB, launch);
        cahnHilliardUpdate(deviceCnew, deviceCold, deviceMu, nx, ny, nz, D,
                           dt, dx, dy, dz, launch);
        std::swap(deviceCold, deviceCnew);
    }

    // Kernel launches are asynchronous, so synchronize before stopping the
    // timer and before copying data back to the host.
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::vector<double> cold(gridSize);
    CUDA_CHECK(cudaMemcpy(cold.data(), deviceCold, bytes,
                          cudaMemcpyDeviceToHost));

    std::printf("Computation time: %ld ms\n", duration.count());

    const double elapsedSeconds = duration.count() / 1000.0;
    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double mcups = elapsedSeconds > 0.0
                             ? cellUpdates / elapsedSeconds / 1e6
                             : 0.0;
    std::printf("Performance: %.3f MCellUpdates/s\n", mcups);

    if (printResults) {
        print_results(cold, "Concentration");
    }

    bool valid = true;
    if (validate) {
        std::printf("Validating result...\n");
        valid = validateResult(cold, nx, ny, nz);
        std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }

    CUDA_CHECK(cudaFree(deviceMu));
    CUDA_CHECK(cudaFree(deviceCnew));
    CUDA_CHECK(cudaFree(deviceCold));

    return validate && !valid ? 1 : 0;
}
