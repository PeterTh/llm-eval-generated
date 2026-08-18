#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned int blockX = 32;
constexpr unsigned int blockY = 4;
constexpr unsigned int blockZ = 4;

[[noreturn]] void cudaFailure(const cudaError_t error, const char* call, const char* file, const int line) {
    fprintf(stderr, "CUDA error at %s:%d (%s): %s\n", file, line, call, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

inline void cudaCheck(const cudaError_t error, const char* call, const char* file, const int line) {
    if (error != cudaSuccess) {
        cudaFailure(error, call, file, line);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call, __FILE__, __LINE__)

// 3D index calculation for the flat row-major field layout used by the host code.
__device__ __forceinline__ size_t idx3Device(const size_t x, const size_t y, const size_t z,
                                             const size_t nx, const size_t planeSize) noexcept {
    return z * planeSize + y * nx + x;
}

__device__ __forceinline__ size_t tileIndex(const unsigned int x, const unsigned int y,
                                            const unsigned int z, const unsigned int tileX,
                                            const unsigned int tileY) noexcept {
    return (static_cast<size_t>(z) * tileY + y) * tileX + x;
}

// Load a block-sized tile plus its one-cell halo. Clamping the global coordinates here
// exactly matches the original boundary condition, including on partial edge blocks.
__device__ __forceinline__ void loadClampedTile(const double* __restrict__ input, double* tile,
                                                const size_t nx, const size_t ny, const size_t nz) {
    const unsigned int tileX = blockDim.x + 2;
    const unsigned int tileY = blockDim.y + 2;
    const unsigned int tileZ = blockDim.z + 2;
    const size_t planeSize = nx * ny;
    const size_t baseX = static_cast<size_t>(blockIdx.x) * blockDim.x;
    const size_t baseY = static_cast<size_t>(blockIdx.y) * blockDim.y;
    const size_t baseZ = static_cast<size_t>(blockIdx.z) * blockDim.z;

    for (unsigned int localZ = threadIdx.z; localZ < tileZ; localZ += blockDim.z) {
        size_t globalZ = baseZ + localZ - 1;
        if (localZ == 0) {
            globalZ = (baseZ == 0) ? 0 : baseZ - 1;
        } else if (globalZ >= nz) {
            globalZ = nz - 1;
        }

        for (unsigned int localY = threadIdx.y; localY < tileY; localY += blockDim.y) {
            size_t globalY = baseY + localY - 1;
            if (localY == 0) {
                globalY = (baseY == 0) ? 0 : baseY - 1;
            } else if (globalY >= ny) {
                globalY = ny - 1;
            }

            for (unsigned int localX = threadIdx.x; localX < tileX; localX += blockDim.x) {
                size_t globalX = baseX + localX - 1;
                if (localX == 0) {
                    globalX = (baseX == 0) ? 0 : baseX - 1;
                } else if (globalX >= nx) {
                    globalX = nx - 1;
                }

                tile[tileIndex(localX, localY, localZ, tileX, tileY)] =
                    input[idx3Device(globalX, globalY, globalZ, nx, planeSize)];
            }
        }
    }

    __syncthreads();
}

__device__ __forceinline__ double tileLaplacian(const double* tile, const unsigned int x,
                                                const unsigned int y, const unsigned int z,
                                                const unsigned int tileX, const unsigned int tileY,
                                                const double invDx2, const double invDy2,
                                                const double invDz2) noexcept {
    const size_t center = tileIndex(x, y, z, tileX, tileY);
    const size_t xOffset = 1;
    const size_t yOffset = tileX;
    const size_t zOffset = static_cast<size_t>(tileX) * tileY;
    const double centerValue = tile[center];

    const double cxx = (tile[center + xOffset] + tile[center - xOffset] - 2.0 * centerValue) * invDx2;
    const double cyy = (tile[center + yOffset] + tile[center - yOffset] - 2.0 * centerValue) * invDy2;
    const double czz = (tile[center + zOffset] + tile[center - zOffset] - 2.0 * centerValue) * invDz2;
    return cxx + cyy + czz;
}

__global__ void initializeConcentrationKernel(double* __restrict__ c, const size_t total,
                                               const size_t volume) {
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t linearId = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         linearId < total; linearId += stride) {
        // Keep the unsigned size_t arithmetic of the original initialization formula.
        const uint64_t pseudoNumerator = (static_cast<uint64_t>(linearId) + 1ULL) * 1299709ULL;
        const double pseudo = static_cast<double>(pseudoNumerator % volume) /
                              static_cast<double>(volume);
        c[linearId] = -1.0 + 2.0 * pseudo;
    }
}

__global__ void computeChemicalPotentialKernel(const double* __restrict__ c,
                                               double* __restrict__ mu,
                                               const size_t nx, const size_t ny, const size_t nz,
                                               const double invDx2, const double invDy2,
                                               const double invDz2, const double gamma,
                                               const double e_AA, const double e_BB,
                                               const double e_AB) {
    extern __shared__ double tile[];
    loadClampedTile(c, tile, nx, ny, nz);

    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const unsigned int tileX = blockDim.x + 2;
        const unsigned int tileY = blockDim.y + 2;
        const unsigned int localX = threadIdx.x + 1;
        const unsigned int localY = threadIdx.y + 1;
        const unsigned int localZ = threadIdx.z + 1;
        const double cv = tile[tileIndex(localX, localY, localZ, tileX, tileY)];
        const double laplacian = tileLaplacian(tile, localX, localY, localZ, tileX, tileY,
                                               invDx2, invDy2, invDz2);
        const size_t idx = idx3Device(x, y, z, nx, nx * ny);

        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                + 3.0 * cv + cv * cv * cv
                - gamma * laplacian;
    }
}

__global__ void cahnHilliardUpdateKernel(double* __restrict__ cnew,
                                         const double* __restrict__ cold,
                                         const double* __restrict__ mu,
                                         const size_t nx, const size_t ny, const size_t nz,
                                         const double D, const double dt,
                                         const double invDx2, const double invDy2,
                                         const double invDz2) {
    extern __shared__ double tile[];
    loadClampedTile(mu, tile, nx, ny, nz);

    const size_t x = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t y = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    const size_t z = static_cast<size_t>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (x < nx && y < ny && z < nz) {
        const unsigned int tileX = blockDim.x + 2;
        const unsigned int tileY = blockDim.y + 2;
        const unsigned int localX = threadIdx.x + 1;
        const unsigned int localY = threadIdx.y + 1;
        const unsigned int localZ = threadIdx.z + 1;
        const double laplacian = tileLaplacian(tile, localX, localY, localZ, tileX, tileY,
                                               invDx2, invDy2, invDz2);
        const size_t idx = idx3Device(x, y, z, nx, nx * ny);
        cnew[idx] = cold[idx] + dt * D * laplacian;
    }
}

} // namespace

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
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

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
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
    
    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
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
    
    size_t gridSize = nx * ny * nz;
    if (nx == 0 || ny == 0 || nz == 0 || gridSize == 0) {
        fprintf(stderr, "Grid dimensions must be greater than zero\n");
        return 1;
    }
    
    // Allocate arrays
    std::vector<double> cold(gridSize);
    const size_t bytes = gridSize * sizeof(double);
    double* d_cold = nullptr;
    double* d_cnew = nullptr;
    double* d_mu = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cold), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_cnew), bytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_mu), bytes));
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    const unsigned int initThreads = 256;
    const size_t requestedInitBlocks = (gridSize - 1) / initThreads + 1;
    const unsigned int initBlocks = static_cast<unsigned int>(
        std::min<size_t>(requestedInitBlocks, 65535));
    initializeConcentrationKernel<<<initBlocks, initThreads>>>(d_cold, gridSize, gridSize);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    const dim3 threads(blockX, blockY, blockZ);
    const dim3 blocks(static_cast<unsigned int>((nx - 1) / blockX + 1),
                      static_cast<unsigned int>((ny - 1) / blockY + 1),
                      static_cast<unsigned int>((nz - 1) / blockZ + 1));
    const size_t sharedBytes = static_cast<size_t>(blockX + 2) * (blockY + 2) * (blockZ + 2) * sizeof(double);
    const double invDx2 = 1.0 / (dx * dx);
    const double invDy2 = 1.0 / (dy * dy);
    const double invDz2 = 1.0 / (dz * dz);

    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    
    for (int t = 0; t < iterations; ++t) {
        // Compute chemical potential
        computeChemicalPotentialKernel<<<blocks, threads, sharedBytes>>>(
            d_cold, d_mu, nx, ny, nz, invDx2, invDy2, invDz2, gamma, e_AA, e_BB, e_AB);
        
        // Update concentration
        cahnHilliardUpdateKernel<<<blocks, threads, sharedBytes>>>(
            d_cnew, d_cold, d_mu, nx, ny, nz, D, dt, invDx2, invDy2, invDz2);
        
        // Swap buffers
        std::swap(d_cold, d_cnew);
    }
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsedMilliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMilliseconds, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    CUDA_CHECK(cudaMemcpy(cold.data(), d_cold, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_cnew));
    CUDA_CHECK(cudaFree(d_cold));
    
    const long durationMilliseconds = static_cast<long>(elapsedMilliseconds);
    printf("Computation time: %ld ms\n", durationMilliseconds);
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = (elapsedMilliseconds > 0.0f)
        ? cellUpdates / (static_cast<double>(elapsedMilliseconds) / 1000.0) / 1e6
        : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
