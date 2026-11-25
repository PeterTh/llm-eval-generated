#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

namespace {

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* concentration, size_t volume) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < volume; i += size_t(gridDim.x) * blockDim.x) {
        const double pseudo = ((i + 1) * 1299709 % volume) / static_cast<double>(volume);
        concentration[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ double chemicalPotential(double c, double laplacian) {
    constexpr double eAA = -2.0 / 9.0, eBB = -2.0 / 9.0, eAB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    double interaction = fma(c + 1.0, eAA, (c - 1.0) * eBB);
    interaction = fma(-2.0 * c, eAB, interaction);
    double bulk = fma(4.5, interaction, 3.0 * c);
    bulk = fma(c * c, c, bulk);
    return fma(-gamma, laplacian, bulk);
}

constexpr unsigned planesPerTile = 4;

// Each warp traverses contiguous x coordinates. A thread carries the two
// overlapping z neighbors in registers as it advances through its tile.
template<bool computePotential>
__global__ void stencil(const double* __restrict__ field, double* __restrict__ output,
                        size_t nx, size_t ny, size_t nz) {
    const size_t x = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (x >= nx) return;
    const size_t plane = nx * ny;
    for (size_t y = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
         y < ny; y += size_t(gridDim.y) * blockDim.y) {
        const size_t left = x > 0, right = x + 1 < nx;
        const size_t down = y > 0 ? nx : 0, up = y + 1 < ny ? nx : 0;
        for (size_t z0 = size_t(blockIdx.z) * planesPerTile;
             z0 < nz; z0 += size_t(gridDim.z) * planesPerTile) {
            size_t i = z0 * plane + y * nx + x;
            double center = field[i];
            double previous = z0 > 0 ? field[i - plane] : center;
            #pragma unroll
            for (unsigned k = 0; k < planesPerTile; ++k) {
                if (z0 + k >= nz) break;
                const double next = z0 + k + 1 < nz ? field[i + plane] : center;
                const double twice = 2.0 * center;
                // Unit mesh spacing in all three dimensions.
                const double xx = (field[i + right] + field[i - left]) - twice;
                const double yy = (field[i + up] + field[i - down]) - twice;
                const double zz = (next + previous) - twice;
                const double laplacian = xx + yy + zz;
                if constexpr (computePotential) {
                    output[i] = chemicalPotential(center, laplacian);
                } else {
                    constexpr double dt = 0.01, diffusivity = 1.0;
                    output[i] = fma(dt * diffusivity, laplacian, output[i]);
                }
                previous = center;
                center = next;
                i += plane;
            }
        }
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
    
    const size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx == 0 || ny == 0 || nz == 0 || nx > maxCells / ny || nx * ny > maxCells / nz) {
        std::fprintf(stderr, "Invalid grid size\n");
        return 1;
    }
    const size_t gridSize = nx * ny * nz;
    const size_t bytes = gridSize * sizeof(double);
    double* concentration;
    double* mu;
    checkCuda(cudaMalloc(&concentration, bytes), "Allocating concentration");
    checkCuda(cudaMalloc(&mu, bytes), "Allocating chemical potential");

    printf("Initializing concentration field...\n");
    const unsigned initBlocks = std::min<size_t>((gridSize + 255) / 256, 65535);
    initializeConcentration<<<initBlocks, 256>>>(concentration, gridSize);
    checkCuda(cudaDeviceSynchronize(), "Initializing concentration");

    const dim3 block(32, 4);
    const dim3 grid((nx + block.x - 1) / block.x,
                    std::min<size_t>((ny + block.y - 1) / block.y, 65535),
                    std::min<size_t>((nz + planesPerTile - 1) / planesPerTile, 65535));
    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        stencil<true><<<grid, block>>>(concentration, mu, nx, ny, nz);
        // The stream completes mu before updating each concentration in place.
        stencil<false><<<grid, block>>>(mu, concentration, nx, ny, nz);
    }
    checkCuda(cudaGetLastError(), "Launching stencil kernels");
    checkCuda(cudaDeviceSynchronize(), "Running stencil kernels");
    const std::chrono::duration<double> duration = std::chrono::steady_clock::now() - start;
    printf("Computation time: %.3f ms\n", duration.count() * 1000.0);
    const double mcups = static_cast<double>(gridSize) * iterations / duration.count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), concentration, bytes, cudaMemcpyDeviceToHost),
                  "Reading final concentration");
    }
    checkCuda(cudaFree(mu), "Freeing chemical potential");
    checkCuda(cudaFree(concentration), "Freeing concentration");

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
