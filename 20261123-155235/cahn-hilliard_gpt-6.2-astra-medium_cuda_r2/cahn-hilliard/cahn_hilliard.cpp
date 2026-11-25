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

constexpr unsigned threads = 256;
constexpr unsigned planesPerTile = 4;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* field, size_t volume) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < volume) {
        const double pseudo = ((i + 1) * 1299709 % volume) / static_cast<double>(volume);
        field[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ double chemicalPotential(double c, double laplacian) {
    constexpr double e_AA = -2.0 / 9.0, e_BB = -2.0 / 9.0, e_AB = 2.0 / 9.0;
    const double interactions = fma(c + 1.0, e_AA, (c - 1.0) * e_BB);
    const double mixing = fma(-2.0 * c, e_AB, interactions);
    const double bulk = fma(c * c, c, fma(4.5, mixing, 3.0 * c));
    return fma(-0.5, laplacian, bulk);
}

// Each thread traverses a short column. Adjacent threads access adjacent
// cells in every plane, including when a warp crosses a row boundary.
template<bool computePotential, unsigned depth, typename Index>
__global__ void stencil(const double* __restrict__ field,
                        double* __restrict__ output, unsigned nx, Index plane, unsigned nz) {
    const Index lane = Index(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = depth == 1 ? plane * nz : plane;
    if (lane >= count) return;
    const Index xy = depth == 1 ? lane % plane : lane;
    const unsigned x = xy % nx;
    const size_t left = x > 0, right = x + 1 < nx;
    const size_t down = xy >= nx ? nx : 0;
    const size_t up = xy < plane - nx ? nx : 0;
    const size_t firstZ = depth == 1 ? lane / plane : size_t(blockIdx.y) * depth;

    for (size_t z0 = firstZ; z0 < nz; z0 += size_t(gridDim.y) * depth) {
        size_t i = depth == 1 ? lane : z0 * plane + xy;
        double center = field[i];
        double below = z0 ? field[i - plane] : center;
        #pragma unroll
        for (unsigned k = 0; k < depth; ++k) {
            const size_t z = z0 + k;
            const double above = z + 1 < nz ? field[i + plane] : center;
            const double twice = 2.0 * center;
            const double xx = (field[i + right] + field[i - left]) - twice;
            const double yy = (field[i + up] + field[i - down]) - twice;
            const double zz = (above + below) - twice;
            const double laplacian = xx + yy + zz;  // Unit grid spacing.
            if constexpr (computePotential)
                output[i] = chemicalPotential(center, laplacian);
            else
                output[i] = fma(0.01, laplacian, output[i]);  // dt * D = 0.01

            if (z + 1 == nz) break;
            below = center;
            center = above;
            i += plane;
        }
        if constexpr (depth == 1) break;
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
    if (!nx || nx > maxCells / ny || nx * ny > maxCells / nz) {
        fprintf(stderr, "Invalid grid size\n");
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *cold, *mu;
    checkCuda(cudaMalloc(&cold, bytes), "Allocate concentration");
    checkCuda(cudaMalloc(&mu, bytes), "Allocate chemical potential");

    printf("Initializing concentration field...\n");
    initializeConcentration<<<(gridSize + threads - 1) / threads, threads>>>(cold, gridSize);
    checkCuda(cudaGetLastError(), "Launch initialization");
    checkCuda(cudaDeviceSynchronize(), "Initialize concentration");

    printf("Running Cahn-Hilliard simulation...\n");
    // Flatten small planes so even very narrow grids fill entire warps.
    const bool flat = plane < 8192 && gridSize <= std::numeric_limits<unsigned>::max();
    const dim3 blocks(((flat ? gridSize : plane) + threads - 1) / threads,
                      flat ? 1 : std::min<size_t>((nz + planesPerTile - 1) / planesPerTile, 65535));
    auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        // With mu complete, the update only needs the old concentration at
        // its own cell, so it can safely update the concentration in place.
        if (flat) {
            stencil<true, 1><<<blocks, threads>>>(cold, mu, nx, static_cast<unsigned>(plane), nz);
            stencil<false, 1><<<blocks, threads>>>(mu, cold, nx, static_cast<unsigned>(plane), nz);
        } else {
            stencil<true, planesPerTile><<<blocks, threads>>>(cold, mu, nx, plane, nz);
            stencil<false, planesPerTile><<<blocks, threads>>>(mu, cold, nx, plane, nz);
        }
    }
    checkCuda(cudaGetLastError(), "Launch simulation");
    checkCuda(cudaDeviceSynchronize(), "Run simulation");
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    printf("Computation time: %.3f ms\n", elapsed.count() * 1000.0);
    printf("Performance: %.3f MCellUpdates/s\n", gridSize * double(iterations) / elapsed.count() / 1e6);

    std::vector<double> result;
    if (printResults || validate) {
        result.resize(gridSize);
        checkCuda(cudaMemcpy(result.data(), cold, bytes, cudaMemcpyDeviceToHost), "Read concentration");
    }
    checkCuda(cudaFree(mu), "Free chemical potential");
    checkCuda(cudaFree(cold), "Free concentration");

    // Print results for external validation
    if (printResults) {
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, nx, ny, nz);
        
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
