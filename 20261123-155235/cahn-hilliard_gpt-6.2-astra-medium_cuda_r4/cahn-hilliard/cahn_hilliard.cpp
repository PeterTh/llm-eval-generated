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
constexpr unsigned threads = 128;
constexpr unsigned depth = 4;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void initializeConcentration(double* c, size_t vol) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < vol; i += size_t(gridDim.x) * blockDim.x) {
        const double pseudo = (((i + 1) * 1299709) % vol) / static_cast<double>(vol);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ double chemicalPotential(double cv, double laplacian) {
    constexpr double e_AA = -(2.0 / 9.0), e_BB = -(2.0 / 9.0), e_AB = 2.0 / 9.0;
    constexpr double gamma = 0.5;
    double energy = fma(cv + 1.0, e_AA, __dmul_rn(cv - 1.0, e_BB));
    energy = fma(-2.0 * cv, e_AB, energy);
    double bulk = fma(4.5, energy, 3.0 * cv);
    bulk = fma(cv * cv, cv, bulk);
    return fma(-gamma, laplacian, bulk);
}

// Each warp reads contiguous cells. Advancing through a short column keeps
// the two overlapping z neighbors in registers for the next Laplacian.
template<bool potential>
__global__ void stencil(const double* __restrict__ field, const double* __restrict__ cold,
                        double* __restrict__ out, unsigned nx, unsigned ny, unsigned nz,
                        size_t plane) {
    constexpr double dt = 0.01, D = 1.0;
    for (size_t xy = size_t(blockIdx.x) * threads + threadIdx.x;
         xy < plane; xy += size_t(gridDim.x) * threads) {
        unsigned x = xy % nx, y = xy / nx;
        for (unsigned firstZ = blockIdx.y * depth; firstZ < nz; firstZ += gridDim.y * depth) {
            unsigned z = firstZ;
            size_t i = size_t(z) * plane + xy;
            const unsigned left = x > 0, right = x + 1 < nx;
            const unsigned down = y > 0 ? nx : 0, up = y + 1 < ny ? nx : 0;
            double center = field[i];
            double below = z > 0 ? field[i - plane] : center;
            #pragma unroll
            for (unsigned k = 0; k < depth; ++k, ++z, i += plane) {
                double above = z + 1 < nz ? field[i + plane] : center;
                // The reference grid has unit spacing in all three dimensions.
                double twice = 2.0 * center;
                double xx = field[i + right] + field[i - left] - twice;
                double yy = field[i + up] + field[i - down] - twice;
                double zz = above + below - twice;
                double laplacian = xx + yy + zz;
                if constexpr (potential) {
                    out[i] = chemicalPotential(center, laplacian);
                } else {
                    out[i] = fma(dt * D, laplacian, cold[i]);
                }
                if (z + 1 == nz) break;
                below = center;
                center = above;
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
    if (!nx || nx > std::numeric_limits<int>::max() || ny > std::numeric_limits<int>::max()
        || nz > std::numeric_limits<int>::max() || ny > maxCells / nx || nz > maxCells / (nx * ny)) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t plane = nx * ny, gridSize = plane * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *current, *next, *mu;
    checkCuda(cudaMalloc(&current, bytes), "Allocate concentration");
    checkCuda(cudaMalloc(&next, bytes), "Allocate updated concentration");
    checkCuda(cudaMalloc(&mu, bytes), "Allocate chemical potential");

    constexpr size_t maxBlocks = 2147483647;
    printf("Initializing concentration field...\n");
    initializeConcentration<<<std::min(maxBlocks, (gridSize + threads - 1) / threads), threads>>>(current, gridSize);
    checkCuda(cudaGetLastError(), "Launch initialization");
    checkCuda(cudaDeviceSynchronize(), "Initialize concentration");

    const size_t tilesPerPlane = (plane + threads - 1) / threads;
    const dim3 blocks(std::min(maxBlocks, tilesPerPlane), std::min<size_t>(65535, (nz + depth - 1) / depth));
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < iterations; ++t) {
        stencil<true><<<blocks, threads>>>(current, nullptr, mu, nx, ny, nz, plane);
        stencil<false><<<blocks, threads>>>(mu, current, next, nx, ny, nz, plane);
        std::swap(current, next);
    }
    checkCuda(cudaGetLastError(), "Launch simulation");
    checkCuda(cudaDeviceSynchronize(), "Run simulation");
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    std::vector<double> cold;
    if (validate || printResults) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), current, bytes, cudaMemcpyDeviceToHost), "Read concentration");
    }
    checkCuda(cudaFree(mu), "Free chemical potential");
    checkCuda(cudaFree(next), "Free updated concentration");
    checkCuda(cudaFree(current), "Free concentration");

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
