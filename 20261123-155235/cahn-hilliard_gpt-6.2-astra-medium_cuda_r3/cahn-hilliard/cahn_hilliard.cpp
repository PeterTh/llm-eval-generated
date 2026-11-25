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
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

constexpr unsigned threads = 128;
constexpr unsigned planesPerThread = 4;

__global__ void initializeConcentration(double* c, size_t volume) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < volume) {
        const double pseudo = (((i + 1) * 1299709) % volume) / static_cast<double>(volume);
        c[i] = -1.0 + 2.0 * pseudo;
    }
}

__device__ inline double chemicalPotential(double c, double laplacian) {
    constexpr double eAA = -2.0 / 9.0, eBB = -2.0 / 9.0, eAB = 2.0 / 9.0;
    const double pairEnergy = fma(c + 1.0, eAA, (c - 1.0) * eBB);
    const double interaction = fma(-2.0 * c, eAB, pairEnergy);
    const double local = fma(c * c, c, fma(4.5, interaction, 3.0 * c));
    return fma(-0.5, laplacian, local);
}

// Adjacent lanes access adjacent cells. Slide a three-plane window along z
// so each new plane needs just one new z-neighbor load per column.
template<bool computePotential>
__global__ void applyStencil(const double* __restrict__ field, double* __restrict__ output,
                             unsigned nx, size_t plane, unsigned nz) {
    const size_t column = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (column >= plane) return;
    const unsigned x = column % nx;
    const unsigned left = x > 0, right = x + 1 < nx;
    const unsigned down = column >= nx ? nx : 0;
    const unsigned up = column < plane - nx ? nx : 0;
    const size_t group = size_t(blockIdx.y) * blockDim.y + threadIdx.y;
    for (size_t first = group * planesPerThread; first < nz;
         first += size_t(gridDim.y) * blockDim.y * planesPerThread) {
        size_t i = first * plane + column;
        double center = field[i];
        double prev = first ? field[i - plane] : center;
        #pragma unroll
        for (unsigned k = 0; k < planesPerThread; ++k, i += plane) {
            if (first + k >= nz) break;
            const double next = first + k + 1 < nz ? field[i + plane] : center;
            const double twice = 2.0 * center;
            const double xx = (field[i + right] + field[i - left]) - twice;
            const double yy = (field[i + up] + field[i - down]) - twice;
            const double zz = (next + prev) - twice;
            const double laplacian = (xx + yy) + zz;  // Unit grid spacing
            if constexpr (computePotential) {
                output[i] = chemicalPotential(center, laplacian);
            } else {
                // Only the potential needs neighbors during the update.
                output[i] = fma(0.01, laplacian, output[i]);
            }
            prev = center;
            center = next;
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
    if (!nx || !ny || !nz || nx > maxCells / ny || nx * ny > maxCells / nz ||
        nx > std::numeric_limits<unsigned>::max() || nz > std::numeric_limits<unsigned>::max()) {
        fprintf(stderr, "Invalid grid dimensions\n");
        return 1;
    }
    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t bytes = gridSize * sizeof(double);
    double *concentration, *potential;
    checkCuda(cudaMalloc(&concentration, bytes), "Allocate concentration");
    checkCuda(cudaMalloc(&potential, bytes), "Allocate chemical potential");

    printf("Initializing concentration field...\n");
    initializeConcentration<<<(gridSize + threads - 1) / threads, threads>>>(concentration, gridSize);
    checkCuda(cudaGetLastError(), "Launch initialization kernel");
    checkCuda(cudaDeviceSynchronize(), "Initialize concentration");

    // For narrow grids, fill the remaining lanes with independent z groups.
    unsigned columns = threads;
    while (columns / 2 >= plane) columns /= 2;
    const dim3 block(columns, threads / columns);
    const unsigned tileDepth = block.y * planesPerThread;
    const dim3 blocks((plane + columns - 1) / columns,
                      std::min<size_t>((nz + tileDepth - 1) / tileDepth, 65535));
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::steady_clock::now();
    for (int t = 0; t < iterations; ++t) {
        applyStencil<true><<<blocks, block>>>(concentration, potential, nx, plane, nz);
        applyStencil<false><<<blocks, block>>>(potential, concentration, nx, plane, nz);
    }
    checkCuda(cudaGetLastError(), "Launch stencil kernels");
    checkCuda(cudaDeviceSynchronize(), "Run simulation");
    const std::chrono::duration<double> elapsed = std::chrono::steady_clock::now() - start;
    printf("Computation time: %.0f ms\n", elapsed.count() * 1000.0);
    const double mcups = static_cast<double>(gridSize) * iterations / elapsed.count() / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        checkCuda(cudaMemcpy(cold.data(), concentration, bytes, cudaMemcpyDeviceToHost),
                  "Copy final concentration");
    }
    checkCuda(cudaFree(potential), "Free chemical potential");
    checkCuda(cudaFree(concentration), "Free concentration");

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
