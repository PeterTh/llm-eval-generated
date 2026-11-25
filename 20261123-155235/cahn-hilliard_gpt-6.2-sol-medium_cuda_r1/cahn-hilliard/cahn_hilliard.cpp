#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

namespace {

constexpr unsigned blockSize = 256;
// The benchmark uses unit grid spacing.
constexpr double dt = 0.01;
constexpr double D = 1.0;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = 2.0 / 9.0;
constexpr double gamma = 0.5;

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

struct DeviceDeleter {
    void operator()(double* pointer) const { cudaFree(pointer); }
};

std::unique_ptr<double, DeviceDeleter> allocateField(size_t bytes) {
    double* pointer = nullptr;
    if (bytes != 0) {
        checkCuda(cudaMalloc(&pointer, bytes), "Allocating CUDA field");
    }
    return std::unique_ptr<double, DeviceDeleter>(pointer);
}

__global__ void initializeConcentration(double* field, size_t cells, double inverseCells) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < cells) {
        const size_t seed = (i + 1) * 1299709;
        const size_t mask = cells - 1;
        // For a power-of-two volume the reciprocal is exact.
        const double pseudo = (cells & mask) == 0
            ? static_cast<double>(seed & mask) * inverseCells
            : static_cast<double>(seed % cells) / static_cast<double>(cells);
        field[i] = __fma_rn(2.0, pseudo, -1.0);
    }
}

template <typename Index>
struct Grid {
    Index width;
    Index plane;
    Index cells;
    uint32_t widthReciprocal;
    uint32_t planeReciprocal;
};

template <bool powerOfTwo, typename Index>
__device__ __forceinline__ Index remainder(Index value, Index divisor,
                                          uint32_t reciprocal) {
    if constexpr (powerOfTwo) {
        return value & (divisor - 1);
    } else if constexpr (sizeof(Index) == sizeof(uint32_t)) {
        // The truncated reciprocal underestimates the quotient by at most one.
        const Index quotient = __umulhi(value, reciprocal);
        const Index rest = value - quotient * divisor;
        return rest >= divisor ? rest - divisor : rest;
    } else {
        return value % divisor;
    }
}

// Apply the seven-point stencil, using the center value at each clamped face.
template <bool powerOfTwo, typename Index>
__device__ __forceinline__ double laplacian(const double* field, Index i, double center,
                                           Grid<Index> grid) {
    const Index inPlane = remainder<powerOfTwo>(i, grid.plane, grid.planeReciprocal);
    const Index x = remainder<powerOfTwo>(inPlane, grid.width, grid.widthReciprocal);

    const Index xp = i + (x < grid.width - 1 ? 1 : 0);
    const Index xn = i - (x > 0 ? 1 : 0);
    const Index yp = i + (inPlane < grid.plane - grid.width ? grid.width : 0);
    const Index yn = i - (inPlane >= grid.width ? grid.width : 0);
    const Index zp = i + (i < grid.cells - grid.plane ? grid.plane : 0);
    const Index zn = i - (i >= grid.plane ? grid.plane : 0);

    const double twiceCenter = 2.0 * center;
    const double cxx = (field[xp] + field[xn]) - twiceCenter;
    const double cyy = (field[yp] + field[yn]) - twiceCenter;
    const double czz = (field[zp] + field[zn]) - twiceCenter;
    return (cxx + cyy) + czz;
}

enum class Step { chemicalPotential, concentrationUpdate };

template <bool powerOfTwo, Step step, typename Index>
__global__ void stencilStep(const double* __restrict__ input,
                            double* __restrict__ output, Grid<Index> grid) {
    const Index i = static_cast<Index>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= grid.cells) {
        return;
    }

    const double center = input[i];
    const double lap = laplacian<powerOfTwo>(input, i, center, grid);
    if constexpr (step == Step::chemicalPotential) {
        // Match the fused operations in the optimized CPU expression.
        const double interaction = __fma_rn(center + 1.0, e_AA, (center - 1.0) * e_BB);
        const double energy = __fma_rn(-2.0 * center, e_AB, interaction);
        const double bulk = __fma_rn(4.5, energy, 3.0 * center);
        output[i] = __fma_rn(-gamma, lap, __fma_rn(center * center, center, bulk));
    } else {
        // Each thread reads only its own concentration, so updating it in place is safe.
        output[i] = __fma_rn(dt * D, lap, output[i]);
    }
}

template <bool powerOfTwo, typename Index>
void evolve(double* concentration, double* mu, Grid<Index> grid,
            unsigned blocks, int iterations) {
    for (int t = 0; t < iterations; ++t) {
        // The default stream orders the stencil passes and successive time steps.
        stencilStep<powerOfTwo, Step::chemicalPotential><<<blocks, blockSize>>>(concentration, mu, grid);
        stencilStep<powerOfTwo, Step::concentrationUpdate><<<blocks, blockSize>>>(mu, concentration, grid);
    }
}

template <typename Index>
void evolveGrid(double* concentration, double* mu, size_t width, size_t plane,
                size_t cells, unsigned blocks, int iterations) {
    constexpr auto maxReciprocal = std::numeric_limits<uint32_t>::max();
    const Grid<Index> grid{static_cast<Index>(width), static_cast<Index>(plane),
                           static_cast<Index>(cells), static_cast<uint32_t>(maxReciprocal / width),
                           static_cast<uint32_t>(maxReciprocal / plane)};
    if ((width & (width - 1)) == 0 && (plane & (plane - 1)) == 0) {
        evolve<true>(concentration, mu, grid, blocks, iterations);
    } else {
        evolve<false>(concentration, mu, grid, blocks, iterations);
    }
}

}  // namespace

bool validateResult(const std::vector<double>& c) {
    if (c.empty()) {
        printf("Validation failed: empty concentration field\n");
        return false;
    }

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

    constexpr size_t maxCells = std::numeric_limits<size_t>::max() / sizeof(double);
    if (nx != 0 && (ny > maxCells / nx || nz > maxCells / (nx * ny))) {
        fprintf(stderr, "Grid is too large\n");
        return 1;
    }

    const size_t plane = nx * ny;
    const size_t gridSize = plane * nz;
    const size_t blocks = (gridSize + blockSize - 1) / blockSize;
    if (blocks > std::numeric_limits<int>::max()) {
        fprintf(stderr, "Grid exceeds CUDA launch limits\n");
        return 1;
    }

    const size_t bytes = gridSize * sizeof(double);
    const auto concentration = allocateField(bytes);
    const auto mu = allocateField(iterations > 0 ? bytes : 0);

    printf("Initializing concentration field...\n");
    if (gridSize != 0) {
        initializeConcentration<<<static_cast<unsigned>(blocks), blockSize>>>(
            concentration.get(), gridSize, 1.0 / static_cast<double>(gridSize));
        checkCuda(cudaGetLastError(), "Launching CUDA initialization");
    }
    checkCuda(cudaDeviceSynchronize(), "Initializing concentration on CUDA");

    printf("Running Cahn-Hilliard simulation...\n");
    const auto start = std::chrono::steady_clock::now();
    if (gridSize != 0 && iterations > 0) {
        if (gridSize <= std::numeric_limits<uint32_t>::max()) {
            evolveGrid<uint32_t>(concentration.get(), mu.get(), nx, plane, gridSize,
                                 static_cast<unsigned>(blocks), iterations);
        } else {
            evolveGrid<size_t>(concentration.get(), mu.get(), nx, plane, gridSize,
                               static_cast<unsigned>(blocks), iterations);
        }
        checkCuda(cudaGetLastError(), "Launching CUDA simulation");
        checkCuda(cudaDeviceSynchronize(), "Running CUDA simulation");
    }
    const auto elapsed = std::chrono::steady_clock::now() - start;
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
    printf("Computation time: %ld ms\n", milliseconds.count());

    const double cellUpdates = static_cast<double>(gridSize) * iterations;
    const double seconds = std::chrono::duration<double>(elapsed).count();
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);

    std::vector<double> hostConcentration;
    if ((printResults || validate) && gridSize != 0) {
        hostConcentration.resize(gridSize);
        checkCuda(cudaMemcpy(hostConcentration.data(), concentration.get(), bytes, cudaMemcpyDeviceToHost),
                  "Copying final concentration from CUDA");
    }

    if (printResults) {
        print_results(hostConcentration, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(hostConcentration);

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
