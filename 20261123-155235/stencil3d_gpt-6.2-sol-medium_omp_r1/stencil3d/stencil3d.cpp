#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// Limit each tile's plane footprint for cache reuse across adjacent Z levels.
constexpr size_t tileWidth = 2048;
constexpr size_t tileDepth = 8;
constexpr size_t tilePlaneCells = 8192;

template <typename ProcessRow>
void forEachTileRow(size_t nx, size_t ny, size_t nz, size_t tileHeight, ProcessRow processRow) {
    #pragma omp for collapse(3) schedule(static)
    for (size_t blockZ = 0; blockZ < nz; blockZ += tileDepth) {
        for (size_t blockY = 0; blockY < ny; blockY += tileHeight) {
            for (size_t blockX = 0; blockX < nx; blockX += tileWidth) {
                const size_t endZ = std::min(nz, blockZ + tileDepth);
                const size_t endY = std::min(ny, blockY + tileHeight);
                const size_t endX = std::min(nx, blockX + tileWidth);
                for (size_t z = blockZ; z < endZ; ++z) {
                    for (size_t y = blockY; y < endY; ++y) {
                        processRow(z, y, blockX, endX);
                    }
                }
            }
        }
    }
}

void initializeGrids(Real* grid1, Real* grid2, size_t nx, size_t ny, size_t nz,
                     size_t tileHeight, bool runIterations) {
    forEachTileRow(nx, ny, nz, tileHeight, [=](size_t z, size_t y, size_t beginX, size_t endX) {
        const size_t row = (z * ny + y) * nx;
        #pragma omp simd
        for (size_t x = beginX; x < endX; ++x) {
            const Real value = static_cast<Real>((row + x) % 19);
            grid1[row + x] = value;
            grid2[row + x] = runIterations ? value : 0.0;
        }
    });
}

void stencilIteration(const Real* input, Real* output, size_t nx, size_t ny, size_t nz,
                      size_t tileHeight) {
    const size_t plane = nx * ny;
    forEachTileRow(nx, ny, nz, tileHeight, [=](size_t z, size_t y, size_t beginX, size_t endX) {
        if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
            return;
        }

        const size_t row = z * plane + y * nx;
        const Real* center = input + row;
        const Real* front = input + (row - nx);
        const Real* back = input + (row + nx);
        const Real* bottom = input + (row - plane);
        const Real* top = input + (row + plane);
        Real* next = output + row;
        const size_t firstX = std::max(beginX, size_t{1});
        const size_t lastX = std::min(endX, nx - 1);

        #pragma omp simd
        for (size_t x = firstX; x < lastX; ++x) {
            next[x] = (center[x] + center[x - 1] + center[x + 1]
                       + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
        }
    });
    // The worksharing barrier completes all writes before threads swap buffers.
}

bool validateResult(std::span<const Real> grid, int threads) {
    if (grid.empty()) {
        printf("Validation failed: empty grid\n");
        return false;
    }

    bool hasNonFinite = false;
    Real minVal = grid.front();
    Real maxVal = grid.front();
    #pragma omp parallel for simd num_threads(threads) schedule(static) \
        reduction(||:hasNonFinite) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real value = grid[i];
        hasNonFinite |= !std::isfinite(value);
        minVal = std::min(minVal, value);
        maxVal = std::max(maxVal, value);
    }

    if (hasNonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    const size_t gridSize = nx * ny * nz;

    // Let the OpenMP workers first-touch both buffers during initialization.
    const auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    const auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);

    const size_t tileHeight = tilePlaneCells / std::clamp(nx, size_t{1}, tileWidth);
    const size_t tiles = ((nx + tileWidth - 1) / tileWidth)
                       * ((ny + tileHeight - 1) / tileHeight)
                       * ((nz + tileDepth - 1) / tileDepth);
    // Keep small grids from spending most of each step at the team barrier.
    constexpr size_t cellsPerThread = 32768;
    const size_t workUnits = std::min(tiles, 1 + gridSize / cellsPerThread);
    const int threads = static_cast<int>(std::clamp(workUnits, size_t{1},
                                                   static_cast<size_t>(omp_get_max_threads())));
    const bool hasInterior = nx > 2 && ny > 2 && nz > 2;
    using Clock = std::chrono::steady_clock;
    Clock::time_point start;

    printf("Initializing grid...\n");
    #pragma omp parallel num_threads(threads)
    {
        // Boundary cells are fixed and have the same values in both buffers.
        initializeGrids(grid1.get(), grid2.get(), nx, ny, nz, tileHeight, iterations > 0);

        #pragma omp master
        {
            printf("Running stencil computation...\n");
            start = Clock::now();
        }
        #pragma omp barrier

        Real* input = grid1.get();
        Real* output = grid2.get();
        if (hasInterior) {
            for (int iter = 0; iter < iterations; ++iter) {
                stencilIteration(input, output, nx, ny, nz, tileHeight);
                std::swap(input, output);
            }
        }
    }

    const auto duration = Clock::now() - start;
    const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(duration);
    printf("Computation time: %ld ms\n", milliseconds.count());

    const double cellUpdates = hasInterior && iterations > 0
        ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations
        : 0.0;
    const double seconds = std::chrono::duration<double>(duration).count();
    printf("Performance: %.3f MCellUpdates/s\n", cellUpdates / seconds / 1e6);

    // Print results for external validation
    const std::span<const Real> finalGrid(
        (iterations % 2 == 0) ? grid1.get() : grid2.get(), gridSize);
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, threads);
        
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
