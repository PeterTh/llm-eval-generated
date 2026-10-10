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

void initializeGrids(std::span<Real> grid1, std::span<Real> grid2,
                     const bool initializeSecond) {
    const size_t count = grid1.size();
    // Parallel first touch distributes both buffers across NUMA nodes. Both
    // buffers have identical fixed boundaries before the first iteration.
    #pragma omp parallel for simd schedule(static)
    for (size_t idx = 0; idx < count; ++idx) {
        const Real value = (idx % 19) * 1.0;
        grid1[idx] = value;
        grid2[idx] = initializeSecond ? value : 0.0;
    }
}

inline void stencilRow(const Real* __restrict source, Real* __restrict destination,
                       const size_t row, const size_t nx, const size_t plane,
                       const size_t xBegin, const size_t xEnd) {
    #pragma omp simd
    for (size_t x = xBegin; x < xEnd; ++x) {
        const size_t idx = row + x;
        destination[idx] = (source[idx] + source[idx - 1] +
            source[idx + 1] + source[idx - nx] +
            source[idx + nx] + source[idx - plane] +
            source[idx + plane]) / 7.0;
    }
}

// 7-point stencil computation using one team for all time steps.
void stencilIterations(std::span<Real> grid1, std::span<Real> grid2,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int iterations) {
    const size_t plane = nx * ny;
    const size_t interiorY = ny > 2 ? ny - 2 : 0;
    const size_t interiorZ = nz > 2 ? nz - 2 : 0;
    const size_t lastX = nx > 1 ? nx - 1 : 1;
    constexpr size_t tileX = 512;
    constexpr size_t tileY = 16;
    constexpr size_t tileZ = 8;
    const size_t tilesX = (lastX - 1) / tileX + ((lastX - 1) % tileX != 0);
    const size_t tilesY = interiorY / tileY + (interiorY % tileY != 0);
    const size_t tilesZ = interiorZ / tileZ + (interiorZ % tileZ != 0);
    // Small planes already reuse well in cache. Tile planes of at least 1 MiB
    // or long rows when there are too few rows to occupy the requested team.
    const bool useTiles = plane >= 1024 * 1024 / sizeof(Real) ||
        (interiorY * interiorZ < static_cast<size_t>(omp_get_max_threads()) && tilesX > 1);

    #pragma omp parallel default(none) shared(grid1, grid2) \
        firstprivate(nx, plane, interiorY, interiorZ, lastX, iterations, tilesX, tilesY, tilesZ, useTiles)
    {
        Real* input = grid1.data();
        Real* output = grid2.data();
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* source = input;
            Real* destination = output;

            if (useTiles) {
                // Each tile's neighboring slices fit in private cache.
                #pragma omp for collapse(3) schedule(static)
                for (size_t bz = 0; bz < tilesZ; ++bz) {
                    for (size_t by = 0; by < tilesY; ++by) {
                        for (size_t bx = 0; bx < tilesX; ++bx) {
                            const size_t xBegin = 1 + bx * tileX;
                            const size_t xEnd = std::min(xBegin + tileX, lastX);
                            const size_t yBegin = 1 + by * tileY;
                            const size_t yEnd = std::min(yBegin + tileY, interiorY + 1);
                            const size_t zBegin = 1 + bz * tileZ;
                            const size_t zEnd = std::min(zBegin + tileZ, interiorZ + 1);
                            for (size_t z = zBegin; z < zEnd; ++z) {
                                for (size_t y = yBegin; y < yEnd; ++y) {
                                    stencilRow(source, destination, z * plane + y * nx,
                                               nx, plane, xBegin, xEnd);
                                }
                            }
                        }
                    }
                }
            } else {
                #pragma omp for collapse(2) schedule(static)
                for (size_t z = 0; z < interiorZ; ++z) {
                    for (size_t y = 0; y < interiorY; ++y) {
                        stencilRow(source, destination, (z + 1) * plane + (y + 1) * nx,
                                   nx, plane, 1, lastX);
                    }
                }
            }
            // The worksharing barrier completes this step before any thread
            // reads the new input. Pointer swaps are private to each thread.
            std::swap(input, output);
        }
    }
}

bool validateResult(std::span<const Real> grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // Check finiteness and bounds in a single parallel pass.
    int invalid = 0;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for simd schedule(static) \
        reduction(|:invalid) reduction(min:minVal) reduction(max:maxVal)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        const Real val = grid[idx];
        invalid |= !std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // Boundary values remain at their initial values in both buffers.
    
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
    
    size_t gridSize = nx * ny * nz;
    
    // Avoid serial value initialization so that the parallel initialization
    // determines page placement for both buffers.
    auto storage1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto storage2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    std::span<Real> grid1(storage1.get(), gridSize);
    std::span<Real> grid2(storage2.get(), gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1, grid2, iterations > 0);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    stencilIterations(grid1, grid2, nx, ny, nz, iterations);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::span<const Real> finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, nx, ny, nz);
        
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
