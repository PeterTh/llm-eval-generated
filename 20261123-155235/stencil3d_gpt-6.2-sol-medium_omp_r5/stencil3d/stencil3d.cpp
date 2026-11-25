#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

struct TileLayout {
    size_t yCount;
    size_t zCount;
};

TileLayout chooseTiles(size_t nx, size_t ny, size_t nz, int threadCount) {
    // Limit each tile's strip of a plane so adjacent strips in z can stay in cache.
    constexpr size_t stripBytes = 96 * 1024;
    constexpr size_t maxPlanesPerTile = 32;
    const size_t tileRows = std::max(size_t{1}, stripBytes / sizeof(Real) / nx);
    const size_t targetTiles = 4 * static_cast<size_t>(threadCount);
    const size_t yCount = std::min(ny, 1 + std::max((ny - 1) / tileRows, (targetTiles - 1) / nz));
    const size_t zCount = std::min(nz, 1 + std::max((nz - 1) / maxPlanesPerTile,
                                                   (targetTiles - 1) / yCount));
    return {yCount, zCount};
}

size_t tileBoundary(size_t tile, size_t count, size_t extent) {
    return tile * (extent / count) + std::min(tile, extent % count);
}

void initializeGrids(Real* grid1, Real* grid2, size_t nx, size_t ny, size_t nz,
                     TileLayout tiles, int iterations, int threadCount) {
    // First-touch both buffers with the same tile distribution as the stencil.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threadCount)
    for (size_t tz = 0; tz < tiles.zCount; ++tz) {
        for (size_t ty = 0; ty < tiles.yCount; ++ty) {
            const size_t zBegin = tileBoundary(tz, tiles.zCount, nz);
            const size_t zEnd = tileBoundary(tz + 1, tiles.zCount, nz);
            const size_t yBegin = tileBoundary(ty, tiles.yCount, ny);
            const size_t yEnd = tileBoundary(ty + 1, tiles.yCount, ny);
            for (size_t z = zBegin; z < zEnd; ++z) {
                for (size_t y = yBegin; y < yEnd; ++y) {
                    const size_t row = (z * ny + y) * nx;
                    // Seed the fixed boundaries in both buffers. A negative
                    // iteration count can select the original zeroed buffer.
                    #pragma omp simd
                    for (size_t x = 0; x < nx; ++x) {
                        const Real value = static_cast<Real>((row + x) % 19);
                        grid1[row + x] = value;
                        grid2[row + x] = iterations < 0 ? 0.0 : value;
                    }
                }
            }
        }
    }
}

void runStencil(Real* grid1, Real* grid2, size_t nx, size_t ny, size_t nz,
                TileLayout tiles, int iterations, int threadCount) {
    if (iterations <= 0 || nx < 3 || ny < 3 || nz < 3) {
        return;
    }

    const size_t plane = nx * ny;
    #pragma omp parallel num_threads(threadCount)
    {
        Real* input = grid1;
        Real* output = grid2;
        for (int iter = 0; iter < iterations; ++iter) {
            #pragma omp for collapse(2) schedule(static)
            for (size_t tz = 0; tz < tiles.zCount; ++tz) {
                for (size_t ty = 0; ty < tiles.yCount; ++ty) {
                    const size_t zBegin = std::max(size_t{1}, tileBoundary(tz, tiles.zCount, nz));
                    const size_t zEnd = std::min(nz - 1, tileBoundary(tz + 1, tiles.zCount, nz));
                    const size_t yBegin = std::max(size_t{1}, tileBoundary(ty, tiles.yCount, ny));
                    const size_t yEnd = std::min(ny - 1, tileBoundary(ty + 1, tiles.yCount, ny));
                    for (size_t z = zBegin; z < zEnd; ++z) {
                        for (size_t y = yBegin; y < yEnd; ++y) {
                            const size_t row = z * plane + y * nx;
                            const Real* center = input + row;
                            const Real* front = input + (row - nx);
                            const Real* back = input + (row + nx);
                            const Real* bottom = input + (row - plane);
                            const Real* top = input + (row + plane);
                            Real* result = output + row;

                            #pragma omp simd
                            for (size_t x = 1; x < nx - 1; ++x) {
                                result[x] = (center[x] + center[x - 1] + center[x + 1] +
                                             front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                            }
                        }
                    }
                }
            }
            // The worksharing barrier completes all writes before the next read.
            std::swap(input, output);
        }
    }
}

bool validateResult(const Real* grid, size_t gridSize, int threadCount) {
    if (gridSize == 0) {
        printf("Validation failed: empty grid\n");
        return false;
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    int nonFinite = 0;
    #pragma omp parallel for simd schedule(static) num_threads(threadCount) \
        reduction(min:minVal) reduction(max:maxVal) reduction(|:nonFinite)
    for (size_t i = 0; i < gridSize; ++i) {
        const Real val = grid[i];
        nonFinite |= !std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (nonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
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
    // Keep enough work per thread to amortize dispatch and iteration barriers.
    constexpr size_t cellsPerThread = 65536;
    const int threadCount = static_cast<int>(std::min({
        static_cast<size_t>(omp_get_max_threads()), std::max(size_t{1}, ny * nz),
        1 + gridSize / cellsPerThread
    }));
    const TileLayout tiles = gridSize == 0 ? TileLayout{1, 1} : chooseTiles(nx, ny, nz, threadCount);

    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    if (gridSize != 0) {
        initializeGrids(grid1.get(), grid2.get(), nx, ny, nz, tiles, iterations, threadCount);
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::steady_clock::now();
    runStencil(grid1.get(), grid2.get(), nx, ny, nz, tiles, iterations, threadCount);
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double cellUpdates = (nx > 2 && ny > 2 && nz > 2 && iterations > 0)
        ? static_cast<double>(nx - 2) * (ny - 2) * (nz - 2) * iterations
        : 0.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const Real* finalGrid = (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        print_results(finalGrid, gridSize, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, gridSize, threadCount);
        
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
