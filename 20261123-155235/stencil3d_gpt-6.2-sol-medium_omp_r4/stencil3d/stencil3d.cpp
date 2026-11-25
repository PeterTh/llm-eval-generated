#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

size_t stencilTileWidth(const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) {
        return std::max(nx, size_t{1});
    }

    // Split long rows only when there are too few interior rows for the team.
    const size_t rows = (ny - 2) * (nz - 2);
    const size_t threads = omp_get_max_threads();
    if (rows >= threads) {
        return nx;
    }
    const size_t tilesPerRow = (threads + rows - 1) / rows;
    const size_t width = (nx + tilesPerRow - 1) / tilesPerRow;
    constexpr size_t minTileWidth = 256;
    return std::max(minTileWidth, width);
}

void initializeGrids(Real* first, Real* second,
                     const size_t nx, const size_t ny, const size_t nz,
                     const size_t tileWidth, const bool zeroSecond) {
    // First-touch both buffers in parallel. The stencil leaves both halos intact.
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t block = 0; block < nx; block += tileWidth) {
                const size_t row = (z * ny + y) * nx;
                const size_t end = std::min(block + tileWidth, nx);
                for (size_t x = block; x < end; ++x) {
                    first[row + x] = static_cast<Real>((row + x) % 19);
                }

                const bool boundaryRow = z == 0 || z == nz - 1 || y == 0 || y == ny - 1;
                if (!zeroSecond && boundaryRow) {
                    std::copy_n(first + row + block, end - block, second + row + block);
                } else {
                    std::fill_n(second + row + block, end - block, Real{0});
                    if (!zeroSecond && block == 0) {
                        second[row] = first[row];
                    }
                    if (!zeroSecond && end == nx) {
                        second[row + nx - 1] = first[row + nx - 1];
                    }
                }
            }
        }
    }
}

void runStencil(Real* first, Real* second, const size_t nx, const size_t ny,
                const size_t nz, const size_t tileWidth, const int iterations) {
    if (nx < 3 || ny < 3 || nz < 3 || iterations <= 0) {
        return;
    }

    const size_t plane = nx * ny;
    #pragma omp parallel
    {
        Real* input = first;
        Real* output = second;
        for (int iter = 0; iter < iterations; ++iter) {
            #pragma omp for collapse(3) schedule(static)
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    for (size_t block = 0; block < nx; block += tileWidth) {
                        const size_t row = z * plane + y * nx;
                        const Real* center = input + row;
                        const Real* front = center - nx;
                        const Real* back = center + nx;
                        const Real* bottom = center - plane;
                        const Real* top = center + plane;
                        Real* next = output + row;
                        const size_t begin = std::max(block, size_t{1});
                        const size_t end = std::min(block + tileWidth, nx - 1);

                        #pragma omp simd
                        for (size_t x = begin; x < end; ++x) {
                            next[x] = (center[x] + center[x - 1] + center[x + 1] +
                                       front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                        }
                    }
                }
            }
            // The worksharing barrier completes all writes before the next step.
            std::swap(input, output);
        }
    }
}

bool validateResult(const Real* grid, const size_t size) {
    if (size == 0) {
        printf("Validation failed: grid is empty\n");
        return false;
    }

    Real minVal = std::numeric_limits<Real>::infinity();
    Real maxVal = -std::numeric_limits<Real>::infinity();
    int nonFinite = 0;
    #pragma omp parallel for simd schedule(static) reduction(min : minVal) reduction(max : maxVal) reduction(| : nonFinite)
    for (size_t idx = 0; idx < size; ++idx) {
        const Real val = grid[idx];
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
    
    // Allocate grids (double buffering)
    std::unique_ptr<Real[]> grid1(new Real[gridSize]);
    std::unique_ptr<Real[]> grid2(new Real[gridSize]);
    const bool finalInFirst = iterations % 2 == 0;
    const size_t tileWidth = stencilTileWidth(nx, ny, nz);
    
    // Initialize
    printf("Initializing grid...\n");
    // A negative odd iteration count selects the untouched, zero-filled buffer.
    const bool zeroSecond = iterations < 0 && !finalInFirst;
    initializeGrids(grid1.get(), grid2.get(), nx, ny, nz, tileWidth, zeroSecond);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    const auto start = std::chrono::steady_clock::now();
    
    runStencil(grid1.get(), grid2.get(), nx, ny, nz, tileWidth, iterations);
    
    const auto end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    const double cellUpdates = nx > 2 && ny > 2 && nz > 2 && iterations > 0
        ? static_cast<double>((nx - 2) * (ny - 2) * (nz - 2)) * iterations
        : 0.0;
    const double seconds = std::chrono::duration<double>(end - start).count();
    const double mcups = seconds > 0 ? cellUpdates / seconds / 1e6 : 0.0;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const Real* finalGrid = finalInFirst ? grid1.get() : grid2.get();
    if (printResults) {
        // The shared results formatter takes a vector. Release the unused buffer
        // before making the copy so peak storage stays at two grids.
        (finalInFirst ? grid2 : grid1).reset();
        print_results(std::vector<Real>(finalGrid, finalGrid + gridSize), "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, gridSize);
        
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
