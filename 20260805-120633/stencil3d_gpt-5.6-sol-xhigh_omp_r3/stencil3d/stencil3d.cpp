#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

void initializeGrids(Real* const grid1, Real* const grid2, const size_t gridSize) {
    // Initialize in parallel so the pages are first-touched by the threads that
    // will subsequently process them.  Both grids start with identical boundary
    // values; stencil iterations only need to write the interior thereafter.
#pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        const Real value = static_cast<Real>(idx % 19);
        grid1[idx] = value;
        grid2[idx] = value;
    }
}

// Run all iterations in one parallel region.  The implicit barrier at the end
// of each worksharing loop is required before the buffers exchange roles.
void runStencil(Real* const grid1, Real* const grid2,
                const size_t nx, const size_t ny, const size_t nz,
                const int iterations) {
    if (iterations <= 0 || nx < 3 || ny < 3 || nz < 3) {
        return;
    }

    const size_t planeStride = nx * ny;

#pragma omp parallel default(none) shared(grid1, grid2, nx, ny, nz, iterations, planeStride)
    {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* const input = (iter % 2 == 0) ? grid1 : grid2;
            Real* const output = (iter % 2 == 0) ? grid2 : grid1;

            // Collapsing z and y exposes enough independent, contiguous rows
            // to scale even when the grid has relatively few z planes.
#pragma omp for collapse(2) schedule(static)
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t rowStart = z * planeStride + y * nx;

                    // x is unit-stride, allowing packed loads and stores while
                    // retaining the original per-cell arithmetic order.
#pragma omp simd
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = rowStart + x;
                        const Real center = input[idx];
                        const Real left = input[idx - 1];
                        const Real right = input[idx + 1];
                        const Real front = input[idx - nx];
                        const Real back = input[idx + nx];
                        const Real bottom = input[idx - planeStride];
                        const Real top = input[idx + planeStride];

                        output[idx] =
                            (center + left + right + front + back + bottom + top) / 7.0;
                    }
                }
            }
        }
    }
}

bool validateResult(const Real* const grid, const size_t gridSize) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (size_t idx = 0; idx < gridSize; ++idx) {
        const Real val = grid[idx];
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (size_t idx = 1; idx < gridSize; ++idx) {
        const Real val = grid[idx];
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    // 3. Boundary values should not change significantly
    // (both buffers retain their identical initialized boundary values)
    
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
    
    // Allocate uninitialized grids.  Avoiding serial value-initialization lets
    // the parallel initializer establish useful NUMA page placement.
    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1.get(), grid2.get(), gridSize);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runStencil(grid1.get(), grid2.get(), nx, ny, nz, iterations);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const Real* const finalGrid = (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        // Result reporting is outside the timed region and expects a vector.
        const std::vector<Real> result(finalGrid, finalGrid + gridSize);
        print_results(result, "Grid");
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
