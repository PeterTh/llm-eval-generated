#include <algorithm>
#include <cinttypes>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>

#include <omp.h>

#include "../common/hash.hpp"

using Real = double;

void initializeGrids(Real* const grid1, Real* const grid2, const size_t gridSize) {
    // Besides making initialization scalable, this gives both grids parallel
    // first-touch placement on NUMA machines.
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        grid1[idx] = (idx % 19) * 1.0;
        grid2[idx] = 0.0;
    }
}

// 7-point stencil computation
void runStencil(Real* const __restrict__ grid1,
                Real* const __restrict__ grid2,
                const size_t nx, const size_t ny, const size_t nz,
                const int iterations) {
    const size_t plane = nx * ny;
    const size_t yEnd = ny - 1;
    const size_t zEnd = nz - 1;

    // Small Y/Z tiles keep the three input-plane bands in cache while still
    // exposing many independent tiles to the OpenMP team. X stays contiguous
    // so the compiler can use full-width SIMD loads and stores.
    constexpr size_t tileY = 8;
    constexpr size_t tileZ = 8;

    // Keep one team alive across all time steps. The implicit barrier at the
    // end of each workshare is the required dependency between time steps.
    #pragma omp parallel default(none) proc_bind(spread) \
            shared(grid1, grid2, nx, ny, nz, plane, yEnd, zEnd, iterations)
    {
        // Boundary values are invariant. Populate the second buffer once;
        // repeated boundary copies in later iterations would write the same
        // values and only consume memory bandwidth.
        if (iterations > 0) {
            #pragma omp for schedule(static) nowait
            for (size_t z = 0; z < nz; ++z) {
                const size_t zOffset = z * plane;
                if (z == 0 || z + 1 == nz) {
                    std::copy_n(grid1 + zOffset, plane, grid2 + zOffset);
                } else {
                    std::copy_n(grid1 + zOffset, nx, grid2 + zOffset);
                    std::copy_n(grid1 + zOffset + (ny - 1) * nx, nx,
                                grid2 + zOffset + (ny - 1) * nx);
                    for (size_t y = 1; y < yEnd; ++y) {
                        const size_t row = zOffset + y * nx;
                        grid2[row] = grid1[row];
                        grid2[row + nx - 1] = grid1[row + nx - 1];
                    }
                }
            }
        }

        for (int iter = 0; iter < iterations; ++iter) {
            const Real* const input = (iter & 1) == 0 ? grid1 : grid2;
            Real* const output = (iter & 1) == 0 ? grid2 : grid1;

            #pragma omp for collapse(2) schedule(static)
            for (size_t zBlock = 1; zBlock < zEnd; zBlock += tileZ) {
                for (size_t yBlock = 1; yBlock < yEnd; yBlock += tileY) {
                    const size_t zStop = std::min(zBlock + tileZ, zEnd);
                    const size_t yStop = std::min(yBlock + tileY, yEnd);

                    for (size_t z = zBlock; z < zStop; ++z) {
                        const size_t zOffset = z * plane;
                        for (size_t y = yBlock; y < yStop; ++y) {
                            const size_t row = zOffset + y * nx;

                            #pragma omp simd
                            for (size_t x = 1; x < nx - 1; ++x) {
                                const size_t idx = row + x;
                                output[idx] = (input[idx] + input[idx - 1] +
                                               input[idx + 1] + input[idx - nx] +
                                               input[idx + nx] + input[idx - plane] +
                                               input[idx + plane]) / 7.0;
                            }
                        }
                    }
                }
            }
        }
    }
}

void printGridResults(const Real* const grid, const size_t gridSize) {
    if (gridSize == 0) {
        printf("=== RESULTS ===\n");
        printf("Name: Grid\n");
        printf("Elements: 0\n");
        printf("=== END RESULTS ===\n");
        return;
    }

    Real sum = 0.0;
    Real correction = 0.0;
    for (size_t idx = 0; idx < gridSize; ++idx) {
        const Real adjusted = grid[idx] - correction;
        const Real next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }

    Real minVal = grid[0];
    Real maxVal = grid[0];
    hash resultHash;
    for (size_t idx = 0; idx < gridSize; ++idx) {
        minVal = std::min(minVal, grid[idx]);
        maxVal = std::max(maxVal, grid[idx]);
        resultHash.add(grid[idx]);
    }

    const size_t indices[5] = {0, gridSize / 4, gridSize / 2,
                               3 * gridSize / 4, gridSize - 1};
    printf("=== RESULTS ===\n");
    printf("Name: Grid\n");
    printf("Elements: %zu\n", gridSize);
    printf("Sum: %.17e\n", static_cast<double>(sum));
    printf("Min: %.17e\n", static_cast<double>(minVal));
    printf("Max: %.17e\n", static_cast<double>(maxVal));
    for (const size_t idx : indices) {
        printf("Sample[%zu]: %.17e\n", idx, static_cast<double>(grid[idx]));
    }
    printf("Hash: %016" PRIx64 "\n", resultHash.get());
    printf("=== END RESULTS ===\n");
}

bool validateResult(const Real* const grid, const size_t gridSize,
                    [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
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
    for (size_t idx = 0; idx < gridSize; ++idx) {
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
    // (they are copied, so they should be close to initial values)
    
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
    // Honor the requested thread count and never let the runtime silently
    // shrink the OpenMP team between runs.
    omp_set_dynamic(0);

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
    
    // Allocate grids (double buffering)
    // Default-initialized arrays avoid serially touching every page before
    // the parallel initializer can distribute them across NUMA nodes.
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
        printGridResults(finalGrid, gridSize);
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, gridSize, nx, ny, nz);
        
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
