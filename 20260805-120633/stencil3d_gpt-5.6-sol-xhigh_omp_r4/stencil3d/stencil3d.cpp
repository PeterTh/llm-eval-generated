#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrids(Real* const grid1, Real* const grid2, const size_t gridSize) {
    // Besides parallelizing initialization, this gives both grids a parallel
    // first touch, which is important for NUMA placement on multisocket CPUs.
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        grid1[idx] = static_cast<Real>(idx % 19);
        grid2[idx] = 0.0;
    }
}

// Run all iterations in one persistent OpenMP team. Rows are partitioned as
// contiguous blocks so every grid shape scales without a division in the hot
// row loop and so each thread keeps the same NUMA-local region each iteration.
void runStencil(Real* const grid1, Real* const grid2,
                const size_t nx, const size_t ny, const size_t nz,
                const int iterations) {
    if (iterations <= 0) {
        return;
    }

    // If any dimension has no interior, each iteration only copies boundaries.
    // Since boundary values are invariant, one parallel copy is equivalent.
    if (nx < 3 || ny < 3 || nz < 3) {
        const size_t gridSize = nx * ny * nz;
        #pragma omp parallel for schedule(static) proc_bind(spread)
        for (size_t idx = 0; idx < gridSize; ++idx) {
            grid2[idx] = grid1[idx];
        }
        return;
    }

    const size_t planeSize = nx * ny;
    const size_t interiorY = ny - 2;
    const size_t interiorRows = (nz - 2) * interiorY;

    #pragma omp parallel default(none) proc_bind(spread) \
        firstprivate(grid1, grid2, nx, ny, nz, iterations, planeSize, interiorY, interiorRows)
    {
        // Boundaries never change. Copy them to the second buffer once instead
        // of scanning the entire volume after every stencil iteration.
        #pragma omp for schedule(static) nowait
        for (size_t z = 0; z < nz; ++z) {
            const size_t plane = z * planeSize;
            if (z == 0 || z + 1 == nz) {
                std::copy_n(grid1 + plane, planeSize, grid2 + plane);
            } else {
                std::copy_n(grid1 + plane, nx, grid2 + plane);
                std::copy_n(grid1 + plane + (ny - 1) * nx,
                            nx, grid2 + plane + (ny - 1) * nx);

                for (size_t y = 1; y + 1 < ny; ++y) {
                    const size_t row = plane + y * nx;
                    grid2[row] = grid1[row];
                    grid2[row + nx - 1] = grid1[row + nx - 1];
                }
            }
        }

        const size_t thread = static_cast<size_t>(omp_get_thread_num());
        const size_t threadCount = static_cast<size_t>(omp_get_num_threads());
        const size_t firstRow = (interiorRows * thread) / threadCount;
        const size_t lastRow = (interiorRows * (thread + 1)) / threadCount;

        for (int iter = 0; iter < iterations; ++iter) {
            const Real* __restrict__ const input = (iter & 1) ? grid2 : grid1;
            Real* __restrict__ const output = (iter & 1) ? grid1 : grid2;

            size_t z = firstRow / interiorY + 1;
            size_t y = firstRow % interiorY + 1;

            for (size_t rowIndex = firstRow; rowIndex < lastRow; ++rowIndex) {
                const size_t row = z * planeSize + y * nx;
                const Real* __restrict__ const in = input + row;
                Real* __restrict__ const out = output + row;

                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    out[x] = (in[x] + in[x - 1] + in[x + 1]
                            + in[x - nx] + in[x + nx]
                            + in[x - planeSize] + in[x + planeSize]) / 7.0;
                }

                if (++y + 1 == ny) {
                    y = 1;
                    ++z;
                }
            }

            // The next iteration reads the buffer just written. The parallel
            // region's final barrier supplies synchronization after the last.
            if (iter + 1 < iterations) {
                #pragma omp barrier
            }
        }
    }
}

bool validateResult(const Real* const grid, const size_t gridSize,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    int hasInvalid = 0;
    Real minVal = grid[0];
    Real maxVal = grid[0];

    #pragma omp parallel for schedule(static) proc_bind(spread) reduction(| : hasInvalid) \
        reduction(min : minVal) reduction(max : maxVal)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        const Real val = grid[idx];
        hasInvalid |= !std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (hasInvalid != 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
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

    // Keep the requested OpenMP team size stable across all parallel regions.
    omp_set_dynamic(0);
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering)
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
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    const double elapsedSeconds = std::chrono::duration<double>(end - start).count();
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / elapsedSeconds / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const Real* const finalGrid = (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        // Keep the established external results format unchanged.
        const std::vector<Real> result(finalGrid, finalGrid + gridSize);
        print_results(result, "Grid");
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
