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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Initializes the grid in parallel. The row partitioning matches the stencil sweep's
// worksharing schedule, so each page is first touched by the thread that later uses it
// (NUMA-friendly).
void initializeGrid(Real* grid, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel for collapse(2) schedule(static) proc_bind(spread)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// One 7-point stencil sweep over the whole grid. Must be called by all threads of an
// enclosing parallel region (the worksharing loop below distributes the work).
// Interior points get the averaging stencil; boundary points are copied from input.
// Each (z, y) row is handled by exactly one thread, so writes never overlap.
inline void stencilSweep(const Real* __restrict input,
                         Real* __restrict output,
                         const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0) return;
    const size_t plane = nx * ny;

    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = z * plane + y * nx;

            if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                // Entire row lies on a domain boundary: copy it.
                std::copy(input + row, input + row + nx, output + row);
                continue;
            }

            const Real* cur = input + row;
            const Real* ym = cur - nx;      // y-1
            const Real* yp = cur + nx;      // y+1
            const Real* zm = cur - plane;   // z-1
            const Real* zp = cur + plane;   // z+1
            Real* dst = output + row;

            // x boundaries
            dst[0] = cur[0];
            dst[nx - 1] = cur[nx - 1];

            // Interior: same operation order as the original code
            for (size_t x = 1; x + 1 < nx; ++x) {
                dst[x] = (cur[x] + cur[x - 1] + cur[x + 1] + ym[x] + yp[x] + zm[x] + zp[x]) / 7.0;
            }
        }
    }
}

// Runs all stencil iterations inside a single parallel region, so threads are
// created once and synchronized with the implicit barrier at the end of each sweep.
void runStencil(Real* a, Real* b,
                const size_t nx, const size_t ny, const size_t nz, const int iterations) {
    // Threads are pinned (spread across cores/sockets) so each thread keeps touching the
    // same memory pages every sweep; this matches the first-touch placement in initializeGrid.
    #pragma omp parallel proc_bind(spread)
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilSweep(a, b, nx, ny, nz);
            } else {
                stencilSweep(b, a, nx, ny, nz);
            }
        }
    }
}

bool validateResult(const Real* g, const size_t n) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    long long nonFinite = 0;
    #pragma omp parallel for reduction(+ : nonFinite) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(g[i]) || std::isinf(g[i])) {
            ++nonFinite;
        }
    }
    if (nonFinite > 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = g[0];
    Real maxVal = g[0];
    #pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, g[i]);
        maxVal = std::max(maxVal, g[i]);
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
    
    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;
    
    // Allocate grids (double buffering). Left uninitialized here so that pages are
    // first touched in parallel by initializeGrid/stencil sweeps (NUMA-friendly).
    std::unique_ptr<Real[]> grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    std::unique_ptr<Real[]> grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1.get(), nx, ny, nz);
    
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
    const Real* finalGrid = (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        // Copy into a vector only for reporting (outside the timed region)
        const std::vector<Real> resultVec(finalGrid, finalGrid + gridSize);
        print_results(resultVec, "Grid");
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
