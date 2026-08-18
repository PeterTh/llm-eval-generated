#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// Initialize both grids in parallel so their pages are first-touched by the
// threads that will subsequently process them. The second grid starts at zero,
// matching value-initialized vector storage in the original implementation.
// Its invariant boundary is copied once instead of on every stencil step.
void initializeGrids(Real* const grid1, Real* const grid2,
                     const size_t gridSize, const size_t nx, const size_t ny,
                     const size_t nz, const bool copyBoundary) {
    const size_t planeSize = nx * ny;

    #pragma omp parallel default(none) proc_bind(spread) shared(grid1, grid2, gridSize, nx, ny, nz, planeSize, copyBoundary)
    {
        #pragma omp for simd schedule(static)
        for (size_t idx = 0; idx < gridSize; ++idx) {
            grid1[idx] = (idx % 19) * 1.0;
            grid2[idx] = 0.0;
        }

        if (copyBoundary) {
            // The two z faces.
            #pragma omp for schedule(static)
            for (size_t offset = 0; offset < planeSize; ++offset) {
                grid2[offset] = grid1[offset];
                if (nz > 1) {
                    const size_t opposite = (nz - 1) * planeSize + offset;
                    grid2[opposite] = grid1[opposite];
                }
            }

            // The two y faces, excluding the z faces already copied.
            if (nz > 2 && ny > 0) {
                #pragma omp for schedule(static)
                for (size_t z = 1; z < nz - 1; ++z) {
                    const size_t firstRow = z * planeSize;
                    const size_t lastRow = firstRow + (ny - 1) * nx;
                    #pragma omp simd
                    for (size_t x = 0; x < nx; ++x) {
                        grid2[firstRow + x] = grid1[firstRow + x];
                        grid2[lastRow + x] = grid1[lastRow + x];
                    }
                }
            }

            // The two x faces, excluding all faces already copied.
            if (nz > 2 && ny > 2 && nx > 0) {
                #pragma omp for collapse(2) schedule(static)
                for (size_t z = 1; z < nz - 1; ++z) {
                    for (size_t y = 1; y < ny - 1; ++y) {
                        const size_t first = z * planeSize + y * nx;
                        const size_t last = first + nx - 1;
                        grid2[first] = grid1[first];
                        grid2[last] = grid1[last];
                    }
                }
            }
        }
    }
}

// All threads in the persistent team call this routine. Restrict-qualified
// pointers expose the non-aliasing double buffers to the vectorizer.
inline void stencilStep(const Real* __restrict__ input,
                        Real* __restrict__ output, const size_t nx,
                        const size_t ny, const size_t nz,
                        const size_t planeSize) {
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            const size_t row = z * planeSize + y * nx;
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = row + x;
                const Real center = input[idx];
                const Real left = input[idx - 1];
                const Real right = input[idx + 1];
                const Real front = input[idx - nx];
                const Real back = input[idx + nx];
                const Real bottom = input[idx - planeSize];
                const Real top = input[idx + planeSize];

                output[idx] =
                    (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

// Keep one OpenMP team alive for the complete time-stepping loop. The implicit
// barrier at the end of each omp-for is the only synchronization each Jacobi
// iteration requires.
void stencilIterations(Real* const grid1, Real* const grid2, const size_t nx,
                       const size_t ny, const size_t nz,
                       const int iterations) {
    if (iterations <= 0 || nx <= 2 || ny <= 2 || nz <= 2) {
        return;
    }

    const size_t planeSize = nx * ny;
    #pragma omp parallel default(none) proc_bind(spread) shared(grid1, grid2, nx, ny, nz, planeSize, iterations)
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if ((iter & 1) == 0) {
                stencilStep(grid1, grid2, nx, ny, nz, planeSize);
            } else {
                stencilStep(grid2, grid1, nx, ny, nz, planeSize);
            }
        }
    }
}

bool validateResult(const Real* const grid, const size_t gridSize,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
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
    
    // Allocate uninitialized double buffers. Parallel first-touch in
    // initializeGrids distributes their pages across NUMA nodes.
    std::unique_ptr<Real[]> grid1(new Real[gridSize]);
    std::unique_ptr<Real[]> grid2(new Real[gridSize]);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1.get(), grid2.get(), gridSize, nx, ny, nz,
                    iterations > 0);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    const double start = omp_get_wtime();
    
    stencilIterations(grid1.get(), grid2.get(), nx, ny, nz, iterations);
    
    const double duration = omp_get_wtime() - start;
    const long durationMs = static_cast<long>(duration * 1000.0);
    
    printf("Computation time: %ld ms\n", durationMs);
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / duration / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const Real* const finalGrid =
        (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        // The common reporting helper accepts a vector. This copy is only made
        // on the explicitly requested reporting path and is outside the timed
        // stencil region.
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
