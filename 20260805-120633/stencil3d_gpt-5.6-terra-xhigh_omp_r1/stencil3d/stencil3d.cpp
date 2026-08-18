#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

void initializeGrid(std::vector<Real>& grid) {
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Run all time steps within one OpenMP parallel region.  Keeping the worker
// team alive avoids per-iteration thread startup overhead, while distributing
// whole rows preserves contiguous accesses in the innermost loop.
void stencilIterations(std::vector<Real>& grid1,
                       std::vector<Real>& grid2,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int iterations) {
    const size_t planeSize = nx * ny;

    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* const input = (iter % 2 == 0) ? grid1.data() : grid2.data();
            Real* const output = (iter % 2 == 0) ? grid2.data() : grid1.data();

            // With a thin dimension every cell is a boundary cell.
            if (nx < 3 || ny < 3 || nz < 3) {
                #pragma omp for schedule(static)
                for (size_t idx = 0; idx < grid1.size(); ++idx) {
                    output[idx] = input[idx];
                }
                continue;
            }

            // The three face loops cover each boundary cell exactly once.
            // They can run concurrently with the interior stencil because
            // their output regions are disjoint from it.
            #pragma omp for schedule(static) nowait
            for (size_t idx = 0; idx < planeSize; ++idx) {
                output[idx] = input[idx];
                const size_t topIdx = (nz - 1) * planeSize + idx;
                output[topIdx] = input[topIdx];
            }

            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t row = z * planeSize + x;
                    output[row] = input[row];
                    output[row + (ny - 1) * nx] = input[row + (ny - 1) * nx];
                }
            }

            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t row = z * planeSize + y * nx;
                    output[row] = input[row];
                    output[row + nx - 1] = input[row + nx - 1];
                }
            }

            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t row = z * planeSize + y * nx;

                    #pragma omp simd
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = row + x;
                        output[idx] = (input[idx] + input[idx - 1] + input[idx + 1]
                                     + input[idx - nx] + input[idx + nx]
                                     + input[idx - planeSize] + input[idx + planeSize]) / 7.0;
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    int hasInvalidValue = 0;
    #pragma omp parallel for reduction(|:hasInvalidValue) schedule(static)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        const Real val = grid[idx];
        if (std::isnan(val) || std::isinf(val)) {
            hasInvalidValue = 1;
        }
    }
    if (hasInvalidValue != 0) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        minVal = std::min(minVal, grid[idx]);
        maxVal = std::max(maxVal, grid[idx]);
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
    
    // Allocate grids (double buffering)
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1);
    
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
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
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
