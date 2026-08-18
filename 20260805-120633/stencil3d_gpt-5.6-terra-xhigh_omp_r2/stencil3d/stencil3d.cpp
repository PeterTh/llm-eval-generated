#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

void initializeGrid(std::vector<Real>& grid) {
    const size_t gridSize = grid.size();

    #pragma omp parallel for simd schedule(static)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        grid[idx] = static_cast<Real>(idx % 19);
    }
}

// Run all time steps inside one OpenMP team.  Each worksharing loop writes a
// disjoint part of the output grid, and the final barrier makes a completed
// output grid the input for the next iteration.
void stencilIterations(std::vector<Real>& grid1,
                       std::vector<Real>& grid2,
                       const size_t nx,
                       const size_t ny,
                       const size_t nz,
                       const int iterations) {
    const size_t planeSize = nx * ny;
    const size_t xInteriorEnd = nx > 1 ? nx - 1 : 1;
    const size_t yInteriorEnd = ny > 1 ? ny - 1 : 1;
    const size_t zInteriorEnd = nz > 1 ? nz - 1 : 1;

    #pragma omp parallel default(none) shared(grid1, grid2, nx, ny, nz, iterations, planeSize, xInteriorEnd, yInteriorEnd, zInteriorEnd)
    {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* const input = (iter % 2 == 0) ? grid1.data() : grid2.data();
            Real* const output = (iter % 2 == 0) ? grid2.data() : grid1.data();

            // The contiguous X direction is SIMD-vectorized.  Static
            // scheduling keeps each thread's rows contiguous for locality.
            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 1; z < zInteriorEnd; ++z) {
                for (size_t y = 1; y < yInteriorEnd; ++y) {
                    const size_t rowStart = z * planeSize + y * nx;

                    #pragma omp simd
                    for (size_t x = 1; x < xInteriorEnd; ++x) {
                        const size_t idx = rowStart + x;
                        output[idx] = (input[idx] + input[idx - 1] + input[idx + 1]
                                       + input[idx - nx] + input[idx + nx]
                                       + input[idx - planeSize] + input[idx + planeSize]) / 7.0;
                    }
                }
            }

            // Copy the six boundary faces without scanning every interior
            // cell.  The face ranges are non-overlapping; the remaining
            // barrier also establishes the time-step dependency.
            if (nz != 0) {
                #pragma omp for schedule(static) nowait
                for (size_t offset = 0; offset < planeSize; ++offset) {
                    output[offset] = input[offset];
                    output[(nz - 1) * planeSize + offset] = input[(nz - 1) * planeSize + offset];
                }
            }

            if (zInteriorEnd > 1 && ny != 0) {
                #pragma omp for collapse(2) schedule(static) nowait
                for (size_t z = 1; z < zInteriorEnd; ++z) {
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t first = z * planeSize + x;
                        const size_t last = first + (ny - 1) * nx;
                        output[first] = input[first];
                        output[last] = input[last];
                    }
                }
            }

            if (zInteriorEnd > 1 && yInteriorEnd > 1 && nx != 0) {
                #pragma omp for collapse(2) schedule(static) nowait
                for (size_t z = 1; z < zInteriorEnd; ++z) {
                    for (size_t y = 1; y < yInteriorEnd; ++y) {
                        const size_t first = z * planeSize + y * nx;
                        const size_t last = first + nx - 1;
                        output[first] = input[first];
                        output[last] = input[last];
                    }
                }
            }

            #pragma omp barrier
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    int hasInvalidValue = 0;
    Real minVal = std::numeric_limits<Real>::max();
    Real maxVal = std::numeric_limits<Real>::lowest();
    const size_t gridSize = grid.size();

    #pragma omp parallel for reduction(|:hasInvalidValue) reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        const Real val = grid[idx];
        hasInvalidValue |= std::isnan(val) || std::isinf(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    // 1. No NaN or Inf values
    if (hasInvalidValue != 0) {
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
