#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t gridSize = nx * ny * nz;

    // The initialization is independent for every element and is also
    // included in the parallel work so large grids do not start serially.
    #pragma omp parallel for simd schedule(static)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        grid[idx] = (idx % 19) * 1.0;
    }
}

// Work for one 7-point stencil iteration. This function is called by every
// thread in the surrounding parallel region, allowing the iteration loop to
// reuse one OpenMP team rather than creating a team for every iteration.
void stencilIterationWork(const std::vector<Real>& input,
                          std::vector<Real>& output,
                          const size_t nx, const size_t ny, const size_t nz) {
    const size_t planeSize = nx * ny;
    const size_t gridSize = planeSize * nz;

    // Process interior points (not on boundaries). The x dimension is
    // contiguous, which makes each row cache-friendly and vectorizable.
    if (nx > 2 && ny > 2 && nz > 2) {
        #pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t row = z * planeSize + y * nx;

                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t idx = row + x;

                    // Keep the original operation and summation order for
                    // equivalent floating-point results.
                    output[idx] = (input[idx] +
                                   input[idx - 1] + input[idx + 1] +
                                   input[idx - nx] + input[idx + nx] +
                                   input[idx - planeSize] + input[idx + planeSize]) / 7.0;
                }
            }
        }
    }

    // Copy the two complete z faces. Splitting boundary faces into compact
    // loops avoids scanning and branching over the whole volume every time.
    if (gridSize != 0) {
        #pragma omp for schedule(static) nowait
        for (size_t offset = 0; offset < planeSize; ++offset) {
            output[offset] = input[offset];
            if (nz > 1) {
                const size_t last = (nz - 1) * planeSize + offset;
                output[last] = input[last];
            }
        }
    }

    if (nz > 2 && nx != 0 && ny != 0) {
        // Copy the two y faces for interior z planes.
        #pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t first = z * planeSize + x;
                output[first] = input[first];
                if (ny > 1) {
                    const size_t last = first + (ny - 1) * nx;
                    output[last] = input[last];
                }
            }
        }

        // Copy the two x faces for rows that are not already covered by the
        // y faces. All boundary loops write disjoint cells.
        if (ny > 2) {
            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t first = z * planeSize + y * nx;
                    output[first] = input[first];
                    if (nx > 1) {
                        const size_t last = first + nx - 1;
                        output[last] = input[last];
                    }
                }
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input,
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel
    {
        stencilIterationWork(input, output, nx, ny, nz);
    }
}

void runStencilIterations(std::vector<Real>& grid1,
                          std::vector<Real>& grid2,
                          const size_t nx, const size_t ny, const size_t nz,
                          const int iterations) {
    // Keep one team alive across all Jacobi iterations. The barrier is
    // required between iterations because the next input is the prior output.
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIterationWork(grid1, grid2, nx, ny, nz);
            } else {
                stencilIterationWork(grid2, grid1, nx, ny, nz);
            }
            #pragma omp barrier
        }
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (const auto& val : grid) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (const auto& val : grid) {
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

    // Keep the requested OpenMP team size from being reduced dynamically.
    omp_set_dynamic(0);
    
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
    initializeGrid(grid1, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    runStencilIterations(grid1, grid2, nx, ny, nz, iterations);
    
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
