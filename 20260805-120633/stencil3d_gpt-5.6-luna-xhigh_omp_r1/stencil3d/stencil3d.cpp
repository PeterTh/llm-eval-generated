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

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t gridSize = nx * ny * nz;
    Real* const data = grid.data();

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        data[idx] = static_cast<Real>(idx % 19);
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const size_t planeSize = nx * ny;
    const Real* const inputData = input.data();
    Real* const outputData = output.data();

    // A single team is reused for the interior and boundary work.  The
    // worksharing loops have no implicit barriers because every loop writes a
    // disjoint part of output; the barrier at the end of this region still
    // completes the iteration before the buffers are exchanged.
    #pragma omp parallel default(none) shared(inputData, outputData, nx, ny, nz, planeSize)
    {
        // Process interior points (not on boundaries).  The collapsed loops
        // expose enough rows to keep a large team busy, while SIMD operates on
        // the contiguous X dimension.
        if (nx > 2 && ny > 2 && nz > 2) {
            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t row = z * planeSize + y * nx;
                    const size_t previousRow = row - nx;
                    const size_t nextRow = row + nx;
                    const size_t previousPlane = row - planeSize;
                    const size_t nextPlane = row + planeSize;

                    #pragma omp simd
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = row + x;
                        const Real center = inputData[idx];
                        const Real left = inputData[idx - 1];
                        const Real right = inputData[idx + 1];
                        const Real front = inputData[previousRow + x];
                        const Real back = inputData[nextRow + x];
                        const Real bottom = inputData[previousPlane + x];
                        const Real top = inputData[nextPlane + x];

                        // Simple averaging stencil
                        outputData[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                    }
                }
            }
        }

        // Copy the two Z faces.  They are disjoint unless nz == 1, which is
        // handled separately so that no location is written by two threads.
        if (nz == 1) {
            #pragma omp for schedule(static) nowait
            for (size_t p = 0; p < planeSize; ++p) {
                outputData[p] = inputData[p];
            }
        } else if (nz > 1) {
            #pragma omp for schedule(static) nowait
            for (size_t p = 0; p < planeSize; ++p) {
                outputData[p] = inputData[p];
                outputData[(nz - 1) * planeSize + p] = inputData[(nz - 1) * planeSize + p];
            }
        }

        if (nz > 2) {
            // Copy Y faces for the interior Z planes.  For a single Y row,
            // that row is both Y faces and must be copied only once.
            if (ny == 1) {
                #pragma omp for collapse(2) schedule(static) nowait
                for (size_t z = 1; z < nz - 1; ++z) {
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t idx = z * planeSize + x;
                        outputData[idx] = inputData[idx];
                    }
                }
            } else if (ny > 1) {
                #pragma omp for collapse(2) schedule(static) nowait
                for (size_t z = 1; z < nz - 1; ++z) {
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t firstRow = z * planeSize + x;
                        const size_t lastRow = z * planeSize + (ny - 1) * nx + x;
                        outputData[firstRow] = inputData[firstRow];
                        outputData[lastRow] = inputData[lastRow];
                    }
                }
            }

            if (ny > 2) {
                // The remaining boundary points are the X faces of interior
                // Y rows.  Handle nx == 1 separately to keep the writes
                // disjoint.
                if (nx == 1) {
                    #pragma omp for collapse(2) schedule(static) nowait
                    for (size_t z = 1; z < nz - 1; ++z) {
                        for (size_t y = 1; y < ny - 1; ++y) {
                            const size_t idx = z * planeSize + y * nx;
                            outputData[idx] = inputData[idx];
                        }
                    }
                } else if (nx > 1) {
                    #pragma omp for collapse(2) schedule(static) nowait
                    for (size_t z = 1; z < nz - 1; ++z) {
                        for (size_t y = 1; y < ny - 1; ++y) {
                            const size_t row = z * planeSize + y * nx;
                            outputData[row] = inputData[row];
                            outputData[row + nx - 1] = inputData[row + nx - 1];
                        }
                    }
                }
            }
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
    
    for (int iter = 0; iter < iterations; ++iter) {
        if (iter % 2 == 0) {
            stencilIteration(grid1, grid2, nx, ny, nz);
        } else {
            stencilIteration(grid2, grid1, nx, ny, nz);
        }
    }
    
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
