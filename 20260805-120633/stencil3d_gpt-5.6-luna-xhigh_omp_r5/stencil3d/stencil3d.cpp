#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t planeSize = nx * ny;

    // The initialization is independent for every cell and gives each thread
    // contiguous chunks of the row-major grid.
    #pragma omp parallel for collapse(3) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = z * planeSize + y * nx + x;
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const size_t planeSize = nx * ny;
    const Real* const inputData = input.data();
    Real* const outputData = output.data();

    // Both worksharing loops use the same team.  The interior and boundary
    // writes are disjoint, so the first loop can omit its barrier.  Static
    // scheduling preserves locality and avoids per-cell scheduling overhead.
    #pragma omp parallel default(none) shared(inputData, outputData, nx, ny, nz, planeSize)
    {
        // Process interior points (not on boundaries).
        if (nx > 2 && ny > 2 && nz > 2) {
            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t rowStart = z * planeSize + y * nx;
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = rowStart + x;

                        const Real center = inputData[idx];
                        const Real left = inputData[idx - 1];
                        const Real right = inputData[idx + 1];
                        const Real front = inputData[idx - nx];
                        const Real back = inputData[idx + nx];
                        const Real bottom = inputData[idx - planeSize];
                        const Real top = inputData[idx + planeSize];

                        // Simple averaging stencil.
                        outputData[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                    }
                }
            }
        }

        // Copy boundary values by faces.  The faces are partitioned so every
        // boundary cell is handled by exactly one iteration, including for
        // grids with a dimension of one or two cells.
        if (nz > 0) {
            // z = 0 and z = nz - 1
            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t y = 0; y < ny; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = y * nx + x;
                    outputData[idx] = inputData[idx];
                    if (nz > 1) {
                        const size_t lastIdx = idx + (nz - 1) * planeSize;
                        outputData[lastIdx] = inputData[lastIdx];
                    }
                }
            }
        }

        if (nz > 2) {
            // y = 0 and y = ny - 1 on z-interior planes
            #pragma omp for collapse(2) schedule(static) nowait
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = z * planeSize + x;
                    outputData[idx] = inputData[idx];
                    if (ny > 1) {
                        const size_t lastIdx = idx + (ny - 1) * nx;
                        outputData[lastIdx] = inputData[lastIdx];
                    }
                }
            }
        }

        if (nz > 2 && ny > 2) {
            // x = 0 and x = nx - 1 on z/y-interior rows
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t idx = z * planeSize + y * nx;
                    outputData[idx] = inputData[idx];
                    if (nx > 1) {
                        const size_t lastIdx = idx + nx - 1;
                        outputData[lastIdx] = inputData[lastIdx];
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
