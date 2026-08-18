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

void initializeGrid(std::vector<Real>& grid,
                    [[maybe_unused]] const size_t nx,
                    [[maybe_unused]] const size_t ny,
                    [[maybe_unused]] const size_t nz) {
    // Initialization is bandwidth-bound too, so distribute the contiguous
    // array directly rather than parallelizing three nested index loops.
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        grid[idx] = static_cast<Real>(idx % 19);
    }
}

// 7-point stencil computation
void stencilIteration(const std::vector<Real>& input, 
                      std::vector<Real>& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) {
        return;
    }

    const Real* const inputData = input.data();
    Real* const outputData = output.data();
    const size_t planeSize = nx * ny;
    constexpr size_t xTile = 64;

    // This function is called by every thread in the persistent team created
    // around the iteration loop in main(). The interior and boundary writes
    // are disjoint, so the first workshare can omit its intermediate barrier.
    // The implicit barrier at the end of the boundary workshare synchronizes
    // the completed output before the next time step swaps the buffers.

    // Process interior points (not on boundaries). Tiling the X dimension
    // gives the runtime enough independent chunks for highly rectangular
    // grids while retaining contiguous, SIMD-friendly accesses.
    if (nx >= 3 && ny >= 3 && nz >= 3) {
        #pragma omp for collapse(3) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                for (size_t xBegin = 1; xBegin < nx - 1; xBegin += xTile) {
                    const size_t rowStart = z * planeSize + y * nx;
                    const size_t xEnd = std::min(xBegin + xTile, nx - 1);

                    #pragma omp simd
                    for (size_t x = xBegin; x < xEnd; ++x) {
                        const size_t idx = rowStart + x;
                        const Real center = inputData[idx];
                        const Real left = inputData[idx - 1];
                        const Real right = inputData[idx + 1];
                        const Real front = inputData[idx - nx];
                        const Real back = inputData[idx + nx];
                        const Real bottom = inputData[idx - planeSize];
                        const Real top = inputData[idx + planeSize];

                        // Simple averaging stencil
                        outputData[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                    }
                }
            }
        }
    }

    // Copy boundary values. Whole boundary rows are copied at once; for
    // interior rows only the two X-face elements need to be handled.
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowStart = z * planeSize + y * nx;
            if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                std::copy_n(inputData + rowStart, nx, outputData + rowStart);
            } else {
                outputData[rowStart] = inputData[rowStart];
                if (nx > 1) {
                    outputData[rowStart + nx - 1] = inputData[rowStart + nx - 1];
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
    // Let the runtime honor the requested OpenMP team size without shrinking
    // it dynamically between worksharing regions.
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
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // Keep the OpenMP team alive across all iterations. This removes repeated
    // fork/join overhead while the workshare barriers preserve the original
    // sequential time-step semantics.
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, nx, ny, nz);
            } else {
                stencilIteration(grid2, grid1, nx, ny, nz);
            }
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
