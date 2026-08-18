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

void initializeGrid(Real* const gridData, const size_t gridSize) {

    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        gridData[idx] = static_cast<Real>(idx % 19);
    }
}

// 7-point stencil computation.  One OpenMP team remains active for all time
// steps; the implicit barrier at each workshare completes one buffer before
// it becomes the input of the following iteration.
void stencilIterations(Real* const grid1, Real* const grid2,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int iterations) {
    if (iterations <= 0) {
        return;
    }

    const size_t planeSize = nx * ny;

    // If any dimension lacks an interior, every point is a boundary point.
    if (nx < 3 || ny < 3 || nz < 3) {
        const size_t gridSize = planeSize * nz;
        #pragma omp parallel
        {
            for (int iter = 0; iter < iterations; ++iter) {
                const Real* const inputData = (iter % 2 == 0) ? grid1 : grid2;
                Real* const outputData = (iter % 2 == 0) ? grid2 : grid1;

                #pragma omp for simd schedule(static)
                for (size_t idx = 0; idx < gridSize; ++idx) {
                    outputData[idx] = inputData[idx];
                }
            }
        }
        return;
    }

    // Each (z, y) row is owned by one iteration.  This keeps all writes
    // disjoint, preserves the boundary values, and exposes contiguous X
    // sweeps for SIMD while the collapsed outer loops balance all threads.
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* const inputData = (iter % 2 == 0) ? grid1 : grid2;
            Real* const outputData = (iter % 2 == 0) ? grid2 : grid1;

            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t row = z * planeSize + y * nx;

                    if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                        #pragma omp simd
                        for (size_t x = 0; x < nx; ++x) {
                            outputData[row + x] = inputData[row + x];
                        }
                    } else {
                        outputData[row] = inputData[row];

                        #pragma omp simd
                        for (size_t x = 1; x < nx - 1; ++x) {
                            const size_t idx = row + x;
                            outputData[idx] = (inputData[idx] + inputData[idx - 1] + inputData[idx + 1]
                                             + inputData[idx - nx] + inputData[idx + nx]
                                             + inputData[idx - planeSize] + inputData[idx + planeSize]) / 7.0;
                        }

                        outputData[row + nx - 1] = inputData[row + nx - 1];
                    }
                }
            }
        }
    }
}

bool validateResult(const Real* const grid, const size_t gridSize) {
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
    
    // Allocate without serial value-initialization.  grid1 is initialized in
    // parallel below, and every stencil iteration writes all of grid2.  This
    // lets the worker threads first-touch their own pages on NUMA systems.
    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1.get(), gridSize);

    // The original value-initialized both buffers.  A negative odd iteration
    // count executes no stencil iterations but selects grid2 as the result,
    // so retain that observable behavior without adding work to normal runs.
    if (iterations < 0 && iterations % 2 != 0) {
        #pragma omp parallel for simd schedule(static)
        for (size_t idx = 0; idx < gridSize; ++idx) {
            grid2[idx] = 0.0;
        }
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    stencilIterations(grid1.get(), grid2.get(), nx, ny, nz, iterations);
    
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
        // print_results accepts a vector; materialize it only for the optional
        // external report, after the timed stencil computation has completed.
        const std::vector<Real> resultGrid(finalGrid, finalGrid + gridSize);
        print_results(resultGrid, "Grid");
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
