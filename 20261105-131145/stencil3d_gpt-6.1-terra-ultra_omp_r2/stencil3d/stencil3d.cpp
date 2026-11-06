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

void initializeGrids(Real* grid1, Real* grid2, const size_t size) {
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < size; ++idx) {
        grid1[idx] = (idx % 19) * 1.0;
        grid2[idx] = 0.0;
    }
}

// 7-point Jacobi stencil. The worksharing barrier completes each time step
// before any thread reads the newly written grid in the next one.
void stencilIterations(Real* grid1, Real* grid2,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int iterations) {
    const size_t plane = nx * ny;
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* __restrict__ input = (iter % 2 == 0) ? grid1 : grid2;
            Real* __restrict__ output = (iter % 2 == 0) ? grid2 : grid1;
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t row = z * plane + y * nx;
                    if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                        #pragma omp simd
                        for (size_t x = 0; x < nx; ++x) {
                            output[row + x] = input[row + x];
                        }
                    } else if (nx != 0) {
                        output[row] = input[row];
                        #pragma omp simd
                        for (size_t x = 1; x < nx - 1; ++x) {
                            const size_t idx = row + x;
                            output[idx] = (input[idx] + input[idx - 1] + input[idx + 1]
                                         + input[idx - nx] + input[idx + nx]
                                         + input[idx - plane] + input[idx + plane]) / 7.0;
                        }
                        output[row + nx - 1] = input[row + nx - 1];
                    }
                }
            }
        }
    }
}

bool validateResult(const Real* grid, const size_t size) {
    // Simple sanity checks
    
    // 1. No NaN or Inf values
    for (size_t i = 0; i < size; ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    for (size_t i = 0; i < size; ++i) {
        const Real val = grid[i];
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
    std::unique_ptr<Real[]> grid1(new Real[gridSize]);
    std::unique_ptr<Real[]> grid2(new Real[gridSize]);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1.get(), grid2.get(), gridSize);
    
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
    const Real* finalGrid = (iterations % 2 == 0) ? grid1.get() : grid2.get();
    if (printResults) {
        std::vector<Real> results(gridSize);
        std::copy_n(finalGrid, gridSize, results.begin());
        print_results(results, "Grid");
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
