#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>

#include "../common/results_output.hpp"

using Real = double;

void initializeGrids(Real* first, Real* second, size_t count, bool initializeSecond) {
    // First-touch both grids in the same contiguous bands used for computation.
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < count; ++i) {
        first[i] = static_cast<Real>(i % 19);
        second[i] = initializeSecond ? first[i] : 0.0;
    }
}

void runStencil(Real* first, Real* second, size_t nx, size_t ny, size_t nz, int iterations) {
    if (nx < 3 || ny < 3 || nz < 3) return;
    const size_t plane = nx * ny;
    #pragma omp parallel
    {
        Real* input = first;
        Real* output = second;
        for (int iter = 0; iter < iterations; ++iter) {
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t row = z * plane + y * nx;
                    #pragma omp simd
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t i = row + x;
                        output[i] = (input[i] + input[i-1] + input[i+1]
                                     + input[i-nx] + input[i+nx]
                                     + input[i-plane] + input[i+plane]) / 7.0;
                    }
                }
            }
            // The worksharing barrier completes this grid before any thread reads it.
            std::swap(input, output);
        }
    }
}

bool validateResult(std::span<const Real> grid) {
    Real minVal = grid.empty() ? 0.0 : grid.front();
    Real maxVal = minVal;
    bool invalid = false;
    #pragma omp parallel for simd schedule(static) reduction(min:minVal) reduction(max:maxVal) reduction(||:invalid)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real val = grid[i];
        invalid = invalid || !std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
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
    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1.get(), grid2.get(), gridSize, iterations >= 0);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runStencil(grid1.get(), grid2.get(), nx, ny, nz, iterations);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = nx > 2 && ny > 2 && nz > 2
        ? static_cast<double>((nx-2) * (ny-2) * (nz-2)) * iterations : 0.0;
    double mcups = cellUpdates / (std::chrono::duration<double>(end - start).count()) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::span<const Real> finalGrid(iterations % 2 == 0 ? grid1.get() : grid2.get(), gridSize);
    if (printResults) {
        print_results(finalGrid, "Grid");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid);
        
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
