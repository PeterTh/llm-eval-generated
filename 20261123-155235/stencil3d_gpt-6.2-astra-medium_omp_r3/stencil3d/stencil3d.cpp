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

// Keep small grids from paying for a large team at every iteration barrier.
int teamSize(size_t cells) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + cells / 16384));
}

void initializeGrids(std::vector<Real>& first, std::vector<Real>& second, int iterations, int threads) {
    #pragma omp parallel for simd schedule(static) num_threads(threads)
    for (size_t i = 0; i < first.size(); ++i) {
        first[i] = static_cast<Real>(i % 19);
        second[i] = iterations > 0 ? first[i] : 0.0;
    }
}

void runStencil(Real* first, Real* second, size_t nx, size_t ny, size_t nz,
                int iterations, int threads) {
    // Both buffers already contain the fixed boundary values.
    if (nx < 3 || ny < 3 || nz < 3 || iterations <= 0) return;

    const size_t plane = nx * ny;
    #pragma omp parallel num_threads(threads)
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
                        output[i] = (input[i] + input[i - 1] + input[i + 1]
                                   + input[i - nx] + input[i + nx]
                                   + input[i - plane] + input[i + plane]) / 7.0;
                    }
                }
            }
            // The worksharing barrier completes this grid before any thread reads it.
            std::swap(input, output);
        }
    }
}

bool validateResult(const std::vector<Real>& grid, int threads) {
    Real minVal = grid.front();
    Real maxVal = grid.front();
    int invalid = 0;
    #pragma omp parallel for simd schedule(static) num_threads(threads) reduction(min:minVal) reduction(max:maxVal) reduction(|:invalid)
    for (size_t i = 0; i < grid.size(); ++i) {
        invalid |= !std::isfinite(grid[i]);
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }
    if (invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);
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
    std::vector<Real> grid1(gridSize);
    std::vector<Real> grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    const int threads = teamSize(gridSize);
    initializeGrids(grid1, grid2, iterations, threads);

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runStencil(grid1.data(), grid2.data(), nx, ny, nz, iterations, threads);

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
        bool valid = validateResult(finalGrid, threads);
        
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
