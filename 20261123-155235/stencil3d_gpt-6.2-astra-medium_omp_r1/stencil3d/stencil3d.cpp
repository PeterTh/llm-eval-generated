#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// Each thread keeps its own buffer pointers. The worksharing barrier makes
// the entire output grid available before the next iteration reads it.
void runStencil(Real* input, Real* output, const size_t nx, const size_t ny,
                const size_t nz, const int iterations, const int threads) {
    if (nx < 3 || ny < 3 || nz < 3 || iterations <= 0) return;

    const size_t plane = nx * ny;
    #pragma omp parallel num_threads(threads) firstprivate(input, output)
    {
        for (int iter = 0; iter < iterations; ++iter) {
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t row = z * plane + y * nx;
                    const Real* center = input + row;
                    const Real* front = center - nx;
                    const Real* back = center + nx;
                    const Real* bottom = center - plane;
                    const Real* top = center + plane;

                    #pragma omp simd
                    for (size_t x = 1; x < nx - 1; ++x) {
                        output[row + x] = (center[x] + center[x-1] + center[x+1]
                                           + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                    }
                }
            }
            std::swap(input, output);
        }
    }
}

bool validateResult(std::span<const Real> grid, const int threads) {
    bool finite = true;
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for simd num_threads(threads) schedule(static) reduction(&&:finite) reduction(min:minVal) reduction(max:maxVal)
    for (size_t idx = 0; idx < grid.size(); ++idx) {
        const Real val = grid[idx];
        finite = finite && std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    if (!finite) {
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
    
    const size_t gridSize = nx * ny * nz;
    // Limit barrier overhead on small grids while respecting the runtime's thread limit.
    constexpr size_t cellsPerThread = 64 * 1024;
    const size_t chunks = gridSize == 0 ? 1 : 1 + (gridSize - 1) / cellsPerThread;
    const int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(), chunks));
    
    // Allocate grids (double buffering)
    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    // Set the fixed boundaries in both buffers once.
    #pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t idx = 0; idx < gridSize; ++idx) {
        grid1[idx] = static_cast<Real>(idx % 19);
        grid2[idx] = iterations > 0 ? grid1[idx] : 0.0;
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runStencil(grid1.get(), grid2.get(), nx, ny, nz, iterations, threads);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / std::chrono::duration<double>(end - start).count() / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::span<const Real> finalGrid(iterations % 2 == 0 ? grid1.get() : grid2.get(), gridSize);
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
