#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>

#include <omp.h>

#ifndef _OPENMP
#error "The stencil3d benchmark requires OpenMP support."
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrids(Real* const grid1, Real* const grid2,
                     const size_t nx, const size_t ny, const size_t nz,
                     const int threads) {
    // Both buffers have identical, immutable boundaries. Touch their pages in
    // parallel so memory is placed near the threads that will update it.
    #pragma omp parallel for collapse(2) schedule(static) num_threads(threads) default(none) shared(grid1, grid2, nx, ny, nz)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = idx3(0, y, z, nx, ny);
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const Real value = static_cast<Real>((row + x) % 19);
                grid1[row + x] = value;
                grid2[row + x] = value;
            }
        }
    }
}

// 7-point stencil computation
void stencilIterations(Real* const grid1, Real* const grid2,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int iterations, const int threads) {
    if (nx < 3 || ny < 3 || nz < 3) return;
    const size_t plane = nx * ny;

    // Reuse one team for all time steps; each thread swaps its own pointers.
    #pragma omp parallel num_threads(threads) default(none) shared(grid1, grid2, nx, ny, nz, plane, iterations)
    {
        Real* input = grid1;
        Real* output = grid2;
        for (int iter = 0; iter < iterations; ++iter) {
            // Static contiguous row ranges keep the x dimension vectorizable
            // and expose parallelism even when the grid has few z planes.
            #pragma omp for collapse(2) schedule(static)
            for (size_t z = 1; z < nz - 1; ++z) {
                for (size_t y = 1; y < ny - 1; ++y) {
                    const size_t row = idx3(0, y, z, nx, ny);
                    #pragma omp simd
                    for (size_t x = 1; x < nx - 1; ++x) {
                        const size_t idx = row + x;
                        // Preserve the original addition order and division.
                        output[idx] = (input[idx] + input[idx - 1] + input[idx + 1]
                                     + input[idx - nx] + input[idx + nx]
                                     + input[idx - plane] + input[idx + plane]) / 7.0;
                    }
                }
            }
            // The omp-for barrier completes all writes before any thread
            // reads this time step as the next step's input.
            std::swap(input, output);
        }
    }
}

bool validateResult(std::span<const Real> grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

    int threads = omp_get_max_threads();
    if (std::getenv("OMP_NUM_THREADS") == nullptr &&
        omp_get_proc_bind() == omp_proc_bind_false) {
        // Large unbound teams add migration and barrier costs to this
        // bandwidth-bound kernel. Keep enough work per thread by default;
        // explicit thread counts or affinity retain the runtime's team size.
        const size_t usefulThreads = std::max<size_t>(1, gridSize / (128 * 1024));
        threads = static_cast<int>(std::min<size_t>(threads,
            std::min<size_t>(32, usefulThreads)));
    }
    
    // Allocate grids (double buffering)
    // Avoid serial zero-filling, which would place all pages on one NUMA node.
    auto grid1 = std::make_unique_for_overwrite<Real[]>(gridSize);
    auto grid2 = std::make_unique_for_overwrite<Real[]>(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1.get(), grid2.get(), nx, ny, nz, threads);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    stencilIterations(grid1.get(), grid2.get(), nx, ny, nz, iterations, threads);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Print results for external validation
    const std::span<const Real> finalGrid{
        (iterations % 2 == 0) ? grid1.get() : grid2.get(), gridSize};
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
