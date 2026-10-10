#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// Leave scalar storage uninitialized so the parallel initialization below
// first-touches each thread's pages instead of placing every page on one NUMA node.
template <typename T>
struct GridAllocator : std::allocator<T> {
    template <typename U>
    struct rebind { using other = GridAllocator<U>; };

    template <typename U>
    void construct(U* ptr) {
        ::new (static_cast<void*>(ptr)) U;
    }
};

using Grid = std::vector<Real, GridAllocator<Real>>;

void initializeGrids(Grid& grid1, Grid& grid2, const int iterations, const int threads) {
    const size_t count = grid1.size();
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t idx = 0; idx < count; ++idx) {
        const Real value = (idx % 19) * 1.0;
        grid1[idx] = value;
        // Preserve the original zero-filled second buffer if no steps run.
        grid2[idx] = iterations > 0 ? value : 0.0;
    }
}

// Called by every thread in the persistent team. Both buffers already have
// identical fixed boundaries, so only interior points need to be updated.
void stencilIteration(const Real* __restrict input,
                      Real* __restrict output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx <= 2 || ny <= 2 || nz <= 2) return;

    const size_t plane = nx * ny;
    // Keep whole rows together when possible; split long rows when there are
    // too few Y/Z rows to occupy the team.
    const size_t xBlock = (ny - 2) * (nz - 2) < static_cast<size_t>(omp_get_num_threads())
        ? size_t{256} : nx - 2;
    #pragma omp for collapse(3) schedule(static)
    for (size_t z = 1; z < nz - 1; ++z) {
        for (size_t y = 1; y < ny - 1; ++y) {
            for (size_t xBegin = 1; xBegin < nx - 1; xBegin += xBlock) {
                const size_t row = z * plane + y * nx;
                const size_t xEnd = std::min(xBegin + xBlock, nx - 1);
                #pragma omp simd
                for (size_t x = xBegin; x < xEnd; ++x) {
                    const size_t idx = row + x;

                    const Real center = input[idx];
                    const Real left = input[idx - 1];
                    const Real right = input[idx + 1];
                    const Real front = input[idx - nx];
                    const Real back = input[idx + nx];
                    const Real bottom = input[idx - plane];
                    const Real top = input[idx + plane];

                    // Preserve the original addition order and division.
                    output[idx] = (center + left + right + front + back + bottom + top) / 7.0;
                }
            }
        }
    }
    // The implicit worksharing barrier finishes all writes before the next
    // iteration reads this buffer, including neighbors owned by other threads.
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
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

    // Amortize team and barrier costs on small grids. The runtime's configured
    // thread count remains the upper bound, and both phases use the same team size.
    constexpr size_t minCellsPerThread = 32768;
    const int threads = static_cast<int>(std::min(
        static_cast<size_t>(omp_get_max_threads()),
        std::max(size_t{1}, gridSize / minCellsPerThread)));
    
    // Allocate grids (double buffering)
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1, grid2, iterations, threads);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(threads)
    {
        Real* input = grid1.data();
        Real* output = grid2.data();
        for (int iter = 0; iter < iterations; ++iter) {
            stencilIteration(input, output, nx, ny, nz);
            // These pointers are private to each thread; no shared swap or
            // extra barrier is needed after the iteration's worksharing barrier.
            std::swap(input, output);
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
    const Grid& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
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
