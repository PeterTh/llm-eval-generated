#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

using Real = double;

// Leave scalar elements uninitialized so the parallel initialization, rather
// than a serial zero-fill, determines NUMA page placement. Every element of
// both grids is initialized before the stencil reads it.
template <typename T>
struct GridAllocator : std::allocator<T> {
    template <typename U>
    struct rebind { using other = GridAllocator<U>; };

    GridAllocator() = default;
    template <typename U>
    GridAllocator(const GridAllocator<U>&) noexcept {}

    template <typename U>
    void construct(U* ptr) { ::new (static_cast<void*>(ptr)) U; }

    template <typename U, typename... Args>
    void construct(U* ptr, Args&&... args) {
        ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
    }
};

using Grid = std::vector<Real, GridAllocator<Real>>;

void initializeGrids(Grid& grid1, Grid& grid2,
                     const size_t nx, const size_t ny, const size_t nz,
                     const bool hasIterations) {
    Real* const first = grid1.data();
    Real* const second = grid2.data();
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t row = (z * ny + y) * nx;
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = row + x;
                const Real value = static_cast<Real>(idx % 19);
                first[idx] = value;
                // Both buffers keep the same fixed boundaries. Interior values
                // in the second buffer are overwritten on the first iteration.
                second[idx] = hasIterations ? value : 0.0;
            }
        }
    }
}

inline void stencilRow(const Real* input, Real* output, const size_t row,
                       const size_t nx, const size_t plane, const size_t innerX) {
    const Real* const center = input + row;
    const Real* const front = center - nx;
    const Real* const back = center + nx;
    const Real* const bottom = center - plane;
    const Real* const top = center + plane;
    Real* const result = output + row;
    #pragma omp simd
    for (size_t x = 1; x <= innerX; ++x) {
        // Preserve the original floating-point addition order.
        result[x] = (center[x] + center[x-1] + center[x+1]
                   + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
    }
}

// 7-point Jacobi stencil. One team handles all iterations, with a single
// worksharing barrier per iteration before reading the newly computed buffer.
void stencilIterations(Grid& grid1, Grid& grid2,
                       const size_t nx, const size_t ny, const size_t nz,
                       const int iterations) {
    const size_t plane = nx * ny;
    const size_t innerX = nx > 2 ? nx - 2 : 0;
    const size_t innerY = ny > 2 ? ny - 2 : 0;
    const size_t innerZ = nz > 2 ? nz - 2 : 0;
    constexpr size_t tileZ = 8;
    constexpr size_t tileY = 16;
    constexpr size_t largePlane = 512 * 1024 / sizeof(Real);
    const size_t zTiles = (innerZ + tileZ - 1) / tileZ;
    const size_t yTiles = (innerY + tileY - 1) / tileY;
    const size_t threads = static_cast<size_t>(omp_get_max_threads());
    // Large planes compete for cache at high thread counts. Blocking keeps
    // neighboring planes reusable, but only when enough tiles remain to keep
    // the team busy. Smaller workloads benefit from simple streaming rows.
    const bool useTiles = plane > largePlane && threads >= 16
                       && zTiles * yTiles >= threads;

    #pragma omp parallel default(none) shared(grid1, grid2) \
        firstprivate(nx, plane, innerX, innerY, innerZ, iterations, \
                     tileZ, tileY, useTiles)
    {
        Real* input = grid1.data();
        Real* output = grid2.data();
        for (int iter = 0; iter < iterations; ++iter) {
            if (useTiles) {
                #pragma omp for collapse(2) schedule(static)
                for (size_t z0 = 0; z0 < innerZ; z0 += tileZ) {
                    for (size_t y0 = 0; y0 < innerY; y0 += tileY) {
                        const size_t zEnd = std::min(z0 + tileZ, innerZ);
                        const size_t yEnd = std::min(y0 + tileY, innerY);
                        for (size_t z = z0; z < zEnd; ++z) {
                            for (size_t y = y0; y < yEnd; ++y) {
                                const size_t row = (z + 1) * plane + (y + 1) * nx;
                                stencilRow(input, output, row, nx, plane, innerX);
                            }
                        }
                    }
                }
            } else {
                // Collapse rows to expose parallelism even for thin grids.
                // Static chunks preserve locality between iterations.
                #pragma omp for collapse(2) schedule(static)
                for (size_t z = 0; z < innerZ; ++z) {
                    for (size_t y = 0; y < innerY; ++y) {
                        const size_t row = (z + 1) * plane + (y + 1) * nx;
                        stencilRow(input, output, row, nx, plane, innerX);
                    }
                }
            }
            // Pointers are private to each thread; no extra swap barrier is
            // needed after the implicit barrier above.
            std::swap(input, output);
        }
    }
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    Real minVal = grid[0];
    Real maxVal = grid[0];
    int nonFinite = 0;
    #pragma omp parallel for schedule(static) \
        reduction(min:minVal) reduction(max:maxVal) reduction(|:nonFinite)
    for (size_t i = 0; i < grid.size(); ++i) {
        const Real val = grid[i];
        nonFinite |= !std::isfinite(val);
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }

    if (nonFinite) {
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
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrids(grid1, grid2, nx, ny, nz, iterations > 0);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    stencilIterations(grid1, grid2, nx, ny, nz, iterations);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance metrics
    double cellUpdates = static_cast<double>(nx > 2 ? nx - 2 : 0)
                       * static_cast<double>(ny > 2 ? ny - 2 : 0)
                       * static_cast<double>(nz > 2 ? nz - 2 : 0) * iterations;
    const double seconds = std::chrono::duration<double>(end - start).count();
    double mcups = seconds > 0.0 ? cellUpdates / seconds / 1e6 : 0.0;
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
