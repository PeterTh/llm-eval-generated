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

#include "../common/results_output.hpp"

using Real = double;

// Allocator whose value-construction is default-initialization (no zero fill).
// This keeps the grid allocation from being first-touched serially on the main
// thread; pages are instead placed by the threads that first write them (NUMA-aware).
// Every grid cell is written before it is read, so the lack of zeroing is safe.
template <typename T>
struct DefaultInitAllocator {
    using value_type = T;

    DefaultInitAllocator() noexcept = default;
    template <typename U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}

    T* allocate(size_t n) { return std::allocator<T>{}.allocate(n); }
    void deallocate(T* p, size_t n) noexcept { std::allocator<T>{}.deallocate(p, n); }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            ::new (static_cast<void*>(p)) U;  // default-initialization
        } else {
            ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    }

    template <typename U>
    bool operator==(const DefaultInitAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const DefaultInitAllocator<U>&) const noexcept { return false; }
};

using Grid = std::vector<Real, DefaultInitAllocator<Real>>;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

void initializeGrid(Grid& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* g = grid.data();
    const size_t total = nx * ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total; ++idx) {
        g[idx] = (idx % 19) * 1.0;
    }
}

// 7-point stencil computation
void stencilIteration(const Grid& input, 
                      Grid& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    if (nx == 0 || ny == 0 || nz == 0) return;

    const Real* in = input.data();
    Real* out = output.data();
    const size_t plane = nx * ny;

    // Single fused pass over all rows (x-lines). Each (z, y) row is independent,
    // so rows are distributed across threads. Boundary cells copy the input;
    // interior cells apply the 7-point average (same operation order as before).
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * plane + y * nx;

            if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                std::copy(in + base, in + base + nx, out + base);
                continue;
            }

            out[base] = in[base];
            out[base + nx - 1] = in[base + nx - 1];

            for (size_t x = 1; x + 1 < nx; ++x) {
                const size_t idx = base + x;
                const Real center = in[idx];
                const Real left = in[idx - 1];
                const Real right = in[idx + 1];
                const Real front = in[idx - nx];
                const Real back = in[idx + nx];
                const Real bottom = in[idx - plane];
                const Real top = in[idx + plane];

                // Simple averaging stencil
                out[idx] = (center + left + right + front + back + bottom + top) / 7.0;
            }
        }
    }
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    const size_t n = grid.size();
    const Real* g = grid.data();

    // 1. No NaN or Inf values
    bool foundNonFinite = false;
    #pragma omp parallel for reduction(|| : foundNonFinite) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(g[i]) || std::isinf(g[i])) {
            foundNonFinite = true;
        }
    }
    if (foundNonFinite) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for reduction(min : minVal) reduction(max : maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, g[i]);
        maxVal = std::max(maxVal, g[i]);
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
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    
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
    const Grid& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        // print_results expects a standard vector; copy outside the timed region
        const std::vector<Real> printable(finalGrid.begin(), finalGrid.end());
        print_results(printable, "Grid");
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
