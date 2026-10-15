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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Allocator that leaves elements uninitialized so that the first touch
// happens in the parallel initialization (NUMA-friendly page placement).
template <typename T>
struct NoInitAllocator {
    using value_type = T;
    NoInitAllocator() = default;
    template <typename U> NoInitAllocator(const NoInitAllocator<U>&) noexcept {}
    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), std::align_val_t(64))); }
    void deallocate(T* p, size_t) noexcept { ::operator delete(p, std::align_val_t(64)); }
    template <typename U> void construct(U*) noexcept {}
    template <typename U, typename... Args> void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
    template <typename U> bool operator==(const NoInitAllocator<U>&) const noexcept { return true; }
    template <typename U> bool operator!=(const NoInitAllocator<U>&) const noexcept { return false; }
};

using Grid = std::vector<Real, NoInitAllocator<Real>>;

void initializeGrid(Grid& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* __restrict g = grid.data();
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                g[base + x] = ((base + x) % 19) * 1.0;
            }
        }
    }
}

// 7-point stencil computation (interior update + boundary copy, fused)
void stencilIteration(const Grid& input, Grid& output,
                      const size_t nx, const size_t ny, const size_t nz) {
    const Real* __restrict in = input.data();
    Real* __restrict out = output.data();
    const size_t sy = nx;
    const size_t sz = nx * ny;

    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = z * sz + y * sy;
            if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || nx < 3) {
                #pragma omp simd
                for (size_t x = 0; x < nx; ++x) out[base + x] = in[base + x];
            } else {
                out[base] = in[base];
                out[base + nx - 1] = in[base + nx - 1];
                #pragma omp simd
                for (size_t x = 1; x < nx - 1; ++x) {
                    const size_t i = base + x;
                    out[i] = (in[i] + in[i - 1] + in[i + 1] + in[i - sy] + in[i + sy] + in[i - sz] + in[i + sz]) / 7.0;
                }
            }
        }
    }
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
    
    // Allocate grids (double buffering)
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    
    // Initialize (both buffers first-touched in parallel)
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    initializeGrid(grid2, nx, ny, nz);
    
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
        std::vector<Real> out(finalGrid.begin(), finalGrid.end());
        print_results(out, "Grid");
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
