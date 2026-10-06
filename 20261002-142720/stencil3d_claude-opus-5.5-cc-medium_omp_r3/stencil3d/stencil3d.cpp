#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Allocator that leaves elements default-initialized (no serial zero-fill),
// so that pages are first touched by the OpenMP threads that later use them
// (NUMA-friendly first-touch placement).
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template <typename U>
    struct rebind { using other = DefaultInitAllocator<U>; };
    DefaultInitAllocator() noexcept = default;
    template <typename U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

using Grid = std::vector<Real, DefaultInitAllocator<Real>>;

// Pin OpenMP threads to distinct CPUs (spread over the allowed CPU set) so that
// first-touch NUMA placement stays valid during the sweeps. Skipped if the user
// already controls thread affinity through the environment.
void bindThreads() {
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY")) return;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) cpus.push_back(c);
    }
    if (cpus.empty()) return;
    #pragma omp parallel
    {
        const size_t nt = omp_get_num_threads();
        const size_t t = omp_get_thread_num();
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[(t * cpus.size() / nt) % cpus.size()], &one);
        sched_setaffinity(0, sizeof(one), &one);
    }
}

// Initialize grid (and first-touch the second buffer) using the same static
// row distribution as the stencil sweep.
void initializeGrid(Grid& grid, Grid& other, const size_t nx, const size_t ny, const size_t nz) {
    const size_t rows = ny * nz;
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * nx;
        Real* __restrict g = grid.data() + base;
        Real* __restrict o = other.data() + base;
        #pragma omp simd
        for (size_t x = 0; x < nx; ++x) {
            g[x] = ((base + x) % 19) * 1.0;
            o[x] = 0.0;
        }
    }
}

// 7-point stencil sweep over one row (z,y); boundary rows/points are copied.
static inline void stencilRow(const Real* __restrict input, Real* __restrict output,
                              const size_t r, const size_t nx, const size_t ny, const size_t nz) {
    const size_t z = r / ny;
    const size_t y = r - z * ny;
    const size_t base = r * nx;
    const Real* __restrict in = input + base;
    Real* __restrict out = output + base;

    if (y == 0 || y == ny - 1 || z == 0 || z == nz - 1) {
        // Boundary row: copy values
        #pragma omp simd
        for (size_t x = 0; x < nx; ++x) out[x] = in[x];
        return;
    }

    const size_t plane = nx * ny;
    const Real* __restrict front = in - nx;
    const Real* __restrict back = in + nx;
    const Real* __restrict bottom = in - plane;
    const Real* __restrict top = in + plane;

    out[0] = in[0];
    #pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        // Simple averaging stencil (same operation order as the serial code)
        out[x] = (in[x] + in[x - 1] + in[x + 1] + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
    }
    out[nx - 1] = in[nx - 1];
}

// Run all stencil iterations inside a single parallel region (double buffering).
void runStencil(Grid& grid1, Grid& grid2, const size_t nx, const size_t ny, const size_t nz,
                const int iterations) {
    const size_t rows = ny * nz;
    Real* const g1 = grid1.data();
    Real* const g2 = grid2.data();
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* input = (iter % 2 == 0) ? g1 : g2;
            Real* output = (iter % 2 == 0) ? g2 : g1;
            #pragma omp for schedule(static)
            for (size_t r = 0; r < rows; ++r) {
                stencilRow(input, output, r, nx, ny, nz);
            }
        }
    }
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();
    const Real* data = grid.data();
    
    // 1. No NaN or Inf values
    bool bad = false;
    #pragma omp parallel for schedule(static) reduction(||:bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
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
    
    bindThreads();
    
    // Allocate grids (double buffering)
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, grid2, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    runStencil(grid1, grid2, nx, ny, nz, iterations);
    
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
        print_results(std::vector<Real>(finalGrid.begin(), finalGrid.end()), "Grid");
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
