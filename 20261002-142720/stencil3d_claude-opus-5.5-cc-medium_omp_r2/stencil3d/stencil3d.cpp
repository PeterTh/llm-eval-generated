#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

#include <immintrin.h>
#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Allocator that default-initializes (does not zero) elements, so that pages are
// first touched by the parallel initialization (NUMA-aware placement).
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

// Both grids are touched with the same static row distribution used by the
// stencil, so each thread's working set resides on its local NUMA node.
void initializeGrid(Grid& grid, Grid& other, const size_t nx, const size_t ny, const size_t nz) {
    const size_t rows = ny * nz;
    Real* __restrict g = grid.data();
    Real* __restrict o = other.data();
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        const size_t base = r * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t idx = base + x;
            g[idx] = (idx % 19) * 1.0;
            o[idx] = 0.0;
        }
    }
}

// 7-point stencil computation (called from within a parallel region)
// Streaming (non-temporal) stores only pay off when the grids do not fit in
// the last-level cache; for cache-resident grids they are much slower.
constexpr size_t kStreamingThresholdBytes = size_t(1) << 30;

// Computes the interior points [1, nx-1) of one row
template <bool Streaming>
inline void stencilRow(const Real* __restrict in, Real* __restrict out,
                       const Real* __restrict inF, const Real* __restrict inB,
                       const Real* __restrict inD, const Real* __restrict inU,
                       const size_t nx) {
    size_t x = 1;
#ifdef __AVX__
    if constexpr (Streaming) {
        // Peel until the output is 32-byte aligned
        for (; x < nx - 1 && (reinterpret_cast<uintptr_t>(out + x) & 31); ++x)
            out[x] = (in[x] + in[x - 1] + in[x + 1] + inF[x] + inB[x] + inD[x] + inU[x]) / 7.0;
        const __m256d seven = _mm256_set1_pd(7.0);
        for (; x + 4 <= nx - 1; x += 4) {
            // Same summation order as the scalar expression
            __m256d s = _mm256_add_pd(_mm256_loadu_pd(in + x), _mm256_loadu_pd(in + x - 1));
            s = _mm256_add_pd(s, _mm256_loadu_pd(in + x + 1));
            s = _mm256_add_pd(s, _mm256_loadu_pd(inF + x));
            s = _mm256_add_pd(s, _mm256_loadu_pd(inB + x));
            s = _mm256_add_pd(s, _mm256_loadu_pd(inD + x));
            s = _mm256_add_pd(s, _mm256_loadu_pd(inU + x));
            _mm256_stream_pd(out + x, _mm256_div_pd(s, seven));
        }
    }
#endif
    #pragma omp simd
    for (size_t xi = x; xi < nx - 1; ++xi) {
        // Simple averaging stencil (same summation order as original)
        out[xi] = (in[xi] + in[xi - 1] + in[xi + 1] + inF[xi] + inB[xi] + inD[xi] + inU[xi]) / 7.0;
    }
}

template <bool Streaming>
inline void stencilIteration(const Real* __restrict input,
                             Real* __restrict output,
                             const size_t nx, const size_t ny, const size_t nz) {
    const size_t rows = ny * nz;
    const size_t plane = nx * ny;
    #pragma omp for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        const size_t z = r / ny;
        const size_t y = r - z * ny;
        const size_t base = r * nx;
        const Real* __restrict in = input + base;
        Real* __restrict out = output + base;

        if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1 || nx < 3) {
            // Boundary row: copy
            for (size_t x = 0; x < nx; ++x) out[x] = in[x];
            continue;
        }

        const Real* __restrict inF = in - nx;
        const Real* __restrict inB = in + nx;
        const Real* __restrict inD = in - plane;
        const Real* __restrict inU = in + plane;

        out[0] = in[0];
        stencilRow<Streaming>(in, out, inF, inB, inD, inU, nx);
        out[nx - 1] = in[nx - 1];
    }
#ifdef __AVX__
    if constexpr (Streaming) _mm_sfence();
#endif
}

// Runs all iterations inside a single parallel region
template <bool Streaming>
void runIterations(Real* g1, Real* g2, const int iterations,
                   const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration<Streaming>(g1, g2, nx, ny, nz);
            } else {
                stencilIteration<Streaming>(g2, g1, nx, ny, nz);
            }
        }
    }
}

// Pin OpenMP worker threads to distinct CPUs (spread over physical cores and
// sockets) unless the user already requested a binding via OMP_PROC_BIND /
// OMP_PLACES. Stable placement keeps each thread next to the memory it first
// touched (NUMA locality), which matters a lot for this bandwidth-bound kernel.
static int readTopo(int cpu, const char* what) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

void pinThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false || omp_get_num_places() > 0) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    struct CpuInfo { int cpu, pkg, core, smt; };
    std::vector<CpuInfo> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) cpus.push_back({c, readTopo(c, "physical_package_id"), readTopo(c, "core_id"), 0});
    }
    if (cpus.size() < 2) return;

    // Rank of each hardware thread within its physical core
    std::sort(cpus.begin(), cpus.end(), [](const CpuInfo& a, const CpuInfo& b) {
        if (a.pkg != b.pkg) return a.pkg < b.pkg;
        if (a.core != b.core) return a.core < b.core;
        return a.cpu < b.cpu;
    });
    for (size_t i = 1; i < cpus.size(); ++i) {
        if (cpus[i].pkg == cpus[i-1].pkg && cpus[i].core == cpus[i-1].core && cpus[i].core >= 0)
            cpus[i].smt = cpus[i-1].smt + 1;
    }

    std::vector<int> primary, all;
    for (const auto& ci : cpus) {
        all.push_back(ci.cpu);
        if (ci.smt == 0) primary.push_back(ci.cpu);
    }

    const int nthreads = omp_get_max_threads();
    const std::vector<int>& pool = (static_cast<size_t>(nthreads) <= primary.size()) ? primary : all;
    const size_t P = pool.size();

    #pragma omp parallel num_threads(nthreads)
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        const size_t n = static_cast<size_t>(omp_get_num_threads());
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pool[(t * P / n) % P], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const size_t n = grid.size();
    const Real* g = grid.data();
    
    // 1. No NaN or Inf values
    bool bad = false;
    #pragma omp parallel for schedule(static) reduction(||:bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(g[i]) || std::isinf(g[i])) bad = true;
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
    
    pinThreads();
    
    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, grid2, nx, ny, nz);
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (2 * gridSize * sizeof(Real) > kStreamingThresholdBytes) {
        runIterations<true>(grid1.data(), grid2.data(), iterations, nx, ny, nz);
    } else {
        runIterations<false>(grid1.data(), grid2.data(), iterations, nx, ny, nz);
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
