#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <sched.h>

#include <omp.h>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

// Allocator that leaves memory uninitialized on resize, so that the grids can be
// first-touched by the OpenMP threads that later own them (NUMA page placement).
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    using std::allocator<T>::allocator;

    template <typename U>
    struct rebind {
        using other = DefaultInitAllocator<U>;
    };

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            ::new (static_cast<void*>(p)) U;  // default-init: no zeroing
        } else {
            ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    }
};

using Grid = std::vector<Real, DefaultInitAllocator<Real>>;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Pin every thread to its own CPU. Thread migration would both destroy the NUMA
// locality established by the first touch and make the per-iteration barrier
// much more expensive, and OpenMP does not bind threads by default. Any affinity
// setting made by the user takes precedence: in that case nothing is done here.
void bindThreads() {
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY")) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    // Split the usable CPUs into one per physical core plus the SMT siblings, so
    // that the siblings are only used once every core carries a thread.
    std::vector<int> cores;
    std::vector<int> siblings;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        int first = cpu;
        if (FILE* f = fopen(path, "r")) {
            int v = cpu;
            if (fscanf(f, "%d", &v) == 1) first = v;
            fclose(f);
        }
        (first == cpu ? cores : siblings).push_back(cpu);
    }
    if (cores.empty()) return;

    std::vector<int> all = cores;
    all.insert(all.end(), siblings.begin(), siblings.end());
    const std::vector<int>& pool =
        ((size_t)omp_get_max_threads() <= cores.size()) ? cores : all;

#pragma omp parallel
    {
        const size_t nthreads = (size_t)omp_get_num_threads();
        const size_t tid = (size_t)omp_get_thread_num();
        // Spread the threads evenly over the pool.
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(pool[(tid * pool.size()) / nthreads], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

// Split [0, n) into `parts` nearly equal pieces and return piece number `idx`.
inline void splitRange(const size_t n, const size_t parts, const size_t idx,
                       size_t& begin, size_t& end) noexcept {
    const size_t q = n / parts;
    const size_t r = n % parts;
    begin = idx * q + (idx < r ? idx : r);
    end = begin + q + (idx < r ? 1 : 0);
}

// The threads are arranged as a pz x py grid of (z-slab, y-band) tiles. A purely
// one-dimensional split would give each thread a very thin slab at high thread
// counts, and the two halo planes it has to read would then dominate its memory
// traffic. Pick the factorization of the team size that minimizes that halo
// overhead for the given grid shape.
size_t chooseZParts(const size_t nthreads, const size_t ny, const size_t nz) noexcept {
    size_t bestPz = 1;
    double bestCost = 1e300;
    for (size_t pz = 1; pz <= nthreads; ++pz) {
        if (nthreads % pz != 0) continue;
        const size_t py = nthreads / pz;
        const double slab = (double)nz / (double)pz;
        const double band = (double)ny / (double)py;
        // Halo traffic overhead, times the penalty for threads that would fall
        // outside the grid and stay idle.
        double cost = (1.0 + 2.0 / std::max(slab, 1.0)) * (1.0 + 2.0 / std::max(band, 1.0));
        if (slab < 1.0) cost *= 1.0 / slab;
        if (band < 1.0) cost *= 1.0 / band;
        if (cost < bestCost) {
            bestCost = cost;
            bestPz = pz;
        }
    }
    return bestPz;
}

// Tile of the (z, y) plane owned by the calling thread.
struct Tile {
    size_t z0, z1, y0, y1;
};

inline Tile myTile(const size_t ny, const size_t nz) noexcept {
    const size_t nthreads = (size_t)omp_get_num_threads();
    const size_t tid = (size_t)omp_get_thread_num();
    const size_t pz = chooseZParts(nthreads, ny, nz);
    const size_t py = nthreads / pz;
    Tile t;
    splitRange(nz, pz, tid / py, t.z0, t.z1);
    splitRange(ny, py, tid % py, t.y0, t.y1);
    return t;
}

void initializeGrid(Grid& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* __restrict const g = grid.data();
    const size_t nxy = nx * ny;
    // Each thread initializes exactly the rows it will own during the stencil
    // iterations, so this first touch places the pages on its NUMA node.
#pragma omp parallel proc_bind(spread)
    {
        const Tile t = myTile(ny, nz);
        for (size_t z = t.z0; z < t.z1; ++z) {
            for (size_t y = t.y0; y < t.y1; ++y) {
                const size_t base = z * nxy + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    g[base + x] = ((base + x) % 19) * 1.0;
                }
            }
        }
    }
}

// First touch of the second buffer with the same page distribution.
void touchGrid(Grid& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* __restrict const g = grid.data();
    const size_t nxy = nx * ny;
#pragma omp parallel proc_bind(spread)
    {
        const Tile t = myTile(ny, nz);
        for (size_t z = t.z0; z < t.z1; ++z) {
            for (size_t y = t.y0; y < t.y1; ++y) {
                const size_t base = z * nxy + y * nx;
                memset(g + base, 0, nx * sizeof(Real));
            }
        }
    }
}

// One row of interior points in x. The result is streamed out with non-temporal
// stores where available: the output row is never read again in this iteration,
// so bypassing the caches avoids the write-allocate traffic on this
// bandwidth-bound kernel. The order of the additions is identical to the scalar
// version, so the results are bit-for-bit the same.
inline void stencilRow(Real* __restrict const o,
                       const Real* __restrict const c,
                       const Real* __restrict const ym, const Real* __restrict const yp,
                       const Real* __restrict const zm, const Real* __restrict const zp,
                       const size_t nx) {
    const size_t last = nx - 1;
    size_t x = 1;
#if defined(__AVX__)
    // Scalar prologue until the store address is 32-byte aligned.
    while (x < last && (reinterpret_cast<uintptr_t>(o + x) & 31u) != 0) {
        o[x] = (c[x] + c[x-1] + c[x+1] + ym[x] + yp[x] + zm[x] + zp[x]) / 7.0;
        ++x;
    }
    const __m256d seven = _mm256_set1_pd(7.0);
    for (; x + 4 <= last; x += 4) {
        __m256d v = _mm256_loadu_pd(c + x);
        v = _mm256_add_pd(v, _mm256_loadu_pd(c + x - 1));
        v = _mm256_add_pd(v, _mm256_loadu_pd(c + x + 1));
        v = _mm256_add_pd(v, _mm256_loadu_pd(ym + x));
        v = _mm256_add_pd(v, _mm256_loadu_pd(yp + x));
        v = _mm256_add_pd(v, _mm256_loadu_pd(zm + x));
        v = _mm256_add_pd(v, _mm256_loadu_pd(zp + x));
        _mm256_stream_pd(o + x, _mm256_div_pd(v, seven));
    }
#endif
#pragma omp simd
    for (size_t xi = x; xi < last; ++xi) {
        // Simple averaging stencil
        o[xi] = (c[xi] + c[xi-1] + c[xi+1] + ym[xi] + yp[xi] + zm[xi] + zp[xi]) / 7.0;
    }
}

// 7-point stencil computation for the rows owned by the calling thread.
inline void stencilIterationTile(const Real* __restrict const in,
                                 Real* __restrict const out,
                                 const Tile& t,
                                 const size_t nx, const size_t ny, const size_t nz) {
    const size_t nxy = nx * ny;

    // Degenerate grids have no interior at all: every point is a boundary point.
    const bool allBoundary = (nx < 3 || ny < 3 || nz < 3);

    for (size_t z = t.z0; z < t.z1; ++z) {
        const bool zBoundary = allBoundary || z == 0 || z == nz - 1;
        for (size_t y = t.y0; y < t.y1; ++y) {
            const size_t base = z * nxy + y * nx;
            const Real* __restrict const c = in + base;
            Real* __restrict const o = out + base;

            if (zBoundary || y == 0 || y == ny - 1) {
                // Copy boundary values
                memcpy(o, c, nx * sizeof(Real));
                continue;
            }

            stencilRow(o, c, c - nx, c + nx, c - nxy, c + nxy, nx);

            // x boundaries of this row
            o[0] = c[0];
            o[nx-1] = c[nx-1];
        }
    }
}

// Run all iterations inside a single parallel region: the team, the tile
// assignment and the thread/data affinity are then set up only once, and the
// only per-iteration synchronization is one barrier.
void runStencil(Grid& grid1, Grid& grid2, const size_t nx, const size_t ny, const size_t nz,
                const int iterations) {
    Real* __restrict const g1 = grid1.data();
    Real* __restrict const g2 = grid2.data();

#pragma omp parallel proc_bind(spread)
    {
        // Every thread owns a fixed tile of rows: the same one it initialized.
        const Tile t = myTile(ny, nz);

        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIterationTile(g1, g2, t, nx, ny, nz);
            } else {
                stencilIterationTile(g2, g1, t, nx, ny, nz);
            }
#if defined(__AVX__)
            // Make the non-temporal stores visible to the threads that read this
            // tile's halo in the next iteration.
            _mm_sfence();
#endif
#pragma omp barrier
        }
    }
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    const Real* __restrict const data = grid.data();
    const size_t n = grid.size();
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(||: bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            bad = true;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min: minVal) reduction(max: maxVal)
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
    
    bindThreads();
    printf("Threads: %d\n", omp_get_max_threads());

    size_t gridSize = nx * ny * nz;

    // Allocate grids (double buffering)
    Grid grid1(gridSize);
    Grid grid2(gridSize);

    // Initialize (in parallel, which also establishes NUMA page placement)
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    touchGrid(grid2, nx, ny, nz);
    
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
