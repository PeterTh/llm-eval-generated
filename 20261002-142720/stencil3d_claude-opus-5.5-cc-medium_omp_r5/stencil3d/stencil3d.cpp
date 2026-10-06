#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#include <omp.h>
#include <sched.h>
#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Allocator that leaves elements default-initialized (i.e. untouched), so that
// pages are first touched by the parallel initialization (NUMA-aware placement).
template <typename T>
struct DefaultInitAllocator {
    using value_type = T;
    static constexpr std::align_val_t alignment{64};
    DefaultInitAllocator() noexcept = default;
    template <typename U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    T* allocate(const size_t n) { return static_cast<T*>(::operator new(n * sizeof(T), alignment)); }
    void deallocate(T* p, size_t) noexcept { ::operator delete(p, alignment); }
    template <typename U>
    void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
    template <typename U>
    bool operator==(const DefaultInitAllocator<U>&) const noexcept { return true; }
};

using Grid = std::vector<Real, DefaultInitAllocator<Real>>;

// Pin OpenMP threads to distinct CPUs (physical cores first) unless the user
// already requested a binding policy. Keeps threads near the memory they
// first-touched, which matters a lot on multi-socket (NUMA) systems.
std::vector<int> buildPinList() {
    std::vector<int> list;
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY") || getenv("KMP_AFFINITY")) {
        return list;
    }
    cpu_set_t mask;
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return list;
    std::vector<int> primary, secondary;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) continue;
        int firstSibling = cpu;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        if (FILE* f = fopen(path, "r")) {
            if (fscanf(f, "%d", &firstSibling) != 1) firstSibling = cpu;
            fclose(f);
        }
        (firstSibling == cpu ? primary : secondary).push_back(cpu);
    }
    list = primary;
    if (static_cast<size_t>(omp_get_max_threads()) > primary.size()) {
        list.insert(list.end(), secondary.begin(), secondary.end());
    }
    return list;
}

void pinCurrentThread(const std::vector<int>& pinList) {
    if (pinList.empty()) return;
    const size_t nthreads = omp_get_num_threads();
    const size_t tid = omp_get_thread_num();
    const size_t ncpus = pinList.size();
    // Spread evenly over the list when there are fewer threads than CPUs
    const size_t slot = (nthreads <= ncpus) ? (tid * ncpus) / nthreads : tid % ncpus;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(pinList[slot], &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// The grid is decomposed into (z-block x y-block) tiles of full x-rows. Inside a
// tile the sweep streams through z, so the three z-planes of the tile stay in
// cache and each input value is fetched from memory (close to) once. Tiles are
// distributed statically and identically for initialization and all sweeps, so
// each thread always works on the memory it touched first.
struct Tiling {
    size_t zb, yb;    // tile extents
    size_t nzt, nyt;  // number of tiles per dimension
    size_t count() const noexcept { return nzt * nyt; }
};

Tiling makeTiling(const size_t nx, const size_t ny, const size_t nz, const size_t nthreads) {
    Tiling t{};
    // Keep ~3 input planes of a tile (plus output rows) within a per-core cache budget
    constexpr size_t cacheBudget = 256 * 1024;
    const size_t rowBytes = std::max<size_t>(1, nx) * sizeof(Real);
    t.yb = std::clamp<size_t>(cacheBudget / (4 * rowBytes), 1, std::max<size_t>(1, ny));
    t.nyt = (ny + t.yb - 1) / t.yb;
    t.yb = (ny + t.nyt - 1) / std::max<size_t>(1, t.nyt);  // balance tile heights
    // Enough z-blocks so that there is at least one tile per thread
    const size_t wantZ = std::max<size_t>(1, (nthreads + t.nyt - 1) / std::max<size_t>(1, t.nyt));
    t.nzt = std::min(std::max<size_t>(1, nz), wantZ);
    t.zb = (nz + t.nzt - 1) / t.nzt;
    t.nzt = (nz + t.zb - 1) / std::max<size_t>(1, t.zb);
    return t;
}

void initializeGrid(Real* __restrict grid, const size_t nx, const size_t ny, const size_t nz, const Tiling& tl) {
    #pragma omp for schedule(static)
    for (size_t t = 0; t < tl.count(); ++t) {
        const size_t z0 = (t / tl.nyt) * tl.zb, z1 = std::min(nz, z0 + tl.zb);
        const size_t y0 = (t % tl.nyt) * tl.yb, y1 = std::min(ny, y0 + tl.yb);
        for (size_t z = z0; z < z1; ++z) {
            for (size_t y = y0; y < y1; ++y) {
                const size_t base = idx3(0, y, z, nx, ny);
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = base + x;
                    grid[idx] = (idx % 19) * 1.0;
                }
            }
        }
    }
}

void zeroGrid(Real* __restrict grid, const size_t nx, const size_t ny, const size_t nz, const Tiling& tl) {
    #pragma omp for schedule(static)
    for (size_t t = 0; t < tl.count(); ++t) {
        const size_t z0 = (t / tl.nyt) * tl.zb, z1 = std::min(nz, z0 + tl.zb);
        const size_t y0 = (t % tl.nyt) * tl.yb, y1 = std::min(ny, y0 + tl.yb);
        for (size_t z = z0; z < z1; ++z) {
            for (size_t y = y0; y < y1; ++y) {
                std::memset(grid + idx3(0, y, z, nx, ny), 0, nx * sizeof(Real));
            }
        }
    }
}

// 7-point stencil computation (called from inside a parallel region).
// Interior points are averaged; boundary points are copied from input.
void stencilIteration(const Real* __restrict input,
                      Real* __restrict output,
                      const size_t nx, const size_t ny, const size_t nz, const Tiling& tl,
                      [[maybe_unused]] const bool streamStores) {
    const size_t plane = nx * ny;
    #pragma omp for schedule(static)
    for (size_t t = 0; t < tl.count(); ++t) {
        const size_t z0 = (t / tl.nyt) * tl.zb, z1 = std::min(nz, z0 + tl.zb);
        const size_t y0 = (t % tl.nyt) * tl.yb, y1 = std::min(ny, y0 + tl.yb);
        for (size_t z = z0; z < z1; ++z) {
            for (size_t y = y0; y < y1; ++y) {
                const size_t base = idx3(0, y, z, nx, ny);
                const Real* __restrict in = input + base;
                Real* __restrict out = output + base;
                if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
                    // Whole row lies on the boundary
                    std::memcpy(out, in, nx * sizeof(Real));
                    continue;
                }
                const Real* __restrict front = in - nx;
                const Real* __restrict back = in + nx;
                const Real* __restrict bottom = in - plane;
                const Real* __restrict top = in + plane;
                out[0] = in[0];
                size_t x = 1;
#if defined(__AVX__)
                if (streamStores) {
                    // Scalar peel until the output is 32-byte aligned
                    for (; x < nx - 1 && (reinterpret_cast<uintptr_t>(out + x) & 31) != 0; ++x) {
                        out[x] = (in[x] + in[x-1] + in[x+1] + front[x] + back[x] + bottom[x] + top[x]) / 7.0;
                    }
                    const __m256d seven = _mm256_set1_pd(7.0);
                    for (; x + 4 <= nx - 1; x += 4) {
                        // Same summation order as the scalar expression -> bitwise identical
                        __m256d v = _mm256_add_pd(_mm256_loadu_pd(in + x), _mm256_loadu_pd(in + x - 1));
                        v = _mm256_add_pd(v, _mm256_loadu_pd(in + x + 1));
                        v = _mm256_add_pd(v, _mm256_loadu_pd(front + x));
                        v = _mm256_add_pd(v, _mm256_loadu_pd(back + x));
                        v = _mm256_add_pd(v, _mm256_loadu_pd(bottom + x));
                        v = _mm256_add_pd(v, _mm256_loadu_pd(top + x));
                        _mm256_stream_pd(out + x, _mm256_div_pd(v, seven));
                    }
                }
#endif
                #pragma omp simd
                for (size_t i = x; i < nx - 1; ++i) {
                    // Simple averaging stencil (same summation order as reference)
                    out[i] = (in[i] + in[i-1] + in[i+1] + front[i] + back[i] + bottom[i] + top[i]) / 7.0;
                }
                out[nx - 1] = in[nx - 1];
            }
        }
    }
#if defined(__AVX__)
    if (streamStores) _mm_sfence();  // order streaming stores before the barrier
#endif
}

bool validateResult(const Grid& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    const Real* data = grid.data();
    const size_t n = grid.size();
    
    // 1. No NaN or Inf values
    bool nonFinite = false;
    #pragma omp parallel for schedule(static) reduction(||:nonFinite)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) nonFinite = true;
    }
    if (nonFinite) {
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
    
    // Allocate grids (double buffering)
    // (elements left untouched here; first touch happens in parallel below)
    Grid grid1(gridSize);
    Grid grid2(gridSize);
    Real* const g1 = grid1.data();
    Real* const g2 = grid2.data();
    
    // Initialize (grid2 is first-touched with the same tile distribution as the sweeps)
    printf("Initializing grid...\n");
    const std::vector<int> pinList = buildPinList();
    const Tiling tiling = makeTiling(nx, ny, nz, omp_get_max_threads());
    // Bypass the caches on stores only when the grids are far larger than the caches
    const bool streamStores = 2 * gridSize * sizeof(Real) > (size_t(512) << 20);
    #pragma omp parallel
    {
        pinCurrentThread(pinList);
        initializeGrid(g1, nx, ny, nz, tiling);
        zeroGrid(g2, nx, ny, nz, tiling);
    }
    
    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            // Implicit barrier at the end of each omp for separates iterations
            if (iter % 2 == 0) {
                stencilIteration(g1, g2, nx, ny, nz, tiling, streamStores);
            } else {
                stencilIteration(g2, g1, nx, ny, nz, tiling, streamStores);
            }
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
        const std::vector<Real> resultCopy(finalGrid.begin(), finalGrid.end());
        print_results(resultCopy, "Grid");
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
