#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#if defined(__x86_64__) && defined(__AVX__)
#include <cstdint>
#include <immintrin.h>
#define USE_NT_STORES 1
#endif

#include "../common/results_output.hpp"

using Real = double;

#ifdef __linux__
// Parse a Linux CPU list such as "0,128" or "0-3,8".
static std::vector<int> parseCpuList(const char* s) {
    std::vector<int> out;
    while (*s) {
        char* end = nullptr;
        long a = strtol(s, &end, 10);
        if (end == s) break;
        s = end;
        long b = a;
        if (*s == '-') {
            ++s;
            b = strtol(s, &end, 10);
            if (end == s) break;
            s = end;
        }
        for (long c = a; c <= b; ++c) out.push_back(static_cast<int>(c));
        while (*s == ',' || *s == '\n' || *s == ' ') ++s;
    }
    return out;
}

// Order the CPUs this process may run on so that all first SMT siblings come
// first (in ascending id order), then all second siblings, and so on. Binding
// consecutive OpenMP threads to this order spreads them across physical cores
// and, on a multi-socket machine, across NUMA nodes in a contiguous fashion
// that matches the static loop schedule used for first touch.
static std::vector<int> buildCpuOrder(size_t& physicalCores) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return {};

    std::vector<int> allowed;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) allowed.push_back(c);
    }
    if (allowed.empty()) return {};

    std::vector<std::vector<int>> byRank;
    for (const int c : allowed) {
        size_t rank = 0;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        if (FILE* f = fopen(path, "r")) {
            char buf[512] = {0};
            if (fgets(buf, sizeof(buf), f)) {
                for (const int sib : parseCpuList(buf)) {
                    if (sib < c && sib < CPU_SETSIZE && CPU_ISSET(sib, &mask)) ++rank;
                }
            }
            fclose(f);
        }
        if (byRank.size() <= rank) byRank.resize(rank + 1);
        byRank[rank].push_back(c);
    }

    std::vector<int> order;
    order.reserve(allowed.size());
    for (const auto& group : byRank) {
        order.insert(order.end(), group.begin(), group.end());
    }
    physicalCores = byRank.empty() ? allowed.size() : byRank[0].size();
    return order;
}
#endif

// Pin OpenMP threads to distinct cores. Without this the default (unbound)
// runtime lets threads migrate, which destroys NUMA locality for this
// bandwidth-bound kernel. An explicit user affinity request is left alone.
static void setupThreadAffinity() {
#ifdef __linux__
    if (getenv("OMP_PROC_BIND") || getenv("GOMP_CPU_AFFINITY") || getenv("OMP_PLACES")) return;

    size_t physicalCores = 0;
    const std::vector<int> order = buildCpuOrder(physicalCores);
    if (order.empty()) return;

    // Default to one thread per physical core; SMT siblings only add
    // contention on a memory-bound stencil.
    if (!getenv("OMP_NUM_THREADS") && physicalCores > 0) {
        omp_set_num_threads(static_cast<int>(physicalCores));
    }

    #pragma omp parallel
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        const size_t n = static_cast<size_t>(omp_get_num_threads());
        // Fewer threads than cores: spread them evenly over the physical cores
        // (and thus over the NUMA nodes) instead of packing one socket.
        const size_t cpu = (n <= physicalCores && physicalCores > 0)
                               ? order[(t * physicalCores) / n]
                               : order[t % order.size()];
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(static_cast<int>(cpu), &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
}

// Sum of the L3 slices visible to this process (one entry per distinct L3
// domain). Used to decide whether the working set is cache resident.
static size_t aggregateL3Bytes() {
#ifdef __linux__
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return 0;

    size_t totalBytes = 0;
    std::vector<int> seen;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &mask)) continue;
        for (int idx = 0; idx < 8; ++idx) {
            char path[160];
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/level", c, idx);
            FILE* f = fopen(path, "r");
            if (!f) break;
            int level = 0;
            const bool okLevel = (fscanf(f, "%d", &level) == 1);
            fclose(f);
            if (!okLevel || level != 3) continue;

            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", c, idx);
            int owner = c;
            if (FILE* g = fopen(path, "r")) {
                char buf[512] = {0};
                if (fgets(buf, sizeof(buf), g)) {
                    const std::vector<int> sharers = parseCpuList(buf);
                    if (!sharers.empty()) owner = *std::min_element(sharers.begin(), sharers.end());
                }
                fclose(g);
            }
            if (std::find(seen.begin(), seen.end(), owner) != seen.end()) continue;

            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/size", c, idx);
            if (FILE* g = fopen(path, "r")) {
                long kb = 0;
                char unit = 'K';
                if (fscanf(g, "%ld%c", &kb, &unit) >= 1 && kb > 0) {
                    size_t bytes = static_cast<size_t>(kb) * 1024;
                    if (unit == 'M') bytes *= 1024;
                    totalBytes += bytes;
                    seen.push_back(owner);
                }
                fclose(g);
            }
        }
    }
    return totalBytes;
#else
    return 0;
#endif
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Thread grid over the interior volume: pz slabs in z, each split into py
// slices in y. z is preferred (contiguous memory, better cache blocking) and y
// is only used to soak up threads that z cannot balance evenly.
struct ThreadGrid {
    size_t pz = 1;
    size_t py = 1;
};

static inline size_t ceilDiv(const size_t a, const size_t b) { return (a + b - 1) / b; }

static ThreadGrid chooseThreadGrid(const size_t interiorZ, const size_t interiorY, const size_t nth) {
    ThreadGrid best{std::max<size_t>(1, std::min(interiorZ, nth)), 1};
    size_t bestCost = ceilDiv(interiorZ, best.pz) * ceilDiv(interiorY, best.py);
    for (size_t py = 1; py <= nth && py <= interiorY; ++py) {
        if (nth % py != 0) continue;
        const size_t pz = nth / py;
        if (pz > interiorZ) continue;
        // Cost = work of the busiest thread, in rows.
        const size_t cost = ceilDiv(interiorZ, pz) * ceilDiv(interiorY, py);
        if (cost < bestCost) {
            bestCost = cost;
            best = {pz, py};
        }
    }
    return best;
}

// Rows of y processed per cache block. The sweep walks z inside a y-block, so
// the live data is three planes of (yb + 2) rows; sizing that to a core's
// private cache turns the z-neighbour accesses into cache hits.
static size_t chooseYBlock(const size_t nx, const size_t ny, const size_t nz) {
    if (nx < 3 || ny < 3 || nz < 3) return 1;
    const size_t targetBytes = 256 * 1024;
    size_t yb = targetBytes / (3 * nx * sizeof(Real));
    if (yb < 4) yb = 4;
    if (yb > 64) yb = 64;
    if (yb > ny - 2) yb = ny - 2;
    return yb;
}

void initializeGrid(std::vector<Real>& grid, const size_t nx, const size_t ny, const size_t nz) {
    Real* __restrict g = grid.data();
    const size_t total = nx * ny * nz;
    // idx == idx3(x, y, z, nx, ny), so the initial value only depends on the
    // linear index; a flat parallel sweep keeps the identical values while
    // matching the first-touch page distribution.
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < total; ++idx) {
        g[idx] = static_cast<Real>(idx % 19) * 1.0;
    }
}

// One interior row of the 7-point stencil. When the grids are far larger than
// the caches the output is written with non-temporal stores: nothing re-reads
// the output plane during the sweep, so skipping the read-for-ownership saves
// a third of the memory traffic of this bandwidth-bound kernel. For grids that
// stay resident in cache the opposite is true, hence the runtime switch.
template <bool NonTemporal>
static inline void stencilRow(const Real* __restrict c, const Real* __restrict fr,
                              const Real* __restrict bk, const Real* __restrict bo,
                              const Real* __restrict tp, Real* __restrict o,
                              const size_t nx) {
    size_t x = 1;
    const size_t xEnd = nx - 1;
#if defined(USE_NT_STORES)
    // Peel until the store address is 32-byte aligned, then stream 4 doubles
    // at a time, keeping the exact left-to-right summation order.
    while (NonTemporal && x < xEnd && (reinterpret_cast<uintptr_t>(o + x) & 31u) != 0) {
        o[x] = (c[x] + c[x-1] + c[x+1] + fr[x] + bk[x] + bo[x] + tp[x]) / 7.0;
        ++x;
    }
    const __m256d seven = _mm256_set1_pd(7.0);
    for (; NonTemporal && x + 4 <= xEnd; x += 4) {
        __m256d s = _mm256_loadu_pd(c + x);
        s = _mm256_add_pd(s, _mm256_loadu_pd(c + x - 1));
        s = _mm256_add_pd(s, _mm256_loadu_pd(c + x + 1));
        s = _mm256_add_pd(s, _mm256_loadu_pd(fr + x));
        s = _mm256_add_pd(s, _mm256_loadu_pd(bk + x));
        s = _mm256_add_pd(s, _mm256_loadu_pd(bo + x));
        s = _mm256_add_pd(s, _mm256_loadu_pd(tp + x));
        _mm256_stream_pd(o + x, _mm256_div_pd(s, seven));
    }
#endif
    #pragma omp simd
    for (size_t i = x; i < xEnd; ++i) {
        o[i] = (c[i] + c[i-1] + c[i+1] + fr[i] + bk[i] + bo[i] + tp[i]) / 7.0;
    }
}

// 7-point stencil computation. Called by every thread of an enclosing
// parallel region; ends with a barrier, so the caller can chain iterations
// without paying a fork/join per iteration.
void stencilIteration(const Real* __restrict in, Real* __restrict out,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t yb, const bool nonTemporal) {
    const size_t plane = nx * ny;
    const size_t total = plane * nz;
    const bool hasInterior = (nx >= 3 && ny >= 3 && nz >= 3);
    const size_t interiorY = hasInterior ? ny - 2 : 0;
    const size_t interiorZ = hasInterior ? nz - 2 : 0;
    const size_t zLast = (nz > 1) ? nz - 1 : 1;  // exclusive bound of interior planes

    {
        // Interior points. The threads are laid out as a pz x py grid over the
        // interior volume: each thread owns a contiguous z-slab (which keeps
        // its data on the NUMA node it first touched), split further in y only
        // when there are more threads than planes. Within its region a thread
        // walks z inside a y-block, so the three active planes of the block
        // stay in cache. The split is even to within one plane/row, so no
        // schedule-induced load imbalance remains.
        if (hasInterior) {
            const size_t nth = static_cast<size_t>(omp_get_num_threads());
            const size_t tid = static_cast<size_t>(omp_get_thread_num());
            const ThreadGrid tg = chooseThreadGrid(interiorZ, interiorY, nth);
            if (tid < tg.pz * tg.py) {
                const size_t zi = tid / tg.py;
                const size_t yi = tid % tg.py;
                const size_t z0 = 1 + (zi * interiorZ) / tg.pz;
                const size_t z1 = 1 + ((zi + 1) * interiorZ) / tg.pz;
                const size_t y0 = 1 + (yi * interiorY) / tg.py;
                const size_t y1 = 1 + ((yi + 1) * interiorY) / tg.py;

                for (size_t yblk = y0; yblk < y1; yblk += yb) {
                    const size_t yStop = std::min(yblk + yb, y1);
                    for (size_t z = z0; z < z1; ++z) {
                        for (size_t y = yblk; y < yStop; ++y) {
                            const size_t base = z * plane + y * nx;
                            const Real* __restrict c = in + base;
                            Real* __restrict o = out + base;
                            if (nonTemporal) {
                                stencilRow<true>(c, c - nx, c + nx, c - plane, c + plane, o, nx);
                            } else {
                                stencilRow<false>(c, c - nx, c + nx, c - plane, c + plane, o, nx);
                            }
                        }
                    }
                }
            }
#if defined(USE_NT_STORES)
            // Make the streaming stores visible before the region's barrier.
            if (nonTemporal) _mm_sfence();
#endif
        }

        // Copy boundary values: the two full z-faces plus the frame of every
        // interior plane (identical to a full-grid scan with a boundary test).
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < std::min(plane, total); ++i) {
            out[i] = in[i];
        }

        if (nz > 1) {
            const size_t off = (nz - 1) * plane;
            #pragma omp for schedule(static) nowait
            for (size_t i = 0; i < plane; ++i) {
                out[off + i] = in[off + i];
            }
        }

        #pragma omp for schedule(static) nowait
        for (size_t z = 1; z < zLast; ++z) {
            const size_t zbase = z * plane;
            // y = 0 and y = ny-1 rows (full rows).
            for (size_t i = 0; i < nx; ++i) {
                out[zbase + i] = in[zbase + i];
            }
            if (ny > 1) {
                const size_t off = zbase + (ny - 1) * nx;
                for (size_t i = 0; i < nx; ++i) {
                    out[off + i] = in[off + i];
                }
            }
            // x = 0 and x = nx-1 columns of the interior rows.
            for (size_t y = 1; y + 1 < ny; ++y) {
                const size_t off = zbase + y * nx;
                out[off] = in[off];
                out[off + nx - 1] = in[off + nx - 1];
            }
        }

        #pragma omp barrier
    }
}

bool validateResult(const std::vector<Real>& grid, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Simple sanity checks
    
    const Real* __restrict g = grid.data();
    const size_t n = grid.size();

    // 1. No NaN or Inf values
    bool bad = false;
    #pragma omp parallel for schedule(static) reduction(||:bad)
    for (size_t i = 0; i < n; ++i) {
        bad = bad || std::isnan(g[i]) || std::isinf(g[i]);
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

    // Query the cache topology before pinning narrows this thread's CPU mask.
    const size_t aggL3 = aggregateL3Bytes();
    setupThreadAffinity();

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    size_t gridSize = nx * ny * nz;

    printf("Threads: %d\n", omp_get_max_threads());

    // Allocate grids (double buffering). The raw storage is reserved first and
    // touched in parallel with the same static distribution used by the
    // stencil sweep, so pages land on the NUMA node of the thread that will
    // work on them.
    std::vector<Real> grid1;
    std::vector<Real> grid2;
    grid1.reserve(gridSize);
    grid2.reserve(gridSize);
    {
        Real* __restrict p1 = grid1.data();
        Real* __restrict p2 = grid2.data();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < gridSize; ++i) {
            p1[i] = 0.0;
            p2[i] = 0.0;
        }
    }
    grid1.resize(gridSize);
    grid2.resize(gridSize);

    const size_t yBlock = chooseYBlock(nx, ny, nz);

    // Non-temporal stores only pay off once the two grids no longer fit in the
    // aggregate last-level cache; below that, keeping the output cached for the
    // next iteration is far cheaper.
    const bool nonTemporal = (aggL3 > 0) && (gridSize * sizeof(Real) > aggL3);

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // One parallel region for the whole run: the per-iteration synchronisation
    // is then a single barrier instead of a fork/join.
    #pragma omp parallel
    {
        Real* const p1 = grid1.data();
        Real* const p2 = grid2.data();
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(p1, p2, nx, ny, nz, yBlock, nonTemporal);
            } else {
                stencilIteration(p2, p1, nx, ny, nz, yBlock, nonTemporal);
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
    const std::vector<Real>& finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
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
