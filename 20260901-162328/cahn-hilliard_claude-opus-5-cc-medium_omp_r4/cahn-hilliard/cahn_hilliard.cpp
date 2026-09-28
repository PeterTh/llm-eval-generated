#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// Pin every OpenMP thread to a distinct CPU, spread over the available cores.
// This kernel is memory bound, so keeping threads (and thus the pages they
// first touch) on a fixed NUMA node is worth several times the performance.
// Only done when the user did not request an affinity policy: libgomp's
// default is OMP_PROC_BIND=false with an empty place list, in which case the
// proc_bind clauses below have no effect and threads are free to migrate.
static void bindThreadsToCores() {
#ifdef __linux__
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &allowed)) cpus.push_back(i);
    }
    if (cpus.empty()) return;
    const int ncpus = static_cast<int>(cpus.size());

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nthreads = omp_get_num_threads();
        // Spread: with fewer threads than CPUs, stride over the CPU list so
        // both sockets are used before any SMT sibling pair is doubled up.
        const int slot = (nthreads <= ncpus)
                             ? static_cast<int>(static_cast<long long>(tid) * ncpus / nthreads)
                             : (tid % ncpus);
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[slot], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
}

// Smallest multiple of the 64 byte cache line that holds `n` bytes.
static inline constexpr size_t roundUpTo64(const size_t n) noexcept {
    return ((n + 63) & ~static_cast<size_t>(63)) + (n == 0 ? 64 : 0);
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// One row (fixed y, z) of the 7-point Laplacian stencil with clamped boundary
// conditions in x. `cc` is the row itself, the other four pointers are the
// already clamped neighbour rows in y and z. `body(index, value, laplacian)`
// produces the output value; `base` is the linear index of the row start.
template <typename Body>
static inline void stencilRow(const double* __restrict cc, const double* __restrict cyp,
                              const double* __restrict cyn, const double* __restrict czp,
                              const double* __restrict czn, double* __restrict o,
                              const size_t nx, const size_t base,
                              const double dx2, const double dy2, const double dz2,
                              Body body) {
    // x = 0 (clamped low edge)
    {
        const double c0 = cc[0];
        const double lap = (cc[(nx > 1) ? 1 : 0] + c0 - 2.0 * c0) / dx2
                         + (cyp[0] + cyn[0] - 2.0 * c0) / dy2
                         + (czp[0] + czn[0] - 2.0 * c0) / dz2;
        o[0] = body(base, c0, lap);
    }

    // interior in x: vectorizable, no boundary checks
    for (size_t x = 1; x + 1 < nx; ++x) {
        const double c0 = cc[x];
        const double lap = (cc[x + 1] + cc[x - 1] - 2.0 * c0) / dx2
                         + (cyp[x] + cyn[x] - 2.0 * c0) / dy2
                         + (czp[x] + czn[x] - 2.0 * c0) / dz2;
        o[x] = body(base + x, c0, lap);
    }

    // x = nx - 1 (clamped high edge)
    if (nx > 1) {
        const size_t x = nx - 1;
        const double c0 = cc[x];
        const double lap = (c0 + cc[x - 1] - 2.0 * c0) / dx2
                         + (cyp[x] + cyn[x] - 2.0 * c0) / dy2
                         + (czp[x] + czn[x] - 2.0 * c0) / dz2;
        o[x] = body(base + x, c0, lap);
    }
}

// Copy a freshly computed row to the grid with non-temporal (streaming)
// stores. The updated concentration is not read again during this time step,
// so bypassing the caches avoids both the read-for-ownership traffic of a
// normal store and the eviction of the data that is still needed. On this
// kernel that is worth about a third of the achievable memory bandwidth.
static inline void storeRowStreaming(double* __restrict dst, const double* __restrict src, const size_t n) {
#if defined(__AVX__)
    size_t i = 0;
    while (i < n && (reinterpret_cast<uintptr_t>(dst + i) & 31u) != 0) {
        dst[i] = src[i];
        ++i;
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_stream_pd(dst + i, _mm256_loadu_pd(src + i));
    }
    for (; i < n; ++i) {
        dst[i] = src[i];
    }
#else
    std::memcpy(dst, src, n * sizeof(double));
#endif
}

// Make streaming stores visible to the other threads before the next barrier.
static inline void streamingFence() {
#if defined(__AVX__)
    _mm_sfence();
#endif
}

// Total last level cache of the machine, in bytes: the size of one L3
// instance times the number of instances. Used to decide whether the working
// set still fits in cache.
static size_t totalLastLevelCache() {
    size_t perInstance = 0;
#ifdef _SC_LEVEL3_CACHE_SIZE
    const long sz = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (sz > 0) perInstance = static_cast<size_t>(sz);
#endif
    if (perInstance == 0) perInstance = 32u << 20;  // sensible default

    // Number of CPUs sharing one L3, to derive the number of instances.
    size_t sharing = 0;
#ifdef __linux__
    if (FILE* f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list", "r")) {
        char buf[512] = {0};
        if (fgets(buf, sizeof(buf), f)) {
            for (const char* p = buf; *p;) {
                char* e = nullptr;
                const long lo = strtol(p, &e, 10);
                if (e == p) break;
                long hi = lo;
                if (*e == '-') hi = strtol(e + 1, &e, 10);
                if (hi >= lo) sharing += static_cast<size_t>(hi - lo + 1);
                p = (*e == ',') ? e + 1 : e;
                if (*e != ',') break;
            }
        }
        fclose(f);
    }
    const long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    if (sharing == 0) sharing = 8;  // typical cores per L3
    if (ncpu > 0) return perInstance * std::max<size_t>(1, static_cast<size_t>(ncpu) / sharing);
#endif
    return perInstance;
}

// Decomposition of the (y, z) plane of grid rows into tiles. The tile grid is
// chosen so that every thread gets the same number of tiles (no static
// scheduling imbalance) and tiles stay roughly square, which keeps the halo
// the fused kernel recomputes small.
struct Tiling {
    size_t by = 1, bz = 1;    // largest tile extent in y and z
    size_t nby = 1, nbz = 1;  // number of tiles per direction
    size_t ntiles = 1;
    bool fused = false;       // worth running the fused (cache blocked) kernel?
};

static Tiling makeTiling(const size_t nx, const size_t ny, const size_t nz, const int nthreads) {
    const size_t nt = std::max(nthreads, 1);

    // The fused kernel keeps three planes of `by + 2` rows of mu live; size the
    // tile so that ring buffer stays inside a private L2 cache.
    const size_t kRingBudget = 384 * 1024;  // bytes
    size_t byMax = kRingBudget / (3 * sizeof(double) * std::max<size_t>(nx, 1));
    byMax = std::max<size_t>(byMax, 3) - 2;

    Tiling t;
    double bestScore = 0.0;

    // Try tile counts that are exact multiples of the thread count, and among
    // their factorizations pick the one with the smallest halo overhead.
    for (size_t mult = 1; mult <= 4 && bestScore == 0.0; ++mult) {
        const size_t total = mult * nt;
        for (size_t nby = 1; nby <= total; ++nby) {
            if (total % nby != 0) continue;
            const size_t nbz = total / nby;
            if (nby > ny || nbz > nz) continue;
            const size_t by = (ny + nby - 1) / nby;
            const size_t bz = (nz + nbz - 1) / nbz;
            if (mult < 4 && by > byMax) continue;  // last round: accept anything
            const double score = static_cast<double>(by + 4) * (bz + 4) / (static_cast<double>(by) * bz);
            if (bestScore == 0.0 || score < bestScore) {
                bestScore = score;
                t.nby = nby;
                t.nbz = nbz;
                t.by = by;
                t.bz = bz;
            }
        }
    }

    if (bestScore == 0.0) {
        // Fewer rows than threads (or a very flat grid): one tile per row.
        t.nby = ny;
        t.nbz = nz;
        t.by = 1;
        t.bz = 1;
        bestScore = 25.0;
    }
    t.ntiles = t.nby * t.nbz;

    // Fusing trades memory traffic (and cache friendliness of the stores) for
    // recomputation of the tile halo. It pays off only once the three fields
    // no longer fit in the last level cache - while they do, the two-pass
    // kernel gets mu for free and streaming stores would push the field out to
    // memory for nothing. The 1.75 factor is where the two cross over in
    // practice. The halo test guards against grid shapes that would make the
    // tiles too thin for the recomputation to be affordable.
    const double workingSet = 3.0 * sizeof(double) * static_cast<double>(nx) * ny * nz;
    t.fused = (bestScore <= 1.7) && (workingSet > 1.75 * static_cast<double>(totalLastLevelCache()));
    return t;
}

// Bounds of tile number `tile` in the (y, z) plane. The split is balanced:
// tile extents differ by at most one row and no tile is empty.
static inline void tileBounds(const Tiling& t, const size_t tile, const size_t ny, const size_t nz,
                              size_t& y0, size_t& y1, size_t& z0, size_t& z1) {
    const size_t ty = tile % t.nby;
    const size_t tz = tile / t.nby;
    y0 = ty * ny / t.nby;
    y1 = (ty + 1) * ny / t.nby;
    z0 = tz * nz / t.nbz;
    z1 = (tz + 1) * nz / t.nbz;
}

// Non-fused reference path: one full sweep of the grid, distributed over the
// same tiles as everything else so that the NUMA placement stays consistent.
//
// This is an orphaned worksharing construct: call it from a parallel region.
template <typename Body>
static inline void stencilSweep(const double* __restrict in, double* __restrict out,
                                const size_t nx, const size_t ny, const size_t nz,
                                const double dx2, const double dy2, const double dz2,
                                const Tiling& t, Body body) {
    const size_t slab = nx * ny;

#pragma omp for schedule(static) nowait
    for (size_t tile = 0; tile < t.ntiles; ++tile) {
        size_t y0, y1, z0, z1;
        tileBounds(t, tile, ny, nz, y0, y1, z0, z1);

        for (size_t z = z0; z < z1; ++z) {
            const size_t zp = ((z < nz - 1) ? z + 1 : z) * slab;
            const size_t zn = ((z > 0) ? z - 1 : 0) * slab;
            for (size_t y = y0; y < y1; ++y) {
                const size_t base = z * slab + y * nx;
                const size_t yp = ((y < ny - 1) ? y + 1 : y) * nx;
                const size_t yn = ((y > 0) ? y - 1 : 0) * nx;
                stencilRow(in + base, in + z * slab + yp, in + z * slab + yn,
                           in + zp + y * nx, in + zn + y * nx, out + base,
                           nx, base, dx2, dy2, dz2, body);
            }
        }
    }
}

// Compute chemical potential (non-fused path)
static inline void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                                            const size_t nx, const size_t ny, const size_t nz,
                                            const double dx2, const double dy2, const double dz2,
                                            const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                            const Tiling& t) {
    stencilSweep(c, mu, nx, ny, nz, dx2, dy2, dz2, t,
                 [=](const size_t, const double cv, const double lap) {
                     return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                            + 3.0 * cv + cv * cv * cv - gamma * lap;
                 });
}

// Cahn-Hilliard update step (non-fused path)
static inline void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                                      const double* __restrict mu,
                                      const size_t nx, const size_t ny, const size_t nz,
                                      const double D, const double dt,
                                      const double dx2, const double dy2, const double dz2,
                                      const Tiling& t) {
    stencilSweep(mu, cnew, nx, ny, nz, dx2, dy2, dz2, t,
                 [=](const size_t idx, const double, const double lap) {
                     return cold[idx] + dt * D * lap;
                 });
}

// Fused time step: for every tile, mu is produced plane by plane into a small
// per-thread ring buffer that stays in cache, and the concentration update
// consumes it immediately. Compared with two full sweeps this removes all
// memory traffic for mu (write + read) and the second read of the
// concentration field, at the cost of recomputing mu on a one row wide halo
// around each tile. The arithmetic per cell is unchanged, so results are
// bit-identical to the non-fused path.
//
// This is an orphaned worksharing construct: call it from a parallel region.
static inline void fusedStep(const double* __restrict cold, double* __restrict cnew,
                             double* __restrict ring, const size_t ringRows,
                             double* __restrict scratch,
                             const size_t nx, const size_t ny, const size_t nz,
                             const double dx2, const double dy2, const double dz2,
                             const double gamma, const double e_AA, const double e_BB, const double e_AB,
                             const double D, const double dt, const Tiling& t) {
    const size_t slab = nx * ny;
    const size_t ringPlane = ringRows * nx;

#pragma omp for schedule(static) nowait
    for (size_t tile = 0; tile < t.ntiles; ++tile) {
        size_t y0, y1, z0, z1;
        tileBounds(t, tile, ny, nz, y0, y1, z0, z1);
        if (y0 >= y1 || z0 >= z1) continue;

        // rows of mu this tile needs (its own rows plus one halo row per side)
        const size_t ylo = (y0 > 0) ? y0 - 1 : 0;
        const size_t yhi = (y1 < ny) ? y1 + 1 : ny;

        const size_t pstart = (z0 > 0) ? z0 - 1 : 0;
        const size_t pend = (z1 < nz) ? z1 : nz - 1;  // inclusive

        for (size_t p = pstart; p <= pend; ++p) {
            // --- produce mu for plane p into ring slot p % 3 ---
            double* __restrict dst = ring + (p % 3) * ringPlane;
            const size_t zp = ((p < nz - 1) ? p + 1 : p) * slab;
            const size_t zn = ((p > 0) ? p - 1 : 0) * slab;
            for (size_t y = ylo; y < yhi; ++y) {
                const size_t base = p * slab + y * nx;
                const size_t yp = ((y < ny - 1) ? y + 1 : y) * nx;
                const size_t yn = ((y > 0) ? y - 1 : 0) * nx;
                stencilRow(cold + base, cold + p * slab + yp, cold + p * slab + yn,
                           cold + zp + y * nx, cold + zn + y * nx,
                           dst + (y - ylo) * nx, nx, 0, dx2, dy2, dz2,
                           [=](const size_t, const double cv, const double lap) {
                               return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                      + 3.0 * cv + cv * cv * cv - gamma * lap;
                           });
            }

            // --- consume: every plane whose mu neighbourhood is complete ---
            // plane p - 1 becomes available once plane p is done; the last
            // plane of the grid clamps upwards and is available immediately.
            for (int pass = 0; pass < 2; ++pass) {
                size_t z;
                if (pass == 0) {
                    if (p < z0 + 1) continue;
                    z = p - 1;
                } else {
                    if (p != nz - 1 || z1 != nz) continue;
                    z = nz - 1;
                }

                const double* __restrict mzp = ring + (((z < nz - 1) ? z + 1 : z) % 3) * ringPlane;
                const double* __restrict mzn = ring + (((z > 0) ? z - 1 : 0) % 3) * ringPlane;
                const double* __restrict mcc = ring + (z % 3) * ringPlane;
                for (size_t y = y0; y < y1; ++y) {
                    const size_t base = z * slab + y * nx;
                    const size_t r = (y - ylo) * nx;
                    const size_t rp = (((y < ny - 1) ? y + 1 : y) - ylo) * nx;
                    const size_t rn = (((y > 0) ? y - 1 : 0) - ylo) * nx;
                    stencilRow(mcc + r, mcc + rp, mcc + rn, mzp + r, mzn + r,
                               scratch, nx, base, dx2, dy2, dz2,
                               [=](const size_t idx, const double, const double lap) {
                                   return cold[idx] + dt * D * lap;
                               });
                    storeRowStreaming(cnew + base, scratch, nx);
                }
            }
        }
    }
    streamingFence();
}

// Initialize concentration field (also performs the NUMA first touch of all
// buffers, using the same tile distribution as the compute kernels).
static void initializeConcentration(double* __restrict c, double* __restrict cnew, double* __restrict mu,
                                    const size_t nx, const size_t ny, const size_t nz, const Tiling& t) {
    const size_t vol = nx * ny * nz;
    const size_t slab = nx * ny;

#pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t tile = 0; tile < t.ntiles; ++tile) {
        size_t y0, y1, z0, z1;
        tileBounds(t, tile, ny, nz, y0, y1, z0, z1);

        for (size_t z = z0; z < z1; ++z) {
            for (size_t y = y0; y < y1; ++y) {
                const size_t base = z * slab + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = base + x;
                    // Generate pseudo-random value in [-1, 1]
                    const size_t linear_id = idx;
                    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                    c[idx] = -1.0 + 2.0 * pseudo;
                    cnew[idx] = 0.0;
                    if (mu) mu[idx] = 0.0;
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    if (n == 0) return true;
    const double* __restrict d = c.data();

    // Check for NaN or Inf
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(|:bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(d[i]) || std::isinf(d[i])) {
            bad = 1;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, d[i]);
        maxVal = std::max(maxVal, d[i]);
    }

    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);

    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 64)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of time steps (default: 20)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 64;
    size_t ny = 0;
    size_t nz = 0;
    int iterations = 20;
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

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Physical parameters
    const double dx = 1.0;
    const double dy = 1.0;
    const double dz = 1.0;
    const double dt = 0.01;
    const double e_AA = -(2.0 / 9.0);
    const double e_BB = -(2.0 / 9.0);
    const double e_AB = (2.0 / 9.0);
    const double gamma = 0.5;
    const double D = 1.0;

    size_t gridSize = nx * ny * nz;

    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    bindThreadsToCores();
    const Tiling tiling = makeTiling(nx, ny, nz, omp_get_max_threads());
    printf("Decomposition: %zu x %zu tiles of %zu x %zu rows (%s)\n",
           tiling.nbz, tiling.nby, tiling.bz, tiling.by, tiling.fused ? "fused" : "two-pass");

    // Allocate arrays. Raw (uninitialized) allocation lets the parallel
    // initialization below place each page on the NUMA node of the thread
    // that will work on it.
    const size_t bytes = roundUpTo64(gridSize * sizeof(double));
    double* cold = static_cast<double*>(std::aligned_alloc(64, bytes));
    double* cnew = static_cast<double*>(std::aligned_alloc(64, bytes));
    // The fused kernel keeps mu in per-thread ring buffers instead.
    double* mu = tiling.fused ? nullptr : static_cast<double*>(std::aligned_alloc(64, bytes));
    if (!cold || !cnew || (!tiling.fused && !mu)) {
        printf("Allocation failed\n");
        return 1;
    }

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, cnew, mu, nx, ny, nz, tiling);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Rows of mu a tile needs: its own plus one halo row on each side.
    const size_t ringRows = std::min(ny, tiling.by + 2);

    // One parallel region for the whole time loop: threads are forked once
    // instead of twice per time step.
#pragma omp parallel proc_bind(spread)
    {
        // Per-thread scratch space of the fused kernel, allocated (and thus
        // first touched) by the thread that uses it.
        double* ring = nullptr;
        double* scratch = nullptr;
        if (tiling.fused) {
            const size_t ringBytes = roundUpTo64(3 * ringRows * nx * sizeof(double));
            const size_t scratchBytes = roundUpTo64(nx * sizeof(double));
            ring = static_cast<double*>(std::aligned_alloc(64, ringBytes));
            scratch = static_cast<double*>(std::aligned_alloc(64, scratchBytes));
            if (!ring || !scratch) {
                printf("Allocation failed\n");
                std::exit(1);
            }
        }

        for (int t = 0; t < iterations; ++t) {
            if (tiling.fused) {
                fusedStep(cold, cnew, ring, ringRows, scratch, nx, ny, nz, dx2, dy2, dz2,
                          gamma, e_AA, e_BB, e_AB, D, dt, tiling);
#pragma omp barrier
            } else {
                // Compute chemical potential
                computeChemicalPotential(cold, mu, nx, ny, nz, dx2, dy2, dz2,
                                         gamma, e_AA, e_BB, e_AB, tiling);
#pragma omp barrier

                // Update concentration
                cahnHilliardUpdate(cnew, cold, mu, nx, ny, nz, D, dt, dx2, dy2, dz2, tiling);
#pragma omp barrier
            }

            // Swap buffers
#pragma omp single
            {
                std::swap(cold, cnew);
            }
        }

        std::free(ring);
        std::free(scratch);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults || validate) {
        const std::vector<double> result(cold, cold + gridSize);

        if (printResults) {
            print_results(result, "Concentration");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(result, nx, ny, nz);

            std::free(cold);
            std::free(cnew);
            std::free(mu);

            if (valid) {
                printf("Validation: PASSED\n");
                return 0;
            } else {
                printf("Validation: FAILED\n");
                return 1;
            }
        }
    }

    std::free(cold);
    std::free(cnew);
    std::free(mu);

    return 0;
}
