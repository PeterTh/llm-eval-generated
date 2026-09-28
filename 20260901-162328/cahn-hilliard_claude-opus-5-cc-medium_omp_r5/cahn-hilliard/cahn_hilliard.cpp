#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <omp.h>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Physical parameters of the benchmark; they are fixed, so they are compile time constants
namespace params {
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gamma = 0.5;
constexpr double D = 1.0;
}  // namespace params

// Laplacian of a single cell from its (already clamped) neighbour values
static inline double laplacianCell(const double v, const double xn, const double xp,
                                   const double yn, const double yp,
                                   const double zn, const double zp) noexcept {
    constexpr double dx = params::dx;
    constexpr double dy = params::dy;
    constexpr double dz = params::dz;

    const double cxx = (xp + xn - 2.0 * v) / (dx * dx);
    const double cyy = (yp + yn - 2.0 * v) / (dy * dy);
    const double czz = (zp + zn - 2.0 * v) / (dz * dz);

    return cxx + cyy + czz;
}

// Chemical potential of a single cell
static inline double muCell(const double cv, const double xn, const double xp,
                            const double yn, const double yp,
                            const double zn, const double zp) noexcept {
    constexpr double e_AA = params::e_AA;
    constexpr double e_BB = params::e_BB;
    constexpr double e_AB = params::e_AB;
    constexpr double gamma = params::gamma;

    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * laplacianCell(cv, xn, xp, yn, yp, zn, zp);
}

// Chemical potential for the x range [xlo, xhi] (inclusive) of one row; the caller supplies the
// row pointers of the (clamped) y/z neighbours, out[0] receives the value of column xlo.
static inline void computeMuRow(const double* __restrict cc, const double* __restrict cyn,
                                const double* __restrict cyp, const double* __restrict czn,
                                const double* __restrict czp, double* __restrict out,
                                const size_t xlo, const size_t xhi, const size_t nx) noexcept {
    size_t xs = xlo;
    size_t xe = xhi;

    // Columns at the domain border use the clamped stencil and are peeled off the vector loop
    if (xlo == 0) {
        out[0] = muCell(cc[0], cc[0], cc[(nx > 1) ? 1 : 0], cyn[0], cyp[0], czn[0], czp[0]);
        xs = 1;
    }
    if (xhi == nx - 1 && nx > 1) {
        out[xhi - xlo] = muCell(cc[nx - 1], cc[nx - 2], cc[nx - 1], cyn[nx - 1], cyp[nx - 1],
                                czn[nx - 1], czp[nx - 1]);
        xe = nx - 2;
    }

#pragma omp simd
    for (size_t x = xs; x <= xe; ++x) {
        out[x - xlo] = muCell(cc[x], cc[x - 1], cc[x + 1], cyn[x], cyp[x], czn[x], czp[x]);
    }
}

// Cahn-Hilliard update for n columns of one row. All mu pointers refer to the first column of
// the segment; the x neighbours are part of the halo of the mu tile, so no clamping is needed.
static inline void updateRow(const double* __restrict cold, const double* __restrict mc,
                             const double* __restrict myn, const double* __restrict myp,
                             const double* __restrict mzn, const double* __restrict mzp,
                             double* __restrict cnew, const size_t n) noexcept {
    constexpr double fac = params::dt * params::D;

    const double* __restrict mxn = mc - 1;
    const double* __restrict mxp = mc + 1;

#pragma omp simd
    for (size_t i = 0; i < n; ++i) {
        cnew[i] = cold[i] + fac * laplacianCell(mc[i], mxn[i], mxp[i], myn[i], myp[i], mzn[i],
                                                mzp[i]);
    }
}

// Copy a freshly computed row into the grid with non temporal stores. Once the grid no longer
// fits into the last level cache, the write allocate traffic of ordinary stores costs more than
// the extra L1 round trip through the row buffer (the values themselves are untouched, so the
// result is bit identical to the direct store path).
static inline void streamRow(double* __restrict dst, const double* __restrict src,
                             const size_t n) noexcept {
    size_t i = 0;
#if defined(__AVX__)
    while (i < n && (reinterpret_cast<uintptr_t>(dst + i) & 31u) != 0) {
        dst[i] = src[i];
        ++i;
    }
    for (; i + 4 <= n; i += 4) {
        _mm256_stream_pd(dst + i, _mm256_loadu_pd(src + i));
    }
#endif
    for (; i < n; ++i) dst[i] = src[i];
}

// Blocking of the iteration space into tiles that are processed by a single thread each.
// The chemical potential is never materialized for the whole grid: every tile keeps a rolling
// window of three mu planes in the private cache, which fuses both sweeps of a time step into a
// single pass over the grid and cuts the memory traffic roughly in half.
struct Tiling {
    size_t xb = 1, yb = 1, zb = 1;     // tile extents
    size_t nxb = 1, nyb = 1, nzb = 1;  // number of tiles per dimension
    size_t rowStride = 1;              // elements per mu row (tile width + halo + padding)
    size_t planeCap = 1;               // elements per mu plane
};

static Tiling makeTiling(const size_t nx, const size_t ny, const size_t nz, const size_t nthreads) {
    // Rolling mu window budget; small enough for two SMT siblings to share an L2 cache
    constexpr size_t bufferBudget = 64 * 1024;
    constexpr size_t maxTileWidth = 256;

    Tiling t;
    if (nx == 0 || ny == 0 || nz == 0) return t;  // empty grid, nothing to tile

    t.nxb = (nx + maxTileWidth - 1) / maxTileWidth;
    t.xb = (nx + t.nxb - 1) / t.nxb;
    // One halo column on each side, padded so that every mu row starts on a cache line
    t.rowStride = (t.xb + 2 + 7) / 8 * 8 + 8;

    t.yb = ny;
    while (t.yb > 1 && 3 * t.rowStride * (t.yb + 2) * sizeof(double) > bufferBudget) {
        t.yb = (t.yb + 1) / 2;
    }

    const auto derive = [&]() {
        t.nyb = (ny + t.yb - 1) / t.yb;
        const size_t planar = t.nxb * t.nyb;
        size_t nzb = (nthreads + planar - 1) / planar;
        if (nzb > nz) nzb = nz;
        if (nzb < 1) nzb = 1;
        t.zb = (nz + nzb - 1) / nzb;
        t.nzb = (nz + t.zb - 1) / t.zb;
    };
    derive();

    // Flat tiles recompute a large fraction of the mu planes at their z borders; getting the
    // required parallelism from thinner y slabs instead is the cheaper trade.
    while (t.zb < 4 && t.yb > 2 && t.nxb * t.nyb * t.nzb < nthreads) {
        t.yb = (t.yb + 1) / 2;
        derive();
    }

    t.planeCap = t.rowStride * (t.yb + 2);
    return t;
}

// Advance one tile by a single time step
template <bool Stream>
static void processTile(const double* __restrict cold, double* __restrict cnew,
                        double* __restrict mubuf, double* __restrict rowbuf, const size_t nx,
                        const size_t ny, const size_t nz, const Tiling& t, const size_t x0,
                        const size_t x1, const size_t y0, const size_t y1, const size_t z0,
                        const size_t z1) noexcept {
    const size_t plane = nx * ny;
    const size_t width = x1 - x0;

    // Halo of the mu tile: one row/plane on each side, clamped at the domain border
    const size_t ylo = (y0 > 0) ? y0 - 1 : 0;
    const size_t yhi = (y1 < ny) ? y1 : ny - 1;
    const size_t xlo = (x0 > 0) ? x0 - 1 : 0;
    const size_t xhi = (x1 < nx) ? x1 : nx - 1;

    // Column x0 of a mu row lives at offset 8, so the vector loops stay cache line aligned
    constexpr size_t off = 8;
    const auto muRow = [&](const size_t zz, const size_t y) noexcept {
        return mubuf + (zz % 3) * t.planeCap + (y - ylo) * t.rowStride + off;
    };

    const auto computeMuPlane = [&](const size_t zz) noexcept {
        const size_t zn = (zz > 0) ? zz - 1 : 0;
        const size_t zp = (zz + 1 < nz) ? zz + 1 : zz;
        const double* __restrict cz = cold + zz * plane;
        for (size_t y = ylo; y <= yhi; ++y) {
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t yp = (y + 1 < ny) ? y + 1 : y;
            double* __restrict out = muRow(zz, y);
            computeMuRow(cz + y * nx, cz + yn * nx, cz + yp * nx, cold + zn * plane + y * nx,
                         cold + zp * plane + y * nx, (x0 > 0) ? out - 1 : out, xlo, xhi, nx);
            // Halo columns of a tile that touches the domain border repeat the clamped value
            if (x0 == 0) out[-1] = out[0];
            if (x1 == nx) out[width] = out[width - 1];
        }
    };

    for (size_t z = z0; z < z1; ++z) {
        if (z == z0) {
            const size_t zstart = (z > 0) ? z - 1 : 0;
            const size_t zend = (z + 1 < nz) ? z + 1 : z;
            for (size_t zz = zstart; zz <= zend; ++zz) computeMuPlane(zz);
        } else if (z + 1 < nz) {
            computeMuPlane(z + 1);
        }

        const size_t zn = (z > 0) ? z - 1 : 0;
        const size_t zp = (z + 1 < nz) ? z + 1 : z;
        for (size_t y = y0; y < y1; ++y) {
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t yp = (y + 1 < ny) ? y + 1 : y;
            const size_t base = z * plane + y * nx + x0;
            if constexpr (Stream) {
                updateRow(cold + base, muRow(z, y), muRow(z, yn), muRow(z, yp), muRow(zn, y),
                          muRow(zp, y), rowbuf, width);
                streamRow(cnew + base, rowbuf, width);
            } else {
                updateRow(cold + base, muRow(z, y), muRow(z, yn), muRow(z, yp), muRow(zn, y),
                          muRow(zp, y), cnew + base, width);
            }
        }
    }

#if defined(__AVX__)
    // Make the streaming stores visible before the barrier releases the next time step
    if constexpr (Stream) _mm_sfence();
#endif
}

// The kernel is bandwidth and cache bound, so it depends on stable thread placement and on the
// first touch NUMA placement staying valid. Unless the runtime pins the threads itself, pin them
// here: one thread per physical core, spread over the sockets, only falling back to the SMT
// siblings once all cores are taken (the default place list of the runtime enumerates the SMT
// siblings of a core before moving on to the next core, which leaves half of the machine idle).
#if defined(__linux__)
#include <sched.h>

struct CpuTopology {
    std::vector<int> order;  // usable cpus, one per physical core first, SMT siblings last
    int cores = 0;           // number of physical cores in the list
    size_t llcBytes = 0;     // aggregated size of all reachable last level caches
};

// First integer of a sysfs cpu attribute, -1 if unavailable
static long sysfsCpuValue(const int cpu, const char* attr) {
    char path[192];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/%s", cpu, attr);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    long v = -1;
    char suffix = 0;
    if (fscanf(f, "%ld%c", &v, &suffix) >= 1) {
        if (suffix == 'K') v *= 1024;
        else if (suffix == 'M') v *= 1024 * 1024;
    }
    fclose(f);
    return v;
}

static CpuTopology cpuTopology() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return {};

    CpuTopology topo;
    std::vector<int> siblings;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;

        // Count every last level cache once, via the first cpu that shares it
        for (int idx = 3; idx >= 1; --idx) {
            char attr[64];
            snprintf(attr, sizeof(attr), "cache/index%d/shared_cpu_list", idx);
            const long owner = sysfsCpuValue(cpu, attr);
            if (owner < 0) continue;
            if (owner == cpu) {
                snprintf(attr, sizeof(attr), "cache/index%d/size", idx);
                const long size = sysfsCpuValue(cpu, attr);
                if (size > 0) topo.llcBytes += static_cast<size_t>(size);
            }
            break;
        }

        const int first = static_cast<int>(sysfsCpuValue(cpu, "topology/thread_siblings_list"));
        if (first < 0 || first == cpu) {
            topo.order.push_back(cpu);
            ++topo.cores;
        } else {
            siblings.push_back(cpu);
        }
    }
    topo.order.insert(topo.order.end(), siblings.begin(), siblings.end());
    return topo;
}

static void bindThreads(const int nthreads, const CpuTopology& topo) {
    // If the runtime binds the threads itself (OMP_PROC_BIND or OMP_PLACES set), that is the
    // user's decision and it is left alone; it would also undo this binding on every team start.
    if (topo.order.empty() || omp_get_proc_bind() != omp_proc_bind_false) return;

    const int ncpus = static_cast<int>(topo.order.size());
    // Use one hardware thread per core as long as there are enough cores
    const int span = (nthreads <= topo.cores) ? topo.cores : ncpus;
#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int slot = (nt >= span) ? (tid % ncpus)
                                      : static_cast<int>((long long)tid * span / nt);
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(topo.order[slot], &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
    }
}
#else
struct CpuTopology {
    std::vector<int> order;
    int cores = 0;
    size_t llcBytes = 0;
};
static CpuTopology cpuTopology() { return {}; }
static void bindThreads(int, const CpuTopology&) {}
#endif

// Initialize concentration field (and first touch both grid buffers with the same tile
// distribution that the time loop uses)
void initializeConcentration(std::vector<double>& c, std::vector<double>& scratch, const size_t nx,
                             const size_t ny, const size_t nz, const Tiling& t) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(3) schedule(static)
    for (size_t zb = 0; zb < t.nzb; ++zb) {
        for (size_t yb = 0; yb < t.nyb; ++yb) {
            for (size_t xb = 0; xb < t.nxb; ++xb) {
                const size_t z1 = std::min(nz, (zb + 1) * t.zb);
                const size_t y1 = std::min(ny, (yb + 1) * t.yb);
                const size_t x1 = std::min(nx, (xb + 1) * t.xb);
                for (size_t z = zb * t.zb; z < z1; ++z) {
                    for (size_t y = yb * t.yb; y < y1; ++y) {
                        for (size_t x = xb * t.xb; x < x1; ++x) {
                            const size_t idx = idx3(x, y, z, nx, ny);
                            // Generate pseudo-random value in [-1, 1]
                            const size_t linear_id = z * (nx * ny) + y * nx + x;
                            const double pseudo =
                                ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                            c[idx] = -1.0 + 2.0 * pseudo;
                            scratch[idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();

    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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

    // Thread placement: hyperthreading only adds cache pressure to this kernel, so unless the
    // user asked for a specific number of threads, run one thread per physical core.
    const CpuTopology topo = cpuTopology();
    if (getenv("OMP_NUM_THREADS") == nullptr && topo.cores > 0 &&
        topo.cores < omp_get_max_threads()) {
        omp_set_num_threads(topo.cores);
    }
    omp_set_dynamic(0);  // the per thread scratch buffers assume a fixed team size
    const int nthreads = omp_get_max_threads();
    bindThreads(nthreads, topo);

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("OpenMP threads: %d\n", nthreads);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    size_t gridSize = nx * ny * nz;

    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);

    const Tiling tiling = makeTiling(nx, ny, nz, static_cast<size_t>(nthreads));

    // Once the two grids together clearly exceed the aggregated last level cache, writing the
    // result with non temporal stores pays off because it saves the write allocate traffic. Below
    // that size the result is still reused from cache in the next time step, which is worth more.
    const bool stream = topo.llcBytes > 0 && 2 * gridSize * sizeof(double) > 2 * topo.llcBytes;

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, cnew, nx, ny, nz, tiling);

    // Scratch space of the worker threads: a rolling window of three chemical potential planes
    // plus the staging row of the streaming store path. Allocated (and first touched) by its
    // owner ahead of the measurement, just like the serial version allocates its mu array.
    std::vector<std::vector<double>> threadScratch(nthreads);
#pragma omp parallel
    {
        threadScratch[omp_get_thread_num()].assign(3 * tiling.planeCap + tiling.xb + 16, 0.0);
    }

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    const double* src = cold.data();
    double* dst = cnew.data();

#pragma omp parallel
    {
        double* mubuf = threadScratch[omp_get_thread_num()].data();
        mubuf += (64 - (reinterpret_cast<uintptr_t>(mubuf) % 64)) % 64 / sizeof(double);
        double* rowbuf = mubuf + 3 * tiling.planeCap;  // staging row of the streaming store path

        for (int t = 0; t < iterations; ++t) {
#pragma omp for collapse(3) schedule(static)
            for (size_t zb = 0; zb < tiling.nzb; ++zb) {
                for (size_t yb = 0; yb < tiling.nyb; ++yb) {
                    for (size_t xb = 0; xb < tiling.nxb; ++xb) {
                        const size_t x0 = xb * tiling.xb;
                        const size_t x1 = std::min(nx, (xb + 1) * tiling.xb);
                        const size_t y0 = yb * tiling.yb;
                        const size_t y1 = std::min(ny, (yb + 1) * tiling.yb);
                        const size_t z0 = zb * tiling.zb;
                        const size_t z1 = std::min(nz, (zb + 1) * tiling.zb);
                        if (stream) {
                            processTile<true>(src, dst, mubuf, rowbuf, nx, ny, nz, tiling, x0, x1,
                                              y0, y1, z0, z1);
                        } else {
                            processTile<false>(src, dst, mubuf, rowbuf, nx, ny, nz, tiling, x0, x1,
                                               y0, y1, z0, z1);
                        }
                    }
                }
            }

            // Swap buffers
#pragma omp single
            {
                const double* tmp = src;
                src = dst;
                dst = const_cast<double*>(tmp);
            }
        }
    }

    // Keep the final state in cold, as the serial version does
    if (src != cold.data()) {
        cold.swap(cnew);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(cold, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, nx, ny, nz);

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
