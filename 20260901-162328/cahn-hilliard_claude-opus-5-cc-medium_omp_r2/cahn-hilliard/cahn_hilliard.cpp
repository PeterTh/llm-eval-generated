#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "../common/results_output.hpp"

// Physical parameters of the simulation. These are compile-time constants so that the
// stencil kernels can fold the (unit) grid spacings away; the values are identical to
// the ones used by the serial reference implementation.
namespace params {
constexpr double dx = 1.0;
constexpr double dy = 1.0;
constexpr double dz = 1.0;
constexpr double dt = 0.01;
constexpr double e_AA = -(2.0 / 9.0);
constexpr double e_BB = -(2.0 / 9.0);
constexpr double e_AB = (2.0 / 9.0);
constexpr double gam = 0.5;
constexpr double D = 1.0;
}  // namespace params

#ifdef __linux__
// Read the hardware threads sharing a physical core with `cpu` (sysfs sibling list).
std::vector<int> readThreadSiblings(const int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    std::vector<int> sibs;
    FILE* f = fopen(path, "r");
    if (f == nullptr) return sibs;
    char buf[512] = {};
    if (fgets(buf, sizeof(buf), f) != nullptr) {
        // Format is a comma separated list of ids and "a-b" ranges.
        const char* p = buf;
        while (*p != '\0') {
            char* endp = nullptr;
            const long lo = strtol(p, &endp, 10);
            if (endp == p) break;
            long hi = lo;
            if (*endp == '-') {
                p = endp + 1;
                hi = strtol(p, &endp, 10);
            }
            for (long c = lo; c <= hi; ++c) sibs.push_back(static_cast<int>(c));
            p = (*endp == ',') ? endp + 1 : endp;
        }
    }
    fclose(f);
    return sibs;
}

// libgomp parses OMP_PROC_BIND/OMP_PLACES before main() runs, so setting them from here is
// too late. Unpinned threads migrate across sockets and destroy the first-touch NUMA
// placement of the fields (measured: several times slower), so bind the team ourselves in a
// "spread" fashion over the physical cores we are allowed to run on. Any user supplied
// affinity configuration takes precedence and disables this.
void bindThreadsSpread() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    // Group the allowed hardware threads into physical cores, in ascending cpu id order.
    std::vector<std::vector<int>> cores;
    std::vector<char> taken(CPU_SETSIZE, 0);
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed) || taken[cpu]) continue;
        std::vector<int> core;
        for (const int s : readThreadSiblings(cpu)) {
            if (s >= 0 && s < CPU_SETSIZE && CPU_ISSET(s, &allowed) && !taken[s]) {
                core.push_back(s);
                taken[s] = 1;
            }
        }
        if (core.empty()) {
            core.push_back(cpu);
            taken[cpu] = 1;
        }
        cores.push_back(std::move(core));
    }
    if (cores.empty()) return;
    const long ncores = static_cast<long>(cores.size());

    // This is also the first parallel region, so it creates the thread pool that all
    // subsequent regions reuse.
#pragma omp parallel default(none) shared(cores, ncores)
    {
        const long tid = omp_get_thread_num();
        const long nthreads = omp_get_num_threads();
        // Spread: consecutive threads land on cores that are far apart, so a partially
        // filled machine still uses all sockets / memory controllers.
        const std::vector<int>& core = cores[static_cast<size_t>(tid * ncores / nthreads)];
        cpu_set_t set;
        CPU_ZERO(&set);
        for (const int cpu : core) CPU_SET(cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}
#else
void bindThreadsSpread() {}
#endif

// Aggregate size of all last level (L3) cache instances in the machine, used to decide
// which time stepping kernel to run. Returns a conservative default if sysfs is unavailable.
size_t totalLastLevelCacheBytes() {
    constexpr size_t fallback = 32ull << 20;
#ifdef __linux__
    size_t total = 0;
    std::vector<long> seenIds;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        char path[160];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d", cpu);
        if (access(path, F_OK) != 0) break;
        for (int index = 0; index < 8; ++index) {
            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/level", cpu, index);
            FILE* f = fopen(path, "r");
            if (f == nullptr) break;
            int level = 0;
            const int got = fscanf(f, "%d", &level);
            fclose(f);
            if (got != 1 || level != 3) continue;

            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/id", cpu, index);
            long id = -1;
            f = fopen(path, "r");
            if (f != nullptr) {
                if (fscanf(f, "%ld", &id) != 1) id = -1;
                fclose(f);
            }
            if (id >= 0 && std::find(seenIds.begin(), seenIds.end(), id) != seenIds.end()) continue;

            snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, index);
            f = fopen(path, "r");
            if (f == nullptr) continue;
            long kb = 0;
            const int gotSize = fscanf(f, "%ldK", &kb);
            fclose(f);
            if (gotSize != 1 || kb <= 0) continue;

            if (id >= 0) seenIds.push_back(id);
            total += static_cast<size_t>(kb) << 10;
        }
    }
    if (total > 0) return total;
#endif
    return fallback;
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// 64-byte aligned, *uninitialized* allocation. Leaving the memory untouched is essential:
// the first write to a page happens inside the OpenMP kernels, so pages end up on the NUMA
// node of the thread that owns them for the rest of the run (first-touch placement).
struct AlignedFree {
    void operator()(double* p) const noexcept { std::free(p); }
};
using AlignedArray = std::unique_ptr<double[], AlignedFree>;

AlignedArray allocateField(const size_t n) {
    const size_t bytes = ((n * sizeof(double)) + 63) & ~static_cast<size_t>(63);
    void* p = std::aligned_alloc(64, bytes);
    if (p == nullptr) {
        printf("Failed to allocate %zu bytes\n", bytes);
        exit(1);
    }
#ifdef __linux__
    // Large fields benefit strongly from transparent huge pages (fewer TLB misses).
    madvise(p, bytes, MADV_HUGEPAGE);
#endif
    return AlignedArray(static_cast<double*>(p));
}

// 7-point Laplacian for a single grid point, with clamped boundary conditions.
// The neighbouring rows/planes are passed in pre-clamped; only the x direction is
// resolved here so that the interior of a row can be vectorized.
[[gnu::always_inline]] inline double laplacian7(const double* __restrict r0, const double* __restrict rym,
                                                const double* __restrict ryp, const double* __restrict rzm,
                                                const double* __restrict rzp, const size_t x, const size_t xn,
                                                const size_t xp) noexcept {
    using namespace params;
    const double cc = r0[x];
    const double cxx = (r0[xp] + r0[xn] - 2.0 * cc) / (dx * dx);
    const double cyy = (ryp[x] + rym[x] - 2.0 * cc) / (dy * dy);
    const double czz = (rzp[x] + rzm[x] - 2.0 * cc) / (dz * dz);
    return cxx + cyy + czz;
}

[[gnu::always_inline]] inline double chemicalPotential(const double cv, const double lap) noexcept {
    using namespace params;
    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gam * lap;
}

// Chemical potential for one row (fixed y, z) of the grid.
inline void chemicalPotentialRow(double* __restrict mu, const double* __restrict r0, const double* __restrict rym,
                                 const double* __restrict ryp, const double* __restrict rzm,
                                 const double* __restrict rzp, const size_t nx) noexcept {
    if (nx == 1) {
        mu[0] = chemicalPotential(r0[0], laplacian7(r0, rym, ryp, rzm, rzp, 0, 0, 0));
        return;
    }
    mu[0] = chemicalPotential(r0[0], laplacian7(r0, rym, ryp, rzm, rzp, 0, 0, 1));
#pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        mu[x] = chemicalPotential(r0[x], laplacian7(r0, rym, ryp, rzm, rzp, x, x - 1, x + 1));
    }
    mu[nx - 1] = chemicalPotential(r0[nx - 1], laplacian7(r0, rym, ryp, rzm, rzp, nx - 1, nx - 2, nx - 1));
}

// Cahn-Hilliard update for one row (fixed y, z) of the grid.
inline void cahnHilliardRow(double* __restrict cnew, const double* __restrict cold, const double* __restrict r0,
                            const double* __restrict rym, const double* __restrict ryp, const double* __restrict rzm,
                            const double* __restrict rzp, const size_t nx) noexcept {
    using namespace params;
    if (nx == 1) {
        cnew[0] = cold[0] + dt * D * laplacian7(r0, rym, ryp, rzm, rzp, 0, 0, 0);
        return;
    }
    cnew[0] = cold[0] + dt * D * laplacian7(r0, rym, ryp, rzm, rzp, 0, 0, 1);
#pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        cnew[x] = cold[x] + dt * D * laplacian7(r0, rym, ryp, rzm, rzp, x, x - 1, x + 1);
    }
    cnew[nx - 1] = cold[nx - 1] + dt * D * laplacian7(r0, rym, ryp, rzm, rzp, nx - 1, nx - 2, nx - 1);
}

// Contiguous, balanced block decomposition of the z planes over the thread team. Every
// loop in the program (initialization, pre-faulting, time stepping) uses this exact same
// mapping, so each thread always touches the same portion of the fields.
inline void zSlab(const size_t nz, const int tid, const int nthreads, size_t& z0, size_t& z1) noexcept {
    const size_t base = nz / static_cast<size_t>(nthreads);
    const size_t rem = nz % static_cast<size_t>(nthreads);
    const size_t t = static_cast<size_t>(tid);
    z0 = t * base + std::min(t, rem);
    z1 = z0 + base + (t < rem ? 1 : 0);
}

// Initialize concentration field
void initializeConcentration(double* __restrict c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    // Contiguous static decomposition of the grid rows, matching the compute kernels, so
    // every page is first touched by the thread that will own it (NUMA placement).
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

// Pre-fault the second field from the thread that will own it. Also mirrors the
// value-initialization the serial reference performed when constructing its std::vectors,
// so page faults are not part of the measured region.
void firstTouch(double* __restrict f, const size_t nx, const size_t ny, const size_t nz) {
#pragma omp parallel for schedule(static) collapse(2)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            double* row = f + z * (nx * ny) + y * nx;
            for (size_t x = 0; x < nx; ++x) row[x] = 0.0;
        }
    }
}

// Width (in y rows) of the strips a thread sweeps through its z slab. The rolling chemical
// potential window holds 3 planes of (wy + 2) rows; keeping it around half of L2 makes the
// fused kernel read the concentration field from cache instead of memory.
constexpr size_t kMuWindowDoubles = 65536;  // 512 KiB

inline size_t chooseStripWidth(const size_t nx, const size_t ny) noexcept {
    size_t wy = kMuWindowDoubles / (3 * nx);
    wy = (wy > 2) ? wy - 2 : 1;
    if (wy > ny) wy = ny;
    if (wy == 0) wy = 1;
    // Balance the strips so none is a tiny leftover.
    const size_t nstrips = (ny + wy - 1) / wy;
    return (ny + nstrips - 1) / nstrips;
}

// Straightforward two-phase time loop: materialise mu for the whole grid, then update the
// concentration from it. Optimal while the three fields still fit in last level cache,
// because then the mu round trip costs nothing and no stencil work is duplicated.
void runSimulationTwoPass(double* __restrict a, double* __restrict b, double* __restrict mu,
                          const size_t nx, const size_t ny, const size_t nz, const int iterations) {
    const size_t plane = nx * ny;

#pragma omp parallel default(none) shared(a, b, mu) firstprivate(nx, ny, nz, iterations, plane)
    {
        for (int t = 0; t < iterations; ++t) {
            const double* __restrict cold = (t & 1) ? b : a;
            double* __restrict cnew = (t & 1) ? a : b;

            // Compute chemical potential
#pragma omp for schedule(static) collapse(2)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t zn = (z > 0) ? z - 1 : 0;
                    const size_t zp = (z < nz - 1) ? z + 1 : z;
                    const size_t yn = (y > 0) ? y - 1 : 0;
                    const size_t yp = (y < ny - 1) ? y + 1 : y;
                    const double* base = cold + z * plane;
                    chemicalPotentialRow(mu + z * plane + y * nx, base + y * nx, base + yn * nx, base + yp * nx,
                                         cold + zn * plane + y * nx, cold + zp * plane + y * nx, nx);
                }
            }

            // Update concentration
#pragma omp for schedule(static) collapse(2)
            for (size_t z = 0; z < nz; ++z) {
                for (size_t y = 0; y < ny; ++y) {
                    const size_t zn = (z > 0) ? z - 1 : 0;
                    const size_t zp = (z < nz - 1) ? z + 1 : z;
                    const size_t yn = (y > 0) ? y - 1 : 0;
                    const size_t yp = (y < ny - 1) ? y + 1 : y;
                    const double* base = mu + z * plane;
                    cahnHilliardRow(cnew + z * plane + y * nx, cold + z * plane + y * nx, base + y * nx,
                                    base + yn * nx, base + yp * nx, mu + zn * plane + y * nx,
                                    mu + zp * plane + y * nx, nx);
                }
            }
        }
    }
}

// Runs the full time loop.
//
// The two phases (chemical potential, concentration update) are fused: instead of
// materialising mu for the whole grid and streaming it back in, each thread sweeps its z
// slab strip by strip and keeps only a rolling window of three mu planes for the current
// strip in cache. That removes one full write and one full read of a grid-sized array per
// time step, at the cost of recomputing mu on the (thin) halo of each strip -- a good trade
// once the kernel is bound by memory bandwidth rather than by arithmetic.
//
// A single parallel region spans all time steps and the buffers are ping-ponged via
// pointer parity instead of swapping, so every thread stays on the same grid planes -- and
// hence the same NUMA pages -- for the whole simulation.
void runSimulationFused(double* __restrict a, double* __restrict b,
                        const size_t nx, const size_t ny, const size_t nz, const int iterations) {
    const size_t plane = nx * ny;
    const size_t wy = chooseStripWidth(nx, ny);
    const size_t bufRows = std::min(ny, wy + 2);

#pragma omp parallel default(none) shared(a, b) firstprivate(nx, ny, nz, iterations, plane, wy, bufRows)
    {
        size_t z0, z1;
        zSlab(nz, omp_get_thread_num(), omp_get_num_threads(), z0, z1);

        // Rolling mu window: 3 planes, indexed by (z % 3), each holding rows [by0, by1].
        AlignedArray window = allocateField(3 * bufRows * nx);
        double* const win = window.get();
        const size_t winStride = bufRows * nx;

        for (int t = 0; t < iterations; ++t) {
            const double* __restrict cold = (t & 1) ? b : a;
            double* __restrict cnew = (t & 1) ? a : b;

            for (size_t ys = 0; ys < ny && z0 < z1; ys += wy) {
                const size_t ye = std::min(ys + wy, ny);
                const size_t by0 = (ys > 0) ? ys - 1 : 0;              // first mu row held
                const size_t by1 = (ye < ny) ? ye : ny - 1;            // last mu row held

                // Fill the window plane for grid plane z (rows by0..by1).
                const auto muPlane = [&](const size_t z) {
                    const size_t zn = (z > 0) ? z - 1 : 0;
                    const size_t zp = (z < nz - 1) ? z + 1 : z;
                    double* dst = win + (z % 3) * winStride;
                    for (size_t y = by0; y <= by1; ++y) {
                        const size_t yn = (y > 0) ? y - 1 : 0;
                        const size_t yp = (y < ny - 1) ? y + 1 : y;
                        const double* base = cold + z * plane;
                        chemicalPotentialRow(dst + (y - by0) * nx, base + y * nx, base + yn * nx, base + yp * nx,
                                             cold + zn * plane + y * nx, cold + zp * plane + y * nx, nx);
                    }
                };

                // Prime the window with the planes needed by the first output plane.
                for (size_t p = (z0 > 0) ? z0 - 1 : 0; p <= std::min(z0 + 1, nz - 1); ++p) muPlane(p);

                for (size_t z = z0; z < z1; ++z) {
                    const size_t zn = (z > 0) ? z - 1 : 0;
                    const size_t zp = (z < nz - 1) ? z + 1 : z;
                    const double* wz = win + (z % 3) * winStride;
                    const double* wzn = win + (zn % 3) * winStride;
                    const double* wzp = win + (zp % 3) * winStride;
                    for (size_t y = ys; y < ye; ++y) {
                        const size_t yn = (y > 0) ? y - 1 : 0;
                        const size_t yp = (y < ny - 1) ? y + 1 : y;
                        cahnHilliardRow(cnew + z * plane + y * nx, cold + z * plane + y * nx, wz + (y - by0) * nx,
                                        wz + (yn - by0) * nx, wz + (yp - by0) * nx, wzn + (y - by0) * nx,
                                        wzp + (y - by0) * nx, nx);
                    }
                    // The plane just consumed frees the slot needed two steps ahead.
                    if (z + 1 < z1 && z + 2 < nz) muPlane(z + 2);
                }
            }

            // Neighbouring slabs must have published their new concentrations.
#pragma omp barrier
        }
    }
}

bool validateResult(const double* c, const size_t gridSize) {
    // Check for NaN or Inf
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(| : bad)
    for (size_t i = 0; i < gridSize; ++i) {
        if (std::isnan(c[i]) || std::isinf(c[i])) {
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
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < gridSize; ++i) {
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

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    bindThreadsSpread();

    size_t gridSize = nx * ny * nz;

    // Kernel choice: fusing the two phases trades a modest amount of duplicated stencil work
    // for one grid-sized array less of memory traffic per step. That only pays off once the
    // working set no longer fits in last level cache, and it needs enough z planes to give
    // every thread a slab of its own.
    const size_t workingSet = 3 * gridSize * sizeof(double);
    const bool fused = workingSet > 2 * totalLastLevelCacheBytes() &&
                       nz >= static_cast<size_t>(omp_get_max_threads());
    printf("Kernel: %s\n", fused ? "fused" : "two-pass");

    // Allocate arrays
    AlignedArray cold = allocateField(gridSize);
    AlignedArray cnew = allocateField(gridSize);
    AlignedArray mu;
    if (!fused) mu = allocateField(gridSize);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold.get(), nx, ny, nz);
    firstTouch(cnew.get(), nx, ny, nz);
    if (!fused) firstTouch(mu.get(), nx, ny, nz);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    if (fused) {
        runSimulationFused(cold.get(), cnew.get(), nx, ny, nz, iterations);
    } else {
        runSimulationTwoPass(cold.get(), cnew.get(), mu.get(), nx, ny, nz, iterations);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // After an odd number of steps the newest field lives in the second buffer.
    if (iterations & 1) cold.swap(cnew);
    const double* result = cold.get();

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(std::vector<double>(result, result + gridSize), "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, gridSize);

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
