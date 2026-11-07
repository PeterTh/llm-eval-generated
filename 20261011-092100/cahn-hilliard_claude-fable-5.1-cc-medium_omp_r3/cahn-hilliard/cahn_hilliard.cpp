#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <pthread.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// ---------------------------------------------------------------------------
// Point-wise arithmetic.  The expression structure is kept identical to the
// original scalar code so that results are bit-for-bit reproducible
// irrespective of thread count or tiling.
// ---------------------------------------------------------------------------

// Grid-spacing factors for the Laplacian.  When 1/(d*d) is exactly
// representable (d*d is a power of two, e.g. the default spacing of 1.0),
// dividing by d*d and multiplying by its reciprocal give bit-identical
// results, so the much cheaper multiply can be used (kExactRecip = true).
struct Spacing {
    double dx2, dy2, dz2;       // d*d
    double rdx2, rdy2, rdz2;    // 1/(d*d), only used when exact
    bool exactRecip;
};

inline bool isPowerOfTwo(const double v) noexcept {
    if (!(v > 0.0) || std::isinf(v)) return false;
    int e;
    return std::frexp(v, &e) == 0.5;
}

inline Spacing makeSpacing(const double dx, const double dy, const double dz) noexcept {
    Spacing s;
    s.dx2 = dx * dx;
    s.dy2 = dy * dy;
    s.dz2 = dz * dz;
    s.rdx2 = 1.0 / s.dx2;
    s.rdy2 = 1.0 / s.dy2;
    s.rdz2 = 1.0 / s.dz2;
    s.exactRecip = isPowerOfTwo(s.dx2) && isPowerOfTwo(s.dy2) && isPowerOfTwo(s.dz2);
    return s;
}

// 7-point Laplacian from already-gathered neighbour values
template <bool kExactRecip>
inline double laplacian7(const double cc, const double cxp, const double cxn,
                         const double cyp, const double cyn, const double czp, const double czn,
                         const Spacing& sp) noexcept {
    if constexpr (kExactRecip) {
        const double cxx = (cxp + cxn - 2.0 * cc) * sp.rdx2;
        const double cyy = (cyp + cyn - 2.0 * cc) * sp.rdy2;
        const double czz = (czp + czn - 2.0 * cc) * sp.rdz2;
        return cxx + cyy + czz;
    } else {
        const double cxx = (cxp + cxn - 2.0 * cc) / sp.dx2;
        const double cyy = (cyp + cyn - 2.0 * cc) / sp.dy2;
        const double czz = (czp + czn - 2.0 * cc) / sp.dz2;
        return cxx + cyy + czz;
    }
}

// Compute Laplacian with clamped boundary conditions (generic, any point)
template <bool kExactRecip>
inline double computeLaplacian(const double* __restrict c, const size_t nx, const size_t ny, const size_t nz,
                               const Spacing& sp,
                               const size_t x, const size_t y, const size_t z) noexcept {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    return laplacian7<kExactRecip>(c[idx3(x, y, z, nx, ny)],
                                   c[idx3(xp, y, z, nx, ny)], c[idx3(xn, y, z, nx, ny)],
                                   c[idx3(x, yp, z, nx, ny)], c[idx3(x, yn, z, nx, ny)],
                                   c[idx3(x, y, zp, nx, ny)], c[idx3(x, y, zn, nx, ny)],
                                   sp);
}

inline double chemicalPotentialPoint(const double cv, const double lap,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB) noexcept {
    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * lap;
}

// ---------------------------------------------------------------------------
// Row kernels
// ---------------------------------------------------------------------------

// Chemical potential for one full row (y, z) of the concentration field c,
// written to out[0..nx).
template <bool kExactRecip>
inline void chemicalPotentialRow(const double* __restrict c, double* __restrict out,
                                 const size_t nx, const size_t ny, const size_t nz,
                                 const size_t y, const size_t z, const Spacing& sp,
                                 const double gamma, const double e_AA, const double e_BB, const double e_AB) noexcept {
    const size_t sxy = nx * ny;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t zn = (z > 0) ? z - 1 : 0;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;

    const double* __restrict row   = c + z * sxy + y * nx;
    const double* __restrict rowYp = c + z * sxy + yp * nx;
    const double* __restrict rowYn = c + z * sxy + yn * nx;
    const double* __restrict rowZp = c + zp * sxy + y * nx;
    const double* __restrict rowZn = c + zn * sxy + y * nx;

    // x = 0 boundary (clamped)
    out[0] = chemicalPotentialPoint(row[0], computeLaplacian<kExactRecip>(c, nx, ny, nz, sp, 0, y, z),
                                    gamma, e_AA, e_BB, e_AB);
    // interior: no clamping, vectorizable
    for (size_t x = 1; x + 1 < nx; ++x) {
        const double cv = row[x];
        const double lap = laplacian7<kExactRecip>(cv, row[x + 1], row[x - 1], rowYp[x], rowYn[x],
                                                   rowZp[x], rowZn[x], sp);
        out[x] = chemicalPotentialPoint(cv, lap, gamma, e_AA, e_BB, e_AB);
    }
    if (nx > 1) {
        out[nx - 1] = chemicalPotentialPoint(row[nx - 1],
                                             computeLaplacian<kExactRecip>(c, nx, ny, nz, sp, nx - 1, y, z),
                                             gamma, e_AA, e_BB, e_AB);
    }
}

// Cahn-Hilliard update for one row given the (already clamped) neighbouring
// rows of the chemical potential.
template <bool kExactRecip>
inline void cahnHilliardRow(double* __restrict out, const double* __restrict old,
                            const double* __restrict m,
                            const double* __restrict mYp, const double* __restrict mYn,
                            const double* __restrict mZp, const double* __restrict mZn,
                            const size_t nx, const double D, const double dt, const Spacing& sp) noexcept {
    {
        const size_t xp = (nx > 1) ? 1 : 0;
        out[0] = old[0] + dt * D *
                 laplacian7<kExactRecip>(m[0], m[xp], m[0], mYp[0], mYn[0], mZp[0], mZn[0], sp);
    }
    for (size_t x = 1; x + 1 < nx; ++x) {
        out[x] = old[x] + dt * D *
                 laplacian7<kExactRecip>(m[x], m[x + 1], m[x - 1], mYp[x], mYn[x], mZp[x], mZn[x], sp);
    }
    if (nx > 1) {
        const size_t x = nx - 1;
        out[x] = old[x] + dt * D *
                 laplacian7<kExactRecip>(m[x], m[x], m[x - 1], mYp[x], mYn[x], mZp[x], mZn[x], sp);
    }
}

// ---------------------------------------------------------------------------
// Tiling of the (z, y) plane set into blocks that are distributed over threads.
// Each tile is processed by one thread as a single fused sweep: the chemical
// potential is computed (with a one-row/one-plane halo) into a small rolling
// buffer and immediately consumed by the concentration update, so each time
// step streams the concentration field through memory only once.
// ---------------------------------------------------------------------------
struct Tiling {
    size_t ytile = 1;   // rows per tile
    size_t ztile = 1;   // planes per tile
    size_t nyt = 1;     // number of tiles in y
    size_t nzs = 1;     // number of tiles in z
};

Tiling chooseTiling(const size_t nx, const size_t ny, const size_t nz, const size_t nthreads) {
    // Keep the rolling mu buffer (3 planes of (ytile + 2) rows) small enough to
    // stay in the private/shared cache: aim for <= 1 MiB.
    const size_t bufLimitBytes = size_t(1) << 20;
    size_t ytileMax = bufLimitBytes / (3 * nx * sizeof(double));
    ytileMax = (ytileMax > 2) ? ytileMax - 2 : 1;
    ytileMax = std::min(ytileMax, ny);
    ytileMax = std::max<size_t>(ytileMax, 1);
    const size_t nytMin = (ny + ytileMax - 1) / ytileMax;

    Tiling best;
    double bestCost = -1.0;
    const size_t nytCap = std::min<size_t>(ny, 8192);
    const size_t nzsCap = std::min<size_t>(nz, 8192);
    for (size_t nyt = nytMin; nyt <= nytCap; ++nyt) {
        const size_t yt = (ny + nyt - 1) / nyt;
        if ((ny + yt - 1) / yt != nyt) continue;   // skip equivalent decompositions
        for (size_t nzs = 1; nzs <= nzsCap; ++nzs) {
            const size_t zt = (nz + nzs - 1) / nzs;
            if ((nz + zt - 1) / zt != nzs) continue;
            const size_t ntiles = nyt * nzs;
            const size_t perThread = (ntiles + nthreads - 1) / nthreads;
            // Work per tile ~ halo-extended mu computation + update
            const double cost = double(perThread) *
                                (double(yt + 2) * double(zt + 2) + double(yt) * double(zt));
            if (bestCost < 0.0 || cost < bestCost) {
                bestCost = cost;
                best.ytile = yt;
                best.ztile = zt;
                best.nyt = nyt;
                best.nzs = nzs;
            }
            if (ntiles >= nthreads * 8) break;   // more tiles only add halo work
        }
    }
    return best;
}

// One fused time step over the whole grid.
// Must be called from inside an OpenMP parallel region (orphaned worksharing
// loop); `ring` is a thread-private buffer of 3 * (ytile + 2) * nx doubles.
template <bool kExactRecip>
void cahnHilliardStepImpl(double* __restrict cnew, const double* __restrict cold,
                          double* __restrict ring, const Tiling& tl,
                          const size_t nx, const size_t ny, const size_t nz,
                          const double D, const double dt, const Spacing& sp,
                          const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t sxy = nx * ny;
    const size_t planeStride = (tl.ytile + 2) * nx;

#pragma omp for collapse(2) schedule(static)
    for (size_t zs = 0; zs < tl.nzs; ++zs) {
        for (size_t yt = 0; yt < tl.nyt; ++yt) {
            const size_t z0 = zs * tl.ztile;
            const size_t z1 = std::min(z0 + tl.ztile, nz);
            const size_t y0 = yt * tl.ytile;
            const size_t y1 = std::min(y0 + tl.ytile, ny);
            if (z0 >= z1 || y0 >= y1) continue;

            const size_t ybeg = (y0 > 0) ? y0 - 1 : 0;
            const size_t yend = std::min(y1 + 1, ny);

            auto muPlane = [&](const size_t zz) {
                double* __restrict slot = ring + (zz % 3) * planeStride;
                for (size_t y = ybeg; y < yend; ++y) {
                    chemicalPotentialRow<kExactRecip>(cold, slot + (y - ybeg) * nx, nx, ny, nz, y, zz,
                                                      sp, gamma, e_AA, e_BB, e_AB);
                }
            };

            if (z0 > 0) muPlane(z0 - 1);
            muPlane(z0);

            for (size_t z = z0; z < z1; ++z) {
                if (z + 1 < nz) muPlane(z + 1);
                const size_t zp = (z < nz - 1) ? z + 1 : z;
                const size_t zn = (z > 0) ? z - 1 : 0;
                const double* __restrict pz  = ring + (z % 3) * planeStride;
                const double* __restrict pzp = ring + (zp % 3) * planeStride;
                const double* __restrict pzn = ring + (zn % 3) * planeStride;

                for (size_t y = y0; y < y1; ++y) {
                    const size_t yp = (y < ny - 1) ? y + 1 : y;
                    const size_t yn = (y > 0) ? y - 1 : 0;
                    const size_t base = z * sxy + y * nx;
                    cahnHilliardRow<kExactRecip>(cnew + base, cold + base,
                                                 pz + (y - ybeg) * nx,
                                                 pz + (yp - ybeg) * nx, pz + (yn - ybeg) * nx,
                                                 pzp + (y - ybeg) * nx, pzn + (y - ybeg) * nx,
                                                 nx, D, dt, sp);
                }
            }
        }
    }
}

void cahnHilliardStep(double* __restrict cnew, const double* __restrict cold,
                      double* __restrict ring, const Tiling& tl,
                      const size_t nx, const size_t ny, const size_t nz,
                      const double D, const double dt, const Spacing& sp,
                      const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    if (sp.exactRecip) {
        cahnHilliardStepImpl<true>(cnew, cold, ring, tl, nx, ny, nz, D, dt, sp, gamma, e_AA, e_BB, e_AB);
    } else {
        cahnHilliardStepImpl<false>(cnew, cold, ring, tl, nx, ny, nz, D, dt, sp, gamma, e_AA, e_BB, e_AB);
    }
}

// Initialize concentration field.  Uses the same tile -> thread mapping as the
// time stepping so that pages are first-touched by the thread that works on them.
void initializeConcentration(double* __restrict cp, const size_t nx, const size_t ny, const size_t nz,
                             const Tiling& tl) {
    const size_t vol = nx * ny * nz;

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t zs = 0; zs < tl.nzs; ++zs) {
        for (size_t yt = 0; yt < tl.nyt; ++yt) {
            const size_t z0 = zs * tl.ztile;
            const size_t z1 = std::min(z0 + tl.ztile, nz);
            const size_t y0 = yt * tl.ytile;
            const size_t y1 = std::min(y0 + tl.ytile, ny);
            for (size_t z = z0; z < z1; ++z) {
                for (size_t y = y0; y < y1; ++y) {
                    for (size_t x = 0; x < nx; ++x) {
                        const size_t idx = idx3(x, y, z, nx, ny);
                        // Generate pseudo-random value in [-1, 1]
                        const size_t linear_id = z * (nx * ny) + y * nx + x;
                        const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                        cp[idx] = -1.0 + 2.0 * pseudo;
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* __restrict cp = c.data();

    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(||:bad)
    for (size_t i = 0; i < n; ++i) {
        const double val = cp[i];
        bad = bad || std::isnan(val) || std::isinf(val);
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
        const double val = cp[i];
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    printf("Concentration range: [%.6f, %.6f]\n", minVal, maxVal);
    
    // Values should generally stay within reasonable bounds
    if (maxVal > 10.0 || minVal < -10.0) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }
    
    return true;
}

// ---------------------------------------------------------------------------
// Thread placement.  Unless the user configured OpenMP affinity themselves,
// use one thread per physical core (hardware threads of the same core are
// skipped) and pin each OpenMP thread to its own CPU.  This keeps the
// first-touched memory local and prevents the OS from migrating threads.
// ---------------------------------------------------------------------------
void setupThreadPlacement() {
#ifdef __linux__
    const bool userBinding = getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY") ||
                             getenv("KMP_AFFINITY");
    const bool userThreads = getenv("OMP_NUM_THREADS") != nullptr;
    if (userBinding) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    std::vector<int> primary;    // first hardware thread of each physical core
    std::vector<int> secondary;  // remaining hardware threads
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        bool isPrimary = true;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        if (FILE* f = fopen(path, "r")) {
            int first = -1;
            if (fscanf(f, "%d", &first) == 1 && first >= 0 && first != cpu && CPU_ISSET(first, &allowed)) {
                isPrimary = false;
            }
            fclose(f);
        }
        (isPrimary ? primary : secondary).push_back(cpu);
    }
    if (primary.empty()) return;

    std::vector<int> order(primary);
    order.insert(order.end(), secondary.begin(), secondary.end());

    int nthreads = omp_get_max_threads();
    if (!userThreads) {
        nthreads = static_cast<int>(primary.size());
        omp_set_num_threads(nthreads);
    }
    if (nthreads > static_cast<int>(order.size())) return;   // oversubscribed: leave unpinned

#pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(order[tid], &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    }
#endif
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
    
    setupThreadPlacement();
    const size_t nthreads = static_cast<size_t>(std::max(1, omp_get_max_threads()));

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %zu\n", nthreads);
    
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

    const Spacing spacing = makeSpacing(dx, dy, dz);
    const Tiling tiling = chooseTiling(nx, ny, nz, nthreads);
    printf("Tiling: %zu x %zu tiles of %zu rows x %zu planes\n", tiling.nyt, tiling.nzs, tiling.ytile, tiling.ztile);
    
    // Allocate arrays.  Default-initialised (not zero-filled) so that the pages
    // are first touched -- and therefore placed on the NUMA node of -- the
    // thread that later works on them.
    std::unique_ptr<double[]> coldBuf(new double[gridSize]);
    std::unique_ptr<double[]> cnewBuf(new double[gridSize]);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(coldBuf.get(), nx, ny, nz, tiling);
    // First-touch the second buffer with the same thread -> tile mapping
    initializeConcentration(cnewBuf.get(), nx, ny, nz, tiling);
    
    // Thread-private rolling buffers for 3 planes of chemical potential,
    // allocated and first-touched by their owning threads before timing starts.
    std::vector<std::vector<double>> rings(nthreads);
#pragma omp parallel default(none) shared(rings, tiling, nx, nthreads)
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        if (tid < nthreads) {
            rings[tid].assign(3 * (tiling.ytile + 2) * nx, 0.0);
        }
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // A single parallel region spans the whole time loop; the implicit barrier
    // at the end of each worksharing loop separates consecutive time steps.
#pragma omp parallel default(none) \
        shared(coldBuf, cnewBuf, rings, tiling, spacing, nx, ny, nz, dt, D, gamma, e_AA, e_BB, e_AB, iterations)
    {
        std::vector<double>& ring = rings[static_cast<size_t>(omp_get_thread_num())];
        if (ring.size() < 3 * (tiling.ytile + 2) * nx) {
            ring.assign(3 * (tiling.ytile + 2) * nx, 0.0);
        }

        double* pold = coldBuf.get();
        double* pnew = cnewBuf.get();

        for (int t = 0; t < iterations; ++t) {
            cahnHilliardStep(pnew, pold, ring.data(), tiling, nx, ny, nz,
                             D, dt, spacing, gamma, e_AA, e_BB, e_AB);
            // Swap buffers (thread-local pointer swap, identical in all threads)
            std::swap(pold, pnew);
        }
    }
    // Make the buffer objects reflect the final buffer assignment
    if (iterations % 2 == 1) {
        std::swap(coldBuf, cnewBuf);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy the final field into a vector for output / validation (not timed)
    std::vector<double> cold;
    if (printResults || validate) {
        cold.resize(gridSize);
        const double* __restrict src = coldBuf.get();
        double* __restrict dst = cold.data();
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < gridSize; ++i) {
            dst[i] = src[i];
        }
    }
    
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
