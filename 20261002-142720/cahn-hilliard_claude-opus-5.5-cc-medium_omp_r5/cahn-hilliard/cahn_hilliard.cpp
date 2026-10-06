#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Thread pinning: distributes OpenMP threads evenly over the allowed CPUs (ordered by
// socket, core, hardware thread) so that the static work partition keeps NUMA locality
// and threads are not migrated. Skipped if the user configured OpenMP binding explicitly.
static std::vector<int> buildPinningOrder() {
    std::vector<int> order;
    if (std::getenv("OMP_PROC_BIND") || std::getenv("OMP_PLACES")) return order;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return order;
    struct CpuInfo { long pkg, core; int cpu; };
    std::vector<CpuInfo> cpus;
    auto readTopo = [](int cpu, const char* what) -> long {
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (FILE* f = fopen(path, "r")) {
            if (fscanf(f, "%ld", &v) != 1) v = -1;
            fclose(f);
        }
        return v;
    };
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &set)) cpus.push_back({readTopo(cpu, "physical_package_id"), readTopo(cpu, "core_id"), cpu});
    }
    std::sort(cpus.begin(), cpus.end(), [](const CpuInfo& a, const CpuInfo& b) {
        if (a.pkg != b.pkg) return a.pkg < b.pkg;
        if (a.core != b.core) return a.core < b.core;
        return a.cpu < b.cpu;
    });
    for (const auto& ci : cpus) order.push_back(ci.cpu);
    return order;
}

// Must be called inside a parallel region by every thread.
static inline void pinCurrentThread(const std::vector<int>& order) {
    if (order.empty()) return;
    const size_t nthreads = static_cast<size_t>(omp_get_num_threads());
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    const size_t n = order.size();
    const int cpu = (nthreads <= n) ? order[tid * n / nthreads] : order[tid % n];
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    sched_setaffinity(0, sizeof(set), &set);
}

// Allocator that default-initializes elements (no zero fill), so that pages are
// first touched by the OpenMP threads that later work on them (NUMA locality).
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template <typename U>
    struct rebind { using other = DefaultInitAllocator<U>; };
    DefaultInitAllocator() noexcept = default;
    template <typename U>
    DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template <typename U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
};

using Field = std::vector<double, DefaultInitAllocator<double>>;

// Laplacian with clamped boundary conditions for one x-row.
// Calls f(x, center, laplacian) for every x in [0, nx).
// rc: current row, ryp/ryn: rows at y+1/y-1 (clamped), rzp/rzn: rows at z+1/z-1 (clamped).
template <typename F>
static inline void laplacianRow(const double* __restrict rc, const double* __restrict ryp,
                                const double* __restrict ryn, const double* __restrict rzp,
                                const double* __restrict rzn, const size_t nx,
                                const double dx2, const double dy2, const double dz2, F&& f) {
    auto point = [&](const size_t x, const size_t xp, const size_t xn) {
        const double cc = rc[x];
        const double cxx = (rc[xp] + rc[xn] - 2.0 * cc) / dx2;
        const double cyy = (ryp[x] + ryn[x] - 2.0 * cc) / dy2;
        const double czz = (rzp[x] + rzn[x] - 2.0 * cc) / dz2;
        f(x, cc, cxx + cyy + czz);
    };
    if (nx == 1) {
        point(0, 0, 0);
        return;
    }
    point(0, 1, 0);
#pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        point(x, x + 1, x - 1);
    }
    point(nx - 1, nx - 1, nx - 2);
}

// Static partition of the nz*ny x-rows into contiguous per-thread ranges [begin, end).
// The same partition is used for initialization (first touch) and the time loop.
struct RowRange { size_t begin, end; };

static inline RowRange threadRows(const size_t rows) {
    const size_t nthreads = static_cast<size_t>(omp_get_num_threads());
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    const size_t q = rows / nthreads;
    const size_t rem = rows % nthreads;
    const size_t begin = tid * q + std::min(tid, rem);
    return {begin, begin + q + (tid < rem ? 1 : 0)};
}

// Calls rowFn(i, iyp, iyn, izp, izn) with the start offsets of x-row r = z*ny + y and its
// clamped y/z neighbor rows.
template <typename RowFn>
static inline void withRow(const size_t r, const size_t nx, const size_t ny, const size_t nz, RowFn&& rowFn) {
    const size_t z = r / ny;
    const size_t y = r - z * ny;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t zn = (z > 0) ? z - 1 : 0;
    rowFn(idx3(0, y, z, nx, ny), idx3(0, yp, z, nx, ny), idx3(0, yn, z, nx, ny),
          idx3(0, y, zp, nx, ny), idx3(0, y, zn, nx, ny));
}

struct Params {
    size_t nx, ny, nz;
    double dx2, dy2, dz2;
    double gamma, e_AA, e_BB, e_AB;
    double D, dt;
};

// Compute chemical potential for x-row r
static inline void computeChemicalPotentialRow(const double* __restrict cp, double* __restrict mp,
                                               const Params& P, const size_t r) {
    const double gamma = P.gamma, e_AA = P.e_AA, e_BB = P.e_BB, e_AB = P.e_AB;
    withRow(r, P.nx, P.ny, P.nz, [&](size_t i, size_t iyp, size_t iyn, size_t izp, size_t izn) {
        double* __restrict out = mp + i;
        laplacianRow(cp + i, cp + iyp, cp + iyn, cp + izp, cp + izn, P.nx, P.dx2, P.dy2, P.dz2,
                     [&](size_t x, double cv, double lap) {
                         out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                  + 3.0 * cv + cv * cv * cv
                                  - gamma * lap;
                     });
    });
}

// Cahn-Hilliard update for x-row r
static inline void cahnHilliardUpdateRow(double* __restrict np, const double* __restrict op,
                                         const double* __restrict mp, const Params& P, const size_t r) {
    const double D = P.D, dt = P.dt;
    withRow(r, P.nx, P.ny, P.nz, [&](size_t i, size_t iyp, size_t iyn, size_t izp, size_t izn) {
        const double* __restrict oldRow = op + i;
        double* __restrict out = np + i;
        laplacianRow(mp + i, mp + iyp, mp + iyn, mp + izp, mp + izn, P.nx, P.dx2, P.dy2, P.dz2,
                     [&](size_t x, double, double lap) {
                         out[x] = oldRow[x] + dt * D * lap;
                     });
    });
}

// One time step (called by all threads inside a parallel region).
// The update of row r needs mu on rows r-ny .. r+ny. Each thread first computes mu on the
// first and last plane-worth (ny rows) of its row range, which are the only mu rows read by
// neighboring threads. After a barrier, it streams through its range computing interior mu
// rows one plane ahead of the update, so that mu is reused from cache.
static inline void timeStep(double* __restrict cnew, const double* __restrict cold, double* __restrict mu,
                            const Params& P, const RowRange rr) {
    const size_t ny = P.ny;
    const size_t headEnd = std::min(rr.begin + ny, rr.end);
    const size_t tailBegin = std::max(rr.end >= ny ? rr.end - ny : 0, headEnd);

    for (size_t r = rr.begin; r < headEnd; ++r) computeChemicalPotentialRow(cold, mu, P, r);
    for (size_t r = tailBegin; r < rr.end; ++r) computeChemicalPotentialRow(cold, mu, P, r);
#pragma omp barrier

    // Interior mu rows: [headEnd, tailBegin)
    for (size_t r = headEnd; r < std::min(rr.begin + ny + 1, tailBegin); ++r) {
        computeChemicalPotentialRow(cold, mu, P, r);
    }
    for (size_t r = rr.begin; r < rr.end; ++r) {
        const size_t m = r + ny + 1;
        if (m < tailBegin) computeChemicalPotentialRow(cold, mu, P, m);
        cahnHilliardUpdateRow(cnew, cold, mu, P, r);
    }
#pragma omp barrier
}

// Initialize concentration field and first-touch all work arrays (called inside a parallel region)
void initializeConcentration(Field& c, Field& cnew, Field& mu, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double* __restrict cp = c.data();
    double* __restrict np = cnew.data();
    double* __restrict mp = mu.data();
    const RowRange rr = threadRows(ny * nz);
    for (size_t r = rr.begin; r < rr.end; ++r) {
        const size_t i = r * nx;
        for (size_t x = 0; x < nx; ++x) {
            // Generate pseudo-random value in [-1, 1]
            const size_t linear_id = i + x;
            const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
            cp[i + x] = -1.0 + 2.0 * pseudo;
            np[i + x] = 0.0;
            mp[i + x] = 0.0;
        }
    }
}

bool validateResult(const Field& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    const size_t n = c.size();
    const double* cp = c.data();
    bool bad = false;
#pragma omp parallel for proc_bind(spread) schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(cp[i]) || std::isinf(cp[i])) bad = true;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
#pragma omp parallel for proc_bind(spread) schedule(static) reduction(min : minVal) reduction(max : maxVal)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, cp[i]);
        maxVal = std::max(maxVal, cp[i]);
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
    
    // Allocate arrays
    Field cold(gridSize);
    Field cnew(gridSize);
    Field mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    const std::vector<int> pinOrder = buildPinningOrder();
#pragma omp parallel proc_bind(spread)
    {
        pinCurrentThread(pinOrder);
        initializeConcentration(cold, cnew, mu, nx, ny, nz);
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
#pragma omp parallel proc_bind(spread)
    {
        pinCurrentThread(pinOrder);
        const Params P{nx, ny, nz, dx * dx, dy * dy, dz * dz, gamma, e_AA, e_BB, e_AB, D, dt};
        const RowRange rr = threadRows(ny * nz);
        double* src = cold.data();
        double* dst = cnew.data();
        for (int t = 0; t < iterations; ++t) {
            // Fused chemical potential + concentration update
            timeStep(dst, src, mu.data(), P, rr);
            
            // Swap buffers (thread-private pointers)
            std::swap(src, dst);
        }
    }
    // Final result is in cnew after an odd number of steps
    if (iterations > 0 && (iterations % 2) == 1) {
        std::swap(cold, cnew);
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
        const std::vector<double> result(cold.begin(), cold.end());
        print_results(result, "Concentration");
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
