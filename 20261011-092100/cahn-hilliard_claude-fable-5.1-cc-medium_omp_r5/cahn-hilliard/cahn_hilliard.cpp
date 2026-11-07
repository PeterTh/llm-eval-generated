#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

// Read a small integer from a sysfs file; returns -1 on failure.
static int readSysfsInt(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

// Pin each OpenMP thread to one CPU from the process's allowed CPU set.
// Without binding, libgomp's thread pool migrates between cores and sockets
// and this barrier-heavy stencil loses most of its scalability. The mapping is
// topology-aware:
//  * thread ids are split into contiguous blocks per package, so the
//    contiguous z-slabs handed out by schedule(static) stay socket-local with
//    a single cross-socket halo;
//  * within a package, the threads are divided evenly over the L3 domains
//    (core complexes) in contiguous blocks, so partial thread counts enjoy the
//    full aggregate last-level cache while neighbouring slabs mostly share an
//    L3;
//  * if the threads fit on distinct physical cores each gets its own core,
//    otherwise SMT siblings receive adjacent thread ids so they share cache on
//    neighbouring slabs.
// User-provided OMP_PROC_BIND / OMP_PLACES settings take precedence and
// disable this.
static void pinOpenMPThreads() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr) return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    struct CpuInfo { int cpu; int pkg; int l3; int core; int smt; };
    std::vector<CpuInfo> cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &allowed)) continue;
        char path[160];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        const int pkg = readSysfsInt(path);
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", i);
        const int core = readSysfsInt(path);
        // L3 domain id: first CPU listed as sharing the L3 cache.
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", i);
        const int l3 = readSysfsInt(path);
        cpus.push_back({i, pkg < 0 ? 0 : pkg, l3 < 0 ? 0 : l3, core < 0 ? i : core, 0});
    }
    if (cpus.empty()) return;

    std::sort(cpus.begin(), cpus.end(), [](const CpuInfo& a, const CpuInfo& b) {
        if (a.pkg != b.pkg) return a.pkg < b.pkg;
        if (a.l3 != b.l3) return a.l3 < b.l3;
        if (a.core != b.core) return a.core < b.core;
        return a.cpu < b.cpu;
    });
    for (size_t i = 1; i < cpus.size(); ++i) {
        if (cpus[i].pkg == cpus[i - 1].pkg && cpus[i].core == cpus[i - 1].core) cpus[i].smt = cpus[i - 1].smt + 1;
    }

    // Split into packages: [begin, end) ranges into 'cpus'.
    std::vector<std::pair<size_t, size_t>> pkgs;
    for (size_t b = 0; b < cpus.size();) {
        size_t e = b;
        while (e < cpus.size() && cpus[e].pkg == cpus[b].pkg) ++e;
        pkgs.push_back({b, e});
        b = e;
    }
    const size_t ncpu = cpus.size();
    const size_t nthreads = static_cast<size_t>(omp_get_max_threads());

    std::vector<int> map(nthreads);
    size_t t = 0;
    for (size_t p = 0; p < pkgs.size(); ++p) {
        const size_t b = pkgs[p].first, e = pkgs[p].second, size = e - b;
        // Threads for this package: proportional share, remainder to the last one.
        const size_t want = (p + 1 == pkgs.size()) ? (nthreads - t)
                                                   : (nthreads * size + ncpu / 2) / ncpu;
        if (want == 0) continue;

        // Collect the cores of this package as (l3, core) -> list of cpus by smt.
        struct Core { int l3; std::vector<int> cpusBySmt; };
        std::vector<Core> cores;
        for (size_t i = b; i < e; ++i) {
            if (i == b || cpus[i].core != cpus[i - 1].core) cores.push_back({cpus[i].l3, {}});
            cores.back().cpusBySmt.push_back(cpus[i].cpu);
        }
        // Group cores by L3 domain and build each group's cpu order.
        std::vector<std::vector<int>> groups;
        for (size_t c = 0; c < cores.size(); ++c) {
            if (c == 0 || cores[c].l3 != cores[c - 1].l3) groups.push_back({});
            groups.back().push_back(static_cast<int>(c));
        }
        const bool coreFirst = (want <= cores.size());
        std::vector<std::vector<int>> groupCpus(groups.size());
        for (size_t g = 0; g < groups.size(); ++g) {
            if (coreFirst) {
                // k-th SMT thread of every core before the (k+1)-th.
                size_t total = 0;
                for (const int c : groups[g]) total += cores[c].cpusBySmt.size();
                for (size_t k = 0; groupCpus[g].size() < total; ++k) {
                    for (const int c : groups[g]) {
                        if (k < cores[c].cpusBySmt.size()) groupCpus[g].push_back(cores[c].cpusBySmt[k]);
                    }
                }
            } else {
                // All SMT threads of a core get consecutive ids.
                for (const int c : groups[g]) {
                    for (const int cpu : cores[c].cpusBySmt) groupCpus[g].push_back(cpu);
                }
            }
        }
        // Block layout: the package's threads are divided evenly over the L3
        // domains, consecutive thread ids staying within one domain.
        size_t chunk = (want + groups.size() - 1) / groups.size();
        std::vector<int> order;
        order.reserve(size);
        for (size_t pass = 0; order.size() < size; ++pass) {
            for (size_t g = 0; g < groups.size(); ++g) {
                const size_t lo = pass * chunk, hi = std::min(groupCpus[g].size(), lo + chunk);
                for (size_t k = lo; k < hi; ++k) order.push_back(groupCpus[g][k]);
            }
        }
        for (size_t k = 0; k < want && t < nthreads; ++k, ++t) map[t] = order[k % size];
    }
    // Any threads left over (only if the rounding above fell short) wrap around.
    for (; t < nthreads; ++t) map[t] = cpus[t % ncpu].cpu;

#pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        if (tid < map.size()) {
            cpu_set_t mine;
            CPU_ZERO(&mine);
            CPU_SET(map[tid], &mine);
            sched_setaffinity(0, sizeof(mine), &mine);
        }
    }
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute the 7-point Laplacian (clamped boundaries) for a single point, given
// pointers to the row itself and its clamped neighbour rows in y and z.
// The arithmetic order matches the original reference implementation exactly.
inline double laplacianPoint(const double* __restrict row, const double* __restrict rowYn,
                             const double* __restrict rowYp, const double* __restrict rowZn,
                             const double* __restrict rowZp, const size_t x, const size_t xn,
                             const size_t xp, const double dx2, const double dy2,
                             const double dz2) noexcept {
    const double cv = row[x];
    const double cxx = (row[xp] + row[xn] - 2.0 * cv) / dx2;
    const double cyy = (rowYp[x] + rowYn[x] - 2.0 * cv) / dy2;
    const double czz = (rowZp[x] + rowZn[x] - 2.0 * cv) / dz2;
    return cxx + cyy + czz;
}

// Apply a per-point functor f(x, laplacian) along one x-row (fixed y, z) of the
// field 'c'. Boundary points are peeled so the interior loop is branch-free
// and vectorizable.
template <typename F>
inline void laplacianRow(const double* __restrict c, const size_t nx, const size_t ny, const size_t nz,
                         const size_t y, const size_t z, const double dx2, const double dy2,
                         const double dz2, F&& f) noexcept {
    const size_t plane = nx * ny;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double* __restrict row   = c + z * plane + y * nx;
    const double* __restrict rowYn = c + z * plane + yn * nx;
    const double* __restrict rowYp = c + z * plane + yp * nx;
    const double* __restrict rowZn = c + zn * plane + y * nx;
    const double* __restrict rowZp = c + zp * plane + y * nx;

    if (nx == 1) {
        f(0, laplacianPoint(row, rowYn, rowYp, rowZn, rowZp, 0, 0, 0, dx2, dy2, dz2));
        return;
    }

    // x = 0 (clamped left neighbour)
    f(0, laplacianPoint(row, rowYn, rowYp, rowZn, rowZp, 0, 0, 1, dx2, dy2, dz2));

    // interior
#pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        f(x, laplacianPoint(row, rowYn, rowYp, rowZn, rowZp, x, x - 1, x + 1, dx2, dy2, dz2));
    }

    // x = nx-1 (clamped right neighbour)
    f(nx - 1, laplacianPoint(row, rowYn, rowYp, rowZn, rowZp, nx - 1, nx - 2, nx - 1, dx2, dy2, dz2));
}

// Compute chemical potential. Must be called from inside an OpenMP parallel
// region; work is distributed with an orphaned worksharing loop.
void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
#pragma omp for collapse(2) schedule(static) nowait
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            const double* __restrict crow = c + base;
            double* __restrict murow = mu + base;
            laplacianRow(c, nx, ny, nz, y, z, dx2, dy2, dz2, [=](const size_t x, const double lap) {
                const double cv = crow[x];
                murow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                           + 3.0 * cv + cv * cv * cv
                           - gamma * lap;
            });
        }
    }
}

// Cahn-Hilliard update step. Must be called from inside an OpenMP parallel region.
void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                        const double* __restrict mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const double dtD = dt * D;
#pragma omp for collapse(2) schedule(static) nowait
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            const double* __restrict oldrow = cold + base;
            double* __restrict newrow = cnew + base;
            laplacianRow(mu, nx, ny, nz, y, z, dx2, dy2, dz2, [=](const size_t x, const double lap) {
                newrow[x] = oldrow[x] + dtD * lap;
            });
        }
    }
}

// Initialize concentration field (parallel; also performs NUMA first-touch)
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double* __restrict cp = c.data();

#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
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

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* __restrict cp = c.data();

    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|:bad)
    for (size_t i = 0; i < n; ++i) {
        bad |= (std::isnan(cp[i]) || std::isinf(cp[i]));
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
    printf("OpenMP threads: %d\n", omp_get_max_threads());
    pinOpenMPThreads();
    
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
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    // First-touch the work buffers with the same distribution as the kernels.
    {
        double* __restrict cn = cnew.data();
        double* __restrict mp = mu.data();
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                const size_t base = idx3(0, y, z, nx, ny);
                for (size_t x = 0; x < nx; ++x) {
                    cn[base + x] = 0.0;
                    mp[base + x] = 0.0;
                }
            }
        }
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    // A single parallel region spans the whole time loop; the two kernels use
    // orphaned 'omp for' worksharing loops, separated by explicit barriers.
    // Each thread keeps its own copy of the buffer pointers and swaps them in
    // lockstep, so no shared state is mutated between steps.
#pragma omp parallel
    {
        double* pold = cold.data();
        double* pnew = cnew.data();
        double* pmu = mu.data();
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotential(pold, pmu, nx, ny, nz, dx, dy, dz,
                                     gamma, e_AA, e_BB, e_AB);
#pragma omp barrier
            // Update concentration
            cahnHilliardUpdate(pnew, pold, pmu, nx, ny, nz, D, dt, dx, dy, dz);
#pragma omp barrier
            // Swap buffers
            std::swap(pold, pnew);
        }
    }
    // Mirror the per-step buffer swaps on the owning vectors.
    if (iterations % 2 != 0) {
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
