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

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Parse a sysfs cpu list such as "0,128" or "0-3,8-11" into cpu ids.
static std::vector<int> parseCpuList(const char* str) {
    std::vector<int> out;
    const char* p = str;
    while (*p) {
        char* endp = nullptr;
        long a = strtol(p, &endp, 10);
        if (endp == p) break;
        long b = a;
        p = endp;
        if (*p == '-') {
            b = strtol(p + 1, &endp, 10);
            p = endp;
        }
        for (long i = a; i <= b && i < CPU_SETSIZE; ++i) out.push_back(static_cast<int>(i));
        while (*p == ',' || *p == ' ' || *p == '\n') ++p;
    }
    return out;
}

static int readSysfsInt(const char* path, int fallback) {
    int v = fallback;
    if (FILE* f = fopen(path, "r")) {
        if (fscanf(f, "%d", &v) != 1) v = fallback;
        fclose(f);
    }
    return v;
}

// Pin each OpenMP worker thread to one CPU of the process's allowed CPU set, unless the
// OpenMP runtime has already been asked to bind threads (OMP_PROC_BIND / OMP_PLACES).
// Unbound threads migrate between cores and NUMA nodes, which badly hurts the
// bandwidth-bound stencil kernels. Placement policy:
//  * threads <= physical cores: one thread per core, cores shared out over the packages
//    (NUMA nodes) so all memory controllers are used; thread ids stay contiguous per package.
//  * otherwise: fill cores package by package with SMT siblings on adjacent thread ids,
//    so siblings sharing a core's caches own neighbouring slabs of the static schedule.
// libgomp keeps its thread pool across regions, so pinning once up front is enough.
void pinThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    // Discover cores (sibling groups) and their package ids from sysfs.
    struct Core { int pkg; std::vector<int> cpus; };
    std::vector<Core> cores;
    std::vector<char> seen(CPU_SETSIZE, 0);
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &allowed) || seen[i]) continue;
        char path[160];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
        std::vector<int> group;
        if (FILE* f = fopen(path, "r")) {
            char buf[512];
            if (fgets(buf, sizeof(buf), f)) group = parseCpuList(buf);
            fclose(f);
        }
        Core core;
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", i);
        core.pkg = readSysfsInt(path, 0);
        for (int c : group) {
            if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(c, &allowed) && !seen[c]) {
                seen[c] = 1;
                core.cpus.push_back(c);
            }
        }
        if (!seen[i]) { seen[i] = 1; core.cpus.push_back(i); }
        cores.push_back(std::move(core));
    }
    if (cores.empty()) return;

    // Distinct packages in order of first appearance; cores per package keep cpu order.
    std::vector<int> pkgIds;
    for (const Core& c : cores) {
        if (std::find(pkgIds.begin(), pkgIds.end(), c.pkg) == pkgIds.end()) pkgIds.push_back(c.pkg);
    }
    std::vector<std::vector<size_t>> pkgCores(pkgIds.size());
    for (size_t ci = 0; ci < cores.size(); ++ci) {
        const size_t pi = static_cast<size_t>(std::find(pkgIds.begin(), pkgIds.end(), cores[ci].pkg) - pkgIds.begin());
        pkgCores[pi].push_back(ci);
    }

    const int nthreads = omp_get_max_threads();
    size_t ncpus = 0;
    for (const Core& c : cores) ncpus += c.cpus.size();
    if (nthreads <= 0 || static_cast<size_t>(nthreads) > ncpus) return;

    // Give each package a share of the threads proportional to its cpu count (thread ids
    // contiguous per package). Inside a package use one thread per core while they fit,
    // otherwise fill cores with SMT siblings on adjacent thread ids.
    std::vector<int> order;
    order.reserve(static_cast<size_t>(nthreads));
    size_t assigned = 0;
    for (size_t pi = 0; pi < pkgIds.size(); ++pi) {
        size_t pkgCpus = 0;
        for (size_t ci : pkgCores[pi]) pkgCpus += cores[ci].cpus.size();
        size_t share = (pi + 1 == pkgIds.size())
            ? static_cast<size_t>(nthreads) - assigned
            : (static_cast<size_t>(nthreads) * pkgCpus + ncpus / 2) / ncpus;
        share = std::min(share, pkgCpus);
        share = std::min(share, static_cast<size_t>(nthreads) - assigned);
        if (share <= pkgCores[pi].size()) {
            for (size_t k = 0; k < share; ++k) order.push_back(cores[pkgCores[pi][k]].cpus[0]);
        } else {
            size_t k = 0;
            for (size_t ci : pkgCores[pi]) {
                for (int cpu : cores[ci].cpus) {
                    if (k < share) { order.push_back(cpu); ++k; }
                }
            }
        }
        assigned += share;
    }
    // Rounding may leave a few threads unplaced: use any remaining cpus in topology order.
    if (assigned < static_cast<size_t>(nthreads)) {
        std::vector<char> used(CPU_SETSIZE, 0);
        for (int cpu : order) used[static_cast<size_t>(cpu)] = 1;
        for (size_t pi = 0; pi < pkgIds.size(); ++pi) {
            for (size_t ci : pkgCores[pi]) {
                for (int cpu : cores[ci].cpus) {
                    if (!used[static_cast<size_t>(cpu)] && assigned < static_cast<size_t>(nthreads)) {
                        used[static_cast<size_t>(cpu)] = 1;
                        order.push_back(cpu);
                        ++assigned;
                    }
                }
            }
        }
    }
    if (order.size() < static_cast<size_t>(nthreads)) return;

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        if (omp_get_num_threads() == nthreads) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(order[static_cast<size_t>(tid)], &one);
            sched_setaffinity(0, sizeof(one), &one);  // Linux: pid 0 means the calling thread
        }
    }
}

// Compute Laplacian with clamped boundary conditions (scalar reference form)
inline double computeLaplacian(const double* __restrict c, const size_t nx, const size_t ny, const size_t nz,
                               const double dx, const double dy, const double dz,
                               const size_t x, const size_t y, const size_t z) noexcept {
    const size_t xp = (x < nx - 1) ? x + 1 : x;
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t xn = (x > 0) ? x - 1 : 0;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double cxx = (c[idx3(xp, y, z, nx, ny)] + c[idx3(xn, y, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dx * dx);
    const double cyy = (c[idx3(x, yp, z, nx, ny)] + c[idx3(x, yn, z, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dy * dy);
    const double czz = (c[idx3(x, y, zp, nx, ny)] + c[idx3(x, y, zn, nx, ny)] -
                  2.0 * c[idx3(x, y, z, nx, ny)]) / (dz * dz);

    return cxx + cyy + czz;
}

// Laplacian for one (y,z) row: the x boundary cells use the clamped scalar form,
// the interior cells use a branch-free, vectorizable form with identical arithmetic.
// Must be called from inside an OpenMP parallel region (uses orphaned worksharing in callers).
inline void laplacianRow(const double* __restrict c, double* __restrict lap,
                         const size_t nx, const size_t ny, const size_t nz,
                         const double dx, const double dy, const double dz,
                         const size_t y, const size_t z) noexcept {
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zn = (z > 0) ? z - 1 : 0;

    const double* __restrict row  = c + idx3(0, y,  z,  nx, ny);
    const double* __restrict rowYp = c + idx3(0, yp, z,  nx, ny);
    const double* __restrict rowYn = c + idx3(0, yn, z,  nx, ny);
    const double* __restrict rowZp = c + idx3(0, y,  zp, nx, ny);
    const double* __restrict rowZn = c + idx3(0, y,  zn, nx, ny);

    const double idx2 = dx * dx;
    const double idy2 = dy * dy;
    const double idz2 = dz * dz;

    // Left boundary
    lap[0] = computeLaplacian(c, nx, ny, nz, dx, dy, dz, 0, y, z);
    // Interior
    const size_t xend = (nx > 1) ? nx - 1 : 1;
    #pragma omp simd
    for (size_t x = 1; x < xend; ++x) {
        const double cv = row[x];
        const double cxx = (row[x + 1] + row[x - 1] - 2.0 * cv) / idx2;
        const double cyy = (rowYp[x] + rowYn[x] - 2.0 * cv) / idy2;
        const double czz = (rowZp[x] + rowZn[x] - 2.0 * cv) / idz2;
        lap[x] = cxx + cyy + czz;
    }
    // Right boundary (also handles nx == 1)
    if (nx > 1) {
        lap[nx - 1] = computeLaplacian(c, nx, ny, nz, dx, dy, dz, nx - 1, y, z);
    }
}

// Compute chemical potential (orphaned worksharing: call inside a parallel region)
void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            double* __restrict murow = mu + base;
            const double* __restrict crow = c + base;
            // Store the Laplacian into mu first, then finish the expression in place.
            laplacianRow(c, murow, nx, ny, nz, dx, dy, dz, y, z);
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                const double cv = crow[x];
                murow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * murow[x];
            }
        }
    }
}

// Cahn-Hilliard update step (orphaned worksharing: call inside a parallel region)
void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                        const double* __restrict mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t base = idx3(0, y, z, nx, ny);
            double* __restrict nrow = cnew + base;
            const double* __restrict orow = cold + base;
            laplacianRow(mu, nrow, nx, ny, nz, dx, dy, dz, y, z);
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                nrow[x] = orow[x] + dt * D * nrow[x];
            }
        }
    }
}

// Zero a field in parallel with the same static (z,y) partitioning as the kernels so that
// its pages are first-touched (and thus NUMA-placed) by the thread that will work on them.
void firstTouchZero(double* __restrict p, const size_t nx, const size_t ny, const size_t nz) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            double* __restrict row = p + idx3(0, y, z, nx, ny);
            for (size_t x = 0; x < nx; ++x) row[x] = 0.0;
        }
    }
}

// Initialize concentration field (parallel, also provides NUMA first-touch placement
// matching the static schedule used by the kernels)
void initializeConcentration(double* __restrict cp, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

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

bool validateResult(const double* __restrict cp, const size_t n) {

    // Check for NaN or Inf
    bool bad = false;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        bad |= (std::isnan(cp[i]) || std::isinf(cp[i]));
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = cp[0];
    double maxVal = cp[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
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
    pinThreads();
    
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
    
    // Allocate arrays. Raw 64-byte aligned buffers are used instead of std::vector so the
    // memory is not touched (zero-filled) by the main thread: pages are first-touched by the
    // parallel initialization below, placing them on the NUMA node of the thread using them.
    const size_t bytes = ((gridSize * sizeof(double) + 63) / 64) * 64;
    double* cold = static_cast<double*>(std::aligned_alloc(64, bytes > 0 ? bytes : 64));
    double* cnew = static_cast<double*>(std::aligned_alloc(64, bytes > 0 ? bytes : 64));
    double* mu   = static_cast<double*>(std::aligned_alloc(64, bytes > 0 ? bytes : 64));
    if (!cold || !cnew || !mu) {
        printf("Allocation failed\n");
        return 1;
    }

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    firstTouchZero(cnew, nx, ny, nz);
    firstTouchZero(mu, nx, ny, nz);
    initializeConcentration(cold, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    {
        double* __restrict pOld = cold;
        double* __restrict pNew = cnew;
        double* __restrict pMu = mu;

        // One parallel region for the whole time loop; the worksharing loops inside
        // the kernels carry implicit barriers that order mu -> cnew -> next step.
        #pragma omp parallel firstprivate(pOld, pNew)
        {
            for (int t = 0; t < iterations; ++t) {
                // Compute chemical potential
                computeChemicalPotential(pOld, pMu, nx, ny, nz, dx, dy, dz,
                                        gamma, e_AA, e_BB, e_AB);

                // Update concentration
                cahnHilliardUpdate(pNew, pOld, pMu, nx, ny, nz, D, dt, dx, dy, dz);

                // Swap buffers (each thread swaps its private pointer copies identically)
                std::swap(pOld, pNew);
            }
        }

        // Make the outer pointers reflect the final buffer assignment
        if (iterations % 2 == 1) {
            std::swap(cold, cnew);
        }
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
        const std::vector<double> result(cold, cold + gridSize);
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, gridSize);

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

    std::free(cold);
    std::free(cnew);
    std::free(mu);
    return 0;
}
