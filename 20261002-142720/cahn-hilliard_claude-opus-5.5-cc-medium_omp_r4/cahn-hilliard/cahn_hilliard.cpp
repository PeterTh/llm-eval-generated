#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
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

// Pin OpenMP threads to CPUs (unless the user controls binding via OMP_PROC_BIND /
// OMP_PLACES), like OMP_PROC_BIND=spread: threads with consecutive numbers (which own
// neighboring slabs of the grid) go to topologically adjacent cores, using distinct
// physical cores before SMT siblings. Stable binding keeps first-touch NUMA placement,
// cache contents and halo exchange local across time steps.
static int readTopo(const int cpu, const char* what) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

void pinThreads() {
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES")) return;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return;

    struct Cpu { int id, pkg, core, smt; };
    std::vector<Cpu> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &mask)) cpus.push_back({c, readTopo(c, "physical_package_id"), readTopo(c, "core_id"), 0});
    }
    if (cpus.size() < 2) return;
    // SMT rank of each logical CPU within its physical core
    for (size_t i = 0; i < cpus.size(); ++i)
        for (size_t j = 0; j < i; ++j)
            if (cpus[j].pkg == cpus[i].pkg && cpus[j].core == cpus[i].core) ++cpus[i].smt;
    std::stable_sort(cpus.begin(), cpus.end(), [](const Cpu& a, const Cpu& b) {
        if (a.pkg != b.pkg) return a.pkg < b.pkg;
        if (a.core != b.core) return a.core < b.core;
        return a.smt < b.smt;
    });
    std::vector<int> all, primary;
    for (const Cpu& c : cpus) {
        all.push_back(c.id);
        if (c.smt == 0) primary.push_back(c.id);
    }

    #pragma omp parallel
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        const size_t nt = static_cast<size_t>(omp_get_num_threads());
        const std::vector<int>& list = (nt <= primary.size()) ? primary : all;
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(list[(t * list.size() / nt) % list.size()], &one);
        sched_setaffinity(0, sizeof(one), &one);
    }
}

// Allocator that default-initializes (no zero fill), so pages are first-touched
// in parallel by the threads that later work on them (NUMA locality).
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template <typename U> struct rebind { using other = DefaultInitAllocator<U>; };
    DefaultInitAllocator() noexcept = default;
    template <typename U> DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template <typename U> void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <typename U, typename... Args> void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};
using Field = std::vector<double, DefaultInitAllocator<double>>;

// Laplacian at x on a row, given pointers to the current row and its clamped neighbor rows
static inline double lap(const double* __restrict r, const double* __restrict ryp, const double* __restrict ryn,
                         const double* __restrict rzp, const double* __restrict rzn,
                         const size_t x, const size_t xp, const size_t xn,
                         const double dx2, const double dy2, const double dz2) {
    const double cc = r[x];
    const double cxx = (r[xp] + r[xn] - 2.0 * cc) / dx2;
    const double cyy = (ryp[x] + ryn[x] - 2.0 * cc) / dy2;
    const double czz = (rzp[x] + rzn[x] - 2.0 * cc) / dz2;
    return cxx + cyy + czz;
}

// Row neighbor pointers with clamped boundary conditions
struct RowPtrs {
    const double *r, *yp, *yn, *zp, *zn;
};
static inline RowPtrs rowPtrs(const double* base, const size_t y, const size_t z,
                              const size_t nx, const size_t ny, const size_t nz) {
    const size_t yp = (y < ny - 1) ? y + 1 : y;
    const size_t yn = (y > 0) ? y - 1 : 0;
    const size_t zp = (z < nz - 1) ? z + 1 : z;
    const size_t zn = (z > 0) ? z - 1 : 0;
    return {base + idx3(0, y, z, nx, ny), base + idx3(0, yp, z, nx, ny), base + idx3(0, yn, z, nx, ny),
            base + idx3(0, y, zp, nx, ny), base + idx3(0, y, zn, nx, ny)};
}

// Compute chemical potential (called inside a parallel region)
void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const RowPtrs p = rowPtrs(c, y, z, nx, ny, nz);
            double* __restrict m = mu + idx3(0, y, z, nx, ny);
            auto point = [&](const size_t x, const size_t xp, const size_t xn) {
                const double cv = p.r[x];
                m[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                       + 3.0 * cv + cv * cv * cv
                       - gamma * lap(p.r, p.yp, p.yn, p.zp, p.zn, x, xp, xn, dx2, dy2, dz2);
            };
            if (nx == 1) { point(0, 0, 0); continue; }
            point(0, 1, 0);
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) point(x, x + 1, x - 1);
            point(nx - 1, nx - 1, nx - 2);
        }
    }
}

// Cahn-Hilliard update step (called inside a parallel region)
void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                        const double* __restrict mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const RowPtrs p = rowPtrs(mu, y, z, nx, ny, nz);
            const double* __restrict co = cold + idx3(0, y, z, nx, ny);
            double* __restrict cn = cnew + idx3(0, y, z, nx, ny);
            auto point = [&](const size_t x, const size_t xp, const size_t xn) {
                cn[x] = co[x] + dt * D * lap(p.r, p.yp, p.yn, p.zp, p.zn, x, xp, xn, dx2, dy2, dz2);
            };
            if (nx == 1) { point(0, 0, 0); continue; }
            point(0, 1, 0);
            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) point(x, x + 1, x - 1);
            point(nx - 1, nx - 1, nx - 2);
        }
    }
}

// Initialize concentration field (same static row distribution as the compute loops
// so that first touch places pages on the NUMA node of the thread that uses them)
void initializeConcentration(Field& c, Field& cnew, Field& mu, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
                cnew[idx] = 0.0;
                mu[idx] = 0.0;
            }
        }
    }
}

bool validateResult(const Field& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* data = c.data();

    // Check for NaN or Inf
    int bad = 0;
    #pragma omp parallel for reduction(|:bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) bad |= 1;
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        minVal = std::min(minVal, data[i]);
        maxVal = std::max(maxVal, data[i]);
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
    
    // Bind threads before first touch so that data placement matches the compute phase
    pinThreads();

    // Allocate arrays
    Field cold(gridSize);
    Field cnew(gridSize);
    Field mu(gridSize);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, cnew, mu, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    {
        double* pold = cold.data();
        double* pnew = cnew.data();
        double* pmu = mu.data();
        #pragma omp parallel firstprivate(pold, pnew)
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotential(pold, pmu, nx, ny, nz, dx, dy, dz, 
                                    gamma, e_AA, e_BB, e_AB);
            
            // Update concentration
            cahnHilliardUpdate(pnew, pold, pmu, nx, ny, nz, D, dt, dx, dy, dz);
            
            // Swap buffers (each thread swaps its private pointers; implicit barrier above)
            std::swap(pold, pnew);
        }
        if (iterations % 2 != 0) std::swap(cold, cnew);
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
        print_results(std::vector<double>(cold.begin(), cold.end()), "Concentration");
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
