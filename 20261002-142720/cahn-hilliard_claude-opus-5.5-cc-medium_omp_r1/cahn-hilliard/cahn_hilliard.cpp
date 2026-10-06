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

// Allocator that leaves elements default-initialized (no serial zero-fill),
// so pages are first touched by the OpenMP threads that later use them.
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

// Pin OpenMP threads (spread over the CPUs available to the process) unless the
// user already requested a binding policy. Must be called inside a parallel region.
// Stable thread placement keeps first-touch NUMA locality and caches warm.
static inline void pinThreadsIfUnbound() {
    static const bool userBinding = std::getenv("OMP_PROC_BIND") || std::getenv("OMP_PLACES") ||
                                    std::getenv("GOMP_CPU_AFFINITY") || omp_get_proc_bind() != omp_proc_bind_false;
    if (userBinding) return;
    cpu_set_t avail;
    CPU_ZERO(&avail);
    if (sched_getaffinity(0, sizeof(avail), &avail) != 0) return;
    int cpus[CPU_SETSIZE];
    int ncpu = 0;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &avail)) cpus[ncpu++] = i;
    }
    if (ncpu == 0) return;
    const long tid = omp_get_thread_num();
    const long nthr = omp_get_num_threads();
    const int target = cpus[(nthr <= ncpu) ? (tid * ncpu) / nthr : tid % ncpu];
    cpu_set_t mine;
    CPU_ZERO(&mine);
    CPU_SET(target, &mine);
    sched_setaffinity(0, sizeof(mine), &mine);
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Compute Laplacian along one x-row with clamped boundary conditions.
// row/ryp/ryn/rzp/rzn point at rows (y,z), (y+1,z), (y-1,z), (y,z+1), (y,z-1) (clamped).
// Calls f(x, cv, laplacian) for every x in the row.
template <typename F>
static inline __attribute__((always_inline))
void laplacianRow(const double* __restrict row, const double* __restrict ryp, const double* __restrict ryn,
                  const double* __restrict rzp, const double* __restrict rzn,
                  const size_t nx, const double dx2, const double dy2, const double dz2, F&& f) {
    auto lapAt = [&](const size_t x, const size_t xp, const size_t xn) {
        const double cc = row[x];
        const double cxx = (row[xp] + row[xn] - 2.0 * cc) / dx2;
        const double cyy = (ryp[x] + ryn[x] - 2.0 * cc) / dy2;
        const double czz = (rzp[x] + rzn[x] - 2.0 * cc) / dz2;
        return cxx + cyy + czz;
    };
    if (nx == 1) {
        f(0, row[0], lapAt(0, 0, 0));
        return;
    }
    f(0, row[0], lapAt(0, 1, 0));
#pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        f(x, row[x], lapAt(x, x + 1, x - 1));
    }
    f(nx - 1, row[nx - 1], lapAt(nx - 1, nx - 1, nx - 2));
}

// Compute chemical potential (called inside a parallel region)
static inline void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
#pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t zn = (z > 0) ? z - 1 : 0;
            const double* row = c + idx3(0, y, z, nx, ny);
            double* __restrict out = mu + idx3(0, y, z, nx, ny);
            laplacianRow(row, c + idx3(0, yp, z, nx, ny), c + idx3(0, yn, z, nx, ny),
                         c + idx3(0, y, zp, nx, ny), c + idx3(0, y, zn, nx, ny),
                         nx, dx2, dy2, dz2,
                         [&](const size_t x, const double cv, const double lap) {
                             out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                      + 3.0 * cv + cv * cv * cv
                                      - gamma * lap;
                         });
        }
    }
}

// Cahn-Hilliard update step (called inside a parallel region)
static inline void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                        const double* __restrict mu,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
#pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t zn = (z > 0) ? z - 1 : 0;
            const size_t base = idx3(0, y, z, nx, ny);
            const double* __restrict oldRow = cold + base;
            double* __restrict out = cnew + base;
            laplacianRow(mu + base, mu + idx3(0, yp, z, nx, ny), mu + idx3(0, yn, z, nx, ny),
                         mu + idx3(0, y, zp, nx, ny), mu + idx3(0, y, zn, nx, ny),
                         nx, dx2, dy2, dz2,
                         [&](const size_t x, const double, const double lap) {
                             out[x] = oldRow[x] + dt * D * lap;
                         });
        }
    }
}

// Initialize concentration field (called inside a parallel region; uses the same
// static row distribution as the compute kernels for NUMA first-touch placement)
static inline void initializeConcentration(double* __restrict c, double* __restrict cnew, double* __restrict mu,
                             const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

#pragma omp for collapse(2) schedule(static)
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

template <typename Vec>
bool validateResult(const Vec& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* data = c.data();
    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(||:bad)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) bad = true;
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
    
    // Allocate arrays
    Field cold(gridSize);
    Field cnew(gridSize);
    Field mu(gridSize);

    double* pOld = cold.data();
    double* pNew = cnew.data();
    double* pMu = mu.data();
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
#pragma omp parallel
    {
        pinThreadsIfUnbound();
        initializeConcentration(pOld, pNew, pMu, nx, ny, nz);
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
#pragma omp parallel firstprivate(pOld, pNew)
    {
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotential(pOld, pMu, nx, ny, nz, dx, dy, dz,
                                    gamma, e_AA, e_BB, e_AB);

            // Update concentration
            cahnHilliardUpdate(pNew, pOld, pMu, nx, ny, nz, D, dt, dx, dy, dz);

            // Swap buffers (thread-local pointers; every thread swaps identically)
            std::swap(pOld, pNew);
        }
    }
    // Make 'cold' hold the final state, matching the original buffer swapping
    if (iterations > 0 && (iterations % 2) != 0) {
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
