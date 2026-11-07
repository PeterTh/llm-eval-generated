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

// Compute Laplacian with clamped boundary conditions (scalar, generic clamped version
// used for the x-boundary cells of each row).
static inline double computeLaplacianClamped(const double* __restrict__ c,
                                             const size_t nx, const size_t ny, const size_t nz,
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

// Compute chemical potential. Must be called from inside an OpenMP parallel region;
// the work is distributed over (z, y) rows with a static schedule so that each thread
// always touches the same slab of memory (cache / NUMA locality).
static void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                                     const size_t nx, const size_t ny, const size_t nz,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const size_t plane = nx * ny;
    const size_t xEnd = (nx > 1) ? nx - 1 : 1;  // interior cells are x in [1, nx-2]

#pragma omp for collapse(2) schedule(static) nowait
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t zn = (z > 0) ? z - 1 : 0;

            const double* __restrict__ row   = c + z * plane + y * nx;
            const double* __restrict__ rowYp = c + z * plane + yp * nx;
            const double* __restrict__ rowYn = c + z * plane + yn * nx;
            const double* __restrict__ rowZp = c + zp * plane + y * nx;
            const double* __restrict__ rowZn = c + zn * plane + y * nx;
            double* __restrict__ out = mu + z * plane + y * nx;

            // x = 0 boundary (clamped)
            {
                const double cv = row[0];
                out[0] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacianClamped(c, nx, ny, nz, dx, dy, dz, 0, y, z);
            }
            // interior: vectorizable, no clamping needed
#pragma omp simd
            for (size_t x = 1; x < xEnd; ++x) {
                const double cv = row[x];
                const double cxx = (row[x + 1] + row[x - 1] - 2.0 * cv) / dx2;
                const double cyy = (rowYp[x] + rowYn[x] - 2.0 * cv) / dy2;
                const double czz = (rowZp[x] + rowZn[x] - 2.0 * cv) / dz2;
                const double lap = cxx + cyy + czz;
                out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * lap;
            }
            // x = nx-1 boundary (clamped)
            if (nx > 1) {
                const size_t x = nx - 1;
                const double cv = row[x];
                out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                         + 3.0 * cv + cv * cv * cv
                         - gamma * computeLaplacianClamped(c, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Cahn-Hilliard update step. Must be called from inside an OpenMP parallel region.
static void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                               const double* __restrict__ mu,
                               const size_t nx, const size_t ny, const size_t nz,
                               const double D, const double dt, const double dx, const double dy, const double dz) {
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;
    const size_t plane = nx * ny;
    const size_t xEnd = (nx > 1) ? nx - 1 : 1;  // interior cells are x in [1, nx-2]

#pragma omp for collapse(2) schedule(static) nowait
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t zp = (z < nz - 1) ? z + 1 : z;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t zn = (z > 0) ? z - 1 : 0;

            const double* __restrict__ row   = mu + z * plane + y * nx;
            const double* __restrict__ rowYp = mu + z * plane + yp * nx;
            const double* __restrict__ rowYn = mu + z * plane + yn * nx;
            const double* __restrict__ rowZp = mu + zp * plane + y * nx;
            const double* __restrict__ rowZn = mu + zn * plane + y * nx;
            const double* __restrict__ oldRow = cold + z * plane + y * nx;
            double* __restrict__ out = cnew + z * plane + y * nx;

            // x = 0 boundary (clamped)
            out[0] = oldRow[0] + dt * D * computeLaplacianClamped(mu, nx, ny, nz, dx, dy, dz, 0, y, z);

            // interior: vectorizable
#pragma omp simd
            for (size_t x = 1; x < xEnd; ++x) {
                const double mv = row[x];
                const double cxx = (row[x + 1] + row[x - 1] - 2.0 * mv) / dx2;
                const double cyy = (rowYp[x] + rowYn[x] - 2.0 * mv) / dy2;
                const double czz = (rowZp[x] + rowZn[x] - 2.0 * mv) / dz2;
                const double lap = cxx + cyy + czz;
                out[x] = oldRow[x] + dt * D * lap;
            }
            // x = nx-1 boundary (clamped)
            if (nx > 1) {
                const size_t x = nx - 1;
                out[x] = oldRow[x] + dt * D * computeLaplacianClamped(mu, nx, ny, nz, dx, dy, dz, x, y, z);
            }
        }
    }
}

// Initialize concentration field (parallel, same static row partitioning as the kernels
// so that pages are first-touched by the thread that will later work on them).
void initializeConcentration(std::vector<double>& c, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    double* __restrict__ cp = c.data();

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
    const double* __restrict__ cp = c.data();

    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < n; ++i) {
        bad = bad || std::isnan(cp[i]) || std::isinf(cp[i]);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = cp[0];
    double maxVal = cp[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
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


// Pin each OpenMP thread to its own hardware thread (taken from the process's allowed
// CPU set, in ascending order). Unpinned threads migrate across cores and NUMA nodes,
// which destroys the first-touch memory placement the kernels rely on. If the user has
// already chosen a binding policy via OMP_PROC_BIND / OMP_PLACES we leave it alone.
static void pinThreads() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr) {
        return;
    }
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }
    // Order the allowed CPUs so that SMT siblings of one core are adjacent (like
    // OMP_PLACES=threads): consecutive threads then get consecutive data chunks on the
    // same core and share its caches. Falls back to plain numeric order without sysfs.
    std::vector<int> cpus;
    std::vector<char> taken(CPU_SETSIZE, 0);
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &allowed) || taken[i]) continue;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
        FILE* f = fopen(path, "r");
        if (f != nullptr) {
            char buf[1024];
            if (fgets(buf, sizeof(buf), f) != nullptr) {
                // Format: comma separated list of numbers or ranges, e.g. "0,128" or "0-1".
                char* save = nullptr;
                for (char* tok = strtok_r(buf, ",\n", &save); tok != nullptr; tok = strtok_r(nullptr, ",\n", &save)) {
                    int lo = 0, hi = 0;
                    if (sscanf(tok, "%d-%d", &lo, &hi) < 2) hi = lo;
                    for (int c = lo; c <= hi; ++c) {
                        if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(c, &allowed) && !taken[c]) {
                            taken[c] = 1;
                            cpus.push_back(c);
                        }
                    }
                }
            }
            fclose(f);
        }
        if (!taken[i]) {  // sysfs unavailable or malformed: just take this CPU
            taken[i] = 1;
            cpus.push_back(i);
        }
    }
    if (cpus.empty()) return;

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int cpu = cpus[static_cast<size_t>(tid) % cpus.size()];
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cpu, &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
    }
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

    // Allocate arrays
    std::vector<double> cold(gridSize);
    std::vector<double> cnew(gridSize);
    std::vector<double> mu(gridSize);

    // Initialize concentration field (first-touch placement for cold; touch cnew/mu
    // with the same partitioning so all three arrays live near their worker threads).
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    {
        double* __restrict__ pn = cnew.data();
        double* __restrict__ pm = mu.data();
#pragma omp parallel for collapse(2) schedule(static)
        for (size_t z = 0; z < nz; ++z) {
            for (size_t y = 0; y < ny; ++y) {
                const size_t base = z * (nx * ny) + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    pn[base + x] = 0.0;
                    pm[base + x] = 0.0;
                }
            }
        }
    }

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // A single parallel region spans the whole time loop; each step consists of two
    // work-shared loops separated by explicit barriers. Every thread keeps its own copy
    // of the buffer pointers and swaps them identically, so no synchronisation is
    // needed for the swap itself.
#pragma omp parallel
    {
        const double* pold = cold.data();
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

            // Swap buffers (thread-local pointer swap, identical on all threads)
            const double* tmp = pold;
            pold = pnew;
            pnew = const_cast<double*>(tmp);
        }
    }
    // Make `cold` hold the final state, matching the original's std::swap per step.
    if (iterations % 2 == 1) {
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
