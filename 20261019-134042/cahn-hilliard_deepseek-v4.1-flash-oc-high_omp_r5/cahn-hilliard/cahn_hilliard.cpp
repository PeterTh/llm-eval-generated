#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstddef>
#include <memory>
#include <vector>

#include <sched.h>
#include <omp.h>

#include "../common/results_output.hpp"

namespace {

std::vector<int> g_all_cpus;
std::vector<int> g_primary_cpus;

void readTopology() {
    g_all_cpus.clear();
    g_primary_cpus.clear();

    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    std::vector<std::uint64_t> core_keys;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        g_all_cpus.push_back(cpu);

        char path[256];
        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        FILE* fp = std::fopen(path, "r");
        int package = -1;
        if (fp != nullptr) {
            if (std::fscanf(fp, "%d", &package) != 1) {
                package = -1;
            }
            std::fclose(fp);
        }

        std::snprintf(path, sizeof(path),
                      "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        fp = std::fopen(path, "r");
        int core = -1;
        if (fp != nullptr) {
            if (std::fscanf(fp, "%d", &core) != 1) {
                core = -1;
            }
            std::fclose(fp);
        }

        if (package < 0 || core < 0) {
            g_primary_cpus.push_back(cpu);
            continue;
        }

        const std::uint64_t key =
            (static_cast<std::uint64_t>(static_cast<std::uint32_t>(package)) << 32) |
            static_cast<std::uint32_t>(core);
        const bool seen = std::find(core_keys.begin(), core_keys.end(), key) != core_keys.end();
        if (!seen) {
            core_keys.push_back(key);
            g_primary_cpus.push_back(cpu);
        }
    }

    if (g_primary_cpus.empty()) {
        g_primary_cpus = g_all_cpus;
    }
}

void pinCurrentThread() {
    const int nt = omp_get_num_threads();
    if (nt <= 1) {
        return;
    }
    const int tid = omp_get_thread_num();

    const std::vector<int>& cpus =
        (static_cast<size_t>(nt) <= g_primary_cpus.size()) ? g_primary_cpus : g_all_cpus;
    const size_t n = cpus.size();
    if (n == 0) {
        return;
    }

    size_t pos = (static_cast<size_t>(tid) * n) / static_cast<size_t>(nt);
    if (pos >= n) {
        pos = n - 1;
    }

    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpus[pos], &set);
    sched_setaffinity(0, sizeof(set), &set);
}

void configureThreads(const size_t rows) {
    const bool userAffinity = (std::getenv("OMP_PROC_BIND") != nullptr ||
                               std::getenv("GOMP_CPU_AFFINITY") != nullptr);
    if (!userAffinity) {
        readTopology();
    }

    // The kernels are memory-bandwidth bound: using SMT siblings adds
    // contention without throughput. Cap the default to physical cores and
    // avoid oversubscribing very small grids where per-thread work is tiny.
    if (std::getenv("OMP_NUM_THREADS") == nullptr && !g_primary_cpus.empty()) {
        int target = std::min(omp_get_max_threads(),
                              static_cast<int>(g_primary_cpus.size()));
        const size_t minRowsPerThread = 4;
        const size_t workThreads = rows / minRowsPerThread;
        if (workThreads > 0 && workThreads < static_cast<size_t>(target)) {
            target = static_cast<int>(workThreads);
        }
        if (target > 0) {
            omp_set_num_threads(target);
        }
    }
}

} // namespace

// Compute the Cahn-Hilliard chemical potential at one grid point
inline double chemicalPotentialAt(const double* __restrict c, const size_t idx,
                                  const size_t xm, const size_t xp,
                                  const size_t ym, const size_t yp,
                                  const size_t zm, const size_t zp,
                                  const double dx2, const double dy2, const double dz2,
                                  const double gamma, const double e_AA,
                                  const double e_BB, const double e_AB) noexcept {
    const double cv = c[idx];

    const double cxx = (c[xp] + c[xm] - 2.0 * cv) / dx2;
    const double cyy = (c[yp] + c[ym] - 2.0 * cv) / dy2;
    const double czz = (c[zp] + c[zm] - 2.0 * cv) / dz2;

    return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
           + 3.0 * cv + cv * cv * cv
           - gamma * (cxx + cyy + czz);
}

// Compute the Laplacian at one grid point
inline double laplacianAt(const double* __restrict c, const size_t idx,
                          const size_t xm, const size_t xp,
                          const size_t ym, const size_t yp,
                          const size_t zm, const size_t zp,
                          const double dx2, const double dy2, const double dz2) noexcept {
    const double cv = c[idx];

    const double cxx = (c[xp] + c[xm] - 2.0 * cv) / dx2;
    const double cyy = (c[yp] + c[ym] - 2.0 * cv) / dy2;
    const double czz = (c[zp] + c[zm] - 2.0 * cv) / dz2;

    return cxx + cyy + czz;
}

// Compute chemical potential
void computeChemicalPotential(const double* __restrict c_ptr, double* __restrict mu_ptr,
                              const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t strideZ = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    const std::ptrdiff_t nz_i = static_cast<std::ptrdiff_t>(nz);
    const std::ptrdiff_t ny_i = static_cast<std::ptrdiff_t>(ny);

    #pragma omp for collapse(2) schedule(static)
    for (std::ptrdiff_t z = 0; z < nz_i; ++z) {
        for (std::ptrdiff_t y = 0; y < ny_i; ++y) {
            const size_t base = static_cast<size_t>(z) * strideZ + static_cast<size_t>(y) * nx;
            const size_t zm = (z > 0) ? base - strideZ : base;
            const size_t zp = (z + 1 < nz_i) ? base + strideZ : base;
            const size_t ym = (y > 0) ? base - nx : base;
            const size_t yp = (y + 1 < ny_i) ? base + nx : base;

            if (nx == 1) {
                mu_ptr[base] = chemicalPotentialAt(c_ptr, base, base, base, ym, yp, zm, zp,
                                                  dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
                continue;
            }

            mu_ptr[base] = chemicalPotentialAt(c_ptr, base, base, base + 1, ym, yp, zm, zp,
                                              dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);

            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = base + x;
                mu_ptr[idx] = chemicalPotentialAt(c_ptr, idx, idx - 1, idx + 1,
                                                  ym + x, yp + x, zm + x, zp + x,
                                                  dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
            }

            const size_t last = base + nx - 1;
            mu_ptr[last] = chemicalPotentialAt(c_ptr, last, last - 1, last,
                                               ym + nx - 1, yp + nx - 1,
                                               zm + nx - 1, zp + nx - 1,
                                               dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
        }
    }
}

// Cahn-Hilliard update step
void cahnHilliardUpdate(double* __restrict cnew_ptr, const double* __restrict cold_ptr,
                        const double* __restrict mu_ptr,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    const size_t strideZ = nx * ny;
    const double dx2 = dx * dx;
    const double dy2 = dy * dy;
    const double dz2 = dz * dz;

    const std::ptrdiff_t nz_i = static_cast<std::ptrdiff_t>(nz);
    const std::ptrdiff_t ny_i = static_cast<std::ptrdiff_t>(ny);

    #pragma omp for collapse(2) schedule(static)
    for (std::ptrdiff_t z = 0; z < nz_i; ++z) {
        for (std::ptrdiff_t y = 0; y < ny_i; ++y) {
            const size_t base = static_cast<size_t>(z) * strideZ + static_cast<size_t>(y) * nx;
            const size_t zm = (z > 0) ? base - strideZ : base;
            const size_t zp = (z + 1 < nz_i) ? base + strideZ : base;
            const size_t ym = (y > 0) ? base - nx : base;
            const size_t yp = (y + 1 < ny_i) ? base + nx : base;

            if (nx == 1) {
                cnew_ptr[base] = cold_ptr[base] + dt * D *
                    laplacianAt(mu_ptr, base, base, base, ym, yp, zm, zp, dx2, dy2, dz2);
                continue;
            }

            cnew_ptr[base] = cold_ptr[base] + dt * D *
                laplacianAt(mu_ptr, base, base, base + 1, ym, yp, zm, zp, dx2, dy2, dz2);

            #pragma omp simd
            for (size_t x = 1; x < nx - 1; ++x) {
                const size_t idx = base + x;
                cnew_ptr[idx] = cold_ptr[idx] + dt * D *
                    laplacianAt(mu_ptr, idx, idx - 1, idx + 1,
                                ym + x, yp + x, zm + x, zp + x, dx2, dy2, dz2);
            }

            const size_t last = base + nx - 1;
            cnew_ptr[last] = cold_ptr[last] + dt * D *
                laplacianAt(mu_ptr, last, last - 1, last,
                            ym + nx - 1, yp + nx - 1,
                            zm + nx - 1, zp + nx - 1, dx2, dy2, dz2);
        }
    }
}

// Initialize concentration field
void initializeConcentration(double* __restrict c_ptr, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t strideZ = nx * ny;
    const std::ptrdiff_t nz_i = static_cast<std::ptrdiff_t>(nz);
    const std::ptrdiff_t ny_i = static_cast<std::ptrdiff_t>(ny);

    // Use the same decomposition as the compute kernels so that NUMA
    // first-touch placement matches the threads that later use the data.
    #pragma omp for collapse(2) schedule(static)
    for (std::ptrdiff_t z = 0; z < nz_i; ++z) {
        for (std::ptrdiff_t y = 0; y < ny_i; ++y) {
            const size_t base = static_cast<size_t>(z) * strideZ + static_cast<size_t>(y) * nx;
            #pragma omp simd
            for (size_t x = 0; x < nx; ++x) {
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = base + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c_ptr[linear_id] = -1.0 + 2.0 * pseudo;
            }
        }
    }
}

bool validateResult(const double* __restrict c_ptr, const size_t n, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {

    // Check for NaN or Inf
    int invalid = 0;
    #pragma omp parallel for reduction(|:invalid) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double val = c_ptr[i];
        if (std::isnan(val) || std::isinf(val)) {
            invalid = 1;
        }
    }
    if (invalid) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c_ptr[0];
    double maxVal = c_ptr[0];
    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const double val = c_ptr[i];
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

    omp_set_dynamic(0);

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

    configureThreads(ny * nz);

    // Allocate arrays without zero-initialization so that pages are first
    // touched by the parallel kernels rather than serially in the allocator.
    std::unique_ptr<double[]> cold(new double[gridSize]);
    std::unique_ptr<double[]> cnew(new double[gridSize]);
    std::unique_ptr<double[]> mu(new double[gridSize]);

    // Initialize concentration field
    printf("Initializing concentration field...\n");

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");

    std::chrono::high_resolution_clock::time_point start;
    std::chrono::high_resolution_clock::time_point end;

    // Keep one persistent team of pinned threads for initialization and all
    // time steps: this ensures matching NUMA placement and removes repeated
    // fork/join overhead from the hot loop.
    #pragma omp parallel
    {
        pinCurrentThread();

        initializeConcentration(cold.get(), nx, ny, nz);

        #pragma omp single
        {
            start = std::chrono::high_resolution_clock::now();
        }

        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            computeChemicalPotential(cold.get(), mu.get(), nx, ny, nz, dx, dy, dz,
                                    gamma, e_AA, e_BB, e_AB);

            // Update concentration
            cahnHilliardUpdate(cnew.get(), cold.get(), mu.get(), nx, ny, nz, D, dt, dx, dy, dz);

            // Swap buffers
            #pragma omp single
            {
                std::swap(cold, cnew);
            }
        }

        #pragma omp single
        {
            end = std::chrono::high_resolution_clock::now();
        }
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        const std::vector<double> result(cold.get(), cold.get() + gridSize);
        print_results(result, "Concentration");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold.get(), gridSize, nx, ny, nz);

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
