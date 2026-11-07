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

// Compute Laplacian with clamped boundary conditions at a single point.
// Used for the x-boundary cells of each row; interior cells use the
// branch-free row kernel below. Arithmetic order is identical in both so
// results are bit-exact with respect to the original implementation.
inline double computeLaplacianPoint(const double* __restrict c, const size_t nx, const size_t ny, const size_t nz,
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

// Compute the Laplacian for one full x-row (fixed y, z) into lap[0..nx).
// Interior cells (0 < x < nx-1) are computed without clamping branches so the
// compiler can vectorize them; the two x-boundary cells use the clamped kernel.
inline void computeLaplacianRow(const double* __restrict c, double* __restrict lap,
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

    // x = 0 boundary (clamped)
    lap[0] = computeLaplacianPoint(c, nx, ny, nz, dx, dy, dz, 0, y, z);

    // Interior: 1 <= x <= nx-2
    const size_t xEnd = (nx > 1) ? nx - 1 : 1;
    #pragma omp simd
    for (size_t x = 1; x < xEnd; ++x) {
        const double cv = row[x];
        const double cxx = (row[x + 1] + row[x - 1] - 2.0 * cv) / idx2;
        const double cyy = (rowYp[x] + rowYn[x] - 2.0 * cv) / idy2;
        const double czz = (rowZp[x] + rowZn[x] - 2.0 * cv) / idz2;
        lap[x] = cxx + cyy + czz;
    }

    // x = nx-1 boundary (clamped); for nx == 1 this coincides with x = 0
    if (nx > 1) {
        lap[nx - 1] = computeLaplacianPoint(c, nx, ny, nz, dx, dy, dz, nx - 1, y, z);
    }
}

// Compute chemical potential for one x-row. Must be called from inside a
// parallel region; distributes rows over the team via a worksharing loop.
inline void computeChemicalPotentialRow(const double* __restrict c, double* __restrict mu,
                                        const size_t nx, const size_t ny, const size_t nz,
                                        const double dx, const double dy, const double dz,
                                        const double gamma, const double e_AA, const double e_BB, const double e_AB,
                                        const size_t y, const size_t z) noexcept {
    double* __restrict muRow = mu + idx3(0, y, z, nx, ny);
    const double* __restrict cRow = c + idx3(0, y, z, nx, ny);

    // Laplacian of c written into mu row first, then combined in-place.
    computeLaplacianRow(c, muRow, nx, ny, nz, dx, dy, dz, y, z);

    #pragma omp simd
    for (size_t x = 0; x < nx; ++x) {
        const double cv = cRow[x];
        muRow[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * muRow[x];
    }
}

// Cahn-Hilliard update for one x-row.
inline void cahnHilliardUpdateRow(double* __restrict cnew, const double* __restrict cold,
                                  const double* __restrict mu,
                                  const size_t nx, const size_t ny, const size_t nz,
                                  const double D, const double dt, const double dx, const double dy, const double dz,
                                  const size_t y, const size_t z) noexcept {
    double* __restrict newRow = cnew + idx3(0, y, z, nx, ny);
    const double* __restrict oldRow = cold + idx3(0, y, z, nx, ny);

    // Laplacian of mu written into cnew row first, then combined in-place.
    computeLaplacianRow(mu, newRow, nx, ny, nz, dx, dy, dz, y, z);

    #pragma omp simd
    for (size_t x = 0; x < nx; ++x) {
        newRow[x] = oldRow[x] + dt * D * newRow[x];
    }
}

// Compute chemical potential over the full grid (must be called inside a parallel region)
inline void computeChemicalPotential(const double* __restrict c, double* __restrict mu,
                                     const size_t nx, const size_t ny, const size_t nz,
                                     const double dx, const double dy, const double dz,
                                     const double gamma, const double e_AA, const double e_BB, const double e_AB) noexcept {
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            computeChemicalPotentialRow(c, mu, nx, ny, nz, dx, dy, dz, gamma, e_AA, e_BB, e_AB, y, z);
        }
    }
}

// Cahn-Hilliard update step over the full grid (must be called inside a parallel region)
inline void cahnHilliardUpdate(double* __restrict cnew, const double* __restrict cold,
                               const double* __restrict mu,
                               const size_t nx, const size_t ny, const size_t nz,
                               const double D, const double dt, const double dx, const double dy, const double dz) noexcept {
    #pragma omp for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            cahnHilliardUpdateRow(cnew, cold, mu, nx, ny, nz, D, dt, dx, dy, dz, y, z);
        }
    }
}

// Parse a sysfs CPU list such as "0,128" or "0-3,64-67" into `out`.
static void parseCpuList(const char* str, std::vector<int>& out) {
    const char* p = str;
    while (*p) {
        char* end = nullptr;
        const long a = strtol(p, &end, 10);
        if (end == p) break;
        long b = a;
        p = end;
        if (*p == '-') {
            ++p;
            b = strtol(p, &end, 10);
            if (end == p) break;
            p = end;
        }
        for (long c = a; c <= b && c < CPU_SETSIZE; ++c) {
            out.push_back(static_cast<int>(c));
        }
        while (*p == ',' || *p == ' ' || *p == '\n') ++p;
    }
}

// Order the allowed CPUs so that SMT siblings of one core are adjacent
// (topology order, like OMP_PLACES=threads). Adjacent threads then own
// adjacent z-slabs and share the core's private caches for their halos.
// Falls back to numeric order when sysfs topology is unavailable.
static std::vector<int> topologyOrderedCpus(const cpu_set_t& allowed) {
    std::vector<int> cpus;
    std::vector<char> placed(CPU_SETSIZE, 0);
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (!CPU_ISSET(i, &allowed) || placed[i]) continue;

        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", i);
        std::vector<int> siblings;
        if (FILE* f = fopen(path, "r")) {
            char buf[1024];
            if (fgets(buf, sizeof(buf), f)) {
                parseCpuList(buf, siblings);
            }
            fclose(f);
        }
        if (siblings.empty()) siblings.push_back(i);

        for (int c : siblings) {
            if (c >= 0 && c < CPU_SETSIZE && CPU_ISSET(c, &allowed) && !placed[c]) {
                placed[c] = 1;
                cpus.push_back(c);
            }
        }
        if (!placed[i]) {  // sibling list did not include i itself
            placed[i] = 1;
            cpus.push_back(i);
        }
    }
    return cpus;
}

// Pin OpenMP threads to distinct CPUs when the runtime has no binding policy.
// Unbound threads migrate between cores and NUMA nodes, which destroys the
// cache and first-touch locality of the static schedule. If the user has set
// OMP_PROC_BIND / OMP_PLACES the runtime's placement is respected instead.
static void pinThreadsIfUnbound() {
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    const std::vector<int> cpus = topologyOrderedCpus(allowed);
    if (cpus.empty()) {
        return;
    }

    #pragma omp parallel
    {
        const int tid = omp_get_thread_num();
        const int nth = omp_get_num_threads();
        // Spread threads evenly over the topology-ordered CPU list
        const size_t pos = (static_cast<size_t>(tid) * cpus.size()) / static_cast<size_t>(nth);
        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpus[pos], &one);
        sched_setaffinity(0, sizeof(one), &one);
    }
}

// Initialize concentration field (parallel, also performs NUMA first-touch)
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

// Touch the other buffers in the same thread/page layout as the concentration field
static void firstTouch(std::vector<double>& v) {
    const size_t n = v.size();
    double* __restrict p = v.data();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        p[i] = 0.0;
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* __restrict p = c.data();

    // Check for NaN or Inf
    bool bad = false;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for (size_t i = 0; i < n; ++i) {
        const double val = p[i];
        bad |= (std::isnan(val) || std::isinf(val));
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = p[0];
    double maxVal = p[0];
    #pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < n; ++i) {
        const double val = p[i];
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

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());

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

    // Pin threads before any first-touch so data stays local to its thread
    pinThreadsIfUnbound();

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz);
    firstTouch(cnew);
    firstTouch(mu);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // One parallel region for the whole time loop: the two worksharing loops
    // per step are separated by their implicit barriers, and every thread
    // swaps its own private copy of the buffer pointers identically.
    {
        double* pOld = cold.data();
        double* pNew = cnew.data();
        double* pMu  = mu.data();

        #pragma omp parallel firstprivate(pOld, pNew)
        {
            for (int t = 0; t < iterations; ++t) {
                // Compute chemical potential
                computeChemicalPotential(pOld, pMu, nx, ny, nz, dx, dy, dz,
                                         gamma, e_AA, e_BB, e_AB);

                // Update concentration
                cahnHilliardUpdate(pNew, pOld, pMu, nx, ny, nz, D, dt, dx, dy, dz);

                // Swap buffers (thread-private pointers, identical on all threads)
                std::swap(pOld, pNew);
            }
        }

        // Mirror the swaps on the vectors so `cold` holds the final result
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
