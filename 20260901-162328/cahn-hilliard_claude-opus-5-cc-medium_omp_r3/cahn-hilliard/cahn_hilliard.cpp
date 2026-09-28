#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <pthread.h>
#include <sched.h>
#endif

#include "../common/results_output.hpp"

#ifdef __linux__
// Read the sibling list of a hardware thread (e.g. "0,128" or "0-1") and return the
// smallest CPU id in it, i.e. a stable representative of the physical core.
int coreRepresentative(const int cpu) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    FILE* f = fopen(path, "r");
    if (f == nullptr) {
        return cpu;
    }
    char buf[512] = {0};
    const size_t len = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    if (len == 0) {
        return cpu;
    }

    int rep = cpu;
    for (const char* p = buf; *p != '\0';) {
        if (*p >= '0' && *p <= '9') {
            const int v = atoi(p);
            rep = std::min(rep, v);
            while (*p >= '0' && *p <= '9') ++p;
        } else {
            ++p;
        }
    }
    return rep;
}
#endif

// Pin every OpenMP thread to a distinct CPU of the process' affinity mask, and by default
// use one thread per physical core.
//
// This benchmark is memory bound, so thread migration (which invalidates the NUMA-local
// first touch of the grid arrays and the private caches) costs more than an order of
// magnitude on large multi-socket machines, and SMT siblings only add contention. If the
// user configured the OpenMP runtime's own binding or thread count we leave that alone;
// otherwise we do it here, since libgomp has already parsed its environment by the time
// main() runs.
void configureThreads() {
#ifdef __linux__
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    // Split the usable CPUs into one hardware thread per physical core plus the remaining
    // SMT siblings.
    std::vector<int> primary;
    std::vector<int> siblings;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        if (coreRepresentative(cpu) == cpu) {
            primary.push_back(cpu);
        } else {
            siblings.push_back(cpu);
        }
    }
    std::vector<int> cpuOrder = primary;
    cpuOrder.insert(cpuOrder.end(), siblings.begin(), siblings.end());
    if (cpuOrder.empty()) {
        return;
    }

    if (getenv("OMP_NUM_THREADS") == nullptr && getenv("OMP_THREAD_LIMIT") == nullptr &&
        !primary.empty()) {
        omp_set_num_threads(static_cast<int>(primary.size()));
    }

#pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t nthreads = static_cast<size_t>(omp_get_num_threads());
        // Teams that fit on the physical cores are spread evenly over them (so that a
        // partially loaded machine uses all caches, memory controllers and sockets);
        // larger teams additionally fill the SMT siblings.
        const int cpu = (nthreads <= primary.size() && !primary.empty())
                            ? primary[tid * primary.size() / nthreads]
                            : cpuOrder[tid % cpuOrder.size()];
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#endif
}

// True if dividing by v is exactly equivalent to multiplying by 1/v, which is the case
// iff v is a (finite, normal) power of two.
inline bool divisionIsExactlyReciprocal(const double v) noexcept {
    int exp = 0;
    return v > 0.0 && std::isnormal(v) && std::frexp(v, &exp) == 0.5;
}

// Decomposition of the grid into (z-slab, y-block) tiles.
//
// Both the stencil sweeps and the initialization walk the grid with this decomposition and
// a static schedule, so every thread always works on - and first touches - the same tiles.
struct GridBlocking {
    size_t yBlock;
    size_t zBlock;
    size_t nyBlocks;
    size_t nzBlocks;
};

// Pick a tile shape that keeps the three y/z neighbour row planes of a tile in cache while
// still handing every thread a few tiles worth of work.
inline GridBlocking makeBlocking(const size_t nx, const size_t ny, const size_t nz, const int nthreads) {
    // ~1 MiB of working set per thread: three planes of nx * yBlock doubles. The block
    // counts are turned back into evenly sized blocks so that all tiles carry the same
    // amount of work.
    const size_t yTarget = std::clamp<size_t>(1024 * 1024 / (3 * nx * sizeof(double)), 4, ny);

    GridBlocking b{};
    b.nyBlocks = (ny + yTarget - 1) / yTarget;
    b.yBlock = (ny + b.nyBlocks - 1) / b.nyBlocks;
    b.nyBlocks = (ny + b.yBlock - 1) / b.yBlock;

    // Over-decompose slightly in z so that the static schedule balances well.
    const size_t wantedTiles = static_cast<size_t>(std::max(nthreads, 1)) * 2;
    size_t nzBlocks = std::clamp<size_t>((wantedTiles + b.nyBlocks - 1) / b.nyBlocks, 1, nz);
    b.zBlock = (nz + nzBlocks - 1) / nzBlocks;
    b.nzBlocks = (nz + b.zBlock - 1) / b.zBlock;
    return b;
}

// Distribute the tiles over the OpenMP team and invoke rowFn(z, y) for every row of every
// tile. Orphaned worksharing construct: must be called from inside a parallel region.
template <typename RowFn>
inline void forEachRow(const GridBlocking& b, const size_t ny, const size_t nz, RowFn rowFn) {
#pragma omp for collapse(2) schedule(static)
    for (size_t zi = 0; zi < b.nzBlocks; ++zi) {
        for (size_t yi = 0; yi < b.nyBlocks; ++yi) {
            const size_t zEnd = std::min(nz, zi * b.zBlock + b.zBlock);
            const size_t yEnd = std::min(ny, yi * b.yBlock + b.yBlock);
            for (size_t z = zi * b.zBlock; z < zEnd; ++z) {
                for (size_t y = yi * b.yBlock; y < yEnd; ++y) {
                    rowFn(z, y);
                }
            }
        }
    }
}

// Apply a stencil-based kernel to every cell of the grid, one (z, y) row at a time.
//
// The stencil uses clamped (zero-gradient) boundary conditions. Instead of clamping every
// single access, the row pointers for the y/z neighbours are clamped once per row and only
// the two x-boundary cells of a row need special treatment. This leaves a contiguous,
// branch-free inner loop over x that the compiler can vectorize.
//
// `body` is invoked as body(idx, laplacian) for every cell. With Recip the (loop
// invariant) divisions by the squared grid spacings are turned into multiplications by
// their reciprocals, which the caller only selects when that is bit-exact.
template <bool Recip, typename F>
inline void stencilLoop(const double* __restrict__ f, const GridBlocking& b,
                        const size_t nx, const size_t ny, const size_t nz,
                        const double dx, const double dy, const double dz, F body) {
    const size_t nxy = nx * ny;
    const double dx2 = Recip ? 1.0 / (dx * dx) : dx * dx;
    const double dy2 = Recip ? 1.0 / (dy * dy) : dy * dy;
    const double dz2 = Recip ? 1.0 / (dz * dz) : dz * dz;

    // Scale a second difference by the corresponding 1/h^2 factor.
    const auto scale = [](const double v, const double h2) {
        if constexpr (Recip) {
            return v * h2;
        } else {
            return v / h2;
        }
    };

    forEachRow(b, ny, nz, [&](const size_t z, const size_t y) {
        const size_t base = z * nxy + y * nx;
        const double* __restrict__ row = f + base;
        const double* __restrict__ rowYp = f + z * nxy + ((y < ny - 1) ? y + 1 : y) * nx;
        const double* __restrict__ rowYn = f + z * nxy + ((y > 0) ? y - 1 : y) * nx;
        const double* __restrict__ rowZp = f + ((z < nz - 1) ? z + 1 : z) * nxy + y * nx;
        const double* __restrict__ rowZn = f + ((z > 0) ? z - 1 : z) * nxy + y * nx;

        // Laplacian at position x of this row, given the clamped x neighbours.
        const auto lap = [&](const size_t x, const size_t xp, const size_t xn) {
            const double cv = row[x];
            const double cxx = scale(row[xp] + row[xn] - 2.0 * cv, dx2);
            const double cyy = scale(rowYp[x] + rowYn[x] - 2.0 * cv, dy2);
            const double czz = scale(rowZp[x] + rowZn[x] - 2.0 * cv, dz2);
            return cxx + cyy + czz;
        };

        if (nx == 1) {
            body(base, lap(0, 0, 0));
            return;
        }

        body(base, lap(0, 1, 0));

#pragma omp simd
        for (size_t x = 1; x < nx - 1; ++x) {
            const double cv = row[x];
            const double cxx = scale(row[x + 1] + row[x - 1] - 2.0 * cv, dx2);
            const double cyy = scale(rowYp[x] + rowYn[x] - 2.0 * cv, dy2);
            const double czz = scale(rowZp[x] + rowZn[x] - 2.0 * cv, dz2);
            body(base + x, cxx + cyy + czz);
        }

        body(base + nx - 1, lap(nx - 1, nx - 1, nx - 2));
    });
}

// Compute chemical potential
template <bool Recip>
void computeChemicalPotential(const double* __restrict__ c, double* __restrict__ mu,
                              const GridBlocking& b, const size_t nx, const size_t ny, const size_t nz,
                              const double dx, const double dy, const double dz,
                              const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    stencilLoop<Recip>(c, b, nx, ny, nz, dx, dy, dz, [=](const size_t idx, const double laplacian) {
        const double cv = c[idx];
        mu[idx] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * laplacian;
    });
}

// Cahn-Hilliard update step
template <bool Recip>
void cahnHilliardUpdate(double* __restrict__ cnew, const double* __restrict__ cold,
                        const double* __restrict__ mu,
                        const GridBlocking& b, const size_t nx, const size_t ny, const size_t nz,
                        const double D, const double dt, const double dx, const double dy, const double dz) {
    stencilLoop<Recip>(mu, b, nx, ny, nz, dx, dy, dz, [=](const size_t idx, const double laplacian) {
        cnew[idx] = cold[idx] + dt * D * laplacian;
    });
}

// Initialize concentration field. The extra buffers are first-touched with the very same
// tile distribution the compute kernels use, so all pages end up on the NUMA node of the
// thread that will work on them.
void initializeConcentration(double* __restrict__ c, double* __restrict__ cnew, double* __restrict__ mu,
                             const GridBlocking& b, const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    const size_t nxy = nx * ny;

#pragma omp parallel
    forEachRow(b, ny, nz, [&](const size_t z, const size_t y) {
        const size_t base = z * nxy + y * nx;
        for (size_t x = 0; x < nx; ++x) {
            const size_t idx = base + x;
            // Generate pseudo-random value in [-1, 1]
            const size_t linear_id = idx;
            const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
            c[idx] = -1.0 + 2.0 * pseudo;
            cnew[idx] = 0.0;
            mu[idx] = 0.0;
        }
    });
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* __restrict__ data = c.data();

    // Check for NaN or Inf
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < n; ++i) {
        bad = bad || std::isnan(data[i]) || std::isinf(data[i]);
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = data[0];
    double maxVal = data[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
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
    configureThreads();
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

    // Allocate arrays. These are deliberately raw, uninitialized allocations: a
    // std::vector would zero the whole grid from the master thread, which places every
    // page on that thread's NUMA node. Instead the arrays are first touched below by the
    // threads that will own them, which is worth over 50% of the achievable bandwidth on
    // a multi-socket machine.
    const size_t bytes = std::max<size_t>(gridSize * sizeof(double), 64);
    double* coldPtr = static_cast<double*>(aligned_alloc(64, (bytes + 63) & ~size_t(63)));
    double* cnewPtr = static_cast<double*>(aligned_alloc(64, (bytes + 63) & ~size_t(63)));
    double* muPtr = static_cast<double*>(aligned_alloc(64, (bytes + 63) & ~size_t(63)));
    if (coldPtr == nullptr || cnewPtr == nullptr || muPtr == nullptr) {
        printf("Failed to allocate grid arrays\n");
        return 1;
    }

    const GridBlocking blocking = makeBlocking(nx, ny, nz, omp_get_max_threads());

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(coldPtr, cnewPtr, muPtr, blocking, nx, ny, nz);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Selected once: multiplying by the reciprocal grid spacings is only used when it
    // yields exactly the same results as dividing by them.
    const bool recip = divisionIsExactlyReciprocal(dx * dx) &&
                       divisionIsExactlyReciprocal(dy * dy) &&
                       divisionIsExactlyReciprocal(dz * dz);

    // A single parallel region spans the whole time loop: the threads are started once and
    // the per-step synchronization is just the implicit barriers of the worksharing
    // constructs inside the two sweeps.
#pragma omp parallel
    {
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential
            if (recip) {
                computeChemicalPotential<true>(coldPtr, muPtr, blocking, nx, ny, nz, dx, dy, dz,
                                               gamma, e_AA, e_BB, e_AB);
            } else {
                computeChemicalPotential<false>(coldPtr, muPtr, blocking, nx, ny, nz, dx, dy, dz,
                                                gamma, e_AA, e_BB, e_AB);
            }

            // Update concentration
            if (recip) {
                cahnHilliardUpdate<true>(cnewPtr, coldPtr, muPtr, blocking, nx, ny, nz, D, dt, dx, dy, dz);
            } else {
                cahnHilliardUpdate<false>(cnewPtr, coldPtr, muPtr, blocking, nx, ny, nz, D, dt, dx, dy, dz);
            }

            // Swap buffers
#pragma omp single
            std::swap(coldPtr, cnewPtr);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Hand the final state over for reporting. The scratch buffers are released first so
    // that this does not increase the peak memory footprint.
    free(muPtr);
    free(cnewPtr);
    std::vector<double> cold(coldPtr, coldPtr + gridSize);
    free(coldPtr);

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
