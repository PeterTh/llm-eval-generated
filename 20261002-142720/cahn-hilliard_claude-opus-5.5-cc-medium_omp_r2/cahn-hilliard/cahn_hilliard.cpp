#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <omp.h>
#ifdef __linux__
#include <sched.h>
#endif
#include <vector>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Laplacian stencil at a single point, given neighbour values (clamped boundary handling
// is done by the caller choosing the neighbour rows/columns). Evaluation order matches the
// reference formulation exactly. With Recip == true the divisors are replaced by their
// reciprocals, which is only used when those are exact (powers of two), so results are
// bitwise identical to dividing.
template <bool Recip>
static inline double laplacianPoint(const double xp, const double xn, const double yp, const double yn,
                                    const double zp, const double zn, const double cc,
                                    const double sx, const double sy, const double sz) noexcept {
    double cxx, cyy, czz;
    if constexpr (Recip) {
        cxx = (xp + xn - 2.0 * cc) * sx;
        cyy = (yp + yn - 2.0 * cc) * sy;
        czz = (zp + zn - 2.0 * cc) * sz;
    } else {
        cxx = (xp + xn - 2.0 * cc) / sx;
        cyy = (yp + yn - 2.0 * cc) / sy;
        czz = (zp + zn - 2.0 * cc) / sz;
    }
    return cxx + cyy + czz;
}

// True if 1/v is exactly representable (v is a normal power of two)
static bool hasExactReciprocal(const double v) {
    int e = 0;
    return std::isnormal(v) && std::frexp(v, &e) == 0.5 && std::isnormal(1.0 / v);
}

// Physical parameters shared by the kernels
struct Params {
    double sx, sy, sz;  // dx^2, dy^2, dz^2 or their exact reciprocals (see laplacianPoint)
    double gamma, e_AA, e_BB, e_AB;
    double D, dt;
};

// Compute chemical potential for one x-row with clamped boundary conditions.
// r is the row itself; ryp/ryn/rzp/rzn are the (already clamped) neighbouring rows.
template <bool Recip>
static inline void chemicalPotentialRow(double* __restrict out, const double* __restrict r,
                                        const double* __restrict ryp, const double* __restrict ryn,
                                        const double* __restrict rzp, const double* __restrict rzn,
                                        const size_t nx, const Params& p) noexcept {
    const double dx2 = p.sx, dy2 = p.sy, dz2 = p.sz;
    const double gamma = p.gamma, e_AA = p.e_AA, e_BB = p.e_BB, e_AB = p.e_AB;

    auto point = [&](const size_t x, const size_t xp, const size_t xn) {
        const double cv = r[x];
        out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * laplacianPoint<Recip>(r[xp], r[xn], ryp[x], ryn[x], rzp[x], rzn[x], cv, dx2, dy2, dz2);
    };

    if (nx == 1) {
        point(0, 0, 0);
        return;
    }
    point(0, 1, 0);
#pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        const double cv = r[x];
        out[x] = 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                 + 3.0 * cv + cv * cv * cv
                 - gamma * laplacianPoint<Recip>(r[x + 1], r[x - 1], ryp[x], ryn[x], rzp[x], rzn[x], cv, dx2, dy2, dz2);
    }
    point(nx - 1, nx - 1, nx - 2);
}

// Cahn-Hilliard update for one x-row with clamped boundary conditions.
// m is the mu row itself; myp/myn/mzp/mzn are the (already clamped) neighbouring mu rows.
template <bool Recip>
static inline void cahnHilliardRow(double* __restrict out, const double* __restrict co,
                                   const double* __restrict m,
                                   const double* __restrict myp, const double* __restrict myn,
                                   const double* __restrict mzp, const double* __restrict mzn,
                                   const size_t nx, const Params& p) noexcept {
    const double dx2 = p.sx, dy2 = p.sy, dz2 = p.sz;
    const double D = p.D, dt = p.dt;

    auto point = [&](const size_t x, const size_t xp, const size_t xn) {
        out[x] = co[x] + dt * D *
                 laplacianPoint<Recip>(m[xp], m[xn], myp[x], myn[x], mzp[x], mzn[x], m[x], dx2, dy2, dz2);
    };

    if (nx == 1) {
        point(0, 0, 0);
        return;
    }
    point(0, 1, 0);
#pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        out[x] = co[x] + dt * D *
                 laplacianPoint<Recip>(m[x + 1], m[x - 1], myp[x], myn[x], mzp[x], mzn[x], m[x], dx2, dy2, dz2);
    }
    point(nx - 1, nx - 1, nx - 2);
}

// Static 2D (z, y) tile decomposition: each thread owns one tile for the whole run, so
// pages first-touched during initialization stay local to the thread that uses them.
struct Tile {
    size_t z0, z1, y0, y1;  // half-open ranges; empty if z0 == z1
};

static void chooseGrid(const size_t nthreads, const size_t ny, const size_t nz, size_t& pz, size_t& py) {
    // Maximize the number of busy threads, then minimize the halo overhead (2/tz + 2/ty)
    pz = 1;
    py = 1;
    size_t bestUsed = 0;
    double bestCost = 1e300;
    for (size_t a = 1; a <= std::min(nthreads, nz); ++a) {
        const size_t b = std::min(nthreads / a, ny);
        if (b == 0) continue;
        const size_t used = a * b;
        const double tz = static_cast<double>(nz) / a;
        const double ty = static_cast<double>(ny) / b;
        const double cost = 2.0 / tz + 2.0 / ty;
        if (used > bestUsed || (used == bestUsed && cost < bestCost)) {
            bestUsed = used;
            bestCost = cost;
            pz = a;
            py = b;
        }
    }
}

static Tile tileFor(const size_t tid, const size_t pz, const size_t py, const size_t ny, const size_t nz) {
    if (tid >= pz * py) return Tile{0, 0, 0, 0};
    const size_t iz = tid / py;
    const size_t iy = tid % py;
    auto lo = [](size_t n, size_t parts, size_t i) { return (n / parts) * i + std::min(i, n % parts); };
    return Tile{lo(nz, pz, iz), lo(nz, pz, iz + 1), lo(ny, py, iy), lo(ny, py, iy + 1)};
}

// Pin the calling OpenMP thread to a single CPU (compact placement over the allowed CPU set)
// unless the user already requested a binding policy via OMP_PROC_BIND / OMP_PLACES.
// Stable placement keeps first-touched pages NUMA-local and avoids thread migration.
// Must be called from inside a parallel region.
static void pinThreadIfUnbound() {
#ifdef __linux__
    if (omp_get_proc_bind() != omp_proc_bind_false) return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    const int ncpu = CPU_COUNT(&allowed);
    if (ncpu <= 0) return;
    int target = omp_get_thread_num() % ncpu;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (target-- == 0) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpu, &one);
            sched_setaffinity(0, sizeof(one), &one);
            return;
        }
    }
#endif
}

// Initialize concentration field over one tile (also first-touches the second buffer)
static void initializeConcentration(double* c, double* other, const Tile& t,
                                    const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;

    for (size_t z = t.z0; z < t.z1; ++z) {
        for (size_t y = t.y0; y < t.y1; ++y) {
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = idx3(x, y, z, nx, ny);
                // Generate pseudo-random value in [-1, 1]
                const size_t linear_id = z * (nx * ny) + y * nx + x;
                const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                c[idx] = -1.0 + 2.0 * pseudo;
                other[idx] = 0.0;
            }
        }
    }
}

// One fused time step over a tile: the chemical potential is computed into a thread-private
// rolling window of 3 z-planes (restricted to the tile's y-range plus a 1-row halo), and the
// concentration update for plane z is applied as soon as mu planes z-1, z, z+1 are available.
// mu values are bitwise identical to a global two-pass evaluation; halo values are simply
// recomputed redundantly by neighbouring tiles.
template <bool Recip>
static void fusedStepTile(double* __restrict cnew, const double* __restrict cold, double* __restrict win,
                          const Tile& t, const size_t nx, const size_t ny, const size_t nz, const Params& p) {
    if (t.z0 >= t.z1 || t.y0 >= t.y1) return;

    const size_t ya = (t.y0 > 0) ? t.y0 - 1 : 0;           // first mu row needed
    const size_t yb = (t.y1 < ny) ? t.y1 : ny - 1;         // last mu row needed (inclusive)
    const size_t rows = yb - ya + 1;
    const size_t planeStride = rows * nx;

    auto slot = [&](size_t zz) -> double* { return win + (zz % 3) * planeStride; };
    auto muRow = [&](size_t zz, size_t y) -> const double* { return slot(zz) + (y - ya) * nx; };

    auto computeMuPlane = [&](size_t zz) {
        const size_t zp = (zz < nz - 1) ? zz + 1 : zz;
        const size_t zn = (zz > 0) ? zz - 1 : 0;
        double* dst = slot(zz);
        for (size_t y = ya; y <= yb; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            chemicalPotentialRow<Recip>(dst + (y - ya) * nx,
                                 cold + idx3(0, y, zz, nx, ny),
                                 cold + idx3(0, yp, zz, nx, ny),
                                 cold + idx3(0, yn, zz, nx, ny),
                                 cold + idx3(0, y, zp, nx, ny),
                                 cold + idx3(0, y, zn, nx, ny),
                                 nx, p);
        }
    };

    // Prime the window with planes z0-1 (clamped) and z0
    size_t nextMu = (t.z0 > 0) ? t.z0 - 1 : 0;
    const size_t lastMu = (t.z1 < nz) ? t.z1 : nz - 1;  // inclusive

    for (size_t z = t.z0; z < t.z1; ++z) {
        const size_t zp = (z < nz - 1) ? z + 1 : z;
        const size_t zn = (z > 0) ? z - 1 : 0;
        while (nextMu <= zp && nextMu <= lastMu) {
            computeMuPlane(nextMu);
            ++nextMu;
        }
        for (size_t y = t.y0; y < t.y1; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const size_t base = idx3(0, y, z, nx, ny);
            cahnHilliardRow<Recip>(cnew + base, cold + base,
                            muRow(z, y), muRow(z, yp), muRow(z, yn),
                            muRow(zp, y), muRow(zn, y), nx, p);
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    // Check for NaN or Inf
    for (const auto& val : c) {
        if (std::isnan(val) || std::isinf(val)) {
            printf("Validation failed: found NaN or Inf value\n");
            return false;
        }
    }
    
    // Check if values are in reasonable range for concentration field
    // After Cahn-Hilliard evolution, values should typically remain bounded
    double minVal = c[0];
    double maxVal = c[0];
    for (const auto& val : c) {
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

    // Reference divides by (dx*dx) etc.; multiply by the reciprocal only when that is exact
    const bool useRecip = hasExactReciprocal(dx * dx) && hasExactReciprocal(dy * dy) && hasExactReciprocal(dz * dz);
    const Params params = useRecip
        ? Params{1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz), gamma, e_AA, e_BB, e_AB, D, dt}
        : Params{dx * dx, dy * dy, dz * dz, gamma, e_AA, e_BB, e_AB, D, dt};
    
    // Allocate arrays (uninitialized; pages are first-touched in parallel for NUMA locality)
    std::unique_ptr<double[]> bufA(new double[gridSize]);
    std::unique_ptr<double[]> bufB(new double[gridSize]);

    // Fixed tile decomposition shared by initialization and the time loop
    const size_t nthreads = static_cast<size_t>(omp_get_max_threads());
    size_t pz = 1, py = 1;
    chooseGrid(nthreads, ny, nz, pz, py);
    
    std::vector<std::vector<double>> windows(nthreads);
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
#pragma omp parallel num_threads(static_cast<int>(nthreads))
    {
        pinThreadIfUnbound();
        const Tile tile = tileFor(static_cast<size_t>(omp_get_thread_num()), pz, py, ny, nz);
        initializeConcentration(bufA.get(), bufB.get(), tile, nx, ny, nz);
        // Thread-private rolling window of 3 mu planes over the tile's y-range plus halo
        // (allocated and first-touched by its owner thread)
        const size_t winRows = (tile.y1 > tile.y0) ? std::min(tile.y1 - tile.y0 + 2, ny) : 0;
        windows[static_cast<size_t>(omp_get_thread_num())].assign(3 * winRows * nx, 0.0);
    }
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
#pragma omp parallel num_threads(static_cast<int>(nthreads))
    {
        const Tile tile = tileFor(static_cast<size_t>(omp_get_thread_num()), pz, py, ny, nz);
        double* window = windows[static_cast<size_t>(omp_get_thread_num())].data();

        double* cold = bufA.get();
        double* cnew = bufB.get();
        
        for (int t = 0; t < iterations; ++t) {
            // Compute chemical potential and update concentration (fused per tile)
            if (useRecip) {
                fusedStepTile<true>(cnew, cold, window, tile, nx, ny, nz, params);
            } else {
                fusedStepTile<false>(cnew, cold, window, tile, nx, ny, nz, params);
            }
            
            // All tiles must finish reading cold before it is overwritten next step
#pragma omp barrier
            
            // Swap buffers (thread-local pointers)
            std::swap(cold, cnew);
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    const double* finalBuf = (iterations > 0 && (iterations % 2) == 1) ? bufB.get() : bufA.get();
    std::vector<double> cold(finalBuf, finalBuf + gridSize);
    
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
