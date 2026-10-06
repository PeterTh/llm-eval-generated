#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>
#include <sched.h>

#include "../common/results_output.hpp"

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Pointers to the five rows needed for a 7-point Laplacian of one row
// (neighbors already clamped at the domain boundaries).
struct RowPtrs {
    const double* r;   // current row
    const double* ryp; // row y+1 (clamped)
    const double* ryn; // row y-1 (clamped)
    const double* rzp; // row z+1 (clamped)
    const double* rzn; // row z-1 (clamped)
};

// Laplacian with clamped boundary conditions at position x of a row,
// with xp/xn the (clamped) x neighbors.
inline double laplacianAt(const RowPtrs& p, const size_t x, const size_t xp, const size_t xn,
                          const double dx2, const double dy2, const double dz2) noexcept {
    const double cxx = (p.r[xp] + p.r[xn] - 2.0 * p.r[x]) / dx2;
    const double cyy = (p.ryp[x] + p.ryn[x] - 2.0 * p.r[x]) / dy2;
    const double czz = (p.rzp[x] + p.rzn[x] - 2.0 * p.r[x]) / dz2;
    return cxx + cyy + czz;
}

// Chemical potential of a single row: m[x] = mu(x, y, z)
inline void chemicalPotentialRow(const RowPtrs& p, double* __restrict m, const size_t nx,
                                 const double dx2, const double dy2, const double dz2,
                                 const double gamma, const double e_AA, const double e_BB, const double e_AB) noexcept {
    auto point = [&](const size_t x, const size_t xp, const size_t xn) {
        const double cv = p.r[x];
        return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
               + 3.0 * cv + cv * cv * cv
               - gamma * laplacianAt(p, x, xp, xn, dx2, dy2, dz2);
    };
    m[0] = point(0, (nx > 1) ? 1 : 0, 0);
    #pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        m[x] = point(x, x + 1, x - 1);
    }
    if (nx > 1) m[nx - 1] = point(nx - 1, nx - 1, nx - 2);
}

// Cahn-Hilliard update of a single row: cn[x] = co[x] + dt * D * lap(mu)(x, y, z)
inline void cahnHilliardRow(const RowPtrs& p, double* __restrict cn, const double* __restrict co, const size_t nx,
                            const double dx2, const double dy2, const double dz2,
                            const double D, const double dt) noexcept {
    auto point = [&](const size_t x, const size_t xp, const size_t xn) {
        return co[x] + dt * D * laplacianAt(p, x, xp, xn, dx2, dy2, dz2);
    };
    cn[0] = point(0, (nx > 1) ? 1 : 0, 0);
    #pragma omp simd
    for (size_t x = 1; x < nx - 1; ++x) {
        cn[x] = point(x, x + 1, x - 1);
    }
    if (nx > 1) cn[nx - 1] = point(nx - 1, nx - 1, nx - 2);
}

// Domain decomposition into tiles of (z-slab x y-block). Each tile is
// processed by one thread, streaming through z while keeping a rolling
// 3-plane window of the chemical potential (for the tile's y-rows plus halo)
// in a small cache-resident buffer. This fuses the chemical-potential and
// update kernels, so mu never has to round-trip through main memory.
struct Tiling {
    size_t yBlock;  // rows per y-block
    size_t nyb;     // number of y-blocks
    size_t zSlab;   // planes per z-slab
    size_t nzs;     // number of z-slabs
    size_t ntiles;
};

Tiling makeTiling(const size_t nx, const size_t ny, const size_t nz, const size_t nthreads) {
    const size_t T = std::max<size_t>(nthreads, 1);
    // Keep the 3-plane mu window (3 * (yBlock + 2) * nx doubles) around 256 KiB
    const size_t rows = (256 * 1024) / (3 * sizeof(double) * std::max<size_t>(nx, 1));
    const size_t ybMax = std::min(std::max<size_t>((rows > 10) ? rows - 2 : 8, 1), std::max<size_t>(ny, 1));
    const size_t nybMin = (ny + ybMax - 1) / ybMax;

    // Search block counts minimizing the per-thread work of the busiest thread
    // (update work plus redundant halo chemical-potential work).
    Tiling best{};
    double bestCost = -1.0;
    const size_t nybMaxSearch = std::min(std::max<size_t>(ny, 1), nybMin * 8);
    for (size_t nyb = nybMin; nyb <= nybMaxSearch; ++nyb) {
        const size_t yb = (ny + nyb - 1) / nyb;
        const size_t nybEff = (ny + yb - 1) / yb;
        if (nybEff != nyb) continue;
        const size_t nzsMax = std::min(std::max<size_t>(nz, 1), (2 * T + nyb - 1) / nyb);
        for (size_t nzs = 1; nzs <= nzsMax; ++nzs) {
            const size_t zs = (nz + nzs - 1) / nzs;
            const size_t nzsEff = (nz + zs - 1) / zs;
            if (nzsEff != nzs) continue;
            const size_t ntiles = nyb * nzs;
            const double tilesPerThread = static_cast<double>((ntiles + T - 1) / T);
            const double tileWork = static_cast<double>(yb * zs)
                                  + static_cast<double>((yb + 2) * (zs + 2));
            const double cost = tilesPerThread * tileWork;
            if (bestCost < 0.0 || cost < bestCost) {
                bestCost = cost;
                best = Tiling{yb, nyb, zs, nzs, ntiles};
            }
        }
    }
    return best;
}

// One full time step on one tile: reads cold, writes cnew.
// muBuf must hold 3 * (yBlock + 2) * nx doubles.
void processTile(const Tiling& tl, const size_t tile, double* __restrict cnew, const double* __restrict cold,
                 double* __restrict muBuf,
                 const size_t nx, const size_t ny, const size_t nz,
                 const double D, const double dt, const double dx, const double dy, const double dz,
                 const double gamma, const double e_AA, const double e_BB, const double e_AB) {
    const size_t z0 = (tile / tl.nyb) * tl.zSlab;
    const size_t z1 = std::min(z0 + tl.zSlab, nz);
    const size_t y0 = (tile % tl.nyb) * tl.yBlock;
    const size_t y1 = std::min(y0 + tl.yBlock, ny);
    if (z0 >= z1 || y0 >= y1) return;

    const double dx2 = dx * dx, dy2 = dy * dy, dz2 = dz * dz;
    const size_t plane = nx * ny;

    // mu rows needed: [rlo, rhi], planes needed: [plo, phi] (inclusive, clamped)
    const size_t rlo = (y0 > 0) ? y0 - 1 : 0;
    const size_t rhi = (y1 < ny) ? y1 : ny - 1;
    const size_t plo = (z0 > 0) ? z0 - 1 : 0;
    const size_t winRows = rhi - rlo + 1;
    const size_t slotSize = winRows * nx;

    auto slot = [&](const size_t z) { return muBuf + (z % 3) * slotSize; };
    auto row = [&](const double* base, const size_t y, const size_t z) { return base + z * plane + y * nx; };

    size_t next = plo;  // next mu plane to compute
    for (size_t z = z0; z < z1; ++z) {
        const size_t zp = (z < nz - 1) ? z + 1 : z;
        const size_t zn = (z > 0) ? z - 1 : 0;

        // Compute mu planes up to zp into the rolling window
        for (; next <= zp; ++next) {
            const size_t mz = next;
            const size_t mzp = (mz < nz - 1) ? mz + 1 : mz;
            const size_t mzn = (mz > 0) ? mz - 1 : 0;
            double* dst = slot(mz);
            for (size_t y = rlo; y <= rhi; ++y) {
                const size_t yp = (y < ny - 1) ? y + 1 : y;
                const size_t yn = (y > 0) ? y - 1 : 0;
                const RowPtrs p{row(cold, y, mz), row(cold, yp, mz), row(cold, yn, mz),
                                row(cold, y, mzp), row(cold, y, mzn)};
                chemicalPotentialRow(p, dst + (y - rlo) * nx, nx, dx2, dy2, dz2, gamma, e_AA, e_BB, e_AB);
            }
        }

        // Update concentration for plane z, rows [y0, y1)
        const double* mc = slot(z);
        const double* mp = slot(zp);
        const double* mn = slot(zn);
        for (size_t y = y0; y < y1; ++y) {
            const size_t yp = (y < ny - 1) ? y + 1 : y;
            const size_t yn = (y > 0) ? y - 1 : 0;
            const RowPtrs p{mc + (y - rlo) * nx, mc + (yp - rlo) * nx, mc + (yn - rlo) * nx,
                            mp + (y - rlo) * nx, mn + (y - rlo) * nx};
            const size_t off = z * plane + y * nx;
            cahnHilliardRow(p, cnew + off, cold + off, nx, dx2, dy2, dz2, D, dt);
        }
    }
}

// Initialize concentration field. Pages of both field buffers are first-touched
// by the thread that will later process the corresponding tile (NUMA locality).
void initializeConcentration(const Tiling& tl, const int nthreads, double* c, double* cnew,
                             const size_t nx, const size_t ny, const size_t nz) {
    const size_t vol = nx * ny * nz;
    
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (size_t tile = 0; tile < tl.ntiles; ++tile) {
        const size_t z0 = (tile / tl.nyb) * tl.zSlab;
        const size_t z1 = std::min(z0 + tl.zSlab, nz);
        const size_t y0 = (tile % tl.nyb) * tl.yBlock;
        const size_t y1 = std::min(y0 + tl.yBlock, ny);
        for (size_t z = z0; z < z1; ++z) {
            for (size_t y = y0; y < y1; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    // Generate pseudo-random value in [-1, 1]
                    const size_t linear_id = z * (nx * ny) + y * nx + x;
                    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                    c[idx] = -1.0 + 2.0 * pseudo;
                    cnew[idx] = 0.0;
                }
            }
        }
    }
}

bool validateResult(const std::vector<double>& c, [[maybe_unused]] const size_t nx, [[maybe_unused]] const size_t ny, [[maybe_unused]] const size_t nz) {
    const size_t n = c.size();
    const double* data = c.data();

    // Check for NaN or Inf
    size_t bad = 0;
    #pragma omp parallel for reduction(+:bad) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) ++bad;
    }
    if (bad > 0) {
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

// Pin OpenMP threads to CPUs if the user has not configured thread affinity
// (OMP_PROC_BIND / OMP_PLACES). A stable thread->CPU mapping keeps the
// first-touch NUMA placement of the grid consistent with the compute loops.
void pinThreadsIfUnbound() {
    if (omp_get_proc_bind() != omp_proc_bind_false || omp_get_num_places() > 0) return;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cpus;
    for (int i = 0; i < CPU_SETSIZE; ++i) {
        if (CPU_ISSET(i, &allowed)) cpus.push_back(i);
    }
    if (cpus.size() < 2) return;
    #pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[tid % cpus.size()], &set);
        sched_setaffinity(0, sizeof(set), &set);
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
    
    pinThreadsIfUnbound();
    
    // Tile decomposition used for both initialization (first touch) and compute
    const int nthreads = omp_get_max_threads();
    const Tiling tiling = makeTiling(nx, ny, nz, static_cast<size_t>(nthreads));
    
    // Allocate arrays (uninitialized; pages are first-touched in parallel
    // during initialization so they are placed close to the computing threads)
    std::unique_ptr<double[]> bufA(new double[gridSize]);
    std::unique_ptr<double[]> bufB(new double[gridSize]);
    double* cold = bufA.get();
    double* cnew = bufB.get();
    
    // Per-thread rolling windows for the chemical potential (page-padded)
    const size_t muStride = ((3 * (tiling.yBlock + 2) * nx + 511) / 512) * 512;
    std::unique_ptr<double[]> muAll(new double[muStride * static_cast<size_t>(nthreads)]);
    #pragma omp parallel num_threads(nthreads)
    {
        double* mine = muAll.get() + muStride * static_cast<size_t>(omp_get_thread_num());
        std::fill(mine, mine + muStride, 0.0);
    }
    
    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(tiling, nthreads, cold, cnew, nx, ny, nz);
    
    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel num_threads(nthreads) firstprivate(cold, cnew)
    {
        double* muBuf = muAll.get() + muStride * static_cast<size_t>(omp_get_thread_num());
        
        for (int t = 0; t < iterations; ++t) {
            // Fused chemical potential + concentration update
            #pragma omp for schedule(static)
            for (size_t tile = 0; tile < tiling.ntiles; ++tile) {
                processTile(tiling, tile, cnew, cold, muBuf, nx, ny, nz,
                            D, dt, dx, dy, dz, gamma, e_AA, e_BB, e_AB);
            }
            
            // Swap buffers (thread-private pointers, swapped identically by all threads)
            std::swap(cold, cnew);
        }
    }
    if (iterations > 0 && (iterations % 2) == 1) std::swap(cold, cnew);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);
    
    // Copy final field into a std::vector for result output / validation
    std::vector<double> result(gridSize);
    {
        double* dst = result.data();
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < gridSize; ++i) dst[i] = cold[i];
    }
    
    // Print results for external validation
    if (printResults) {
        print_results(result, "Concentration");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(result, nx, ny, nz);
        
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
