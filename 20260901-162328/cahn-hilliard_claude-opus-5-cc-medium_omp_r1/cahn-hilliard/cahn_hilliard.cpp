#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include "../common/results_output.hpp"

// This benchmark is purely memory bound, so thread placement dominates its
// performance on multi socket machines. If the user did not configure the OpenMP
// affinity, pin the threads ourselves: one thread per physical core, spread over
// all packages, using the SMT siblings only once every core is occupied. The stencil
// already saturates the memory bandwidth with one thread per core, so the default
// team size is capped at the number of cores unless the user asks for more.
static void bindThreadsIfUnconfigured() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    // Group the usable CPUs into "rounds": round r holds the r-th SMT sibling of
    // every physical core, so round 0 already covers all cores exactly once.
    std::vector<std::vector<int>> rounds;
    std::vector<std::pair<long, long>> seenCores; // (package, core) key per physical core
    std::vector<int> siblingCount;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;

        long pkg = 0;
        long core = cpu;
        char path[128];
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        if (FILE* f = fopen(path, "r")) {
            if (fscanf(f, "%ld", &pkg) != 1) pkg = 0;
            fclose(f);
        }
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        if (FILE* f = fopen(path, "r")) {
            if (fscanf(f, "%ld", &core) != 1) core = cpu;
            fclose(f);
        }

        size_t g = 0;
        for (; g < seenCores.size(); ++g) {
            if (seenCores[g].first == pkg && seenCores[g].second == core) break;
        }
        if (g == seenCores.size()) {
            seenCores.emplace_back(pkg, core);
            siblingCount.push_back(0);
        }
        const size_t r = static_cast<size_t>(siblingCount[g]++);
        if (r >= rounds.size()) rounds.resize(r + 1);
        rounds[r].push_back(cpu);
    }
    if (rounds.empty() || rounds[0].empty()) return;

    const size_t nrounds = rounds.size();
    const size_t ncores = rounds[0].size();

    if (getenv("OMP_NUM_THREADS") == nullptr && static_cast<size_t>(omp_get_max_threads()) > ncores) {
        omp_set_num_threads(static_cast<int>(ncores));
    }

    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr || getenv("KMP_AFFINITY") != nullptr) {
        return; // the user configured the placement, leave it alone
    }

    #pragma omp parallel
    {
        const size_t n = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());

        size_t r = 0;
        size_t i = tid;
        size_t perRound = n;
        if (n > ncores) {
            perRound = (n + nrounds - 1) / nrounds;
            r = std::min(tid / perRound, nrounds - 1);
            i = tid % perRound;
        }
        const std::vector<int>& ring = rounds[r];
        const int cpu = ring[std::min(i * ring.size() / std::max<size_t>(perRound, 1), ring.size() - 1)];

        cpu_set_t one;
        CPU_ZERO(&one);
        CPU_SET(cpu, &one);
        pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
    }
}

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// True if v is an exact power of two, i.e. multiplying by 1/v gives bit-identical
// results to dividing by v. Used to turn the stencil divisions into multiplications
// without changing the produced values.
static bool isPowerOfTwo(const double v) noexcept {
    if (!(v > 0.0) || std::isinf(v)) return false;
    int e = 0;
    return std::frexp(v, &e) == 0.5;
}

// Constant simulation parameters, shared by all threads
struct Params {
    size_t nx, ny, nz, nxy;
    double dx2, dy2, dz2;
    double rdx2, rdy2, rdz2;
    double gamma, e_AA, e_BB, e_AB;
    double dtD;
};

// Applies the 7-point Laplacian to one row of nx cells and writes op(x, center, lap)
// to out. The neighbour rows in y and z are passed in already clamped; only the x
// direction needs its boundary handling, which is peeled off the vectorized body.
template <bool Recip, typename Op>
static inline void rowSweep(const Params& p,
                            const double* __restrict__ c0, const double* __restrict__ cyn,
                            const double* __restrict__ cyp, const double* __restrict__ czn,
                            const double* __restrict__ czp, double* __restrict__ out, Op op) {
    const size_t nx = p.nx;

    const auto edge = [&](const size_t x, const size_t xn, const size_t xp) {
        const double cv = c0[x];
        const double sxx = c0[xp] + c0[xn] - 2.0 * cv;
        const double syy = cyp[x] + cyn[x] - 2.0 * cv;
        const double szz = czp[x] + czn[x] - 2.0 * cv;
        const double cxx = Recip ? sxx * p.rdx2 : sxx / p.dx2;
        const double cyy = Recip ? syy * p.rdy2 : syy / p.dy2;
        const double czz = Recip ? szz * p.rdz2 : szz / p.dz2;
        out[x] = op(x, cv, cxx + cyy + czz);
    };

    if (nx == 1) {
        edge(0, 0, 0);
        return;
    }

    edge(0, 0, 1);
    const size_t xend = nx - 1;
    #pragma omp simd
    for (size_t x = 1; x < xend; ++x) {
        const double cv = c0[x];
        const double sxx = c0[x + 1] + c0[x - 1] - 2.0 * cv;
        const double syy = cyp[x] + cyn[x] - 2.0 * cv;
        const double szz = czp[x] + czn[x] - 2.0 * cv;
        const double cxx = Recip ? sxx * p.rdx2 : sxx / p.dx2;
        const double cyy = Recip ? syy * p.rdy2 : syy / p.dy2;
        const double czz = Recip ? szz * p.rdz2 : szz / p.dz2;
        out[x] = op(x, cv, cxx + cyy + czz);
    }
    edge(nx - 1, nx - 2, nx - 1);
}

// A block of the grid owned by one thread: rows [y0, y1) of planes [z0, z1).
struct Tile {
    size_t y0, y1, z0, z1;
};

// Chemical potential for the rows [yFirst, yLast) of plane z. The destination rows
// are laid out contiguously, dst + (y - yFirst) * nx.
template <bool Recip>
static inline void muPlaneRows(const Params& p, const double* __restrict__ cold, double* __restrict__ dst,
                               const size_t z, const size_t yFirst, const size_t yLast) {
    const size_t nx = p.nx, ny = p.ny, nz = p.nz, nxy = p.nxy;
    const size_t zn = (z > 0) ? z - 1 : 0;
    const size_t zp = (z + 1 < nz) ? z + 1 : z;
    const double gamma = p.gamma, e_AA = p.e_AA, e_BB = p.e_BB, e_AB = p.e_AB;

    for (size_t y = yFirst; y < yLast; ++y) {
        const size_t yn = (y > 0) ? y - 1 : 0;
        const size_t yp = (y + 1 < ny) ? y + 1 : y;
        rowSweep<Recip>(p, cold + z * nxy + y * nx, cold + z * nxy + yn * nx,
                        cold + z * nxy + yp * nx, cold + zn * nxy + y * nx,
                        cold + zp * nxy + y * nx, dst + (y - yFirst) * nx,
                        [=](const size_t, const double cv, const double lap) {
                            return 4.5 * ((cv + 1.0) * e_AA + (cv - 1.0) * e_BB - 2.0 * cv * e_AB)
                                   + 3.0 * cv + cv * cv * cv - gamma * lap;
                        });
    }
}

// Concentration update for the rows [y0, y1) of plane z. m0/mzn/mzp point at the mu
// planes z, z-1 and z+1; their row for y sits at (y - muFirst) * nx.
template <bool Recip>
static inline void updatePlaneRows(const Params& p, const double* __restrict__ cold, double* __restrict__ cnew,
                                   const double* __restrict__ m0, const double* __restrict__ mzn,
                                   const double* __restrict__ mzp, const size_t muFirst,
                                   const size_t z, const size_t y0, const size_t y1) {
    const size_t nx = p.nx, ny = p.ny, nxy = p.nxy;
    const double dtD = p.dtD;

    for (size_t y = y0; y < y1; ++y) {
        const size_t yn = (y > 0) ? y - 1 : 0;
        const size_t yp = (y + 1 < ny) ? y + 1 : y;
        const double* __restrict__ coldRow = cold + z * nxy + y * nx;
        rowSweep<Recip>(p, m0 + (y - muFirst) * nx, m0 + (yn - muFirst) * nx, m0 + (yp - muFirst) * nx,
                        mzn + (y - muFirst) * nx, mzp + (y - muFirst) * nx, cnew + z * nxy + y * nx,
                        [=](const size_t x, const double, const double lap) {
                            return coldRow[x] + dtD * lap;
                        });
    }
}

// Fused Cahn-Hilliard time step over one tile.
//
// Instead of materializing the chemical potential for the whole grid (which costs a
// full write plus a full read of a grid sized array per time step), mu is produced
// plane by plane into a small rolling buffer of three planes that stays in cache: the
// update of plane z only needs mu on planes z-1, z and z+1. Per cell this leaves one
// streaming read of cold and one streaming write of cnew, which is what makes the
// difference once the grid no longer fits into the last level cache.
//
// mubuf holds the tile rows [ylo, yhi) of three consecutive planes, plane z living in
// slot z % 3. The mu rows are computed one row wider than the tile on each side in y
// (and one plane wider in z), because the update needs mu at y+-1 and z+-1.
template <bool Recip>
static void fusedTileStep(const Params& p, const Tile& tile,
                          const double* __restrict__ cold, double* __restrict__ cnew,
                          double* __restrict__ mubuf) {
    const size_t nx = p.nx, ny = p.ny, nz = p.nz;
    const size_t ylo = (tile.y0 > 0) ? tile.y0 - 1 : 0;
    const size_t yhi = (tile.y1 < ny) ? tile.y1 + 1 : ny;
    const size_t planeStride = nx * (yhi - ylo);

    // Walk the planes of the tile, keeping the rolling mu buffer one plane ahead.
    size_t next = (tile.z0 > 0) ? tile.z0 - 1 : 0;
    for (size_t z = tile.z0; z < tile.z1; ++z) {
        const size_t need = (z + 1 < nz) ? z + 1 : nz - 1;
        for (; next <= need; ++next) {
            muPlaneRows<Recip>(p, cold, mubuf + (next % 3) * planeStride, next, ylo, yhi);
        }
        const size_t zn = (z > 0) ? z - 1 : 0;
        const size_t zp = (z + 1 < nz) ? z + 1 : z;
        updatePlaneRows<Recip>(p, cold, cnew, mubuf + (z % 3) * planeStride,
                               mubuf + (zn % 3) * planeStride, mubuf + (zp % 3) * planeStride,
                               ylo, z, tile.y0, tile.y1);
    }
}

// Chemical potential half step over one tile, into the grid sized mu array. Used for
// grids that fit into the caches, where the extra array costs little and the fused
// variant would pay for recomputing the tile halos.
template <bool Recip>
static void tileMuStep(const Params& p, const Tile& tile, const double* __restrict__ cold,
                       double* __restrict__ mu) {
    for (size_t z = tile.z0; z < tile.z1; ++z) {
        muPlaneRows<Recip>(p, cold, mu + z * p.nxy + tile.y0 * p.nx, z, tile.y0, tile.y1);
    }
}

// Concentration half step over one tile, from the grid sized mu array
template <bool Recip>
static void tileUpdateStep(const Params& p, const Tile& tile, const double* __restrict__ cold,
                           double* __restrict__ cnew, const double* __restrict__ mu) {
    for (size_t z = tile.z0; z < tile.z1; ++z) {
        const size_t zn = (z > 0) ? z - 1 : 0;
        const size_t zp = (z + 1 < p.nz) ? z + 1 : z;
        updatePlaneRows<Recip>(p, cold, cnew, mu + z * p.nxy, mu + zn * p.nxy, mu + zp * p.nxy,
                               0, z, tile.y0, tile.y1);
    }
}

// Splits the grid into one tile per thread. Threads are handed contiguous blocks of
// y rows; if there are more threads than y blocks, the threads sharing a y block
// split its planes among themselves. For the fused kernel the y blocks are kept small
// enough for the three rolling mu planes to stay cache resident, otherwise they are
// as large as possible so that each thread works on contiguous memory.
static std::vector<Tile> buildTiles(const size_t nx, const size_t ny, const size_t nz,
                                    const size_t nthreads, const bool fused) {
    constexpr size_t planeBudget = 8192; // elements per rolling mu plane
    size_t yb = ny;
    if (fused) {
        yb = std::min(ny, std::max<size_t>(1, planeBudget / std::max<size_t>(nx, 1)));
    }
    size_t nyb = (ny + yb - 1) / yb;

    // shrink the blocks if that is what it takes to keep every thread busy
    const size_t neededBlocks = (nthreads + nz - 1) / nz;
    while (nyb < neededBlocks && yb > 1) {
        yb = (yb + 1) / 2;
        nyb = (ny + yb - 1) / yb;
    }

    std::vector<Tile> tiles(nthreads, Tile{0, 0, 0, 0});
    if (nthreads <= nyb) {
        // one thread covers a contiguous range of y blocks over all planes
        for (size_t t = 0; t < nthreads; ++t) {
            const size_t b0 = t * nyb / nthreads;
            const size_t b1 = (t + 1) * nyb / nthreads;
            tiles[t] = Tile{std::min(b0 * yb, ny), std::min(b1 * yb, ny), 0, nz};
        }
    } else {
        // the threads sharing a y block split its planes among themselves
        for (size_t b = 0; b < nyb; ++b) {
            const size_t first = b * nthreads / nyb;      // first thread of the group
            const size_t last = (b + 1) * nthreads / nyb; // one past the last
            const size_t group = last - first;
            for (size_t t = first; t < last; ++t) {
                const size_t k = t - first;
                tiles[t] = Tile{std::min(b * yb, ny), std::min((b + 1) * yb, ny),
                                k * nz / group, (k + 1) * nz / group};
            }
        }
    }
    return tiles;
}

// Aggregate size of the last level caches usable by this process, in bytes
static size_t aggregateLastLevelCache(const size_t nthreads) {
    size_t total = 0;
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) {
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
            if (!CPU_ISSET(cpu, &allowed)) continue;
            for (int index = 3; index >= 1; --index) {
                char path[160];
                snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/size", cpu, index);
                FILE* f = fopen(path, "r");
                if (f == nullptr) continue;
                long kb = 0;
                const bool ok = (fscanf(f, "%ldK", &kb) == 1);
                fclose(f);
                if (!ok) continue;

                // the cache is shared, count only this CPU's share of it
                snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index%d/shared_cpu_list", cpu, index);
                size_t sharers = 1;
                if (FILE* g = fopen(path, "r")) {
                    char list[512] = {0};
                    if (fgets(list, sizeof(list), g) != nullptr) {
                        sharers = 0;
                        for (const char* q = list; *q != '\0';) {
                            long a = 0, b = 0;
                            int consumed = 0;
                            if (sscanf(q, "%ld-%ld%n", &a, &b, &consumed) == 2) {
                                sharers += static_cast<size_t>(b - a + 1);
                            } else if (sscanf(q, "%ld%n", &a, &consumed) == 1) {
                                sharers += 1;
                            } else {
                                break;
                            }
                            q += consumed;
                            if (*q == ',') ++q;
                        }
                        if (sharers == 0) sharers = 1;
                    }
                    fclose(g);
                }
                total += static_cast<size_t>(kb) * 1024 / sharers;
                break;
            }
        }
    }
    if (total == 0) total = nthreads * (2u << 20); // no topology info: assume 2 MiB per thread
    return total;
}

// Initialize concentration field (this is also the NUMA first touch of the buffer)
void initializeConcentration(double* __restrict__ c, const size_t nx, const size_t ny, const size_t nz,
                             const std::vector<Tile>& tiles) {
    const size_t vol = nx * ny * nz;

    #pragma omp parallel num_threads(static_cast<int>(tiles.size()))
    {
        const Tile& tile = tiles[omp_get_thread_num()];
        for (size_t z = tile.z0; z < tile.z1; ++z) {
            for (size_t y = tile.y0; y < tile.y1; ++y) {
                for (size_t x = 0; x < nx; ++x) {
                    const size_t idx = idx3(x, y, z, nx, ny);
                    // Generate pseudo-random value in [-1, 1]
                    const size_t linear_id = z * (nx * ny) + y * nx + x;
                    const double pseudo = ((((linear_id + 1) * 1299709) % vol) / static_cast<double>(vol));
                    c[idx] = -1.0 + 2.0 * pseudo;
                }
            }
        }
    }
}

// Touch a buffer with the tile distribution used by the kernel, so that its pages end
// up on the NUMA node of the thread that will work on them.
static void firstTouch(double* __restrict__ p, const size_t nx, const size_t ny, const std::vector<Tile>& tiles) {
    #pragma omp parallel num_threads(static_cast<int>(tiles.size()))
    {
        const Tile& tile = tiles[omp_get_thread_num()];
        for (size_t z = tile.z0; z < tile.z1; ++z) {
            for (size_t y = tile.y0; y < tile.y1; ++y) {
                double* __restrict__ row = p + z * nx * ny + y * nx;
                for (size_t x = 0; x < nx; ++x) {
                    row[x] = 0.0;
                }
            }
        }
    }
}

bool validateResult(const double* __restrict__ c, const size_t gridSize) {
    // Check for NaN or Inf
    int bad = 0;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for (size_t i = 0; i < gridSize; ++i) {
        bad |= (std::isnan(c[i]) || std::isinf(c[i])) ? 1 : 0;
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
    for (size_t i = 0; i < gridSize; ++i) {
        minVal = std::min(minVal, c[i]);
        maxVal = std::max(maxVal, c[i]);
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

    bindThreadsIfUnconfigured();
    omp_set_dynamic(0); // the tiles are precomputed for a fixed team size

    printf("Cahn-Hilliard Phase Separation Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Time steps: %d\n", iterations);
    printf("Threads: %d\n", omp_get_max_threads());
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
    if (gridSize == 0) {
        printf("Empty grid\n");
        return 1;
    }

    Params params;
    params.nx = nx;
    params.ny = ny;
    params.nz = nz;
    params.nxy = nx * ny;
    params.dx2 = dx * dx;
    params.dy2 = dy * dy;
    params.dz2 = dz * dz;
    params.rdx2 = 1.0 / params.dx2;
    params.rdy2 = 1.0 / params.dy2;
    params.rdz2 = 1.0 / params.dz2;
    params.gamma = gamma;
    params.e_AA = e_AA;
    params.e_BB = e_BB;
    params.e_AB = e_AB;
    params.dtD = dt * D;
    const bool recip = isPowerOfTwo(params.dx2) && isPowerOfTwo(params.dy2) && isPowerOfTwo(params.dz2);

    const size_t nthreads = static_cast<size_t>(omp_get_max_threads());

    // Two grid sized arrays (cold and cnew) are unavoidable. If a third one for mu
    // would still fit into the caches, keeping it around is cheaper than the halo
    // recomputation of the fused kernel; otherwise the fused kernel wins, because it
    // removes mu from the memory traffic completely.
    const size_t footprint = 3 * gridSize * sizeof(double);
    const bool fused = footprint > aggregateLastLevelCache(nthreads);
    const std::vector<Tile> tiles = buildTiles(nx, ny, nz, nthreads, fused);

    // Allocate arrays. Raw allocations are used so that the pages are first touched by
    // the thread that later works on them (NUMA locality).
    const size_t bytes = ((gridSize * sizeof(double)) + 63) & ~size_t(63);
    double* cold = static_cast<double*>(std::aligned_alloc(64, bytes));
    double* cnew = static_cast<double*>(std::aligned_alloc(64, bytes));
    double* mu = fused ? nullptr : static_cast<double*>(std::aligned_alloc(64, bytes));
    if (cold == nullptr || cnew == nullptr || (!fused && mu == nullptr)) {
        printf("Allocation failed\n");
        return 1;
    }
    firstTouch(cnew, nx, ny, tiles);
    if (!fused) firstTouch(mu, nx, ny, tiles);

    // Initialize concentration field
    printf("Initializing concentration field...\n");
    initializeConcentration(cold, nx, ny, nz, tiles);

    // Run simulation
    printf("Running Cahn-Hilliard simulation...\n");
    std::chrono::high_resolution_clock::time_point start, end;

    // A single parallel region covers the whole time loop, so the team is created once
    // and the per thread mu buffers are allocated (and first touched) only once.
    #pragma omp parallel num_threads(nthreads)
    {
        const Tile& tile = tiles[omp_get_thread_num()];
        const bool empty = (tile.z0 >= tile.z1) || (tile.y0 >= tile.y1);
        const size_t ylo = (tile.y0 > 0) ? tile.y0 - 1 : 0;
        const size_t yhi = (tile.y1 < ny) ? tile.y1 + 1 : ny;
        std::vector<double> mubuf((fused && !empty) ? 3 * nx * (yhi - ylo) : 0, 0.0);
        double* buf = mubuf.data();

        #pragma omp barrier
        #pragma omp master
        start = std::chrono::high_resolution_clock::now();
        #pragma omp barrier

        for (int t = 0; t < iterations; ++t) {
            if (fused) {
                if (!empty) {
                    if (recip) {
                        fusedTileStep<true>(params, tile, cold, cnew, buf);
                    } else {
                        fusedTileStep<false>(params, tile, cold, cnew, buf);
                    }
                }
            } else {
                if (!empty) {
                    if (recip) {
                        tileMuStep<true>(params, tile, cold, mu);
                    } else {
                        tileMuStep<false>(params, tile, cold, mu);
                    }
                }
                // mu of the neighbouring tiles is needed by the update below
                #pragma omp barrier
                if (!empty) {
                    if (recip) {
                        tileUpdateStep<true>(params, tile, cold, cnew, mu);
                    } else {
                        tileUpdateStep<false>(params, tile, cold, cnew, mu);
                    }
                }
            }

            // Swap the buffers once every thread is done with the current step
            #pragma omp barrier
            #pragma omp single
            std::swap(cold, cnew);
        }

        #pragma omp barrier
        #pragma omp master
        end = std::chrono::high_resolution_clock::now();
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance
    double cellUpdates = (double)gridSize * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    if (printResults) {
        print_results(std::vector<double>(cold, cold + gridSize), "Concentration");
    }

    // Validation
    int rc = 0;
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(cold, gridSize);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    std::free(cold);
    std::free(cnew);
    std::free(mu);
    return rc;
}
