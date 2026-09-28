#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <unistd.h>

#include "../common/results_output.hpp"

using Real = double;

// The stencil is memory bound and relies on first-touch NUMA placement, so an
// unpinned run (threads migrating between sockets) loses more than an order of
// magnitude of throughput.  Unless the user configured OpenMP explicitly we
// therefore choose the team size (one thread per physical core) and pin the
// threads ourselves, spread evenly over all cores/sockets.

// Parses a sysfs CPU list such as "0,128" or "0-3,8" into `out`.
static void parseCpuList(const char* path, std::vector<int>& out) {
    out.clear();
    FILE* f = std::fopen(path, "r");
    if (f == nullptr) return;
    char buf[512];
    if (std::fgets(buf, sizeof(buf), f) != nullptr) {
        const char* p = buf;
        while (*p != '\0') {
            char* endp = nullptr;
            const long first = std::strtol(p, &endp, 10);
            if (endp == p) break;
            long last = first;
            p = endp;
            if (*p == '-') {
                last = std::strtol(p + 1, &endp, 10);
                p = endp;
            }
            for (long c = first; c <= last; ++c) out.push_back(static_cast<int>(c));
            if (*p == ',') ++p; else break;
        }
    }
    std::fclose(f);
}

// Cache capacity a single core can count on (its L2 plus its share of the
// shared last level cache); 0 if it could not be determined.
static size_t g_perCoreCache = 0;

static void configureThreads() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    // Group the usable CPUs by physical core.
    std::vector<std::vector<int>> cores;
    std::vector<char> claimed(CPU_SETSIZE, 0);
    std::vector<int> siblings;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed) || claimed[c]) continue;
        claimed[c] = 1;
        std::vector<int> core{c};
        char path[128];
        std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", c);
        parseCpuList(path, siblings);
        for (const int s : siblings) {
            if (s < 0 || s >= CPU_SETSIZE || claimed[s] || !CPU_ISSET(s, &allowed)) continue;
            claimed[s] = 1;
            core.push_back(s);
        }
        cores.push_back(std::move(core));
    }
    if (cores.empty()) return;

    // Count the distinct last level caches covering those cores.
    const long l3PerDomain = sysconf(_SC_LEVEL3_CACHE_SIZE);
    if (l3PerDomain > 0) {
        std::fill(claimed.begin(), claimed.end(), 0);
        size_t domains = 0;
        for (const auto& core : cores) {
            if (claimed[core[0]]) continue;
            ++domains;
            char path[128];
            std::snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", core[0]);
            parseCpuList(path, siblings);
            if (siblings.empty()) siblings.assign(1, core[0]);
            for (const int s : siblings) {
                if (s >= 0 && s < CPU_SETSIZE) claimed[s] = 1;
            }
        }
        long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (l2 < 0) l2 = 0;
        g_perCoreCache = static_cast<size_t>(l2) + domains * static_cast<size_t>(l3PerDomain) / cores.size();
    }

    // SMT siblings buy very little for a bandwidth bound kernel and cost cache
    // capacity, so default to one thread per physical core.
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        omp_set_num_threads(static_cast<int>(cores.size()));
    }
    const size_t numThreads = static_cast<size_t>(omp_get_max_threads());

    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr || getenv("KMP_AFFINITY") != nullptr) {
        return;  // an explicit placement policy was requested; leave it alone
    }

    // Candidate CPUs, one entry per physical core; SMT siblings are appended
    // core-by-core so that an oversubscribed team places *neighbouring* threads
    // (which work on neighbouring data) on the same core.
    std::vector<int> order;
    for (const auto& core : cores) order.push_back(core[0]);
    if (numThreads > cores.size()) {
        order.clear();
        for (const auto& core : cores) order.insert(order.end(), core.begin(), core.end());
    }
    const size_t numCpus = order.size();

#pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        // Spread a small team evenly over the machine instead of crowding the
        // first socket; wrap around once every CPU carries a thread.
        const size_t slot = (numThreads < numCpus) ? (tid * numCpus) / numThreads : tid % numCpus;
        cpu_set_t self;
        CPU_ZERO(&self);
        CPU_SET(order[slot], &self);
        sched_setaffinity(0, sizeof(self), &self);
    }
}

// The grid is partitioned over threads by rows of the x-fastest layout, i.e. by
// the flattened index r = z * ny + y.  Every loop over the grid (initialization
// included) uses the very same static partitioning, so that the pages of each
// row are first-touched by the thread that will keep operating on them.  On a
// multi-socket machine this keeps the (bandwidth bound) stencil traffic local to
// the NUMA node of the owning thread.

// Contiguous, balanced block of rows owned by the calling thread.  Used by every
// grid traversal so that ownership -- and therefore page placement -- is stable.
static inline void rowRange(const size_t numRows, size_t& begin, size_t& end) {
    const size_t numThreads = static_cast<size_t>(omp_get_num_threads());
    const size_t tid = static_cast<size_t>(omp_get_thread_num());
    const size_t quot = numRows / numThreads;
    const size_t rem = numRows % numThreads;
    begin = tid * quot + std::min(tid, rem);
    end = begin + quot + (tid < rem ? 1 : 0);
}

void initializeGrid(Real* __restrict grid, const size_t nx, const size_t ny, const size_t nz) {
    const size_t numRows = ny * nz;
#pragma omp parallel
    {
        size_t rBegin, rEnd;
        rowRange(numRows, rBegin, rEnd);
        for (size_t idx = rBegin * nx; idx < rEnd * nx; ++idx) {
            grid[idx] = (idx % 19) * 1.0;
        }
    }
}

// 7-point stencil computation.  Must be called from inside a parallel region and
// by all of its threads: the closing barrier is what separates two consecutive
// sweeps, so the iteration loop never has to fork a team.
void stencilIteration(const Real* __restrict input,
                      Real* __restrict output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const size_t blockX) {
    const size_t plane = nx * ny;
    const size_t numRows = ny * nz;

    size_t rBegin, rEnd;
    rowRange(numRows, rBegin, rEnd);

    // Degenerate grids have no interior points at all: every cell is a boundary
    // cell and is simply copied over.
    if (nx < 3 || ny < 3 || nz < 3) {
        if (rEnd > rBegin) {
            std::memcpy(output + rBegin * nx, input + rBegin * nx, (rEnd - rBegin) * nx * sizeof(Real));
        }
#pragma omp barrier
        return;
    }

    // A whole row (fixed y, z) is either entirely boundary -- and then just
    // copied -- or interior except for its two end points.  This replaces the
    // original full-grid boundary sweep, which re-scanned every cell.
    for (size_t r = rBegin; r < rEnd; ++r) {
        const size_t z = r / ny;
        const size_t y = r - z * ny;
        const Real* __restrict in = input + r * nx;
        Real* __restrict out = output + r * nx;
        if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) {
            std::memcpy(out, in, nx * sizeof(Real));
        }
    }

    // The interior sweep may be blocked in x: within one column block the values
    // of the z-neighbour planes stay in cache until they are reused as centre
    // values, instead of being evicted by a whole intervening x-y plane.
    for (size_t xb = 1; xb + 1 < nx; xb += blockX) {
        const size_t xEnd = std::min(xb + blockX, nx - 1);
        for (size_t r = rBegin; r < rEnd; ++r) {
            const size_t z = r / ny;
            const size_t y = r - z * ny;
            if (z == 0 || z == nz - 1 || y == 0 || y == ny - 1) continue;

            const Real* __restrict in = input + r * nx;
            Real* __restrict out = output + r * nx;

            if (xb == 1) {  // the two end points of an interior row are boundary cells
                out[0] = in[0];
                out[nx - 1] = in[nx - 1];
            }

            // Same summation order and division as the scalar reference, so
            // results are bit-for-bit identical.
#pragma omp simd
            for (size_t x = xb; x < xEnd; ++x) {
                out[x] = (in[x] + in[x - 1] + in[x + 1] + in[x - nx] + in[x + nx] + in[x - plane] + in[x + plane]) / 7.0;
            }
        }
    }

#pragma omp barrier
}

bool validateResult(const Real* __restrict grid, const size_t gridSize) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    int bad = 0;
#pragma omp parallel for schedule(static) reduction(|:bad)
    for (size_t i = 0; i < gridSize; ++i) {
        if (std::isnan(grid[i]) || std::isinf(grid[i])) {
            bad = 1;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < gridSize; ++i) {
        minVal = std::min(minVal, grid[i]);
        maxVal = std::max(maxVal, grid[i]);
    }

    printf("Value range: [%.6f, %.6f]\n", minVal, maxVal);

    // After averaging, values should be somewhat bounded
    if (maxVal > 1e6 || minVal < -1e6) {
        printf("Validation failed: values out of expected range\n");
        return false;
    }

    // 3. Boundary values should not change significantly
    // (they are copied, so they should be close to initial values)

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -x <num>     Grid size in X dimension (default: 128)\n");
    printf("  -y <num>     Grid size in Y dimension (default: same as X)\n");
    printf("  -z <num>     Grid size in Z dimension (default: same as X)\n");
    printf("  -i <num>     Number of iterations (default: 10)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t nx = 128;
    size_t ny = 0;  // Will be set to nx if not specified
    size_t nz = 0;  // Will be set to nx if not specified
    int iterations = 10;
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

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    configureThreads();
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    size_t gridSize = nx * ny * nz;

    // Allocate grids (double buffering).  Raw aligned allocations are used
    // instead of std::vector so that the buffers are not pre-faulted by a
    // single-threaded zero fill; the parallel first touch below distributes the
    // pages across the NUMA nodes according to the loop partitioning.
    constexpr size_t kAlign = 64;
    const size_t bytes = ((gridSize * sizeof(Real) + kAlign - 1) / kAlign) * kAlign;
    Real* grid1 = static_cast<Real*>(std::aligned_alloc(kAlign, bytes));
    Real* grid2 = static_cast<Real*>(std::aligned_alloc(kAlign, bytes));
    if (gridSize != 0 && (grid1 == nullptr || grid2 == nullptr)) {
        printf("Failed to allocate grids\n");
        return 1;
    }

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    initializeGrid(grid2, nx, ny, nz);  // first touch of the second buffer

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // A full-row sweep keeps the three active x-y planes live between the moment
    // a value is read as a z-neighbour and the moment it is reused as a centre
    // value.  While those planes fit into a core's cache that is the fastest
    // traversal.  Once they do not, the x-blocked variant -- narrow enough for
    // the three column slabs to sit in L2 -- restores that reuse.
    size_t blockX = (nx > 0) ? nx : 1;  // one block == no blocking
    if (g_perCoreCache != 0 && ny > 0 && 3 * nx * ny * sizeof(Real) > g_perCoreCache) {
        long l2 = sysconf(_SC_LEVEL2_CACHE_SIZE);
        if (l2 <= 0) l2 = 512 * 1024;
        blockX = std::max<size_t>(32, static_cast<size_t>(l2) / (3 * ny * sizeof(Real)));
    }

#pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            if (iter % 2 == 0) {
                stencilIteration(grid1, grid2, nx, ny, nz, blockX);
            } else {
                stencilIteration(grid2, grid1, nx, ny, nz, blockX);
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate performance metrics
    double cellUpdates = (double)((nx-2) * (ny-2) * (nz-2)) * iterations;
    double mcups = cellUpdates / (duration.count() / 1000.0) / 1e6;  // Million cell updates per second
    printf("Performance: %.3f MCellUpdates/s\n", mcups);

    // Print results for external validation
    const Real* finalGrid = (iterations % 2 == 0) ? grid1 : grid2;
    if (printResults) {
        const std::vector<Real> finalGridVec(finalGrid, finalGrid + gridSize);
        print_results(finalGridVec, "Grid");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(finalGrid, gridSize);

        if (valid) {
            printf("Validation: PASSED\n");
            std::free(grid1);
            std::free(grid2);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            std::free(grid1);
            std::free(grid2);
            return 1;
        }
    }

    std::free(grid1);
    std::free(grid2);
    return 0;
}
