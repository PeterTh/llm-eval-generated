#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <memory>
#include <vector>

#include <omp.h>

#if defined(__AVX__)
#include <immintrin.h>
#endif

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

#include "../common/results_output.hpp"

using Real = double;

// ---------------------------------------------------------------------------
// Thread placement
//
// This kernel is memory-bandwidth bound, so its performance depends entirely on
// threads staying on the core whose NUMA node holds the data they first touched.
// If the user did not request a placement policy via OMP_PROC_BIND, pin the
// threads ourselves: one thread per physical core, stretched evenly over the
// cores we are allowed to run on (so that fewer threads than cores still use
// all NUMA nodes) and only falling back to SMT siblings when oversubscribed.
// ---------------------------------------------------------------------------
namespace placement {

// CPUs we may run on, ordered physical-core-first ("levels" of SMT siblings).
// coreCount is the number of entries belonging to the first level.
struct CpuOrder {
    std::vector<int> cpus;
    size_t coreCount = 0;
};

#ifdef __linux__
static int readTopologyId(const int cpu, const char* file) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, file);
    FILE* f = fopen(path, "r");
    if (f == nullptr) return -1;
    int value = -1;
    if (fscanf(f, "%d", &value) != 1) value = -1;
    fclose(f);
    return value;
}

static CpuOrder buildCpuOrder() {
    CpuOrder order;
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return order;

    std::vector<int> allowed;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &mask)) allowed.push_back(cpu);
    }
    if (allowed.empty()) return order;

    // Group the allowed CPUs by physical core, keeping package/core order.
    struct Core {
        long long key;
        std::vector<int> cpus;
    };
    std::vector<Core> cores;
    bool haveTopology = true;
    for (const int cpu : allowed) {
        const int pkg = readTopologyId(cpu, "physical_package_id");
        const int core = readTopologyId(cpu, "core_id");
        if (pkg < 0 || core < 0) {
            haveTopology = false;
            break;
        }
        const long long key = (long long)pkg * 1000000 + core;
        auto it = std::find_if(cores.begin(), cores.end(), [key](const Core& c) { return c.key == key; });
        if (it == cores.end()) {
            cores.push_back(Core{key, {cpu}});
        } else {
            it->cpus.push_back(cpu);
        }
    }

    if (!haveTopology) {
        // No topology information: use the plain CPU list.
        order.cpus = allowed;
        order.coreCount = allowed.size();
        return order;
    }

    std::sort(cores.begin(), cores.end(), [](const Core& a, const Core& b) { return a.key < b.key; });
    size_t maxSiblings = 0;
    for (const Core& c : cores) maxSiblings = std::max(maxSiblings, c.cpus.size());

    for (size_t level = 0; level < maxSiblings; ++level) {
        for (const Core& c : cores) {
            if (level < c.cpus.size()) order.cpus.push_back(c.cpus[level]);
        }
        if (level == 0) order.coreCount = order.cpus.size();
    }
    return order;
}

static void pinThread(const CpuOrder& order, const int threadId, const int numThreads) {
    if (order.cpus.empty() || numThreads <= 0) return;
    // Stretch the threads evenly over the physical cores; use the full list
    // (including SMT siblings) only when there are more threads than cores.
    const size_t pool = ((size_t)numThreads <= order.coreCount && order.coreCount > 0) ? order.coreCount : order.cpus.size();
    const size_t slot = ((size_t)threadId * pool) / (size_t)numThreads;
    const int cpu = order.cpus[std::min(slot, pool - 1)];

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(cpu, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}
#endif  // __linux__

// Choose the thread count and pin the threads. Both steps are skipped if the
// user expressed a preference through the environment.
static void configureThreads() {
#ifdef __linux__
    const CpuOrder order = buildCpuOrder();

    if (getenv("OMP_NUM_THREADS") == nullptr && order.coreCount > 0) {
        // Default to one thread per physical core: SMT siblings share the
        // core's load/store resources and do not add memory bandwidth.
        omp_set_num_threads((int)std::min(order.coreCount, (size_t)omp_get_max_threads()));
    }

    if (omp_get_proc_bind() != omp_proc_bind_false) return;

#pragma omp parallel
    { pinThread(order, omp_get_thread_num(), omp_get_num_threads()); }
#endif
}

// Total last level cache of the machine, or 0 if it cannot be determined.
// Used only to decide between cached and streaming stores.
static size_t aggregateLastLevelCache() {
#ifdef __linux__
    const long lineSharedSize = sysconf(_SC_LEVEL3_CACHE_SIZE);
    const long onlineCpus = sysconf(_SC_NPROCESSORS_ONLN);
    if (lineSharedSize <= 0 || onlineCpus <= 0) return 0;

    // Number of CPUs sharing one last level cache, e.g. "0-7,128-135".
    FILE* f = fopen("/sys/devices/system/cpu/cpu0/cache/index3/shared_cpu_list", "r");
    if (f == nullptr) return 0;
    char list[512] = {0};
    const bool ok = fgets(list, sizeof(list), f) != nullptr;
    fclose(f);
    if (!ok) return 0;

    long sharing = 0;
    const char* p = list;
    while (*p != '\0') {
        char* endLow = nullptr;
        const long low = strtol(p, &endLow, 10);
        if (endLow == p) break;
        long high = low;
        p = endLow;
        if (*p == '-') {
            char* endHigh = nullptr;
            high = strtol(p + 1, &endHigh, 10);
            p = endHigh;
        }
        sharing += high - low + 1;
        while (*p == ',' || *p == ' ') ++p;
    }
    if (sharing <= 0) return 0;

    const long domains = std::max(1L, onlineCpus / sharing);
    return (size_t)lineSharedSize * (size_t)domains;
#else
    return 0;
#endif
}

}  // namespace placement

// 3D index calculation
inline constexpr size_t idx3(const size_t x, const size_t y, const size_t z, const size_t nx, const size_t ny) noexcept {
    return z * (nx * ny) + y * nx + x;
}

// Aligned allocation of an untouched buffer; pages are faulted in later by the
// threads that own them (first touch), which keeps the data NUMA-local.
static Real* allocateGrid(const size_t n) {
    constexpr size_t alignment = 64;
    const size_t bytes = ((n * sizeof(Real) + alignment - 1) / alignment) * alignment;
    void* p = std::aligned_alloc(alignment, bytes);
    if (p == nullptr) {
        printf("Allocation of %zu bytes failed\n", bytes);
        exit(1);
    }
    return static_cast<Real*>(p);
}

// The distribution used for first touch must match the one used by the stencil
// loops, so both use a static schedule over the collapsed (z, y) plane rows.
void initializeGrid(Real* __restrict grid, const size_t nx, const size_t ny, const size_t nz) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowBase = idx3(0, y, z, nx, ny);
            for (size_t x = 0; x < nx; ++x) {
                const size_t idx = rowBase + x;
                grid[idx] = (idx % 19) * 1.0;
            }
        }
    }
}

// Touch the second buffer with the same distribution as the first one so that
// its pages are placed on the NUMA node of the thread that will write them.
void touchGrid(Real* __restrict grid, const size_t nx, const size_t ny, const size_t nz) {
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        for (size_t y = 0; y < ny; ++y) {
            const size_t rowBase = idx3(0, y, z, nx, ny);
            for (size_t x = 0; x < nx; ++x) {
                grid[rowBase + x] = 0.0;
            }
        }
    }
}

// Update one interior row [1, nx-1) with ordinary (cached) stores.
static inline void stencilRow(const Real* __restrict in, Real* __restrict out,
                              const size_t nx, const size_t slice, const size_t start) {
#pragma omp simd
    for (size_t x = start; x < nx - 1; ++x) {
        const Real center = in[x];
        const Real left = in[x - 1];
        const Real right = in[x + 1];
        const Real front = in[x - nx];
        const Real back = in[x + nx];
        const Real bottom = in[x - slice];
        const Real top = in[x + slice];

        // Simple averaging stencil
        out[x] = (center + left + right + front + back + bottom + top) / 7.0;
    }
}

#if defined(__AVX__)
// Same computation (same operand order, hence bit-identical results) but with
// streaming stores, which avoid pulling the output cache lines in for writing.
// Only worthwhile when the grids do not fit in the last level caches.
static inline void stencilRowStreaming(const Real* __restrict in, Real* __restrict out,
                                       const size_t nx, const size_t slice) {
    constexpr size_t W = 4;  // doubles per 256-bit vector
    size_t x = 1;
    // Peel until the store address is vector aligned.
    while (x < nx - 1 && (reinterpret_cast<uintptr_t>(out + x) % (W * sizeof(Real))) != 0) {
        out[x] = (in[x] + in[x - 1] + in[x + 1] + in[x - nx] + in[x + nx] + in[x - slice] + in[x + slice]) / 7.0;
        ++x;
    }
    const __m256d seven = _mm256_set1_pd(7.0);
    for (; x + W <= nx - 1; x += W) {
        __m256d s = _mm256_loadu_pd(in + x);
        s = _mm256_add_pd(s, _mm256_loadu_pd(in + x - 1));
        s = _mm256_add_pd(s, _mm256_loadu_pd(in + x + 1));
        s = _mm256_add_pd(s, _mm256_loadu_pd(in + x - nx));
        s = _mm256_add_pd(s, _mm256_loadu_pd(in + x + nx));
        s = _mm256_add_pd(s, _mm256_loadu_pd(in + x - slice));
        s = _mm256_add_pd(s, _mm256_loadu_pd(in + x + slice));
        _mm256_stream_pd(out + x, _mm256_div_pd(s, seven));
    }
    // Remainder
    stencilRow(in, out, nx, slice, x);
}
#endif

// 7-point stencil computation.
// Contains orphaned OpenMP worksharing directives: it is meant to be called
// from within a parallel region so that the region can be reused across
// iterations (no repeated fork/join).
void stencilIteration(const Real* __restrict input,
                      Real* __restrict output,
                      const size_t nx, const size_t ny, const size_t nz,
                      const bool streamingStores) {
    const size_t slice = nx * ny;

    // Process interior points (not on boundaries)
    if (nx > 2 && ny > 2 && nz > 2) {
#pragma omp for collapse(2) schedule(static) nowait
        for (size_t z = 1; z < nz - 1; ++z) {
            for (size_t y = 1; y < ny - 1; ++y) {
                const size_t rowBase = idx3(0, y, z, nx, ny);
                const Real* __restrict in = input + rowBase;
                Real* __restrict out = output + rowBase;
#if defined(__AVX__)
                if (streamingStores) {
                    stencilRowStreaming(in, out, nx, slice);
                    continue;
                }
#else
                (void)streamingStores;
#endif
                stencilRow(in, out, nx, slice, 1);
            }
        }
#if defined(__AVX__)
        // Streaming stores are weakly ordered; make them visible before the
        // barrier at the end of the boundary loop below.
        if (streamingStores) _mm_sfence();
#endif
    }

    // Copy boundary values (disjoint from the interior points written above,
    // hence the nowait on the loop before this one). The implicit barrier at
    // the end of this loop separates the iterations.
#pragma omp for schedule(static)
    for (size_t z = 0; z < nz; ++z) {
        if (z == 0 || z == nz - 1) {
            // Full plane
            const size_t base = z * slice;
            for (size_t i = 0; i < slice; ++i) {
                output[base + i] = input[base + i];
            }
        } else {
            // y boundary rows
            for (size_t y = 0; y < ny; y += (ny > 1 ? ny - 1 : 1)) {
                const size_t base = idx3(0, y, z, nx, ny);
                for (size_t x = 0; x < nx; ++x) {
                    output[base + x] = input[base + x];
                }
            }
            // x boundary columns of the interior rows
            for (size_t y = 1; y + 1 < ny; ++y) {
                const size_t base = idx3(0, y, z, nx, ny);
                output[base] = input[base];
                output[base + nx - 1] = input[base + nx - 1];
            }
        }
    }
}

bool validateResult(const Real* __restrict grid, const size_t gridSize) {
    // Simple sanity checks

    // 1. No NaN or Inf values
    bool bad = false;
#pragma omp parallel for schedule(static) reduction(|| : bad)
    for (size_t i = 0; i < gridSize; ++i) {
        const Real val = grid[i];
        if (std::isnan(val) || std::isinf(val)) {
            bad = true;
        }
    }
    if (bad) {
        printf("Validation failed: found NaN or Inf value\n");
        return false;
    }

    // 2. Values should be reasonable (bounded)
    Real minVal = grid[0];
    Real maxVal = grid[0];
#pragma omp parallel for schedule(static) reduction(min : minVal) reduction(max : maxVal)
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

    // Must happen before the grids are first touched.
    placement::configureThreads();

    printf("3D Stencil Benchmark\n");
    printf("Grid size: %zu x %zu x %zu\n", nx, ny, nz);
    printf("Iterations: %d\n", iterations);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    size_t gridSize = nx * ny * nz;

    // Allocate grids (double buffering)
    Real* grid1 = allocateGrid(gridSize);
    Real* grid2 = allocateGrid(gridSize);

    // Initialize
    printf("Initializing grid...\n");
    initializeGrid(grid1, nx, ny, nz);
    touchGrid(grid2, nx, ny, nz);

    // Streaming stores pay off once the two grids are far too large for the
    // caches; below that threshold the write allocate they avoid is served by
    // cache anyway and bypassing it would only cost bandwidth.
    const size_t lastLevelCache = placement::aggregateLastLevelCache();
    const bool streamingStores = lastLevelCache > 0 && 2 * gridSize * sizeof(Real) > 3 * lastLevelCache;

    // Run stencil iterations
    printf("Running stencil computation...\n");
    auto start = std::chrono::high_resolution_clock::now();

    // Single parallel region for all iterations: the worksharing loops inside
    // stencilIteration bind to it, so threads are forked only once.
#pragma omp parallel
    {
        for (int iter = 0; iter < iterations; ++iter) {
            const Real* in = (iter % 2 == 0) ? grid1 : grid2;
            Real* out = (iter % 2 == 0) ? grid2 : grid1;
            stencilIteration(in, out, nx, ny, nz, streamingStores);
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
        const std::vector<Real> finalVec(finalGrid, finalGrid + gridSize);
        print_results(finalVec, "Grid");
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
