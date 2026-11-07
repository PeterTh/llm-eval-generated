#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <utility>
#include <vector>

#include <omp.h>
#if defined(__linux__)
#include <sched.h>
#endif

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// ---------------------------------------------------------------------------
// Low-level helpers for the inter-thread synchronization of floydWarshall().
// ---------------------------------------------------------------------------

// Cache-line padded atomic counter so that per-thread progress counters never
// share a cache line (avoids false sharing between writers).
struct alignas(128) PaddedCounter {
    std::atomic<size_t> value{0};
};

inline void cpuRelax() noexcept {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#elif defined(__aarch64__)
    asm volatile("yield" ::: "memory");
#endif
}

// Spin until pred() becomes true. Spins with pause instructions first and
// falls back to yielding the CPU so the program still makes progress when
// there are more OpenMP threads than hardware threads.
template <typename Pred>
inline void spinUntil(Pred&& pred) noexcept {
    unsigned spins = 0;
    while (!pred()) {
        if (++spins < 4096) {
            cpuRelax();
        } else {
            std::this_thread::yield();
        }
    }
}

// Relax one row i of the distance matrix with pivot row k:
//   dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]), path updated on
//   strict improvement. Written branch-free so the compiler vectorizes it.
static inline void relaxRow(unsigned int* rowI, unsigned int* pathI,
                            const unsigned int* rowK, const unsigned int distIK,
                            const unsigned int kk, const size_t n) noexcept {
#pragma omp simd
    for (size_t j = 0; j < n; ++j) {
        const unsigned int distIJ = rowI[j];
        const unsigned int newDist = distIK + rowK[j];
        const bool better = newDist < distIJ;
        rowI[j] = better ? newDist : distIJ;
        pathI[j] = better ? kk : pathI[j];
    }
}

// ---------------------------------------------------------------------------
// Thread placement.
//
// Pinning the OpenMP threads to distinct cores is essential for the point-to-
// point synchronization below (migrating threads lose their cache-resident
// rows and the spin-waits become erratic). The OpenMP runtime only pins if the
// user asked for it via the environment, so if no affinity-related variable is
// set we pin the threads ourselves: one thread per physical core first (using
// the Linux sysfs topology when available), SMT siblings afterwards. Only CPUs
// in the process' allowed set (e.g. from taskset/cgroups) are used.
// ---------------------------------------------------------------------------

static bool userControlsAffinity() {
    static const char* const vars[] = {"OMP_PROC_BIND", "OMP_PLACES", "GOMP_CPU_AFFINITY",
                                       "KMP_AFFINITY", "OMP_DISPLAY_AFFINITY"};
    for (const char* v : vars) {
        if (std::getenv(v) != nullptr) return true;
    }
    return false;
}

struct CpuPlacement {
    std::vector<int> cpus;      // CPUs in pinning order (empty: do not pin)
    size_t numPhysicalCores = 0; // distinct physical cores in cpus (0: unknown)
};

// Builds the ordered list of CPUs threads are pinned to. Returns an empty
// list if the information could not be obtained (then no pinning is done).
static CpuPlacement buildCpuPlacement() {
    CpuPlacement placement;
    std::vector<int>& cpus = placement.cpus;
#if defined(__linux__)
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return placement;

    std::vector<int> allowedCpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) allowedCpus.push_back(c);
    }
    if (allowedCpus.empty()) return placement;

    // Group allowed CPUs by physical core (package id, core id) using sysfs.
    std::vector<std::pair<long, long>> coreKey(allowedCpus.size(), {-1, -1});
    bool topologyOk = true;
    for (size_t i = 0; i < allowedCpus.size() && topologyOk; ++i) {
        char fname[128];
        long ids[2] = {-1, -1};
        const char* names[2] = {"physical_package_id", "core_id"};
        for (int f = 0; f < 2; ++f) {
            snprintf(fname, sizeof(fname), "/sys/devices/system/cpu/cpu%d/topology/%s",
                     allowedCpus[i], names[f]);
            FILE* fp = fopen(fname, "r");
            if (fp == nullptr || fscanf(fp, "%ld", &ids[f]) != 1) topologyOk = false;
            if (fp != nullptr) fclose(fp);
        }
        coreKey[i] = {ids[0], ids[1]};
    }

    if (!topologyOk) {
        cpus = allowedCpus;   // no topology info: plain CPU order
        return placement;
    }

    // Round-robin over physical cores: first the lowest-numbered CPU of each
    // core, then the second sibling of each core, and so on.
    std::vector<bool> used(allowedCpus.size(), false);
    size_t remaining = allowedCpus.size();
    bool firstPass = true;
    while (remaining > 0) {
        std::vector<std::pair<long, long>> seen;
        for (size_t i = 0; i < allowedCpus.size(); ++i) {
            if (used[i]) continue;
            if (std::find(seen.begin(), seen.end(), coreKey[i]) != seen.end()) continue;
            seen.push_back(coreKey[i]);
            used[i] = true;
            cpus.push_back(allowedCpus[i]);
            --remaining;
        }
        if (firstPass) {
            placement.numPhysicalCores = seen.size();
            firstPass = false;
        }
    }
#endif
    return placement;
}

// The placement is computed once, from the process' original CPU mask, before
// any thread has been pinned (pinning narrows the mask of the calling thread).
static const CpuPlacement& cpuPlacement() {
    static const CpuPlacement placement =
        userControlsAffinity() ? CpuPlacement{} : buildCpuPlacement();
    return placement;
}

static void pinCurrentThread(const int cpu) {
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    (void)sched_setaffinity(0, sizeof(set), &set);
#else
    (void)cpu;
#endif
}

// Number of threads used for a problem of the given size. Every thread needs
// a reasonable amount of rows per k-iteration for the per-pivot costs (pivot
// row broadcast, flag polling) to be amortized; for small graphs fewer threads
// are faster. When we pin the threads ourselves, we also prefer one thread per
// physical core unless the user explicitly requested a thread count.
static int chooseThreadCount(const size_t numNodes) {
    constexpr size_t MIN_ROWS_PER_THREAD = 16;
    size_t limit = static_cast<size_t>(omp_get_max_threads());
    const size_t numPhysicalCores = cpuPlacement().numPhysicalCores;
    if (numPhysicalCores > 0 && std::getenv("OMP_NUM_THREADS") == nullptr &&
        numPhysicalCores < limit) {
        limit = numPhysicalCores;
    }
    const size_t bySize = std::max<size_t>(1, numNodes / MIN_ROWS_PER_THREAD);
    if (bySize < limit) limit = bySize;
    return static_cast<int>(limit);
}

// Pins the calling OpenMP thread according to the placement (no-op if the
// user controls affinity through the environment).
static void pinThisOpenMPThread() {
    const std::vector<int>& cpus = cpuPlacement().cpus;
    if (!cpus.empty()) {
        pinCurrentThread(cpus[static_cast<size_t>(omp_get_thread_num()) % cpus.size()]);
    }
}

// ---------------------------------------------------------------------------
// Workspace for floydWarshall().
//
// The algorithm works on page-aligned copies of the matrices whose row stride
// is a multiple of the 4 KiB page size (see floydWarshall() for why). The
// workspace is allocated once per process and reused; releasing it is left to
// process exit. Allocation and page faulting are set-up costs, like the
// allocation of the matrices themselves.
// ---------------------------------------------------------------------------

constexpr size_t PAGE_BYTES = 4096;
constexpr size_t PAGE_ELEMS = PAGE_BYTES / sizeof(unsigned int);

// Leading dimension (row stride in elements) for an n x n working matrix.
inline constexpr size_t paddedStride(const size_t n) noexcept {
    return ((n + PAGE_ELEMS - 1) / PAGE_ELEMS) * PAGE_ELEMS;
}

struct Workspace {
    unsigned int* dist = nullptr;   // n rows of paddedStride(n) elements
    unsigned int* path = nullptr;
    size_t capacityElems = 0;       // per matrix
};

static Workspace& workspace() {
    static Workspace ws;
    return ws;
}

static void ensureWorkspace(const size_t n) {
    Workspace& ws = workspace();
    const size_t needed = paddedStride(n) * n;
    if (needed == 0 || needed <= ws.capacityElems) return;
    std::free(ws.dist);
    std::free(ws.path);
    const size_t bytes = ((needed * sizeof(unsigned int) + PAGE_BYTES - 1) / PAGE_BYTES) * PAGE_BYTES;
    ws.dist = static_cast<unsigned int*>(std::aligned_alloc(PAGE_BYTES, bytes));
    ws.path = static_cast<unsigned int*>(std::aligned_alloc(PAGE_BYTES, bytes));
    if (ws.dist == nullptr || ws.path == nullptr) {
        fprintf(stderr, "Out of memory\n");
        std::abort();
    }
    ws.capacityElems = needed;
}

// Static contiguous block partition of the rows among the threads.
inline void rowRange(const size_t n, const size_t numThreads, const size_t tid,
                     size_t& lo, size_t& hi) noexcept {
    const size_t rowsPerThread = n / numThreads;
    const size_t remainder = n % numThreads;
    lo = tid * rowsPerThread + std::min(tid, remainder);
    hi = lo + rowsPerThread + (tid < remainder ? 1 : 0);
}

// Set-up for floydWarshall(): creates the OpenMP thread pool, pins its
// threads, and allocates the workspace, touching each row with the thread
// that will own it (NUMA first touch). Done once, before the timed region,
// so that it does not pay for thread creation, migration and page faults.
static void prepareFloydWarshall(const size_t numNodes) {
    ensureWorkspace(numNodes);
    Workspace& ws = workspace();
    const size_t ld = paddedStride(numNodes);
    const int computeThreads = chooseThreadCount(numNodes);
    // Use exactly this many threads program-wide: surplus pool threads would
    // otherwise sit spinning on the SMT siblings of the compute cores.
    omp_set_num_threads(computeThreads);
#pragma omp parallel num_threads(computeThreads)
    {
        pinThisOpenMPThread();
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        if (tid < static_cast<size_t>(computeThreads)) {
            size_t lo, hi;
            rowRange(numNodes, static_cast<size_t>(computeThreads), tid, lo, hi);
            if (hi > lo) {
                std::memset(ws.dist + lo * ld, 0, (hi - lo) * ld * sizeof(unsigned int));
                std::memset(ws.path + lo * ld, 0, (hi - lo) * ld * sizeof(unsigned int));
            }
        }
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Sequential semantics: path[i][j] = i for i != j (the later write in the
    // inner loop wins), and path[j][j] = j. Equivalent: path[row][col] = row.
    unsigned int* __restrict__ p = path.data();
#pragma omp parallel
    {
        pinThisOpenMPThread();
#pragma omp for schedule(static)
        for (size_t row = 0; row < numNodes; ++row) {
            unsigned int* rowPtr = p + row * numNodes;
            const unsigned int v = static_cast<unsigned int>(row);
            for (size_t col = 0; col < numNodes; ++col) {
                rowPtr[col] = v;
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    unsigned int* __restrict__ d = dist.data();
    unsigned int* __restrict__ p = path.data();
    const size_t n = numNodes;
    if (n == 0) return;

    // Classic Floyd-Warshall algorithm, parallelized with OpenMP.
    //
    // Data layout: dist[idx2(j, i, n)] == d[i * n + j], i.e. row i of the
    // logical matrix is contiguous. For a fixed pivot k, updating row i reads
    // only row i itself (incl. the column element dist[i][k]) and pivot row k.
    // Row k itself (and column k) is not changed by iteration k because
    // dist[k][k] == 0 and the comparison is strict, so iteration k skips it.
    //
    // Decomposition: the rows are statically partitioned into contiguous
    // blocks [lo, hi), one block per thread, and each row is only ever written
    // by its owning thread. Each row must receive its updates in increasing
    // pivot order, exactly as in the sequential loop nest. The only
    // cross-thread data flow is through pivot rows, with two conditions:
    //   (1) "pivot ready": row k must have received all pivots < k before any
    //       thread uses it as pivot k. Tracked by the monotone counter
    //       pivotReady (pivotReady > k  <=>  row k is ready).
    //   (2) "all done": once row k has been published as pivot, its owner must
    //       not modify it again until every thread has finished reading it,
    //       i.e. has completed iteration k. Tracked by progress[t] = number
    //       of iterations thread t has finished reading pivots for.
    // No barrier is used. Every cell observes exactly the same sequence of
    // updates as in the sequential algorithm, so results are bit-identical.
    //
    // Per-thread schedule (rows [lo, hi) owned):
    //   Phase A, k < lo (pivots owned by earlier blocks): relax all own rows
    //       with pivot k. At k == lo-1 row lo is relaxed first and published.
    //   Phase B, lo <= k < hi (own rows are the pivots): relax only the rows
    //       above the pivot (k+1 first, then publish it immediately, so the
    //       pivot chain advances with the latency of a single row relaxation).
    //       Rows lo..k-1 are former pivots: their updates with pivots k are
    //       deferred (condition 2). At the end of the phase they are caught
    //       up in increasing row order: row r receives pivots r+1..hi-1,
    //       all of which are own rows that are still in their pivot-time
    //       state because they are caught up (modified) only later.
    //   Phase C, k >= hi: like phase A without publishing.
    // The owner of the current pivot block therefore does less work per
    // iteration than the other threads, and the pivot chain never waits for
    // the bulk of the work; the deferred work is done while the chain has
    // already moved on to the next block.

    // The work is done on page-aligned copies of the matrices whose row
    // stride is a multiple of the page size, for two reasons:
    //   * Readers of pivot row k stream through it sequentially and their
    //     hardware prefetchers would otherwise run on into rows k+1, k+2, ...
    //     which the owner is writing at that very moment. The resulting
    //     invalidation traffic slows both the owner (3x) and the readers'
    //     pivot fetch (3x). Prefetchers do not cross 4 KiB page boundaries,
    //     so giving every row its own page(s) removes the interference.
    //   * The vectors' storage is only 16-byte aligned, which makes half of
    //     the vector loads/stores straddle cache lines.
    // The O(n^2) copies are negligible against the O(n^3) work and are done
    // by the owning threads. See ensureWorkspace()/prepareFloydWarshall().
    ensureWorkspace(n);
    const size_t ld = paddedStride(n);   // leading dimension of the copies
    unsigned int* __restrict__ wd = workspace().dist;
    unsigned int* __restrict__ wp = workspace().path;

    const int requestedThreads = chooseThreadCount(n);
    std::vector<PaddedCounter> progress(static_cast<size_t>(requestedThreads));
    alignas(128) std::atomic<size_t> pivotReady{1};   // row 0 is ready initially

#pragma omp parallel num_threads(requestedThreads) default(none) \
        shared(d, p, wd, wp, n, ld, progress, pivotReady)
    {
        const size_t numThreads = static_cast<size_t>(omp_get_num_threads());
        const size_t tid = static_cast<size_t>(omp_get_thread_num());

        pinThisOpenMPThread();

        // Contiguous static block partition of the rows.
        size_t lo, hi;
        rowRange(n, numThreads, tid, lo, hi);

        // Copy own rows into the working arrays.
        for (size_t i = lo; i < hi; ++i) {
            std::memcpy(wd + i * ld, d + i * n, n * sizeof(unsigned int));
            std::memcpy(wp + i * ld, p + i * n, n * sizeof(unsigned int));
        }
        // Pivot row 0 is published implicitly (pivotReady == 1): make sure it
        // has been copied before anyone proceeds past this point.
#pragma omp barrier

        auto relax = [&](const size_t i, const size_t k) noexcept {
            relaxRow(wd + i * ld, wp + i * ld, wd + k * ld, wd[i * ld + k],
                     static_cast<unsigned int>(k), n);
        };
        auto waitPivot = [&](const size_t k) noexcept {
            spinUntil([&]() noexcept { return pivotReady.load(std::memory_order_acquire) > k; });
        };
        auto finishIteration = [&](const size_t k) noexcept {
            progress[tid].value.store(k + 1, std::memory_order_release);
        };

        // ---- Phase A: pivots before the own block ----
        for (size_t k = 0; k < lo; ++k) {
            waitPivot(k);
            size_t first = lo;
            if (k + 1 == lo) {
                relax(lo, k);
                pivotReady.store(lo + 1, std::memory_order_release);
                first = lo + 1;
            }
            for (size_t i = first; i < hi; ++i) relax(i, k);
            finishIteration(k);
        }

        // ---- Phase B: own rows are the pivots ----
        for (size_t k = lo; k < hi; ++k) {
            waitPivot(k);
            if (k + 1 < hi) {
                relax(k + 1, k);
                pivotReady.store(k + 2, std::memory_order_release);
                for (size_t i = k + 2; i < hi; ++i) relax(i, k);
            }
            finishIteration(k);
        }

        // Catch up the deferred updates of the former pivot rows. Row r may
        // only be modified once all other threads have finished iteration r.
        if (hi > lo) {
            size_t othersDone = 0;   // min progress over all other threads (cached)
            for (size_t r = lo; r + 1 < hi; ++r) {
                if (othersDone < r + 1) {
                    for (size_t t = 0; t < numThreads; ++t) {
                        __builtin_prefetch(&progress[t].value, 0, 0);
                    }
                    size_t m = n;
                    for (size_t t = 0; t < numThreads; ++t) {
                        if (t == tid) continue;
                        spinUntil([&]() noexcept {
                            return progress[t].value.load(std::memory_order_acquire) >= r + 1;
                        });
                        m = std::min(m, progress[t].value.load(std::memory_order_acquire));
                    }
                    othersDone = m;
                }
                for (size_t j = r + 1; j < hi; ++j) relax(r, j);
            }
        }

        // ---- Phase C: pivots after the own block ----
        for (size_t k = hi; k < n; ++k) {
            waitPivot(k);
            for (size_t i = lo; i < hi; ++i) relax(i, k);
            finishIteration(k);
        }

        // Copy own (final) rows back. Own rows are complete once the own
        // phase C loop has run; nobody else writes them.
        for (size_t i = lo; i < hi; ++i) {
            std::memcpy(d + i * n, wd + i * ld, n * sizeof(unsigned int));
            std::memcpy(p + i * n, wp + i * ld, n * sizeof(unsigned int));
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
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
    
    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    printf("Number of nodes: %zu\n", numNodes);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Create and pin the OpenMP worker threads and allocate the workspace.
    prepareFloydWarshall(numNodes);
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
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
