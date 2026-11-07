#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>
#include <sched.h>
#if defined(__SSE2__)
#include <immintrin.h>
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

namespace {

// Cache-line padded atomic counter to avoid false sharing between threads.
struct alignas(64) PaddedCounter {
    std::atomic<size_t> v{0};
};

// Bounded spin, then yield the CPU: when all hardware threads are busy a
// waiting thread must not starve the (possibly preempted) thread it waits for.
struct Backoff {
    unsigned spins = 0;
    void wait() noexcept {
        if (spins < 128) {
            ++spins;
#if defined(__x86_64__) || defined(__i386__)
            __builtin_ia32_pause();
#endif
        } else {
            sched_yield();
        }
    }
};

// Pin the calling OpenMP thread to one CPU of the process' affinity mask
// (thread t -> t-th allowed CPU, wrapping around). Thread migration destroys
// the per-thread cache locality this kernel relies on, and without an
// explicit OMP_PROC_BIND / OMP_PLACES setting the runtime does not bind.
void pinThread(const size_t t) noexcept {
#if defined(__linux__)
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    const int count = CPU_COUNT(&allowed);
    if (count <= 1) return;
    const int target = static_cast<int>(t % static_cast<size_t>(count));
    int seen = 0;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        if (seen == target) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(c, &one);
            sched_setaffinity(0, sizeof(one), &one);
            return;
        }
        ++seen;
    }
#else
    (void)t;
#endif
}

}  // namespace

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // The original loop nest writes every element path[idx2(j, i, n)] with
    // value i (each off-diagonal element twice, both times with the same
    // value), so each row i of the (i, j) matrix is filled with i. Write every
    // element exactly once, with rows distributed over threads.
    unsigned int* __restrict__ p = path.data();
#pragma omp parallel
    {
        // Bind the whole thread pool once (unless the user chose a binding
        // policy) so that threads left idle by a later, smaller parallel
        // region do not compete for the CPUs of the working threads.
        if (omp_get_num_threads() > 1 && omp_get_proc_bind() == omp_proc_bind_false) {
            pinThread(static_cast<size_t>(omp_get_thread_num()));
        }
#pragma omp for schedule(static)
        for (size_t i = 0; i < numNodes; ++i) {
            const unsigned int v = static_cast<unsigned int>(i);
            unsigned int* row = p + i * numNodes;
            for (size_t j = 0; j < numNodes; ++j) {
                row[j] = v;
            }
        }
    }
}


void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm, parallelized with OpenMP.
    //
    // Memory layout: idx2(j, i, n) == i * n + j, so row i of the (i, j)
    // matrix is contiguous. Applying step k to row i only needs row i itself
    // (including dist[i][k]) plus row k as it was after step k-1 (row k and
    // column k do not change during step k because dist[k][k] == 0). The
    // result is identical to the sequential algorithm as long as every row
    // receives the steps k = 0 .. n-1 in increasing order and step k reads
    // the "after step k-1" snapshot of row k.
    //
    // Rows are partitioned statically into one contiguous block per thread
    // and each thread sweeps its rows with a contiguous, branch-free,
    // vectorizable j loop.
    //
    // Synchronization is point-to-point instead of a barrier per step: the
    // owner of row k publishes its "after step k-1" snapshot into a
    // write-once snapshot buffer (one slot per row, so a publish never has
    // to invalidate lines cached by other readers) and everybody waits only
    // on that row's flag. Inside its own block [lo, hi) a thread needs
    // nothing from other threads for steps lo..hi-1, so it runs ahead and
    // publishes rows lo+1..hi-1 as early as possible (a few row updates
    // each) and backfills the remaining updates of its block afterwards.
    // This keeps the per-step critical path at roughly one row update
    // instead of a full per-thread step plus barrier latency.
    const size_t n = numNodes;
    if (n == 0) return;

    unsigned int* __restrict__ d = dist.data();
    unsigned int* __restrict__ p = path.data();

    // Snapshot buffer: row k ("after step k-1") is stored at snap + k * stride.
    // Allocated uninitialized and first-touched by the publishing thread.
    const size_t stride = (n + 15) & ~static_cast<size_t>(15);  // 64-byte row pitch
    std::unique_ptr<unsigned int[]> snapStorage(new unsigned int[n * stride + 16]);
    unsigned int* __restrict__ snap = reinterpret_cast<unsigned int*>(
        (reinterpret_cast<uintptr_t>(snapStorage.get()) + 63) & ~static_cast<uintptr_t>(63));
    // pub[k] == 1 once row k has been published.
    std::unique_ptr<PaddedCounter[]> pubStorage(new PaddedCounter[n]);
    PaddedCounter* pubArr = pubStorage.get();

    // Give every thread at least a few rows: with fewer rows per thread the
    // per-step synchronization dominates and extra threads only slow it down.
    const int numThreads = static_cast<int>(
        std::max<size_t>(1, std::min<size_t>(static_cast<size_t>(omp_get_max_threads()), n / 8)));

#pragma omp parallel num_threads(numThreads)
    {
        const size_t T = static_cast<size_t>(omp_get_num_threads());
        const size_t t = static_cast<size_t>(omp_get_thread_num());

        // Bind threads unless the user already chose a binding policy.
        if (T > 1 && omp_get_proc_bind() == omp_proc_bind_false) pinThread(t);

        // Static contiguous row partition: rows [lo, hi) belong to thread t.
        const size_t base = n / T;
        const size_t rem = n % T;
        const size_t lo = t * base + std::min(t, rem);
        const size_t hi = lo + base + (t < rem ? 1 : 0);

        // First-touch the snapshot slots this thread will publish, so the
        // page faults happen in parallel and off the critical path.
        if (lo < hi) std::memset(snap + lo * stride, 0, (hi - lo) * stride * sizeof(unsigned int));

        auto snapRow = [&](size_t k) noexcept -> const unsigned int* {
            return snap + k * stride;
        };

        auto prefetchRow = [&](size_t k) noexcept {
            const unsigned int* rowK = snapRow(k);
            for (size_t off = 0; off < n; off += 16) __builtin_prefetch(rowK + off, 0, 3);
        };

        // Copy row r (as it is right now) into its snapshot slot and flag it.
        // The snapshot is written with streaming stores: hundreds of readers
        // then pull it from memory through all channels instead of hammering
        // the publisher's cache / its CCD link with a single-source broadcast.
        auto publish = [&](size_t r) {
            unsigned int* __restrict__ dst = snap + r * stride;
            const unsigned int* __restrict__ src = d + r * n;
            size_t j = 0;
#if defined(__AVX2__)
            for (; j + 8 <= n; j += 8) {
                _mm256_stream_si256(reinterpret_cast<__m256i*>(dst + j),
                                    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(src + j)));
            }
#elif defined(__SSE2__)
            for (; j + 4 <= n; j += 4) {
                _mm_stream_si128(reinterpret_cast<__m128i*>(dst + j),
                                 _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + j)));
            }
#endif
            for (; j < n; ++j) dst[j] = src[j];
#if defined(__SSE2__)
            _mm_sfence();  // streaming stores are weakly ordered: drain before the flag
#endif
            pubArr[r].v.store(1, std::memory_order_release);
        };

        // Apply step k to row i using the published snapshot of row k.
        auto updateRow = [&](size_t i, size_t k, const unsigned int* __restrict__ rowK) {
            const unsigned int distIK = d[i * n + k];
            const unsigned int kk = static_cast<unsigned int>(k);
            unsigned int* __restrict__ rowI = d + i * n;
            unsigned int* __restrict__ pathI = p + i * n;
#pragma omp simd
            for (size_t j = 0; j < n; ++j) {
                const unsigned int distIJ = rowI[j];
                const unsigned int newDist = distIK + rowK[j];
                const bool better = newDist < distIJ;
                rowI[j] = better ? newDist : distIJ;
                pathI[j] = better ? kk : pathI[j];
            }
        };

        // Steps whose rows come from other threads are consumed in chunks of
        // consecutive already-published rows [k, k + K): each own row then
        // receives the K steps (in increasing order) while it stays in L1,
        // instead of streaming the whole block through the cache once per
        // step. Only row k is ever waited for; K adapts to what is available.
        // kMax keeps the K snapshot rows within about 256 KB of L2.
        const size_t kMax = std::max<size_t>(1, std::min<size_t>(64, (256u * 1024u) / (n * sizeof(unsigned int))));

        // Wait for row k, then return the number of consecutively published
        // rows starting at k, capped at limit. Prefetches the flags and the
        // first rows of the chunk so the fetches overlap with computation.
        auto acquireChunk = [&](size_t k, size_t limit) -> size_t {
            for (size_t q = 1; q < limit; ++q) __builtin_prefetch(&pubArr[k + q], 0, 3);
            Backoff backoff;
            while (pubArr[k].v.load(std::memory_order_acquire) == 0) backoff.wait();
            size_t K = 1;
            while (K < limit && pubArr[k + K].v.load(std::memory_order_acquire) != 0) ++K;
            prefetchRow(k);
            if (K > 1) prefetchRow(k + 1);
            return K;
        };

        // Apply steps [k, k + K) to all own rows (row-major), publishing row
        // lo as soon as it has reached the state "after step lo - 1".
        auto consumeChunk = [&](size_t k, size_t K) {
            for (size_t i = lo; i < hi; ++i) {
                for (size_t q = 0; q < K; ++q) updateRow(i, k + q, snapRow(k + q));
                if (i == lo && k + K == lo) publish(lo);  // row lo is needed by everybody next
            }
        };

        if (lo < hi) {
            if (lo == 0) publish(0);

            // Phase A: steps k < lo, row k comes from another thread.
            for (size_t k = 0; k < lo;) {
                const size_t K = acquireChunk(k, std::min(kMax, lo - k));
                consumeChunk(k, K);
                k += K;
            }

            // Phase B: steps lo <= k < hi, all required rows are our own.
            // Run ahead: row r only needs steps lo..r-1, whose snapshots are
            // ours, so publish rows lo+1..hi-1 right away (row r = lo + m
            // costs m updates), then backfill the remaining updates. Every
            // row still receives its steps in increasing order: row r gets
            // steps lo..r-1 here and steps r..hi-1 in the backfill below.
            for (size_t r = lo + 1; r < hi; ++r) {
                for (size_t k = lo; k < r; ++k) updateRow(r, k, snapRow(k));
                publish(r);
            }
            for (size_t i = lo; i < hi; ++i) {
                for (size_t k = i; k < hi; ++k) updateRow(i, k, snapRow(k));
            }

            // Phase C: steps k >= hi, row k comes from another thread.
            for (size_t k = hi; k < n;) {
                const size_t K = acquireChunk(k, std::min(kMax, n - k));
                consumeChunk(k, K);
                k += K;
            }
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
