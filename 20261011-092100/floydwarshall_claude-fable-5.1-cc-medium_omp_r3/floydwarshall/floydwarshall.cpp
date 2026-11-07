#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <omp.h>

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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Equivalent to the original double-write initialization: the final value of
    // path[idx2(i, j, n)] is j for every (i, j), including the diagonal.
    unsigned int* __restrict p = path.data();
#pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        unsigned int* __restrict rowJ = p + j * numNodes;
        for (size_t i = 0; i < numNodes; ++i) {
            rowJ[i] = static_cast<unsigned int>(j);
        }
    }
}

// Cache-line padded atomic counter, one per thread / ring slot, to avoid
// false sharing between the synchronization variables.
struct alignas(64) PaddedCounter {
    std::atomic<size_t> value{0};
    char pad[64 - sizeof(std::atomic<size_t>)];
};

// Spin-wait until *counter >= target (acquire semantics). Falls back to
// yielding after a while so that an oversubscribed machine still progresses.
static inline void waitAtLeast(const std::atomic<size_t>& counter, const size_t target) noexcept {
    unsigned spins = 0;
    while (counter.load(std::memory_order_acquire) < target) {
        if (++spins > 2048) {
            std::this_thread::yield();
            spins = 0;
        }
    }
}

// Relax one row i with intermediate node k: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]).
// rowK is a snapshot of row k as it is at the start of iteration k.
static inline void relaxRow(unsigned int* __restrict rowI,
                            unsigned int* __restrict pathI,
                            const unsigned int* __restrict rowK,
                            const unsigned int distIK,
                            const unsigned int kk,
                            const size_t n) noexcept {
#pragma omp simd
    for (size_t j = 0; j < n; ++j) {
        const unsigned int distIJ = rowI[j];
        const unsigned int newDist = distIK + rowK[j];
        if (newDist < distIJ) {
            rowI[j] = newDist;
            pathI[j] = kk;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm, parallelized over the source node i
    // with OpenMP.
    //
    // Memory layout: dist[idx2(j, i, n)] == dist[i * n + j], so row i (all
    // destinations j) is contiguous. For a fixed k, updating (i, j) reads only
    // dist[i][k] (own row) and dist[k][j] (pivot row k). Row k and column k do
    // not change during iteration k because dist[k][k] == 0, so rows are
    // independent within an iteration and each thread owns a fixed contiguous
    // block of rows (which then stays hot in that core's private caches).
    //
    // Instead of a full barrier per k, iterations are pipelined: the owner of
    // row k+1 relaxes that row first, publishes a snapshot of it into a ring
    // buffer and raises a ready flag; all other threads only wait on that flag
    // before starting iteration k+1. Threads may therefore drift apart by up to
    // RING_SLOTS iterations; the owner of a pivot row waits for every thread to
    // have finished with the slot before overwriting it. Every element is
    // still updated exactly once per k, in increasing k order, using the
    // row-k values of the sequential algorithm, so dist and path are
    // bit-identical to the sequential code.
    unsigned int* __restrict d = dist.data();
    unsigned int* __restrict pth = path.data();
    const size_t n = numNodes;
    if (n == 0) return;

    // Do not spread tiny problems over an excessive number of threads: give
    // each thread at least a few rows so synchronization does not dominate.
    constexpr size_t MIN_ROWS_PER_THREAD = 4;
    const int maxThreads = omp_get_max_threads();
    const size_t wantThreads = std::max<size_t>(1, n / MIN_ROWS_PER_THREAD);
    const int numThreads = static_cast<int>(std::min<size_t>(static_cast<size_t>(maxThreads), wantThreads));

    constexpr size_t RING_SLOTS = 16;
    std::vector<unsigned int> ring(RING_SLOTS * n);
    std::vector<PaddedCounter> ringReady(RING_SLOTS);     // value = k+1 of the row stored in the slot
    std::vector<PaddedCounter> iterationsDone(static_cast<size_t>(numThreads));  // per thread: # of k iterations finished

    // Pivot row for k = 0 is simply row 0 of the input.
    std::copy(d, d + n, ring.data());
    ringReady[0].value.store(1, std::memory_order_release);

#pragma omp parallel num_threads(numThreads)
    {
        const size_t T = static_cast<size_t>(omp_get_num_threads());
        const size_t t = static_cast<size_t>(omp_get_thread_num());

        // Static contiguous row partition [lo, hi) for this thread.
        const size_t lo = (n * t) / T;
        const size_t hi = (n * (t + 1)) / T;

        for (size_t k = 0; k < n; ++k) {
            const size_t slot = k % RING_SLOTS;
            const unsigned int kk = static_cast<unsigned int>(k);

            // Wait until the snapshot of row k is published.
            waitAtLeast(ringReady[slot].value, k + 1);
            const unsigned int* __restrict rowK = ring.data() + slot * n;

            // 1. If we own the next pivot row, relax and publish it first so
            //    other threads can move on to iteration k+1 as soon as possible.
            const size_t next = k + 1;
            const bool ownsNext = (next < n) && (next >= lo) && (next < hi);
            if (ownsNext) {
                unsigned int* __restrict rowNext = d + next * n;
                relaxRow(rowNext, pth + next * n, rowK, rowNext[k], kk, n);

                // The target slot was last used by iteration next - RING_SLOTS;
                // make sure every thread has finished that iteration.
                const size_t nextSlot = next % RING_SLOTS;
                if (next >= RING_SLOTS) {
                    const size_t needDone = next - RING_SLOTS + 1;
                    for (size_t u = 0; u < T; ++u) {
                        waitAtLeast(iterationsDone[u].value, needDone);
                    }
                }
                std::copy(rowNext, rowNext + n, ring.data() + nextSlot * n);
                ringReady[nextSlot].value.store(next + 1, std::memory_order_release);
            }

            // 2. Relax the rest of our rows.
            for (size_t i = lo; i < hi; ++i) {
                if (i == next) continue;  // already done above
                unsigned int* __restrict rowI = d + i * n;
                relaxRow(rowI, pth + i * n, rowK, rowI[k], kk, n);
            }

            // Signal that we are finished reading ring slot `slot` for this k.
            iterationsDone[t].value.store(k + 1, std::memory_order_release);
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
