#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <atomic>
#include <thread>

#include <omp.h>
#include <sched.h>

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
// Blocked (tiled) Floyd-Warshall, parallelized with OpenMP.
//
// The matrices are copied into a padded, tile-contiguous layout (each BxB tile
// is stored contiguously) so every tile kernel works on dense, cache-friendly
// data regardless of numNodes (avoids cache-set conflicts for power-of-two n).
// Padding nodes have distance INF to/from every real node, so they can never
// shorten a real path and the real results are unaffected.
// ---------------------------------------------------------------------------

// Tile update where C may alias A and/or B (diagonal tile and row/column tiles).
// k is outermost; row k and column k of the diagonal tile do not change while k
// is being processed (dist[k][k] == 0), so this matches the sequential order.
template <size_t B>
static inline void tileDependent(unsigned int* C, unsigned int* P,
                                 const unsigned int* A, const unsigned int* Bm,
                                 const unsigned int kBase) {
    for (size_t k = 0; k < B; ++k) {
        const unsigned int kv = kBase + static_cast<unsigned int>(k);
        const unsigned int* rowK = Bm + k * B;
        for (size_t i = 0; i < B; ++i) {
            const unsigned int distIK = A[i * B + k];
            unsigned int* rowI = C + i * B;
            unsigned int* pathI = P + i * B;
            #pragma omp simd
            for (size_t j = 0; j < B; ++j) {
                const unsigned int newDist = distIK + rowK[j];
                const bool better = newDist < rowI[j];
                rowI[j] = better ? newDist : rowI[j];
                pathI[j] = better ? kv : pathI[j];
            }
        }
    }
}

// Tile update where C is disjoint from A and B. Each chunk of a row of C is kept
// in registers across the whole k loop (k is still processed in increasing order
// for every element).
template <size_t B>
static inline void tileIndependent(unsigned int* __restrict C, unsigned int* __restrict P,
                                   const unsigned int* __restrict A,
                                   const unsigned int* __restrict Bm,
                                   const unsigned int kBase) {
    constexpr size_t CHUNK = (B < 32) ? B : 32;
    for (size_t i = 0; i < B; ++i) {
        const unsigned int* rowA = A + i * B;
        for (size_t jc = 0; jc < B; jc += CHUNK) {
            unsigned int cd[CHUNK];
            unsigned int cp[CHUNK];
            #pragma omp simd
            for (size_t j = 0; j < CHUNK; ++j) {
                cd[j] = C[i * B + jc + j];
                cp[j] = P[i * B + jc + j];
            }
            for (size_t k = 0; k < B; ++k) {
                const unsigned int distIK = rowA[k];
                const unsigned int kv = kBase + static_cast<unsigned int>(k);
                const unsigned int* rowK = Bm + k * B + jc;
                #pragma omp simd
                for (size_t j = 0; j < CHUNK; ++j) {
                    const unsigned int newDist = distIK + rowK[j];
                    const bool better = newDist < cd[j];
                    cd[j] = better ? newDist : cd[j];
                    cp[j] = better ? kv : cp[j];
                }
            }
            #pragma omp simd
            for (size_t j = 0; j < CHUNK; ++j) {
                C[i * B + jc + j] = cd[j];
                P[i * B + jc + j] = cp[j];
            }
        }
    }
}

// Sense-reversing spin barrier used between the phases of each round. The rounds
// are short, so spinning avoids the sleep/wake-up latency of the default OpenMP
// barrier; it yields after a while to stay well-behaved when oversubscribed.
class SpinBarrier {
public:
    explicit SpinBarrier(const int n) : numThreads(n) {}

    void wait(bool& localSense) {
        localSense = !localSense;
        if (count.fetch_add(1, std::memory_order_acq_rel) == numThreads - 1) {
            count.store(0, std::memory_order_relaxed);
            sense.store(localSense, std::memory_order_release);
        } else {
            unsigned int spins = 0;
            while (sense.load(std::memory_order_acquire) != localSense) {
                if (++spins > (1u << 16)) {
                    std::this_thread::yield();
                } else {
#if defined(__x86_64__) || defined(__i386__)
                    __builtin_ia32_pause();
#endif
                }
            }
        }
    }

private:
    alignas(64) std::atomic<int> count{0};
    alignas(64) std::atomic<bool> sense{false};
    const int numThreads;
};

// Pin the calling OpenMP thread to one CPU of the process' original affinity mask
// (thread t -> t-th allowed CPU). Only done when no OpenMP place list is configured,
// so user-provided OMP_PLACES/OMP_PROC_BIND settings take precedence. Pinning keeps
// the spin barrier and the tile-to-thread data locality stable between rounds.
static void pinThreadToCpu(const int tid) {
#ifdef __linux__
    static cpu_set_t allowed;
    static int numAllowed = -1;
    #pragma omp single
    {
        if (numAllowed < 0) {
            CPU_ZERO(&allowed);
            numAllowed = (sched_getaffinity(0, sizeof(allowed), &allowed) == 0) ? CPU_COUNT(&allowed) : 0;
        }
    }
    // implicit barrier
    if (numAllowed <= 0 || omp_get_num_places() > 0) return;
    int target = tid % numAllowed;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed) && target-- == 0) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpu, &one);
            sched_setaffinity(0, sizeof(one), &one);
            break;
        }
    }
#else
    (void)tid;
#endif
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Equivalent to the original nested assignment: element (row r, col c),
    // stored at r * numNodes + c, ends up holding r.
    unsigned int* __restrict p = path.data();
    #pragma omp parallel
    {
        // Also pins the OpenMP worker threads ahead of the timed computation
        pinThreadToCpu(omp_get_thread_num());
        #pragma omp for schedule(static)
        for (size_t r = 0; r < numNodes; ++r) {
            for (size_t c = 0; c < numNodes; ++c) {
                p[r * numNodes + c] = static_cast<unsigned int>(r);
            }
        }
    }
}

template <size_t B>
static void floydWarshallBlocked(std::vector<unsigned int>& dist,
                                 std::vector<unsigned int>& path,
                                 const size_t n) {
    const size_t nb = (n + B - 1) / B;
    const size_t tileElems = B * B;
    const size_t numTiles = nb * nb;
    const size_t total = numTiles * tileElems;

    // Uninitialized, cache-line aligned buffers; first touched in parallel below
    unsigned int* td = static_cast<unsigned int*>(std::aligned_alloc(64, total * sizeof(unsigned int)));
    unsigned int* tp = static_cast<unsigned int*>(std::aligned_alloc(64, total * sizeof(unsigned int)));
    if (!td || !tp) {
        fprintf(stderr, "Allocation failed\n");
        std::exit(1);
    }
    unsigned int* const d = dist.data();
    unsigned int* const p = path.data();

    // Never use more threads than there are independent tiles per round
    const size_t maxUseful = std::max<size_t>(1, (nb - 1) * (nb - 1));
    const int numThreads = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(omp_get_max_threads()), maxUseful));

    SpinBarrier* barrier = nullptr;

    #pragma omp parallel num_threads(numThreads)
    {
        pinThreadToCpu(omp_get_thread_num());

        // Pack into tiled layout (same static tile distribution as phase 3 -> NUMA locality)
        #pragma omp for schedule(static)
        for (size_t t = 0; t < numTiles; ++t) {
            const size_t ib = t / nb, jb = t % nb;
            unsigned int* C = td + t * tileElems;
            unsigned int* P = tp + t * tileElems;
            for (size_t r = 0; r < B; ++r) {
                const size_t gi = ib * B + r;
                for (size_t c = 0; c < B; ++c) {
                    const size_t gj = jb * B + c;
                    if (gi < n && gj < n) {
                        C[r * B + c] = d[gi * n + gj];
                        P[r * B + c] = p[gi * n + gj];
                    } else {
                        C[r * B + c] = (gi == gj) ? 0u : INF;
                        P[r * B + c] = static_cast<unsigned int>(gi);
                    }
                }
            }
        }
        // implicit barrier

        const int nthr = omp_get_num_threads();
        #pragma omp single
        barrier = new SpinBarrier(nthr);
        // implicit barrier
        bool localSense = false;
        const size_t nOther = nb - 1;

        for (size_t kb = 0; kb < nb; ++kb) {
            const unsigned int kBase = static_cast<unsigned int>(kb * B);
            unsigned int* const D = td + (kb * nb + kb) * tileElems;

            // Phase 1: diagonal tile
            if (omp_get_thread_num() == 0) {
                tileDependent<B>(D, tp + (kb * nb + kb) * tileElems, D, D, kBase);
            }
            barrier->wait(localSense);

            // Phase 2: tiles in tile-row kb and tile-column kb
            #pragma omp for schedule(static, 1) nowait
            for (size_t t = 0; t < 2 * nOther; ++t) {
                const size_t b = (t >> 1) + ((t >> 1) >= kb ? 1 : 0);
                if (t & 1) {
                    // column tile (b, kb): C[i][j] = min(C[i][j], C[i][k] + D[k][j])
                    const size_t ti = b * nb + kb;
                    unsigned int* C = td + ti * tileElems;
                    tileDependent<B>(C, tp + ti * tileElems, C, D, kBase);
                } else {
                    // row tile (kb, b): C[i][j] = min(C[i][j], D[i][k] + C[k][j])
                    const size_t ti = kb * nb + b;
                    unsigned int* C = td + ti * tileElems;
                    tileDependent<B>(C, tp + ti * tileElems, D, C, kBase);
                }
            }
            barrier->wait(localSense);

            // Phase 3: all remaining tiles are independent of each other. Iterating
            // over all tiles with a static schedule gives every thread a fixed set of
            // tiles across rounds, so they stay in that thread's caches / NUMA node.
            #pragma omp for schedule(static) nowait
            for (size_t t = 0; t < numTiles; ++t) {
                const size_t ib = t / nb, jb = t % nb;
                if (ib == kb || jb == kb) continue;
                tileIndependent<B>(td + t * tileElems, tp + t * tileElems,
                                   td + (ib * nb + kb) * tileElems,
                                   td + (kb * nb + jb) * tileElems, kBase);
            }
            barrier->wait(localSense);
        }

        // Unpack back to the row-major layout
        #pragma omp for schedule(static)
        for (size_t gi = 0; gi < n; ++gi) {
            const size_t ib = gi / B, r = gi % B;
            for (size_t jb = 0; jb < nb; ++jb) {
                const size_t off = (ib * nb + jb) * tileElems + r * B;
                const size_t cEnd = std::min(B, n - jb * B);
                for (size_t c = 0; c < cEnd; ++c) {
                    d[gi * n + jb * B + c] = td[off + c];
                    p[gi * n + jb * B + c] = tp[off + c];
                }
            }
        }
    }

    delete barrier;
    std::free(td);
    std::free(tp);
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;
    // Smaller tiles expose more parallelism for small graphs
    if (numNodes <= 1024) {
        floydWarshallBlocked<32>(dist, path, numNodes);
    } else {
        floydWarshallBlocked<64>(dist, path, numNodes);
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
