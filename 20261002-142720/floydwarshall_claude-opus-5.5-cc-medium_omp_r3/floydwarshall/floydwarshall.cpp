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
#include <sys/mman.h>
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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Final state of the original sequential initialization: path[j*n+i] = j
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
        }
    }
}

// Number of consecutive k iterations processed per pass over the matrix
constexpr long KBLOCK = 32;
// Column tile width for the bulk update (keeps the pivot-row snapshot tile in L1/L2)
constexpr long JTILE = 256;
// Register block: RB rows x VW columns kept in SIMD registers across a k block
constexpr long RB = 4;
constexpr long VW = 8;
// Column tile width for updating the pivot rows
constexpr long CTILE = 64;

typedef unsigned int vu8 __attribute__((vector_size(VW * sizeof(unsigned int))));

static inline vu8 loadu(const unsigned int* p) {
    vu8 v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
static inline void storeu(unsigned int* p, const vu8 v) {
    std::memcpy(p, &v, sizeof(v));
}

// Pin the calling OpenMP thread to one CPU of the process affinity mask (in
// order of thread number) unless the user controls placement via the
// environment. Keeps threads (and their first-touched memory) on fixed cores.
static void pinThreadIfUnplaced(const int tid) {
#ifdef __linux__
    static const bool userPlacement = std::getenv("OMP_PLACES") != nullptr ||
                                      std::getenv("GOMP_CPU_AFFINITY") != nullptr;
    if (userPlacement) return;
    static cpu_set_t allowed;
    static int numAllowed = [] {
        CPU_ZERO(&allowed);
        if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return 0;
        return CPU_COUNT(&allowed);
    }();
    if (numAllowed <= 0) return;
    const int target = tid % numAllowed;
    for (int cpu = 0, seen = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) continue;
        if (seen++ == target) {
            cpu_set_t one;
            CPU_ZERO(&one);
            CPU_SET(cpu, &one);
            pthread_setaffinity_np(pthread_self(), sizeof(one), &one);
            return;
        }
    }
#else
    (void)tid;
#endif
}

// Cache-line aligned, uninitialized buffer (pages are first-touched by the
// threads that use them)
class AlignedBuffer {
public:
    explicit AlignedBuffer(const size_t count) {
        const size_t bytes = std::max<size_t>(count * sizeof(unsigned int), 64);
        // Large buffers: 2 MiB alignment so transparent huge pages can be used
        // (fewer page faults during first touch and fewer TLB misses).
        const size_t align = bytes >= (size_t(4) << 20) ? (size_t(2) << 20) : 64;
        ptr_ = static_cast<unsigned int*>(std::aligned_alloc(align, (bytes + align - 1) / align * align));
        if (!ptr_) {
            fprintf(stderr, "Allocation failed\n");
            std::exit(1);
        }
#ifdef MADV_HUGEPAGE
        if (align > 64) madvise(ptr_, (bytes + align - 1) / align * align, MADV_HUGEPAGE);
#endif
    }
    ~AlignedBuffer() { std::free(ptr_); }
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;
    unsigned int* get() const noexcept { return ptr_; }

private:
    unsigned int* ptr_;
};

// Apply iteration k to row segment [j0, j1): exactly the classic relaxation
static inline void relaxSegment(unsigned int* __restrict rowI, unsigned int* __restrict pathI,
                                const unsigned int* __restrict rowK, const unsigned int distIK,
                                const unsigned int k, const long j0, const long j1) {
    #pragma omp simd
    for (long j = j0; j < j1; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const bool better = newDist < rowI[j];
        rowI[j] = better ? newDist : rowI[j];
        pathI[j] = better ? k : pathI[j];
    }
}

// Apply iterations kb..kb+kn-1 (in order) to an RB x VW block at column j.
// aik[r * KBLOCK + kk] holds dist[i_r][kb+kk] as it was before iteration kb+kk.
static inline void relaxRegisterBlock(unsigned int* const* rows, unsigned int* const* paths,
                                      const unsigned int* __restrict S, const long ld,
                                      const unsigned int* __restrict aik,
                                      const long kb, const long kn, const long j) {
    vu8 dv[RB], pv[RB];
    for (long r = 0; r < RB; ++r) {
        dv[r] = loadu(rows[r] + j);
        pv[r] = loadu(paths[r] + j);
    }
    for (long kk = 0; kk < kn; ++kk) {
        const vu8 sk = loadu(S + kk * ld + j);
        const vu8 kv = vu8{} + static_cast<unsigned int>(kb + kk);
        for (long r = 0; r < RB; ++r) {
            const vu8 nd = aik[r * KBLOCK + kk] + sk;
            const vu8 mn = (nd < dv[r]) ? nd : dv[r];   // dist update (strictly-less)
            pv[r] = (mn == dv[r]) ? pv[r] : kv;         // path set to k iff improved
            dv[r] = mn;
        }
    }
    for (long r = 0; r < RB; ++r) {
        storeu(rows[r] + j, dv[r]);
        storeu(paths[r] + j, pv[r]);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Floyd-Warshall, parallelized with OpenMP and blocked over k.
    //
    // The classic algorithm updates every element (i,j) for k = 0..n-1 in order,
    // using dist[i][k] and dist[k][j] as they are after iteration k-1 (which equal
    // their values after iteration k, since dist[k][k] == 0). The blocked scheme
    // below performs exactly the same sequence of comparisons/updates on every
    // element, so both dist and path are bit-identical to the sequential version:
    //   For a block of pivots [kb, ke):
    //   0. Diagonal block (pivot rows x pivot columns), in order, recording the
    //      pivot-column values A[i][k] and pivot-row snapshots at stage k-1.
    //   1. Pivot rows, remaining columns (parallel over column tiles), saving a
    //      snapshot of each pivot row k right before iteration k is applied.
    //   2. All other rows (each thread owns a cyclic set of row groups) for k in
    //      [kb, ke) in order,
    //      reading pivot row values from the snapshots.
    const long n = static_cast<long>(numNodes);
    if (n == 0) return;

    // NUMA-friendly working copies: first-touched by the threads that own the rows.
    // Rows are padded to a multiple of a cache line and kept away from large
    // power-of-two strides to avoid false sharing and cache-set conflicts.
    long ld = (n + 15) / 16 * 16;
    if (ld % 512 == 0) ld += 16;
    const size_t nn = static_cast<size_t>(n) * ld;
    AlignedBuffer dBuf(nn), pBuf(nn);
    unsigned int* __restrict d = dBuf.get();
    unsigned int* __restrict p = pBuf.get();
    unsigned int* const dSrc = dist.data();
    unsigned int* const pSrc = path.data();

    // Snapshots of pivot rows (KBLOCK x n) and pivot-column values of the pivot
    // rows (KBLOCK x KBLOCK), both at stage k-1.
    AlignedBuffer snap(static_cast<size_t>(KBLOCK) * ld);
    unsigned int* __restrict S = snap.get();
    unsigned int A[KBLOCK][KBLOCK];

    const long numGroups = (n + RB - 1) / RB;

    #pragma omp parallel default(none) shared(d, p, dSrc, pSrc, n, ld, S, A, numGroups) proc_bind(close)
    {
        // Row groups (RB consecutive rows) are distributed cyclically among
        // threads, so the pivot rows of each k block are spread over many
        // threads' caches. The thread owning a group always processes it.
        const long nth = omp_get_num_threads();
        const long tid = omp_get_thread_num();
        pinThreadIfUnplaced(static_cast<int>(tid));
        const long myGroups = tid < numGroups ? (numGroups - 1 - tid) / nth + 1 : 0;

        for (long G = tid; G < numGroups; G += nth) {
            for (long i = G * RB; i < std::min(G * RB + RB, n); ++i) {
                std::memcpy(d + i * ld, dSrc + i * n, n * sizeof(unsigned int));
                std::memcpy(p + i * ld, pSrc + i * n, n * sizeof(unsigned int));
            }
        }
        // aik values for this thread's rows (stage k-1 pivot-column values)
        std::vector<unsigned int> aikBuf(static_cast<size_t>(myGroups) * RB * KBLOCK);
        #pragma omp barrier

        for (long kb = 0; kb < n; kb += KBLOCK) {
            const long ke = std::min(kb + KBLOCK, n);
            const long kn = ke - kb;

            // Step 0: diagonal block, sequential.
            #pragma omp single
            {
                for (long k = kb; k < ke; ++k) {
                    const unsigned int* rowK = d + k * ld;
                    for (long j = kb; j < ke; ++j) S[(k - kb) * ld + j] = rowK[j];
                    for (long i = kb; i < ke; ++i) {
                        unsigned int* rowI = d + i * ld;
                        const unsigned int distIK = rowI[k];
                        A[i - kb][k - kb] = distIK;
                        relaxSegment(rowI, p + i * ld, S + (k - kb) * ld, distIK,
                                     static_cast<unsigned int>(k), kb, ke);
                    }
                }
            } // implicit barrier

            // Step 1: pivot rows, remaining columns; parallel over column tiles.
            {
                const long rest = n - kn;
                const long ntiles = (rest + CTILE - 1) / CTILE;
                #pragma omp for schedule(static)
                for (long t = 0; t < ntiles; ++t) {
                    const long c0 = t * CTILE;
                    const long c1 = std::min(c0 + CTILE, rest);
                    // Map [0, rest) onto columns outside [kb, ke)
                    for (int seg = 0; seg < 2; ++seg) {
                        const long j0 = seg == 0 ? c0 : std::max(c0, kb) + kn;
                        const long j1 = seg == 0 ? std::min(c1, kb) : c1 + kn;
                        if (j0 >= j1) continue;
                        for (long k = kb; k < ke; ++k) {
                            const unsigned int* rowK = d + k * ld;
                            unsigned int* sK = S + (k - kb) * ld;
                            for (long j = j0; j < j1; ++j) sK[j] = rowK[j];
                            for (long i = kb; i < ke; ++i) {
                                relaxSegment(d + i * ld, p + i * ld, sK, A[i - kb][k - kb],
                                             static_cast<unsigned int>(k), j0, j1);
                            }
                        }
                    }
                }
            } // implicit barrier

            // Step 2: all non-pivot rows owned by this thread.
            // 2a: pivot columns first, recording dist[i][k] at stage k-1.
            for (long G = tid, L = 0; G < numGroups; G += nth, ++L) {
                for (long i = G * RB; i < std::min(G * RB + RB, n); ++i) {
                    if (i >= kb && i < ke) continue;
                    unsigned int* rowI = d + i * ld;
                    unsigned int* pathI = p + i * ld;
                    unsigned int* a = aikBuf.data() + (L * RB + (i - G * RB)) * KBLOCK;
                    for (long k = kb; k < ke; ++k) {
                        a[k - kb] = rowI[k];
                        relaxSegment(rowI, pathI, S + (k - kb) * ld, rowI[k],
                                     static_cast<unsigned int>(k), kb, ke);
                    }
                }
            }

            // 2b: remaining columns, tiled over columns, register-blocked.
            for (long t0 = 0; t0 < n; t0 += JTILE) {
                const long t1 = std::min(t0 + JTILE, n);
                for (long G = tid, L = 0; G < numGroups; G += nth, ++L) {
                    const long g = G * RB;
                    const long gEnd = std::min(g + RB, n);
                    const unsigned int* aG = aikBuf.data() + L * RB * KBLOCK;
                    const bool fullGroup = (gEnd - g == RB) && (gEnd <= kb || g >= ke);
                    for (int seg = 0; seg < 2; ++seg) {
                        const long j0 = seg == 0 ? t0 : std::max(t0, ke);
                        const long j1 = seg == 0 ? std::min(t1, kb) : t1;
                        if (j0 >= j1) continue;
                        long jv = j0;
                        if (fullGroup) {
                            unsigned int* rows[RB];
                            unsigned int* paths[RB];
                            for (long r = 0; r < RB; ++r) {
                                rows[r] = d + (g + r) * ld;
                                paths[r] = p + (g + r) * ld;
                            }
                            for (; jv + VW <= j1; jv += VW) {
                                relaxRegisterBlock(rows, paths, S, ld, aG, kb, kn, jv);
                            }
                        }
                        // Remainder (column tail, partial groups, groups with pivot rows)
                        if (jv < j1) {
                            for (long i = g; i < gEnd; ++i) {
                                if (i >= kb && i < ke) continue;
                                const unsigned int* a = aG + (i - g) * KBLOCK;
                                for (long k = kb; k < ke; ++k) {
                                    relaxSegment(d + i * ld, p + i * ld, S + (k - kb) * ld,
                                                 a[k - kb], static_cast<unsigned int>(k), jv, j1);
                                }
                            }
                        }
                    }
                }
            }
            #pragma omp barrier
        }

        // Copy results back
        for (long G = tid; G < numGroups; G += nth) {
            for (long i = G * RB; i < std::min(G * RB + RB, n); ++i) {
                std::memcpy(dSrc + i * n, d + i * ld, n * sizeof(unsigned int));
                std::memcpy(pSrc + i * n, p + i * ld, n * sizeof(unsigned int));
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
