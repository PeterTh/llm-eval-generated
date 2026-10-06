#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

// Pin each OpenMP thread to one CPU of the process affinity mask (in Linux CPU
// numbering, i.e. physical cores first, then SMT siblings). Unbound threads migrate
// and lose their cache-resident tiles, which badly hurts the blocked kernel. Skipped
// if the user controls binding through OMP_PLACES / OMP_PROC_BIND.
void bindThreadsToCpus() {
    if (getenv("OMP_PLACES") || getenv("OMP_PROC_BIND")) return;
    cpu_set_t allowed;
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;
    std::vector<int> cpus;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed)) cpus.push_back(c);
    }
    if (cpus.empty()) return;
    #pragma omp parallel
    {
        const int t = omp_get_thread_num();
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[t % cpus.size()], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // The original sequential loop leaves path[row j][col i] = j for every entry
    // (both of its writes store the row index). Parallel by rows; this also
    // warms up the OpenMP thread pool before the timed region.
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
        }
    }
}

// Relax one row segment through intermediate node k:
//   rowI[j] = min(rowI[j], distIK + rowK[j]), recording k in pathI on improvement.
// Requires rowI != rowK (the i == k case never changes anything since dist[k][k] == 0).
static inline void relaxRow(unsigned int* __restrict rowI,
                            unsigned int* __restrict pathI,
                            const unsigned int* __restrict rowK,
                            const unsigned int distIK,
                            const unsigned int k,
                            const size_t len) noexcept {
#pragma omp simd
    for (size_t j = 0; j < len; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const bool better = newDist < rowI[j];
        rowI[j] = better ? newDist : rowI[j];
        pathI[j] = better ? k : pathI[j];
    }
}

// Relax a REGCHUNK-wide row segment through nk consecutive intermediates
// k0..k0+nk-1, keeping the segment in registers across all k.
constexpr size_t REGCHUNK = 64;
static inline void relaxRowRegs(unsigned int* __restrict rowI,
                                unsigned int* __restrict pathI,
                                const unsigned int* __restrict rowK0,
                                const unsigned int* __restrict distIK,
                                const size_t k0, const size_t nk,
                                const size_t n) noexcept {
    unsigned int d[REGCHUNK], p[REGCHUNK];
#pragma omp simd
    for (size_t j = 0; j < REGCHUNK; ++j) {
        d[j] = rowI[j];
        p[j] = pathI[j];
    }
    for (size_t kk = 0; kk < nk; ++kk) {
        const unsigned int dik = distIK[kk];
        const unsigned int k = (unsigned int)(k0 + kk);
        const unsigned int* rowK = rowK0 + kk * n;
#pragma omp simd
        for (size_t j = 0; j < REGCHUNK; ++j) {
            const unsigned int newDist = dik + rowK[j];
            const bool better = newDist < d[j];
            d[j] = better ? newDist : d[j];
            p[j] = better ? k : p[j];
        }
    }
#pragma omp simd
    for (size_t j = 0; j < REGCHUNK; ++j) {
        rowI[j] = d[j];
        pathI[j] = p[j];
    }
}

// Blocked (tiled) Floyd-Warshall, parallelized with OpenMP.
// Per block round kb: (1) the diagonal tile, (2) tiles in pivot row/column,
// (3) all remaining tiles. Produces the same shortest distances as the classic
// triple loop.
void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    const size_t n = numNodes;
    if (n == 0) return;
    constexpr size_t BS = 64;           // tile size
    constexpr size_t COLCHUNK = 32;     // column chunk width for pivot-row phase
    static_assert(BS % REGCHUNK == 0, "tile size must be a multiple of REGCHUNK");
    const size_t numBlocks = (n + BS - 1) / BS;
    unsigned int* const D = dist.data();
    unsigned int* const P = path.data();

    #pragma omp parallel default(none) shared(D, P) firstprivate(n, numBlocks)
    {
        for (size_t kb = 0; kb < numBlocks; ++kb) {
            const size_t k0 = kb * BS;
            const size_t k1 = std::min(k0 + BS, n);

            // Phase 1: diagonal tile (small, sequential dependency on k)
            #pragma omp single
            {
                for (size_t k = k0; k < k1; ++k) {
                    const unsigned int* rowK = D + k * n + k0;
                    for (size_t i = k0; i < k1; ++i) {
                        if (i == k) continue;
                        relaxRow(D + i * n + k0, P + i * n + k0, rowK,
                                 D[i * n + k], (unsigned int)k, k1 - k0);
                    }
                }
            }

            // Phase 2: pivot row (rows k0..k1, all other columns) and
            // pivot column (all other rows, columns k0..k1).
            const size_t numColChunks = (n + COLCHUNK - 1) / COLCHUNK;
            #pragma omp for schedule(dynamic, 1) nowait
            for (size_t c = 0; c < numColChunks; ++c) {
                const size_t j0 = c * COLCHUNK;
                const size_t j1 = std::min(j0 + COLCHUNK, n);
                if (j1 <= k0 || j0 >= k1) {
                    for (size_t k = k0; k < k1; ++k) {
                        const unsigned int* rowK = D + k * n + j0;
                        for (size_t i = k0; i < k1; ++i) {
                            if (i == k) continue;
                            relaxRow(D + i * n + j0, P + i * n + j0, rowK,
                                     D[i * n + k], (unsigned int)k, j1 - j0);
                        }
                    }
                } else {
                    // Chunk overlaps the diagonal tile: process only parts outside it
                    const size_t a1 = std::min(j1, k0);
                    const size_t b0 = std::max(j0, k1);
                    for (size_t k = k0; k < k1; ++k) {
                        for (size_t i = k0; i < k1; ++i) {
                            if (i == k) continue;
                            const unsigned int dik = D[i * n + k];
                            if (a1 > j0)
                                relaxRow(D + i * n + j0, P + i * n + j0, D + k * n + j0,
                                         dik, (unsigned int)k, a1 - j0);
                            if (j1 > b0)
                                relaxRow(D + i * n + b0, P + i * n + b0, D + k * n + b0,
                                         dik, (unsigned int)k, j1 - b0);
                        }
                    }
                }
            }
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                if (i >= k0 && i < k1) continue;
                unsigned int* rowI = D + i * n + k0;
                unsigned int* pathI = P + i * n + k0;
                for (size_t k = k0; k < k1; ++k) {
                    relaxRow(rowI, pathI, D + k * n + k0, rowI[k - k0],
                             (unsigned int)k, k1 - k0);
                }
            }

            // Phase 3: remaining tiles. Each (row i, column tile jb) is independent;
            // rows of the pivot block row and the pivot column tile are fixed now.
            #pragma omp for collapse(2) schedule(static)
            for (size_t jb = 0; jb < numBlocks; ++jb) {
                for (size_t i = 0; i < n; ++i) {
                    if (jb == kb || (i >= k0 && i < k1)) continue;
                    const size_t j0 = jb * BS;
                    const size_t len = std::min(j0 + BS, n) - j0;
                    unsigned int* rowI = D + i * n + j0;
                    unsigned int* pathI = P + i * n + j0;
                    const unsigned int* dIK = D + i * n;
                    if (len == BS) {
                        for (size_t c = 0; c < BS; c += REGCHUNK) {
                            relaxRowRegs(rowI + c, pathI + c, D + k0 * n + j0 + c,
                                         dIK + k0, k0, k1 - k0, n);
                        }
                    } else {
                        for (size_t k = k0; k < k1; ++k) {
                            relaxRow(rowI, pathI, D + k * n + j0, dIK[k],
                                     (unsigned int)k, len);
                        }
                    }
                }
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
    bindThreadsToCpus();
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
