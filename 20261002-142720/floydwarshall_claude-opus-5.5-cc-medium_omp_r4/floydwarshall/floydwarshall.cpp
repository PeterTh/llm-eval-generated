#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    // Equivalent to the original sequential double loop: the last write to
    // path[idx2(c, r, n)] always stores the row index r.
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < numNodes; ++r) {
        for (size_t c = 0; c < numNodes; ++c) {
            path[idx2(c, r, numNodes)] = r;
        }
    }
}

// Tile size for the blocked Floyd-Warshall and sub-tile size used for the
// (serial) pivot tile and for splitting the pivot row/column into tasks.
constexpr size_t BS = 128;
constexpr size_t SB = 32;

// All kernels apply relax(i, j, k): if (d[i][k] + d[k][j] < d[i][j]) then
// d[i][j] = d[i][k] + d[k][j] and path[i][j] = k, with k processed in increasing
// order for every element. Since dist[k][k] == 0 and weights are non-negative,
// row k and column k of the pivot are invariant during step k, which makes the
// in-place variants safe.

// C (M x W) independent from A (M x K) and B (K x W). Register-blocked.
template <size_t M, size_t W, size_t K>
static inline void kernelIndep(unsigned int* __restrict C, unsigned int* __restrict P,
                               const unsigned int* __restrict A,
                               const unsigned int* __restrict B,
                               const size_t ld, const unsigned int kBase) {
    constexpr size_t RI = 4;
    constexpr size_t JW = 16;
    static_assert(M % RI == 0 && W % JW == 0, "tile shape");
    for (size_t i0 = 0; i0 < M; i0 += RI) {
        for (size_t j0 = 0; j0 < W; j0 += JW) {
            unsigned int c[RI][JW];
            unsigned int p[RI][JW];
            for (size_t r = 0; r < RI; ++r) {
                for (size_t j = 0; j < JW; ++j) {
                    c[r][j] = C[(i0 + r) * ld + j0 + j];
                    p[r][j] = P[(i0 + r) * ld + j0 + j];
                }
            }
            for (size_t k = 0; k < K; ++k) {
                const unsigned int kk = kBase + static_cast<unsigned int>(k);
                const unsigned int* bRow = B + k * ld + j0;
                for (size_t r = 0; r < RI; ++r) {
                    const unsigned int aik = A[(i0 + r) * ld + k];
                    #pragma omp simd
                    for (size_t j = 0; j < JW; ++j) {
                        const unsigned int nd = aik + bRow[j];
                        const bool lt = nd < c[r][j];
                        c[r][j] = lt ? nd : c[r][j];
                        p[r][j] = lt ? kk : p[r][j];
                    }
                }
            }
            for (size_t r = 0; r < RI; ++r) {
                for (size_t j = 0; j < JW; ++j) {
                    C[(i0 + r) * ld + j0 + j] = c[r][j];
                    P[(i0 + r) * ld + j0 + j] = p[r][j];
                }
            }
        }
    }
}

// Pivot-row type: C (K x W) with B == C (rows k of C itself), A (K x K) final
// pivot. Columns of C are independent, so callers may split C into strips.
template <size_t K, size_t W>
static inline void kernelRowDep(unsigned int* C, unsigned int* P, const unsigned int* A,
                                const size_t ld, const unsigned int kBase) {
    for (size_t k = 0; k < K; ++k) {
        const unsigned int kk = kBase + static_cast<unsigned int>(k);
        alignas(64) unsigned int b[W];
        for (size_t j = 0; j < W; ++j) b[j] = C[k * ld + j];
        for (size_t i = 0; i < K; ++i) {
            const unsigned int aik = A[i * ld + k];
            unsigned int* cRow = C + i * ld;
            unsigned int* pRow = P + i * ld;
            #pragma omp simd
            for (size_t j = 0; j < W; ++j) {
                const unsigned int nd = aik + b[j];
                const bool lt = nd < cRow[j];
                cRow[j] = lt ? nd : cRow[j];
                pRow[j] = lt ? kk : pRow[j];
            }
        }
    }
}

// Pivot-column type: C (M x K) with A == C (columns k of C itself), B (K x K)
// final pivot. Rows of C are independent, so callers may split C into row groups.
template <size_t M, size_t K>
static inline void kernelColDep(unsigned int* C, unsigned int* P, const unsigned int* B,
                                const size_t ld, const unsigned int kBase) {
    for (size_t i = 0; i < M; ++i) {
        alignas(64) unsigned int c[K];
        alignas(64) unsigned int p[K];
        unsigned int* cRow = C + i * ld;
        unsigned int* pRow = P + i * ld;
        for (size_t j = 0; j < K; ++j) { c[j] = cRow[j]; p[j] = pRow[j]; }
        for (size_t k = 0; k < K; ++k) {
            const unsigned int kk = kBase + static_cast<unsigned int>(k);
            const unsigned int cik = c[k];
            const unsigned int* bRow = B + k * ld;
            #pragma omp simd
            for (size_t j = 0; j < K; ++j) {
                const unsigned int nd = cik + bRow[j];
                const bool lt = nd < c[j];
                c[j] = lt ? nd : c[j];
                p[j] = lt ? kk : p[j];
            }
        }
        for (size_t j = 0; j < K; ++j) { cRow[j] = c[j]; pRow[j] = p[j]; }
    }
}

// Classic in-place Floyd-Warshall on an S x S tile.
template <size_t S>
static inline void kernelPivot(unsigned int* C, unsigned int* P,
                               const size_t ld, const unsigned int kBase) {
    for (size_t k = 0; k < S; ++k) {
        const unsigned int kk = kBase + static_cast<unsigned int>(k);
        alignas(64) unsigned int b[S];
        for (size_t j = 0; j < S; ++j) b[j] = C[k * ld + j];
        for (size_t i = 0; i < S; ++i) {
            const unsigned int aik = C[i * ld + k];
            unsigned int* cRow = C + i * ld;
            unsigned int* pRow = P + i * ld;
            #pragma omp simd
            for (size_t j = 0; j < S; ++j) {
                const unsigned int nd = aik + b[j];
                const bool lt = nd < cRow[j];
                cRow[j] = lt ? nd : cRow[j];
                pRow[j] = lt ? kk : pRow[j];
            }
        }
    }
}

// Blocked Floyd-Warshall of one BS x BS pivot tile using SB x SB sub-tiles.
static void pivotTile(unsigned int* C, unsigned int* P, const size_t ld, const unsigned int kBase) {
    constexpr size_t ns = BS / SB;
    for (size_t s = 0; s < ns; ++s) {
        const unsigned int kb = kBase + static_cast<unsigned int>(s * SB);
        unsigned int* Css = C + s * SB * ld + s * SB;
        unsigned int* Pss = P + s * SB * ld + s * SB;
        kernelPivot<SB>(Css, Pss, ld, kb);
        for (size_t t = 0; t < ns; ++t) {
            if (t == s) continue;
            kernelRowDep<SB, SB>(C + s * SB * ld + t * SB, P + s * SB * ld + t * SB, Css, ld, kb);
            kernelColDep<SB, SB>(C + t * SB * ld + s * SB, P + t * SB * ld + s * SB, Css, ld, kb);
        }
        for (size_t u = 0; u < ns; ++u) {
            if (u == s) continue;
            for (size_t t = 0; t < ns; ++t) {
                if (t == s) continue;
                kernelIndep<SB, SB, SB>(C + u * SB * ld + t * SB, P + u * SB * ld + t * SB,
                                        C + u * SB * ld + s * SB, C + s * SB * ld + t * SB,
                                        ld, kb);
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Blocked Floyd-Warshall. Row-major view: element (i, j) lives at
    // idx2(j, i, n) = i * n + j. The matrix is padded to a multiple of BS with
    // unreachable (INF) dummy nodes, which never improve any real path.
    const size_t n = numNodes;
    if (n == 0) return;
    const size_t nb = (n + BS - 1) / BS;
    const size_t N = nb * BS;
    // Leading dimension padded by one cache line so that consecutive rows of a
    // tile map to different cache sets (avoids power-of-two stride conflicts).
    const size_t ld = N + 16;

    unsigned int* D = static_cast<unsigned int*>(std::aligned_alloc(64, N * ld * sizeof(unsigned int)));
    unsigned int* P = static_cast<unsigned int*>(std::aligned_alloc(64, N * ld * sizeof(unsigned int)));

    // Do not launch more threads than there are tiles to work on
    const size_t nOther = nb - 1;
    constexpr size_t parts = BS / SB;
    const size_t maxWork = std::max<size_t>({nOther * nOther, 2 * nOther * parts, 1});
    const int numThreads = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(omp_get_max_threads()), maxWork));

    #pragma omp parallel num_threads(numThreads)
    {
        // Copy in (first touch distributed across threads)
        #pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            unsigned int* dRow = D + i * ld;
            unsigned int* pRow = P + i * ld;
            if (i < n) {
                std::memcpy(dRow, dist.data() + i * n, n * sizeof(unsigned int));
                std::memcpy(pRow, path.data() + i * n, n * sizeof(unsigned int));
                for (size_t j = n; j < N; ++j) { dRow[j] = INF; pRow[j] = 0; }
            } else {
                for (size_t j = 0; j < N; ++j) { dRow[j] = INF; pRow[j] = 0; }
                dRow[i] = 0;
            }
        }

        for (size_t kb = 0; kb < nb; ++kb) {
            const unsigned int kBase = static_cast<unsigned int>(kb * BS);
            unsigned int* Dkk = D + kb * BS * ld + kb * BS;
            unsigned int* Pkk = P + kb * BS * ld + kb * BS;

            // Phase 1: pivot tile
            #pragma omp single
            pivotTile(Dkk, Pkk, ld, kBase);

            // Phase 2: pivot row tiles (split into column strips) and pivot
            // column tiles (split into row groups)
            #pragma omp for schedule(dynamic, 1)
            for (size_t t = 0; t < 2 * nOther * parts; ++t) {
                const size_t part = t % parts;
                const size_t tile = t / parts;
                size_t o = tile % nOther;
                if (o >= kb) ++o;
                if (tile < nOther) {
                    // Row tile (kb, o), columns [part*SB, (part+1)*SB)
                    const size_t off = kb * BS * ld + o * BS + part * SB;
                    kernelRowDep<BS, SB>(D + off, P + off, Dkk, ld, kBase);
                } else {
                    // Column tile (o, kb), rows [part*SB, (part+1)*SB)
                    const size_t off = (o * BS + part * SB) * ld + kb * BS;
                    kernelColDep<SB, BS>(D + off, P + off, Dkk, ld, kBase);
                }
            }

            // Phase 3: all remaining tiles
            #pragma omp for schedule(static)
            for (size_t t = 0; t < nOther * nOther; ++t) {
                size_t ib = t / nOther;
                size_t jb = t % nOther;
                if (ib >= kb) ++ib;
                if (jb >= kb) ++jb;
                kernelIndep<BS, BS, BS>(D + ib * BS * ld + jb * BS,
                                        P + ib * BS * ld + jb * BS,
                                        D + ib * BS * ld + kb * BS,
                                        D + kb * BS * ld + jb * BS,
                                        ld, kBase);
            }
        }

        // Copy out
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            std::memcpy(dist.data() + i * n, D + i * ld, n * sizeof(unsigned int));
            std::memcpy(path.data() + i * n, P + i * ld, n * sizeof(unsigned int));
        }
    }

    std::free(D);
    std::free(P);
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
