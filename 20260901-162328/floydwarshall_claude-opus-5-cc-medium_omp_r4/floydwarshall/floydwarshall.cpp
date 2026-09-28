#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
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

// Pin the OpenMP threads, one per physical core first and only then to the SMT
// siblings. Binding matters a lot here: the block steps are separated by
// barriers, and threads that the scheduler migrates make every barrier slow.
// An explicit thread affinity configuration of the user always wins (the
// proc_bind clauses of the parallel regions are only honoured if the OpenMP
// runtime has thread affinity enabled, which is not the default).
static void bindThreads() {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    // CPUs that are the first SMT sibling of their core come first.
    std::vector<int> cores;
    std::vector<int> siblings;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }

        char fileName[128];
        snprintf(fileName, sizeof(fileName),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        int firstSibling = cpu;
        if (FILE* f = fopen(fileName, "r")) {
            if (fscanf(f, "%d", &firstSibling) != 1) {
                firstSibling = cpu;
            }
            fclose(f);
        }

        if (firstSibling == cpu) {
            cores.push_back(cpu);
        } else {
            siblings.push_back(cpu);
        }
    }
    cores.insert(cores.end(), siblings.begin(), siblings.end());
    if (cores.empty()) {
        return;
    }

    #pragma omp parallel
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        CPU_SET(cores[(size_t)omp_get_thread_num() % cores.size()], &mask);
        sched_setaffinity(0, sizeof(mask), &mask);
    }
}

// The kernel accesses the matrices as rows of length numNodes (dist[idx2(j, i, n)]
// is element j of row i). Rows are distributed over the threads with a static
// schedule so that every thread keeps working on the same rows for all k
// iterations, which maximizes cache reuse and (together with the parallel
// first-touch initialization below) keeps the data NUMA-local.
static void firstTouch(unsigned int* __restrict data, const size_t numNodes) {
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t i = 0; i < numNodes; ++i) {
        unsigned int* __restrict row = data + i * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = 0;
        }
    }
}

void initializeDistanceMatrix(unsigned int* __restrict dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Kept sequential: rand_r carries a serial dependency through its seed, so
    // this reproduces the exact same graph as the original code.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(unsigned int* __restrict path, const size_t numNodes) {
    // The original nested loops assign path[idx2(i, j)] = j and path[idx2(j, i)] = i,
    // i.e. every element of the row starting at j * numNodes ends up holding j.
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t j = 0; j < numNodes; ++j) {
        unsigned int* __restrict row = path + j * numNodes;
        const unsigned int value = (unsigned int)j;
        for (size_t i = 0; i < numNodes; ++i) {
            row[i] = value;
        }
    }
}

// Block size of the tiled Floyd-Warshall. A tile of BLOCK x BLOCK unsigned ints
// is 16 KiB, so the three tiles a block update touches stay resident in L2.
constexpr size_t BLOCK = 64;

// Width of the j chunk that is kept in vector registers by the phase 3 kernel.
constexpr size_t CHUNK = 32;
static_assert(BLOCK % CHUNK == 0, "a full tile row must be a whole number of chunks");

// Relax one row against the row of the intermediate node (branch-free so that it
// vectorizes into compares plus blends). Rows i and k may be the same row, but
// then dist[k][k] == 0 makes every update a no-op, so the iterations of the loop
// stay independent.
#define FW_ROW_UPDATE(distI, pathI, distK, distIK, kVal, jLen)          \
    _Pragma("omp simd")                                                 \
    for (size_t j = 0; j < (jLen); ++j) {                               \
        const unsigned int distIJ = (distI)[j];                         \
        const unsigned int newDist = (distIK) + (distK)[j];             \
        const bool better = newDist < distIJ;                           \
        (distI)[j] = better ? newDist : distIJ;                         \
        (pathI)[j] = better ? (kVal) : (pathI)[j];                      \
    }

// The three tile shapes of a block step. The diagonal tile depends on itself,
// the row panel additionally reads the diagonal tile as source of dist[i][k],
// and the column panel reads it as source of dist[k][j].
enum TileKind { TILE_DIAGONAL, TILE_ROW_PANEL, TILE_COLUMN_PANEL };

// Kernel for the tiles of phase 1 and 2, which overlap with the diagonal tile
// and therefore have to keep the k loop outermost. The tiles are copied into
// contiguous scratch buffers first, because in the matrix itself the rows of a
// tile are a multiple of 4 KiB apart for the usual matrix sizes and would all
// map onto the same L1 cache sets. JLEN is the compile time width of the tile
// (0 selects the runtime width used for the remainder tiles of a matrix whose
// size is not a multiple of BLOCK).
template <TileKind KIND, size_t JLEN>
static void floydWarshallPanelTile(unsigned int* __restrict dist, unsigned int* __restrict path,
                                   const size_t numNodes,
                                   const size_t iStart, const size_t iEnd,
                                   const size_t jStart, const size_t jEnd,
                                   const size_t kStart, const size_t kEnd) {
    const size_t iLen = iEnd - iStart;
    const size_t jLen = JLEN != 0 ? JLEN : jEnd - jStart;
    const size_t kLen = kEnd - kStart;

    alignas(64) unsigned int tileDist[BLOCK * BLOCK];
    alignas(64) unsigned int tilePath[BLOCK * BLOCK];
    alignas(64) unsigned int tileOther[BLOCK * BLOCK];

    for (size_t ii = 0; ii < iLen; ++ii) {
        const unsigned int* __restrict srcDist = dist + (iStart + ii) * numNodes + jStart;
        const unsigned int* __restrict srcPath = path + (iStart + ii) * numNodes + jStart;
        #pragma omp simd
        for (size_t j = 0; j < jLen; ++j) {
            tileDist[ii * BLOCK + j] = srcDist[j];
            tilePath[ii * BLOCK + j] = srcPath[j];
        }
    }

    if (KIND == TILE_ROW_PANEL) {
        // dist[i][k] comes from the diagonal tile (rows iStart.., columns kStart..)
        for (size_t ii = 0; ii < iLen; ++ii) {
            const unsigned int* __restrict src = dist + (iStart + ii) * numNodes + kStart;
            #pragma omp simd
            for (size_t k = 0; k < kLen; ++k) {
                tileOther[ii * BLOCK + k] = src[k];
            }
        }
    } else if (KIND == TILE_COLUMN_PANEL) {
        // dist[k][j] comes from the diagonal tile (rows kStart.., columns jStart..)
        for (size_t kk = 0; kk < kLen; ++kk) {
            const unsigned int* __restrict src = dist + (kStart + kk) * numNodes + jStart;
            #pragma omp simd
            for (size_t j = 0; j < jLen; ++j) {
                tileOther[kk * BLOCK + j] = src[j];
            }
        }
    }

    for (size_t kk = 0; kk < kLen; ++kk) {
        const unsigned int* __restrict distK =
            (KIND == TILE_COLUMN_PANEL ? tileOther : tileDist) + kk * BLOCK;
        const unsigned int kVal = (unsigned int)(kStart + kk);

        for (size_t ii = 0; ii < iLen; ++ii) {
            unsigned int* __restrict distI = tileDist + ii * BLOCK;
            unsigned int* __restrict pathI = tilePath + ii * BLOCK;
            const unsigned int distIK =
                (KIND == TILE_ROW_PANEL ? tileOther : tileDist)[ii * BLOCK + kk];

            FW_ROW_UPDATE(distI, pathI, distK, distIK, kVal, jLen)
        }
    }

    for (size_t ii = 0; ii < iLen; ++ii) {
        unsigned int* __restrict dstDist = dist + (iStart + ii) * numNodes + jStart;
        unsigned int* __restrict dstPath = path + (iStart + ii) * numNodes + jStart;
        #pragma omp simd
        for (size_t j = 0; j < jLen; ++j) {
            dstDist[j] = tileDist[ii * BLOCK + j];
            dstPath[j] = tilePath[ii * BLOCK + j];
        }
    }
}

// Kernel for the independent tiles of phase 3, where the three participating
// tiles (i,j), (i,k) and (k,j) do not overlap. Because every pair (i, j) is then
// relaxed independently, the loops can be reordered with i outermost, which
// keeps the target row in L1 while it is relaxed against the whole k block.
// The (k,j) tile is packed into a contiguous buffer first: reading it in place
// would walk the matrix with a row stride that is a multiple of 4 KiB for the
// usual matrix sizes and therefore maps every row onto the same L1 cache sets.
template <size_t JLEN>
static inline void floydWarshallBlockTile(unsigned int* __restrict dist, unsigned int* __restrict path,
                                          const size_t numNodes,
                                          const size_t iStart, const size_t iEnd,
                                          const size_t jStart, const size_t jEnd,
                                          const size_t kStart, const size_t kEnd) {
    const size_t jLen = JLEN != 0 ? JLEN : jEnd - jStart;
    const size_t kLen = kEnd - kStart;

    alignas(64) unsigned int distKPacked[BLOCK * BLOCK];
    for (size_t kk = 0; kk < kLen; ++kk) {
        const unsigned int* __restrict src = dist + (kStart + kk) * numNodes + jStart;
        unsigned int* __restrict dstRow = distKPacked + kk * jLen;
        #pragma omp simd
        for (size_t j = 0; j < jLen; ++j) {
            dstRow[j] = src[j];
        }
    }

    for (size_t i = iStart; i < iEnd; ++i) {
        unsigned int* __restrict distI = dist + i * numNodes + jStart;
        unsigned int* __restrict pathI = path + i * numNodes + jStart;
        const unsigned int* __restrict distIK = dist + i * numNodes + kStart;

        if (JLEN != 0) {
            // The row is processed in chunks that are small enough to stay in
            // vector registers while the whole k block is applied to them, so
            // the k loop only streams the packed (k,j) tile.
            for (size_t jc = 0; jc < jLen; jc += CHUNK) {
                unsigned int distChunk[CHUNK];
                unsigned int pathChunk[CHUNK];
                #pragma omp simd
                for (size_t t = 0; t < CHUNK; ++t) {
                    distChunk[t] = distI[jc + t];
                    pathChunk[t] = pathI[jc + t];
                }

                for (size_t kk = 0; kk < kLen; ++kk) {
                    const unsigned int* __restrict distK = distKPacked + kk * jLen + jc;
                    const unsigned int dik = distIK[kk];
                    const unsigned int kVal = (unsigned int)(kStart + kk);

                    #pragma omp simd
                    for (size_t t = 0; t < CHUNK; ++t) {
                        const unsigned int distIJ = distChunk[t];
                        const unsigned int newDist = dik + distK[t];
                        const bool better = newDist < distIJ;

                        distChunk[t] = better ? newDist : distIJ;
                        pathChunk[t] = better ? kVal : pathChunk[t];
                    }
                }

                #pragma omp simd
                for (size_t t = 0; t < CHUNK; ++t) {
                    distI[jc + t] = distChunk[t];
                    pathI[jc + t] = pathChunk[t];
                }
            }
        } else {
            for (size_t kk = 0; kk < kLen; ++kk) {
                const unsigned int* __restrict distK = distKPacked + kk * jLen;
                const unsigned int dik = distIK[kk];
                const unsigned int kVal = (unsigned int)(kStart + kk);

                FW_ROW_UPDATE(distI, pathI, distK, dik, kVal, jLen)
            }
        }
    }
}

// Dispatch to the compile time sized kernel whenever the tile is a full block.
template <TileKind KIND>
static inline void floydWarshallPanelTileDispatch(unsigned int* __restrict dist, unsigned int* __restrict path,
                                                  const size_t numNodes,
                                                  const size_t iStart, const size_t iEnd,
                                                  const size_t jStart, const size_t jEnd,
                                                  const size_t kStart, const size_t kEnd) {
    if (jEnd - jStart == BLOCK) {
        floydWarshallPanelTile<KIND, BLOCK>(dist, path, numNodes, iStart, iEnd, jStart, jEnd, kStart, kEnd);
    } else {
        floydWarshallPanelTile<KIND, 0>(dist, path, numNodes, iStart, iEnd, jStart, jEnd, kStart, kEnd);
    }
}

static inline void floydWarshallBlockTileDispatch(unsigned int* __restrict dist, unsigned int* __restrict path,
                                                  const size_t numNodes,
                                                  const size_t iStart, const size_t iEnd,
                                                  const size_t jStart, const size_t jEnd,
                                                  const size_t kStart, const size_t kEnd) {
    if (jEnd - jStart == BLOCK) {
        floydWarshallBlockTile<BLOCK>(dist, path, numNodes, iStart, iEnd, jStart, jEnd, kStart, kEnd);
    } else {
        floydWarshallBlockTile<0>(dist, path, numNodes, iStart, iEnd, jStart, jEnd, kStart, kEnd);
    }
}

void floydWarshall(unsigned int* dist,
                   unsigned int* path,
                   const size_t numNodes) {
    // Blocked Floyd-Warshall: the k loop is split into blocks and every block
    // step consists of three phases (diagonal tile, the row/column panels of
    // the diagonal tile, and all remaining tiles). This computes the same
    // distance matrix as the classic triple loop, but reuses each tile BLOCK
    // times while it is in cache and needs only three barriers per block step
    // instead of one per k.
    const size_t numBlocks = (numNodes + BLOCK - 1) / BLOCK;

    #pragma omp parallel proc_bind(spread)
    {
        for (size_t kb = 0; kb < numBlocks; ++kb) {
            const size_t kStart = kb * BLOCK;
            const size_t kEnd = std::min(kStart + BLOCK, numNodes);

            // Phase 1: the diagonal tile depends on itself and is done serially.
            #pragma omp single
            {
                floydWarshallPanelTileDispatch<TILE_DIAGONAL>(dist, path, numNodes, kStart, kEnd, kStart, kEnd, kStart, kEnd);
            }

            // Phase 2: the tiles sharing rows or columns with the diagonal tile.
            #pragma omp for schedule(static)
            for (size_t t = 0; t < 2 * numBlocks; ++t) {
                const size_t b = t >= numBlocks ? t - numBlocks : t;
                if (b == kb) {
                    continue;
                }
                const size_t bStart = b * BLOCK;
                const size_t bEnd = std::min(bStart + BLOCK, numNodes);

                if (t < numBlocks) {
                    // Row panel: tile (kb, b)
                    floydWarshallPanelTileDispatch<TILE_ROW_PANEL>(dist, path, numNodes, kStart, kEnd, bStart, bEnd, kStart, kEnd);
                } else {
                    // Column panel: tile (b, kb)
                    floydWarshallPanelTileDispatch<TILE_COLUMN_PANEL>(dist, path, numNodes, bStart, bEnd, kStart, kEnd, kStart, kEnd);
                }
            }

            // Phase 3: all remaining tiles are independent.
            #pragma omp for collapse(2) schedule(static)
            for (size_t ib = 0; ib < numBlocks; ++ib) {
                for (size_t jb = 0; jb < numBlocks; ++jb) {
                    if (ib == kb || jb == kb) {
                        continue;
                    }
                    const size_t iStart = ib * BLOCK;
                    const size_t iEnd = std::min(iStart + BLOCK, numNodes);
                    const size_t jStart = jb * BLOCK;
                    const size_t jEnd = std::min(jStart + BLOCK, numNodes);

                    floydWarshallBlockTileDispatch(dist, path, numNodes, iStart, iEnd, jStart, jEnd, kStart, kEnd);
                }
            }
        }
    }
}

bool validateResult(const unsigned int* dist, const size_t numNodes) {
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
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    bindThreads();

    // Allocate matrices (uninitialized, so that the parallel first touch below
    // decides the NUMA placement of the pages)
    std::unique_ptr<unsigned int[]> distStorage(new unsigned int[numNodes * numNodes]);
    std::unique_ptr<unsigned int[]> pathStorage(new unsigned int[numNodes * numNodes]);
    unsigned int* dist = distStorage.get();
    unsigned int* path = pathStorage.get();
    firstTouch(dist, numNodes);
    firstTouch(path, numNodes);

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
        const std::vector<unsigned int> distVec(dist, dist + numNodes * numNodes);
        print_results_int(distVec, "DistanceMatrix");
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
