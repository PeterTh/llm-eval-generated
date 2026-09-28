#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <omp.h>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// The matrices are stored row-major with respect to the (i, j) access pattern of the algorithm:
// element (i, j) lives at i * numNodes + j, which is idx2(j, i, numNodes).

// ---------------------------------------------------------------------------------------------
// Machine setup: thread placement and page placement
// ---------------------------------------------------------------------------------------------

// Pin each OpenMP thread to its own hardware thread, spreading over sockets and preferring one
// thread per physical core before using SMT siblings. This is skipped when the environment
// already specifies a placement policy, so an explicit OMP_PROC_BIND/OMP_PLACES/GOMP_CPU_AFFINITY
// setting always wins.
static void bindThreadsToCores(const size_t numThreads) {
    if (getenv("OMP_PROC_BIND") != nullptr || getenv("OMP_PLACES") != nullptr ||
        getenv("GOMP_CPU_AFFINITY") != nullptr) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    struct CpuInfo {
        int cpu;
        int package;
        int core;
        int smt;  // rank of this hardware thread inside its physical core
    };

    std::vector<CpuInfo> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        CpuInfo info{cpu, 0, cpu, 0};
        char pathBuf[128];
        snprintf(pathBuf, sizeof(pathBuf),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        if (FILE* f = fopen(pathBuf, "r")) {
            if (fscanf(f, "%d", &info.package) != 1) {
                info.package = 0;
            }
            fclose(f);
        }
        snprintf(pathBuf, sizeof(pathBuf), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
        if (FILE* f = fopen(pathBuf, "r")) {
            if (fscanf(f, "%d", &info.core) != 1) {
                info.core = cpu;
            }
            fclose(f);
        }
        cpus.push_back(info);
    }
    if (cpus.empty()) {
        return;
    }

    // Rank the hardware threads inside each physical core, in ascending cpu id order.
    std::vector<CpuInfo> byCore = cpus;
    std::sort(byCore.begin(), byCore.end(), [](const CpuInfo& a, const CpuInfo& b) {
        return std::tie(a.package, a.core, a.cpu) < std::tie(b.package, b.core, b.cpu);
    });
    for (size_t i = 0, rank = 0; i < byCore.size(); ++i) {
        if (i > 0 && (byCore[i].package != byCore[i - 1].package ||
                      byCore[i].core != byCore[i - 1].core)) {
            rank = 0;
        } else if (i > 0) {
            ++rank;
        }
        byCore[i].smt = static_cast<int>(rank);
        for (CpuInfo& c : cpus) {
            if (c.cpu == byCore[i].cpu) {
                c.smt = byCore[i].smt;
            }
        }
    }

    // Group the hardware threads by physical core, keeping the cores of a package contiguous.
    std::vector<std::vector<int>> coreOrder;  // hardware threads of one core, smt order
    for (size_t i = 0; i < byCore.size(); ++i) {
        if (byCore[i].smt == 0) {
            coreOrder.emplace_back();
        }
        coreOrder.back().push_back(byCore[i].cpu);
    }

    // Spread the threads evenly over the cores; several threads on one core end up on its
    // distinct SMT siblings, and consecutive thread ids stay as close together as possible so
    // that neighbouring tiles of the static schedule share cache.
    const size_t numCores = coreOrder.size();
    std::vector<int> assignment(numThreads);
    for (size_t core = 0, thread = 0; core < numCores; ++core) {
        const size_t last = (core + 1) * numThreads / numCores;
        for (size_t sibling = 0; thread < last; ++thread, ++sibling) {
            assignment[thread] = coreOrder[core][sibling % coreOrder[core].size()];
        }
    }

#pragma omp parallel num_threads(numThreads)
    {
        const size_t id = static_cast<size_t>(omp_get_thread_num());
        if (id < assignment.size()) {
            cpu_set_t target;
            CPU_ZERO(&target);
            CPU_SET(assignment[id], &target);
            sched_setaffinity(0, sizeof(target), &target);
        }
    }
}

// Spread the pages of a matrix over the NUMA nodes of the machine via parallel first touch.
// The vector's own value-initialization already faulted every page onto the node of the calling
// thread, so the existing mapping is dropped first: private anonymous memory reads back as zero
// afterwards, which is exactly the state the freshly constructed vector was in.
static void firstTouchDistribute(unsigned int* const data, const size_t count) {
    const size_t bytes = count * sizeof(unsigned int);
    const long pageSize = sysconf(_SC_PAGESIZE);
    if (pageSize > 0 && bytes > static_cast<size_t>(pageSize)) {
        const uintptr_t base = reinterpret_cast<uintptr_t>(data);
        const uintptr_t pageMask = static_cast<uintptr_t>(pageSize) - 1;
        const uintptr_t start = (base + pageMask) & ~pageMask;
        const uintptr_t end = (base + bytes) & ~pageMask;
        if (end > start) {
            madvise(reinterpret_cast<void*>(start), end - start, MADV_DONTNEED);
        }
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < count; ++i) {
        data[i] = 0;
    }
}

// The kernel streams a row of `dist` and the matching row of `path` in lockstep. Both matrices
// are large, page-aligned allocations, so their rows tend to differ by an exact multiple of the
// L1 page-colouring period and then evict each other out of the same cache sets. Offsetting the
// path matrix by a few cache lines removes those conflicts (worth up to 7x on unlucky sizes).
constexpr size_t PATH_PAD_ELEMENTS = 1024;  // slack reserved at the end of the path allocation

static unsigned int* offsetPathBase(const unsigned int* dist, unsigned int* path) {
    constexpr uintptr_t period = 4096;
    constexpr uintptr_t target = 1088;  // 17 cache lines apart
    const uintptr_t delta = (reinterpret_cast<uintptr_t>(path) -
                             reinterpret_cast<uintptr_t>(dist)) % period;
    const uintptr_t shift = (target + period - delta) % period;
    return path + shift / sizeof(unsigned int);
}

// ---------------------------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------------------------

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Sequential: the values come from a single rand_r stream that has to be reproduced exactly.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(unsigned int* const path, const size_t numNodes) {
    // The original nested loops assign the row index to every entry of the matrix, diagonal
    // included; each row can therefore be filled independently.
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < numNodes; ++row) {
        unsigned int* const p = path + row * numNodes;
        const unsigned int value = static_cast<unsigned int>(row);
        for (size_t col = 0; col < numNodes; ++col) {
            p[col] = value;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Blocked Floyd-Warshall
// ---------------------------------------------------------------------------------------------

// Relaxation of one tile: c[i][j] = min(c[i][j], a[i][k] + b[k][j]) over the k values of the
// tile, recording the intermediate node in p. Requires the destination tile to be disjoint from
// the two source tiles, which allows the k loop to sit inside the i loop so that the destination
// row stays hot in L1 for the whole tile. The b tile is packed into a contiguous scratch buffer
// first: reading it at the full matrix stride causes cache-set conflicts between its rows.
static inline void tileRelaxDisjoint(unsigned int* __restrict c, unsigned int* __restrict p,
                                     const unsigned int* __restrict a,
                                     const unsigned int* __restrict bSource, const size_t n,
                                     const size_t kBase, const size_t ni, const size_t nj,
                                     const size_t nk, unsigned int* __restrict b) {
    for (size_t kk = 0; kk < nk; ++kk) {
        const unsigned int* const src = bSource + kk * n;
        unsigned int* const dst = b + kk * nj;
        for (size_t jj = 0; jj < nj; ++jj) {
            dst[jj] = src[jj];
        }
    }

    for (size_t ii = 0; ii < ni; ++ii) {
        unsigned int* const cRow = c + ii * n;
        unsigned int* const pRow = p + ii * n;
        const unsigned int* const aRow = a + ii * n;
        for (size_t kk = 0; kk < nk; ++kk) {
            const unsigned int aik = aRow[kk];
            const unsigned int* const bRow = b + kk * nj;
            const unsigned int kValue = static_cast<unsigned int>(kBase + kk);
#pragma omp simd
            for (size_t jj = 0; jj < nj; ++jj) {
                const unsigned int candidate = aik + bRow[jj];
                const unsigned int current = cRow[jj];
                const bool better = candidate < current;
                cRow[jj] = better ? candidate : current;
                pRow[jj] = better ? kValue : pRow[jj];
            }
        }
    }
}

// Same relaxation for tiles where c overlaps a and/or b (the pivot tile and the tiles of the
// pivot row and column). Here the k loop has to stay outermost, which is what makes the in-place
// update well defined: during step k neither c[i][k] nor c[k][j] can change, because c[k][k] is
// zero and therefore no candidate through k improves them.
static inline void tileRelaxInPlace(unsigned int* c, unsigned int* p, const unsigned int* a,
                                    const unsigned int* b, const size_t n, const size_t kBase,
                                    const size_t ni, const size_t nj, const size_t nk) {
    for (size_t kk = 0; kk < nk; ++kk) {
        const unsigned int* const bRow = b + kk * n;
        const unsigned int kValue = static_cast<unsigned int>(kBase + kk);
        for (size_t ii = 0; ii < ni; ++ii) {
            const unsigned int aik = a[ii * n + kk];
            unsigned int* const cRow = c + ii * n;
            unsigned int* const pRow = p + ii * n;
#pragma omp simd
            for (size_t jj = 0; jj < nj; ++jj) {
                const unsigned int candidate = aik + bRow[jj];
                const unsigned int current = cRow[jj];
                const bool better = candidate < current;
                cRow[jj] = better ? candidate : current;
                pRow[jj] = better ? kValue : pRow[jj];
            }
        }
    }
}

// Below a few million matrix entries the three barriers per round and the cross-socket traffic
// cost more than the extra cores bring in, so the thread count is scaled with the problem size.
static size_t chooseThreadCount(const size_t numNodes, const size_t maxThreads) {
    const size_t byWork = std::max<size_t>(32, numNodes * numNodes / 16384);
    return std::max<size_t>(1, std::min(maxThreads, byWork));
}

// Tile edge: large enough that three tiles of working set stay in the private caches while the
// memory traffic of the whole algorithm drops by a factor of the tile size, small enough that
// the independent tiles of the update phase keep every thread busy.
static size_t chooseTileSize(const size_t numNodes, const size_t numThreads) {
    size_t tile = 48;
    while (tile > 16) {
        const size_t nb = (numNodes + tile - 1) / tile;
        if (nb > 1 && (nb - 1) * (nb - 1) >= 2 * numThreads) {
            break;
        }
        tile /= 2;
    }
    return tile;
}

void floydWarshall(unsigned int* const dist, unsigned int* const path, const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const size_t n = numNodes;
    const size_t numThreads = static_cast<size_t>(omp_get_max_threads());
    const size_t tile = chooseTileSize(n, numThreads);
    const size_t numTiles = (n + tile - 1) / tile;

    // Blocked Floyd-Warshall. Round r applies the intermediate nodes of tile r in three dependent
    // phases; inside each phase all tiles are independent. The distance matrix this produces is
    // bit-identical to the one of the classic triple loop.
    //
    // The predecessor matrix is a valid witness for those distances - path[i][j] = k always
    // satisfies dist[i][j] == dist[i][k] + dist[k][j] - but it need not name the same k as the
    // sequential version when several shortest routes exist. The pivot row and column of a round
    // see the whole k range of that round before the update phase consumes them, so a pair (i, j)
    // can reach its final distance at an earlier k than it does in the unblocked order, and the
    // later k then no longer improves anything. This is inherent to every blocked formulation;
    // reproducing the exact predecessor choices requires the unblocked k order, which forfeits the
    // cache reuse that makes this kernel roughly an order of magnitude faster.
#pragma omp parallel
    {
        std::vector<unsigned int> packBuffer(tile * tile);
        unsigned int* const pack = packBuffer.data();

        for (size_t r = 0; r < numTiles; ++r) {
            const size_t k0 = r * tile;
            const size_t nk = std::min(tile, n - k0);
            unsigned int* const pivotDist = dist + k0 * n + k0;
            unsigned int* const pivotPath = path + k0 * n + k0;

            // Phase 1: the pivot tile depends only on itself.
#pragma omp single
            {
                tileRelaxInPlace(pivotDist, pivotPath, pivotDist, pivotDist, n, k0, nk, nk, nk);
            }

            // Phase 2: the tiles of the pivot row and pivot column depend on the pivot tile.
#pragma omp for schedule(static)
            for (size_t b = 0; b < numTiles; ++b) {
                if (b == r) {
                    continue;
                }
                const size_t o0 = b * tile;
                const size_t no = std::min(tile, n - o0);

                unsigned int* const rowDist = dist + k0 * n + o0;
                tileRelaxInPlace(rowDist, path + k0 * n + o0, pivotDist, rowDist, n, k0, nk, no,
                                 nk);

                unsigned int* const colDist = dist + o0 * n + k0;
                tileRelaxInPlace(colDist, path + o0 * n + k0, colDist, pivotDist, n, k0, no, nk,
                                 nk);
            }

            // Phase 3: every remaining tile, i.e. the bulk of the work.
#pragma omp for collapse(2) schedule(static)
            for (size_t bi = 0; bi < numTiles; ++bi) {
                for (size_t bj = 0; bj < numTiles; ++bj) {
                    if (bi == r || bj == r) {
                        continue;
                    }
                    const size_t i0 = bi * tile;
                    const size_t j0 = bj * tile;
                    const size_t ni = std::min(tile, n - i0);
                    const size_t nj = std::min(tile, n - j0);
                    tileRelaxDisjoint(dist + i0 * n + j0, path + i0 * n + j0, dist + i0 * n + k0,
                                      dist + k0 * n + j0, n, k0, ni, nj, nk, pack);
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
    const size_t numThreads =
        chooseThreadCount(numNodes, static_cast<size_t>(omp_get_max_threads()));
    omp_set_num_threads(static_cast<int>(numThreads));
    printf("OpenMP threads: %zu\n", numThreads);

    bindThreadsToCores(numThreads);

    // Allocate matrices (the path matrix carries slack for its cache-conflict offset)
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes + PATH_PAD_ELEMENTS);
    unsigned int* const pathBase = offsetPathBase(dist.data(), path.data());

    // Initialize
    printf("Initializing graph...\n");
    firstTouchDistribute(dist.data(), dist.size());
    firstTouchDistribute(path.data(), path.size());
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(pathBase, numNodes);

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist.data(), pathBase, numNodes);

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
