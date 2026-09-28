#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <vector>

#include <omp.h>
#include <pthread.h>
#include <sched.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// ---------------------------------------------------------------------------
// Thread placement
//
// Floyd-Warshall is heavily memory bound, so leaving the OpenMP threads
// unbound (the libgomp default) costs a large factor on a multi-socket / SMT
// machine: threads migrate, and the NUMA placement established by first touch
// stops matching the thread that uses the data. We therefore pin the threads
// ourselves, one physical core at a time and round-robin over packages, only
// falling back to SMT siblings once every core is taken. Any explicit user
// affinity request wins: we stay inside the inherited CPU mask and do nothing
// at all if OMP_PROC_BIND / OMP_PLACES / GOMP_CPU_AFFINITY is set.
// ---------------------------------------------------------------------------

static int readTopologyId(const int cpu, const char* what) {
    char pathBuf[128];
    snprintf(pathBuf, sizeof(pathBuf), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
    FILE* f = fopen(pathBuf, "r");
    if (!f) {
        return -1;
    }
    int value = -1;
    if (fscanf(f, "%d", &value) != 1) {
        value = -1;
    }
    fclose(f);
    return value;
}

// Build the thread -> CPU map for `numThreads` threads: the threads are split
// into one contiguous id range per package (so a static loop schedule keeps
// neighbouring chunks on the same NUMA node), and inside a package they fill
// distinct physical cores before doubling up on SMT siblings.
static std::vector<int> buildCpuOrder(const size_t numThreads) {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return {};
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            cpus.push_back(cpu);
        }
    }

    // (package, core) -> its logical CPUs, in increasing order.
    std::map<std::pair<int, int>, std::vector<int>> cores;
    for (const int cpu : cpus) {
        const int pkg = readTopologyId(cpu, "physical_package_id");
        const int core = readTopologyId(cpu, "core_id");
        if (pkg < 0 || core < 0) {
            return cpus;  // no usable topology info, keep the natural order
        }
        cores[{pkg, core}].push_back(cpu);
    }

    // Cores grouped per package, packages in increasing id order.
    std::map<int, std::vector<const std::vector<int>*>> byPackage;
    for (const auto& entry : cores) {
        byPackage[entry.first.first].push_back(&entry.second);
    }
    const size_t numPackages = byPackage.size();

    std::vector<int> order;
    order.reserve(numThreads);
    size_t pkgIndex = 0;
    for (const auto& pkg : byPackage) {
        // Even split of the threads over the packages.
        const size_t share = numThreads / numPackages +
                             (pkgIndex < numThreads % numPackages ? 1 : 0);
        ++pkgIndex;

        // Walk the package's cores at a constant stride: with fewer threads
        // than cores this spreads them over all L3 slices, and with more it
        // gives consecutive thread ids to the SMT siblings of one core, so
        // that a static schedule hands neighbouring work to a shared cache.
        const size_t numCores = pkg.second.size();
        if (numCores == 0) {
            continue;
        }
        size_t previousCore = (size_t)-1;
        size_t onThisCore = 0;
        for (size_t t = 0; t < share; ++t) {
            const size_t c = t * numCores / share;
            if (c != previousCore) {
                previousCore = c;
                onThisCore = 0;
            }
            const std::vector<int>& coreCpus = *pkg.second[c];
            order.push_back(coreCpus[onThisCore % coreCpus.size()]);
            ++onThisCore;
        }
    }
    return order.empty() ? cpus : order;
}

static void bindThreads() {
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY")) {
        return;  // respect the placement the user asked for
    }
    const std::vector<int> order = buildCpuOrder((size_t)omp_get_max_threads());
    if (order.empty()) {
        return;
    }

    #pragma omp parallel
    {
        const int cpu = order[(size_t)omp_get_thread_num() % order.size()];
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpu, &set);
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
}

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// The storage layout above places logical element M[r][c] at r * n + c, i.e. a
// logical matrix row is a contiguous span of n elements. The matrices are
// stored with a padded leading dimension `ld` instead of `n`: a row stride
// that is a large power of two (which is what n * sizeof(unsigned int) is for
// the usual benchmark sizes) makes all rows of a tile land in the same cache
// sets and destroys the blocking. `ld` is passed as the stride argument of
// idx2, so the index arithmetic itself is unchanged.

// Padded row stride: 64-byte aligned rows, offset by one cache line so that
// rows of a tile spread over the cache sets.
static size_t paddedStride(const size_t numNodes) {
    if (numNodes == 0) {
        return 1;
    }
    return ((numNodes + 15) & ~static_cast<size_t>(15)) + 16;
}

void initializeDistanceMatrix(unsigned int* dist, const size_t numNodes, const size_t ld,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Element i of the logical (unpadded) array is row i / numNodes, column
    // i % numNodes, so this walks the random sequence in the original order.
    for (size_t r = 0; r < numNodes; ++r) {
        for (size_t c = 0; c < numNodes; ++c) {
            dist[idx2(c, r, ld)] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, ld)] = 0;
    }
}

void initializePathMatrix(unsigned int* path, const size_t numNodes, const size_t ld) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, ld)] = j;
            path[idx2(j, i, ld)] = i;
        }
        path[idx2(j, j, ld)] = j;
    }
}

// Relax the block of destination rows [i0,i1) x columns [j0,j1) against all
// intermediate nodes k in [k0,k1). This is exactly the body of the classic
// triple loop restricted to a sub-range of (k, i, j).
static inline void relaxBlock(unsigned int* __restrict dist, unsigned int* __restrict path,
                              const size_t ld,
                              const size_t k0, const size_t k1,
                              const size_t i0, const size_t i1,
                              const size_t j0, const size_t j1) {
    for (size_t k = k0; k < k1; ++k) {
        const unsigned int* __restrict rowK = dist + k * ld;
        for (size_t i = i0; i < i1; ++i) {
            unsigned int* __restrict rowI = dist + i * ld;
            unsigned int* __restrict pathI = path + i * ld;
            const unsigned int distIK = rowI[k];

            #pragma omp simd
            for (size_t j = j0; j < j1; ++j) {
                const unsigned int newDist = distIK + rowK[j];
                const unsigned int distIJ = rowI[j];
                const bool better = newDist < distIJ;
                rowI[j] = better ? newDist : distIJ;
                pathI[j] = better ? (unsigned int)k : pathI[j];
            }
        }
    }
}

// Pick a tile size that keeps the three active tiles resident in cache while
// still exposing enough independent tiles to keep every thread busy. Bigger
// tiles amortise the per-tile overhead better, so take the largest one that
// still leaves a comfortable number of tiles per thread for phase 3.
static size_t chooseBlockSize(const size_t numNodes, const int numThreads) {
    const size_t candidates[] = {96, 64, 48, 32};
    const size_t minTiles = 8 * (size_t)numThreads;
    size_t chosen = candidates[sizeof(candidates) / sizeof(candidates[0]) - 1];
    for (const size_t b : candidates) {
        const size_t nb = (numNodes + b - 1) / b;
        if (nb * nb >= minTiles) {
            chosen = b;
            break;
        }
    }
    return std::min(chosen, std::max<size_t>(numNodes, 1));
}

void floydWarshall(unsigned int* dist, unsigned int* path, const size_t numNodes, const size_t ld) {
    if (numNodes == 0) {
        return;
    }

    const int numThreads = omp_get_max_threads();
    const size_t B = chooseBlockSize(numNodes, numThreads);
    const size_t nb = (numNodes + B - 1) / B;

    #pragma omp parallel
    {
        for (size_t kb = 0; kb < nb; ++kb) {
            const size_t k0 = kb * B;
            const size_t k1 = std::min(k0 + B, numNodes);

            // Phase 1: the pivot tile depends only on itself and must be
            // relaxed sequentially in increasing k.
            #pragma omp single
            {
                relaxBlock(dist, path, ld, k0, k1, k0, k1, k0, k1);
            }
            // implicit barrier

            // Phase 2: the pivot row-panel and column-panel of tiles; each of
            // them depends on the pivot tile only, so they are independent.
            #pragma omp for schedule(static)
            for (size_t b = 0; b < nb; ++b) {
                if (b == kb) {
                    continue;
                }
                const size_t o0 = b * B;
                const size_t o1 = std::min(o0 + B, numNodes);
                // Row panel: rows [k0,k1), columns [o0,o1)
                relaxBlock(dist, path, ld, k0, k1, k0, k1, o0, o1);
                // Column panel: rows [o0,o1), columns [k0,k1)
                relaxBlock(dist, path, ld, k0, k1, o0, o1, k0, k1);
            }
            // implicit barrier

            // Phase 3: every remaining tile depends only on the two panels.
            #pragma omp for collapse(2) schedule(static)
            for (size_t ib = 0; ib < nb; ++ib) {
                for (size_t jb = 0; jb < nb; ++jb) {
                    if (ib == kb || jb == kb) {
                        continue;
                    }
                    const size_t i0 = ib * B;
                    const size_t i1 = std::min(i0 + B, numNodes);
                    const size_t j0 = jb * B;
                    const size_t j1 = std::min(j0 + B, numNodes);
                    relaxBlock(dist, path, ld, k0, k1, i0, i1, j0, j1);
                }
            }
            // implicit barrier
        }
    }
}

bool validateResult(const unsigned int* dist, const size_t numNodes, const size_t ld) {
    // Basic sanity checks
    bool valid = true;

    // 1. Diagonal should be zero
    #pragma omp parallel for reduction(&& : valid) schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, ld)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            valid = false;
        }
    }
    if (!valid) {
        return false;
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    const size_t limit = std::min(numNodes, static_cast<size_t>(10));
    #pragma omp parallel for collapse(2) reduction(&& : valid) schedule(static)
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, ld)];
                const unsigned int distIK = dist[idx2(k, i, ld)];
                const unsigned int distKJ = dist[idx2(j, k, ld)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        valid = false;
                    }
                }
            }
        }
    }

    return valid;
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

    // Allocate matrices. Raw (uninitialized) storage lets the first write to
    // every page come from the thread that will later work on that row range,
    // so the pages end up on the right NUMA node.
    const size_t elems = numNodes * numNodes;
    const size_t ld = paddedStride(numNodes);
    const size_t bytes = (numNodes ? numNodes * ld : 1) * sizeof(unsigned int);
    auto deleter = [](unsigned int* p) { std::free(p); };
    std::unique_ptr<unsigned int[], decltype(deleter)> distStore(
        static_cast<unsigned int*>(std::aligned_alloc(64, bytes)), deleter);
    std::unique_ptr<unsigned int[], decltype(deleter)> pathStore(
        static_cast<unsigned int*>(std::aligned_alloc(64, bytes)), deleter);
    if (!distStore || !pathStore) {
        printf("Allocation failed\n");
        return 1;
    }
    unsigned int* const dist = distStore.get();
    unsigned int* const path = pathStore.get();

    // NUMA first touch, distributed exactly like the row blocks of the solver.
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < numNodes; ++r) {
        std::memset(dist + r * ld, 0, ld * sizeof(unsigned int));
        std::memset(path + r * ld, 0, ld * sizeof(unsigned int));
    }

    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, ld, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, ld);

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, ld);

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
        // Drop the row padding again so the hash matches the dense layout.
        std::vector<unsigned int> distView(elems);
        for (size_t r = 0; r < numNodes; ++r) {
            std::memcpy(distView.data() + r * numNodes, dist + r * ld, numNodes * sizeof(unsigned int));
        }
        print_results_int(distView, "DistanceMatrix");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes, ld);

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
