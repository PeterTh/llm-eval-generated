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

// Blocking factor of the innermost (j) loop; one block of each matrix stays in L1.
constexpr size_t BLOCK_J = 512;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Number of physical cores usable by this process. The kernel here is bandwidth and
// barrier bound, so running two threads per physical core only adds contention.
// Returns 0 if the topology cannot be determined.
static size_t detectPhysicalCores() {
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return 0;
    }

    std::vector<int> coreIds;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &allowed)) {
            continue;
        }
        // The first entry of the sibling list identifies the physical core.
        char path[160];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        int coreId = cpu;
        if (FILE* f = fopen(path, "r")) {
            int first = 0;
            if (fscanf(f, "%d", &first) == 1) {
                coreId = first;
            }
            fclose(f);
        }
        coreIds.push_back(coreId);
    }

    std::sort(coreIds.begin(), coreIds.end());
    return static_cast<size_t>(std::unique(coreIds.begin(), coreIds.end()) - coreIds.begin());
}

// The k-loop of Floyd-Warshall requires a global barrier after every step, so the amount
// of work available per step (n^2 element updates) has to be large enough to amortize
// that barrier across the team. For small graphs a smaller team is faster than a large
// one, so cap the team size accordingly - never above what the runtime / OMP_NUM_THREADS
// allows, so thread scaling studies still work.
static int chooseThreadCount(const size_t numNodes) {
    size_t maxThreads = static_cast<size_t>(std::max(1, omp_get_max_threads()));
    if (const size_t cores = detectPhysicalCores(); cores > 0) {
        maxThreads = std::min(maxThreads, cores);
    }
    if (numNodes == 0) {
        return 1;
    }
    // Roughly 16K element updates per thread and per k-step.
    size_t threads = (numNodes * numNodes) / 16384;
    threads = std::max<size_t>(threads, 1);
    threads = std::min(threads, maxThreads);
    threads = std::min(threads, numNodes);  // no more threads than matrix rows
    return static_cast<int>(threads);
}

// Touch every row from the thread that will later own it, so that the pages of the
// matrices are placed on the NUMA node of that thread (first-touch policy).
static void firstTouch(unsigned int* dist, unsigned int* path, const size_t numNodes,
                       const int numThreads) {
    #pragma omp parallel for schedule(static) num_threads(numThreads) proc_bind(spread)
    for (size_t i = 0; i < numNodes; ++i) {
        std::memset(dist + i * numNodes, 0, numNodes * sizeof(unsigned int));
        std::memset(path + i * numNodes, 0, numNodes * sizeof(unsigned int));
    }
}

void initializeDistanceMatrix(unsigned int* dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Kept serial: rand_r() carries a loop-carried dependency through the seed and the
    // generated sequence is part of the benchmark's defined input.
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(unsigned int* path, const size_t numNodes, const int numThreads) {
    // The original nested loops assign path[idx2(i, j, n)] = j and path[idx2(j, i, n)] = i,
    // i.e. every element of row r (including the diagonal) ends up holding r.
    #pragma omp parallel for schedule(static) num_threads(numThreads) proc_bind(spread)
    for (size_t j = 0; j < numNodes; ++j) {
        unsigned int* __restrict row = path + j * numNodes;
        const unsigned int value = static_cast<unsigned int>(j);
        for (size_t i = 0; i < numNodes; ++i) {
            row[i] = value;
        }
    }
}

void floydWarshall(unsigned int* __restrict dist,
                   unsigned int* __restrict path,
                   const size_t numNodes,
                   const int numThreads) {
    // Classic Floyd-Warshall algorithm.
    //
    // With idx2(j, i, n) == i * n + j the matrices are row-major in i, so the inner j
    // loop walks contiguous memory and rows are the natural unit of parallel work:
    //   row_i[j] = min(row_i[j], row_i[k] + row_k[j])
    // For a fixed k every row i is updated independently and only row k is read by all
    // threads. Row k itself is provably invariant during step k (dist[k][k] == 0, and
    // the edge weights are positive so it can never drop below zero), so the rows can be
    // distributed across the team without any further synchronization beyond one barrier
    // per k. The barrier is required because step k+1 reads row k+1 as written in step k.
    //
    // A single parallel region spans the whole k loop so the team is created only once.
    #pragma omp parallel num_threads(numThreads) proc_bind(spread)
    {
        for (size_t k = 0; k < numNodes; ++k) {
            const unsigned int* __restrict distK = dist + k * numNodes;

            // For each source node i (implicit barrier at the end of the loop)
            #pragma omp for schedule(static)
            for (size_t i = 0; i < numNodes; ++i) {
                if (i == k) {
                    continue;  // row k is unchanged in step k
                }

                unsigned int* __restrict distI = dist + i * numNodes;
                unsigned int* __restrict pathI = path + i * numNodes;
                const unsigned int distIK = distI[k];

                // For each destination node j, in L1-sized blocks.
                //
                // Only a small fraction of the (k, i) pairs improves anything once the
                // first few k steps are done, so each block is first scanned branch-free
                // for "does anything change at all". If nothing does, the block's stores
                // are skipped entirely and the path row is not even touched - that removes
                // the bulk of the write traffic and half of the read traffic, which is
                // what limits this kernel on a many-core machine.
                for (size_t j0 = 0; j0 < numNodes; j0 += BLOCK_J) {
                    const size_t jEnd = std::min(j0 + BLOCK_J, numNodes);

                    unsigned int changed = 0;
                    #pragma omp simd reduction(| : changed)
                    for (size_t j = j0; j < jEnd; ++j) {
                        changed |= static_cast<unsigned int>(distIK + distK[j] < distI[j]);
                    }
                    if (changed == 0) {
                        continue;
                    }

                    // Branch-free so it vectorizes cleanly; writing back unconditionally
                    // is equivalent to the conditional store because the unchanged value
                    // is written when the test fails.
                    #pragma omp simd
                    for (size_t j = j0; j < jEnd; ++j) {
                        const unsigned int distIJ = distI[j];
                        const unsigned int newDist = distIK + distK[j];
                        const bool better = newDist < distIJ;

                        distI[j] = better ? newDist : distIJ;
                        pathI[j] = better ? static_cast<unsigned int>(k) : pathI[j];
                    }
                }
            }
        }
    }
}

bool validateResult(const unsigned int* dist, const size_t numNodes) {
    // Basic sanity checks
    bool valid = true;

    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    const size_t limit = std::min(numNodes, static_cast<size_t>(10));
    #pragma omp parallel for collapse(2) schedule(static)
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            if (!valid) {
                continue;
            }
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        #pragma omp atomic write
                        valid = false;
                        break;
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

    const int numThreads = chooseThreadCount(numNodes);
    printf("OpenMP threads: %d\n", numThreads);

    if (numNodes == 0) {
        printf("Computation time: 0 ms\n");
        return 0;
    }

    // Allocate matrices (cache-line aligned, left uninitialized so that the parallel
    // first-touch pass below decides the NUMA placement of the pages)
    const size_t elements = numNodes * numNodes;
    const size_t bytes = ((elements * sizeof(unsigned int)) + 63) & ~static_cast<size_t>(63);
    unsigned int* dist = static_cast<unsigned int*>(std::aligned_alloc(64, bytes));
    unsigned int* path = static_cast<unsigned int*>(std::aligned_alloc(64, bytes));
    if (dist == nullptr || path == nullptr) {
        printf("Allocation failed\n");
        std::free(dist);
        std::free(path);
        return 1;
    }
    firstTouch(dist, path, numNodes, numThreads);

    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, numThreads);

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, numThreads);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    const double seconds = std::chrono::duration<double>(end - start).count();
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / seconds / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);

    // Print results for external validation (integer hash-based)
    if (printResults) {
        print_results_int(std::vector<unsigned int>(dist, dist + elements), "DistanceMatrix");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);

        std::free(dist);
        std::free(path);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    std::free(dist);
    std::free(path);

    return 0;
}
