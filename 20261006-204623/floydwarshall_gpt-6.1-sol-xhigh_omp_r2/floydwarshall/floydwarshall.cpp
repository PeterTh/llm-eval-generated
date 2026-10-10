#include <algorithm>
#include <atomic>
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

namespace {
// Avoid idle workers and synchronization dominating small graphs.
int openmpThreadCount(const size_t numNodes) noexcept {
    const size_t available = static_cast<size_t>(
        std::min(omp_get_max_threads(), omp_get_num_procs()));
    return static_cast<int>(std::min(available, std::max(size_t{1}, numNodes / 16)));
}

inline void relaxRow(unsigned int* const row, unsigned int* const pathRow,
                     const unsigned int* const pivot, const unsigned int distIK,
                     const unsigned int intermediate, const size_t begin,
                     const size_t end) noexcept {
    #pragma omp simd
    for (size_t j = begin; j < end; ++j) {
        const unsigned int newDist = distIK + pivot[j];
        if (newDist < row[j]) {
            row[j] = newDist;
            pathRow[j] = intermediate;
        }
    }
}
} // namespace

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
    #pragma omp parallel for schedule(static) num_threads(openmpThreadCount(numNodes))
    for (size_t i = 0; i < numNodes; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes,
                    static_cast<unsigned int>(i));
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    constexpr size_t pivotBlockSize = 32;
    constexpr size_t columnBlockSize = 256;

    // Each snapshot is the pivot row after exactly k preceding pivots.
    // Immutable snapshots let workers advance without a barrier at every k.
    std::vector<unsigned int> pivotRows(pivotBlockSize * numNodes);
    struct alignas(64) PivotReady {
        std::atomic<size_t> value{0};
    };
    std::vector<PivotReady> ready(pivotBlockSize);
    unsigned int* const distances = dist.data();
    unsigned int* const paths = path.data();
    const int teamSize = openmpThreadCount(numNodes);

    #pragma omp parallel num_threads(teamSize)
    {
        const size_t threads = static_cast<size_t>(omp_get_num_threads());
        const size_t thread = static_cast<size_t>(omp_get_thread_num());
        const size_t rowsPerThread = numNodes / threads;
        const size_t extraRows = numNodes % threads;
        const size_t firstRow = thread * rowsPerThread + std::min(thread, extraRows);
        const size_t lastRow = firstRow + rowsPerThread + (thread < extraRows);

        // Stream rows while their working set is small; otherwise keep a
        // column tile in cache across several ordered pivots.
        const bool cacheBlocked = numNodes >= 4096 &&
                                  rowsPerThread * numNodes >= 1024 * 1024;

        for (size_t firstPivot = 0; firstPivot < numNodes; firstPivot += pivotBlockSize) {
            const size_t lastPivot = std::min(numNodes, firstPivot + pivotBlockSize);

            if (!cacheBlocked) {
                for (size_t k = firstPivot; k < lastPivot; ++k) {
                    const size_t slot = k - firstPivot;
                    unsigned int* const pivot = pivotRows.data() + slot * numNodes;
                    if (k >= firstRow && k < lastRow) {
                        std::copy_n(distances + k * numNodes, numNodes, pivot);
                        ready[slot].value.store(k + 1, std::memory_order_release);
                    } else {
                        while (ready[slot].value.load(std::memory_order_acquire) != k + 1) {
                        }
                    }
                    for (size_t i = firstRow; i < lastRow; ++i) {
                        // With the benchmark's zero diagonal, the pivot row
                        // never improves during its own step.
                        if (i == k) {
                            continue;
                        }
                        unsigned int* const row = distances + i * numNodes;
                        relaxRow(row, paths + i * numNodes, pivot, row[k],
                                 static_cast<unsigned int>(k), 0, numNodes);
                    }
                }
            } else {
                // Distribute the block's pivot rows cyclically. Only these
                // rows must be advanced before later snapshots can be made.
                if (thread < lastPivot - firstPivot) {
                    for (size_t k = firstPivot; k < lastPivot; ++k) {
                        const size_t slot = k - firstPivot;
                        unsigned int* const pivot = pivotRows.data() + slot * numNodes;
                        if (slot % threads == thread) {
                            std::copy_n(distances + k * numNodes, numNodes, pivot);
                            ready[slot].value.store(k + 1, std::memory_order_release);
                        } else {
                            while (ready[slot].value.load(std::memory_order_acquire) != k + 1) {
                            }
                        }
                        for (size_t i = firstPivot + thread; i < lastPivot; i += threads) {
                            if (i == k) {
                                continue;
                            }
                            unsigned int* const row = distances + i * numNodes;
                            relaxRow(row, paths + i * numNodes, pivot, row[k],
                                     static_cast<unsigned int>(k), 0, numNodes);
                        }
                    }
                }

                for (size_t i = firstRow; i < lastRow; ++i) {
                    if (i >= firstPivot && i < lastPivot) {
                        continue;
                    }
                    unsigned int* const row = distances + i * numNodes;
                    unsigned int* const pathRow = paths + i * numNodes;
                    unsigned int distToPivot[pivotBlockSize];

                    // First advance the pivot columns, saving dist[i][k] at
                    // each original k step. Other columns cannot affect them.
                    for (size_t k = firstPivot; k < lastPivot; ++k) {
                        const size_t slot = k - firstPivot;
                        while (ready[slot].value.load(std::memory_order_acquire) != k + 1) {
                        }
                        const unsigned int* const pivot = pivotRows.data() + slot * numNodes;
                        distToPivot[slot] = row[k];
                        relaxRow(row, pathRow, pivot, distToPivot[slot],
                                 static_cast<unsigned int>(k), firstPivot, lastPivot);
                    }

                    // Reuse each distance/path tile for every pivot in the
                    // block. Strict comparisons and ascending k preserve ties
                    // and the original path matrix, as well as distances.
                    for (size_t range = 0; range < 2; ++range) {
                        const size_t begin = range == 0 ? 0 : lastPivot;
                        const size_t end = range == 0 ? firstPivot : numNodes;
                        for (size_t firstColumn = begin; firstColumn < end;
                             firstColumn += columnBlockSize) {
                            const size_t lastColumn = std::min(end, firstColumn + columnBlockSize);
                            for (size_t k = firstPivot; k < lastPivot; ++k) {
                                const size_t slot = k - firstPivot;
                                const unsigned int* const pivot = pivotRows.data() + slot * numNodes;
                                relaxRow(row, pathRow, pivot, distToPivot[slot],
                                         static_cast<unsigned int>(k), firstColumn, lastColumn);
                            }
                        }
                    }
                }
            }
            // Finish every row before reusing snapshots or changing row owners.
            #pragma omp barrier
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
