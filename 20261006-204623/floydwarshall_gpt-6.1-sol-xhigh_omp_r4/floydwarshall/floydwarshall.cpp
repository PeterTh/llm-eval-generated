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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i) {
        unsigned int* row = path.data() + i * numNodes;
        #pragma omp simd
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = static_cast<unsigned int>(i);
        }
    }
}

// Every destination in a row is independent for a fixed pivot. The caller
// supplies an immutable pivot row and the source-to-pivot distance.
inline void updateDistanceRow(unsigned int* row, unsigned int* pathRow,
                              const unsigned int* pivot, const unsigned int distIK,
                              const unsigned int intermediate,
                              const size_t begin, const size_t end) {
    #pragma omp simd
    for (size_t j = begin; j < end; ++j) {
        const unsigned int newDist = distIK + pivot[j];
        if (newDist < row[j]) {
            row[j] = newDist;
            pathRow[j] = intermediate;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    unsigned int* distances = dist.data();
    unsigned int* paths = path.data();

    constexpr size_t maxPivots = 32;
    constexpr size_t columnTile = 256;
    const size_t maxThreads = static_cast<size_t>(omp_get_max_threads());
    // Batching saves cache traffic for large row partitions. Smaller ones
    // benefit from direct pivots, without snapshotting or serial preparation.
    const bool useBatches = numNodes >= 4096 &&
        numNodes * numNodes / maxThreads >= (size_t{1} << 20);
    const size_t pivotBatch = useBatches
        ? std::min(maxPivots, std::max(size_t{1}, numNodes / (4 * maxThreads)))
        : 1;
    std::vector<unsigned int> pivotStorage(useBatches ? pivotBatch * numNodes : 0);
    unsigned int* pivots = pivotStorage.data();

    // Reuse one team for the entire computation. Static row ownership keeps
    // writes disjoint and makes successive updates reuse the same caches.
    #pragma omp parallel
    {
        if (!useBatches) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int* pivot = distances + k * numNodes;
                #pragma omp for schedule(static)
                for (size_t i = 0; i < numNodes; ++i) {
                    // Nonnegative weights and a zero diagonal mean this row
                    // never improves. Keep it read-only for the whole team.
                    if (i == k) {
                        continue;
                    }
                    unsigned int* row = distances + i * numNodes;
                    updateDistanceRow(row, paths + i * numNodes, pivot, row[k],
                                      static_cast<unsigned int>(k), 0, numNodes);
                }
                // The implicit barrier completes this pivot before the next.
            }
        } else {
            for (size_t first = 0; first < numNodes; first += pivotBatch) {
                const size_t last = std::min(first + pivotBatch, numNodes);

                // Prepare only the source rows belonging to this batch.
                // Snapshot row k after pivots < k, before later pivots alter it.
                #pragma omp single
                {
                    for (size_t k = first; k < last; ++k) {
                        unsigned int* pivot = pivots + (k - first) * numNodes;
                        std::copy_n(distances + k * numNodes, numNodes, pivot);
                        for (size_t i = first; i < last; ++i) {
                            if (i == k) {
                                continue;
                            }
                            unsigned int* row = distances + i * numNodes;
                            updateDistanceRow(row, paths + i * numNodes, pivot,
                                              row[k], static_cast<unsigned int>(k),
                                              0, numNodes);
                        }
                    }
                } // The implicit barrier publishes all snapshots.

                #pragma omp for schedule(static)
                for (size_t i = 0; i < numNodes; ++i) {
                    if (i >= first && i < last) {
                        continue;
                    }
                    unsigned int* row = distances + i * numNodes;
                    unsigned int* pathRow = paths + i * numNodes;
                    unsigned int distToPivot[maxPivots];

                    // Update the pivot columns first, recording each distance
                    // at exactly the point the original k loop would read it.
                    for (size_t k = first; k < last; ++k) {
                        const unsigned int* pivot = pivots + (k - first) * numNodes;
                        const unsigned int distIK = row[k];
                        distToPivot[k - first] = distIK;
                        updateDistanceRow(row, pathRow, pivot, distIK,
                                          static_cast<unsigned int>(k), first, last);
                    }

                    // Each destination tile stays hot through the whole batch.
                    // Preserve increasing k and strict comparisons, including
                    // the original path entries when distances tie.
                    for (size_t column = 0; column < numNodes;) {
                        if (column == first) {
                            column = last;
                            continue;
                        }
                        const size_t columnEnd = std::min(column + columnTile,
                            column < first ? first : numNodes);
                        for (size_t k = first; k < last; ++k) {
                            const unsigned int* pivot = pivots + (k - first) * numNodes;
                            updateDistanceRow(row, pathRow, pivot,
                                              distToPivot[k - first],
                                              static_cast<unsigned int>(k),
                                              column, columnEnd);
                        }
                        column = columnEnd;
                    }
                } // Finish every source row before preparing the next batch.
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
