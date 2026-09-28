#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Give each thread at least this many rows per k step; more threads than
// that just adds per-step barrier cost without useful work.
constexpr size_t MIN_ROWS_PER_THREAD = 32;

int numThreadsFor(const size_t numNodes) {
    const size_t maxThreads = static_cast<size_t>(omp_get_max_threads());
    return static_cast<int>(std::clamp(numNodes / MIN_ROWS_PER_THREAD,
                                       static_cast<size_t>(1), maxThreads));
}

// First-touch pages with the same row distribution as the compute loops so
// each thread's rows land on its local NUMA node.
void firstTouch(unsigned int* data, const size_t numNodes) {
    #pragma omp parallel for schedule(static) num_threads(numThreadsFor(numNodes))
    for (size_t i = 0; i < numNodes; ++i) {
        memset(&data[idx2(0, i, numNodes)], 0, numNodes * sizeof(unsigned int));
    }
}

void initializeDistanceMatrix(unsigned int* dist, const size_t numNodes,
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

void initializePathMatrix(unsigned int* path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

void floydWarshall(unsigned int* const distData,
                   unsigned int* const pathData,
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm, parallelized over source nodes.
    // The k loop carries a dependency and stays sequential; within a k step,
    // rows are independent (row k itself is never improved at step k, so
    // concurrent reads of it are safe).
    #pragma omp parallel num_threads(numThreadsFor(numNodes))
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        const unsigned int* const rowK = &distData[idx2(0, k, numNodes)];

        // For each source node i (static schedule matches the first-touch
        // distribution, keeping rows NUMA-local across k steps)
        #pragma omp for schedule(static)
        for (size_t i = 0; i < numNodes; ++i) {
            unsigned int* const rowI = &distData[idx2(0, i, numNodes)];
            unsigned int* const pathRowI = &pathData[idx2(0, i, numNodes)];
            const unsigned int distIK = rowI[k];

            // For each destination node j
            #pragma omp simd
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rowK[j];

                if (newDist < rowI[j]) {
                    rowI[j] = newDist;
                    pathRowI[j] = k;
                }
            }
        }
        // Implicit barrier at the end of the omp for keeps k steps ordered.
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
    
    // Allocate matrices uninitialized, then first-touch pages in parallel so
    // they are distributed across NUMA nodes before the sequential init fills them
    std::unique_ptr<unsigned int[]> dist(new unsigned int[numNodes * numNodes]);
    std::unique_ptr<unsigned int[]> path(new unsigned int[numNodes * numNodes]);
    firstTouch(dist.get(), numNodes);
    firstTouch(path.get(), numNodes);

    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist.get(), numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path.get(), numNodes);
    
    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist.get(), path.get(), numNodes);
    
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
        const std::vector<unsigned int> distCopy(dist.get(), dist.get() + numNodes * numNodes);
        print_results_int(distCopy, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist.get(), numNodes);
        
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
