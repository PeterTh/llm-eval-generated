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
constexpr size_t FW_BLOCK_SIZE = 16;

inline int matrixThreadCount(const size_t numNodes) noexcept {
    const size_t blocks = numNodes / FW_BLOCK_SIZE + (numNodes % FW_BLOCK_SIZE != 0);
    // The pivot-row phase has one independent column tile per block. Avoid
    // starting more workers than that phase can use, especially on small graphs.
    return static_cast<int>(std::min(static_cast<size_t>(omp_get_max_threads()),
                                     std::max(size_t{1}, blocks)));
}

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
    const int numThreads = matrixThreadCount(numNodes);
    #pragma omp parallel for num_threads(numThreads) schedule(static) default(none) shared(path, numNodes)
    for (size_t i = 0; i < numNodes; ++i) {
        // Both assignments in the original initialization set each entry to
        // its source row. Fill each row once, with contiguous stores.
        #pragma omp simd
        for (size_t j = 0; j < numNodes; ++j) {
            path[i * numNodes + j] = static_cast<unsigned int>(i);
        }
    }
}

inline void relaxRow(unsigned int* row, unsigned int* pathRow,
                     const unsigned int* pivot, const unsigned int distanceIK,
                     const unsigned int k, const size_t count) {
    #pragma omp simd
    for (size_t j = 0; j < count; ++j) {
        const unsigned int candidate = distanceIK + pivot[j];
        if (candidate < row[j]) {
            row[j] = candidate;
            pathRow[j] = k;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    const size_t numBlocks = numNodes / FW_BLOCK_SIZE + (numNodes % FW_BLOCK_SIZE != 0);
    const int numThreads = matrixThreadCount(numNodes);
    // Keep the pivot rows as they appeared at their individual k iterations.
    // Using their final values after an ordinary blocked closure would change
    // the original strict-improvement path entries, even with equal distances.
    std::vector<unsigned int> pivotStorage(FW_BLOCK_SIZE * numNodes);
    unsigned int pivotColumns[FW_BLOCK_SIZE * FW_BLOCK_SIZE];
    unsigned int* distances = dist.data();
    unsigned int* paths = path.data();
    unsigned int* pivots = pivotStorage.data();

    #pragma omp parallel num_threads(numThreads) default(none) shared(distances, paths, pivots, pivotColumns, numNodes, numBlocks, FW_BLOCK_SIZE)
    {
        for (size_t block = 0; block < numBlocks; ++block) {
            const size_t begin = block * FW_BLOCK_SIZE;
            const size_t width = std::min(FW_BLOCK_SIZE, numNodes - begin);

            // First process the small diagonal tile in the original k order,
            // saving both the pivot row and source-to-pivot column at each k.
            #pragma omp single
            {
                for (size_t offset = 0; offset < width; ++offset) {
                    const size_t k = begin + offset;
                    const unsigned int* pivot = distances + k * numNodes;
                    unsigned int* snapshot = pivots + offset * numNodes;
                    std::copy_n(pivot + begin, width, snapshot + begin);
                    for (size_t rowOffset = 0; rowOffset < width; ++rowOffset) {
                        pivotColumns[offset * FW_BLOCK_SIZE + rowOffset] =
                            distances[(begin + rowOffset) * numNodes + k];
                    }
                    for (size_t rowOffset = 0; rowOffset < width; ++rowOffset) {
                        // A zero diagonal makes the pivot row unchanged at k.
                        if (rowOffset == offset) continue;
                        const size_t i = begin + rowOffset;
                        relaxRow(distances + i * numNodes + begin,
                                 paths + i * numNodes + begin, snapshot + begin,
                                 pivotColumns[offset * FW_BLOCK_SIZE + rowOffset],
                                 static_cast<unsigned int>(k), width);
                    }
                }
            }

            // Each worker owns disjoint columns of the pivot rows. Complete
            // their updates and snapshots without any per-pivot barriers.
            #pragma omp for schedule(static)
            for (size_t columnBlock = 0; columnBlock < numBlocks; ++columnBlock) {
                if (columnBlock == block) continue;
                const size_t column = columnBlock * FW_BLOCK_SIZE;
                const size_t count = std::min(FW_BLOCK_SIZE, numNodes - column);
                for (size_t offset = 0; offset < width; ++offset) {
                    const size_t k = begin + offset;
                    unsigned int* snapshot = pivots + offset * numNodes + column;
                    std::copy_n(distances + k * numNodes + column, count, snapshot);
                    for (size_t rowOffset = 0; rowOffset < width; ++rowOffset) {
                        if (rowOffset == offset) continue;
                        const size_t i = begin + rowOffset;
                        relaxRow(distances + i * numNodes + column,
                                 paths + i * numNodes + column, snapshot,
                                 pivotColumns[offset * FW_BLOCK_SIZE + rowOffset],
                                 static_cast<unsigned int>(k), count);
                    }
                }
            }

            // The barrier above publishes every immutable pivot snapshot.
            // Other rows are now independent: retain each row in cache while
            // applying all pivots in order, preserving distances and paths.
            #pragma omp for schedule(static)
            for (size_t i = 0; i < numNodes; ++i) {
                if (i >= begin && i < begin + width) continue;
                unsigned int* row = distances + i * numNodes;
                unsigned int* pathRow = paths + i * numNodes;
                for (size_t offset = 0; offset < width; ++offset) {
                    const size_t k = begin + offset;
                    relaxRow(row, pathRow, pivots + offset * numNodes,
                             row[k], static_cast<unsigned int>(k), numNodes);
                }
            }
            // All rows must finish before the next block reads its pivots.
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
