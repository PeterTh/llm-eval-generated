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
    for (size_t i = 0; i < numNodes; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes, static_cast<unsigned int>(i));
    }
}

static void relaxRow(unsigned int* row, unsigned int* pathRow, const unsigned int* pivot,
                     const size_t numNodes, const size_t k) {
    const unsigned int toPivot = row[k];
    const auto intermediate = static_cast<unsigned int>(k);
    #pragma omp simd
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int candidate = toPivot + pivot[j];
        const unsigned int previous = row[j];
        // An all-bit mask updates the path only on a strict improvement.
        const unsigned int mask = 0u - static_cast<unsigned int>(candidate < previous);
        pathRow[j] ^= (pathRow[j] ^ intermediate) & mask;
        row[j] = std::min(candidate, previous);
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    constexpr size_t pivotsPerBatch = 8;
    constexpr size_t batchesPerMaxRefresh = 4;
    std::vector<unsigned int> pivotRows(std::min(pivotsPerBatch, numNodes) * numNodes);
    std::vector<unsigned int> rowMax(numNodes);
    std::vector<unsigned int> rowMinEdge(numNodes);
    // Keep enough rows per worker to amortize the barriers between phases.
    constexpr size_t rowsPerThread = 64;
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, (numNodes + rowsPerThread - 1) / rowsPerThread)));

    // The benchmark supplies a zero diagonal and weights at most INF, so
    // pivot rows and columns are unchanged by their own relaxation.
    #pragma omp parallel num_threads(threads)
    {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < numNodes; ++i) {
            const auto* row = dist.data() + i * numNodes;
            unsigned int maxDistance = 0;
            unsigned int minEdge = INF;
            #pragma omp simd reduction(max:maxDistance) reduction(min:minEdge)
            for (size_t j = 0; j < numNodes; ++j) {
                maxDistance = std::max(maxDistance, row[j]);
                if (j != i) {
                    minEdge = std::min(minEdge, row[j]);
                }
            }
            rowMax[i] = maxDistance;
            rowMinEdge[i] = minEdge;
        }

        for (size_t first = 0; first < numNodes; first += pivotsPerBatch) {
            const size_t end = std::min(first + pivotsPerBatch, numNodes);
            // Save each pivot row after applying its preceding pivots in
            // this batch. Only the later pivot rows need these updates now.
            #pragma omp single
            {
                for (size_t k = first; k < end; ++k) {
                    auto* pivot = pivotRows.data() + (k - first) * numNodes;
                    std::copy_n(dist.data() + k * numNodes, numNodes, pivot);
                    for (size_t i = k + 1; i < end; ++i) {
                        auto* row = dist.data() + i * numNodes;
                        if (row[k] + rowMinEdge[k] < rowMax[i]) {
                            relaxRow(row, path.data() + i * numNodes, pivot, numNodes, k);
                        }
                    }
                }
            }

            // The snapshots are read-only here. Each worker can apply the
            // remaining pivots in order while its destination row stays hot.
            #pragma omp for schedule(static)
            for (size_t i = 0; i < numNodes; ++i) {
                auto* row = dist.data() + i * numNodes;
                auto* pathRow = path.data() + i * numNodes;
                // The pivot's own row cannot improve via its zero diagonal.
                const size_t start = i >= first && i < end ? i + 1 : first;
                for (size_t k = start; k < end; ++k) {
                    // Every path from k to a different vertex costs at least
                    // its cheapest original edge. The pivot column stays put.
                    if (row[k] + rowMinEdge[k] < rowMax[i]) {
                        const auto* pivot = pivotRows.data() + (k - first) * numNodes;
                        relaxRow(row, pathRow, pivot, numNodes, k);
                    }
                }
                // Distances only decrease, so the saved maximum remains an
                // upper bound until the next refresh.
                if (first % (pivotsPerBatch * batchesPerMaxRefresh) == 0) {
                    unsigned int maxDistance = 0;
                    #pragma omp simd reduction(max:maxDistance)
                    for (size_t j = 0; j < numNodes; ++j) {
                        maxDistance = std::max(maxDistance, row[j]);
                    }
                    rowMax[i] = maxDistance;
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
