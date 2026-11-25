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

static inline void relaxRow(unsigned int* distances, unsigned int* paths,
                            const unsigned int* pivotRow, const size_t width,
                            const unsigned int toPivot, const unsigned int pivot) {
    #pragma omp simd
    for (size_t j = 0; j < width; ++j) {
        const unsigned int candidate = toPivot + pivotRow[j];
        if (candidate < distances[j]) {
            distances[j] = candidate;
            paths[j] = pivot;
        }
    }
}

static inline unsigned int savePivotRow(unsigned int* saved, const unsigned int* row,
                                        const size_t width) {
    unsigned int minimum = row[0];
    #pragma omp simd reduction(min:minimum)
    for (size_t j = 0; j < width; ++j) {
        saved[j] = row[j];
        minimum = std::min(minimum, row[j]);
    }
    return minimum;
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const size_t tileSize = numNodes <= 128 ? 16 : numNodes <= 256 ? 32 : 64;
    const size_t tileCount = (numNodes + tileSize - 1) / tileSize;
    const size_t otherTiles = tileCount - 1;
    // There are two independent panel tiles per non-pivot tile index.
    const int threads = std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, 2 * otherTiles));
    unsigned int* const distances = dist.data();
    unsigned int* const paths = path.data();

    // Store each pivot row and column as it was when its pivot was processed.
    // Applying final panel values early can change which pivot is recorded in path.
    // Pivot rows are grouped by column tile and pivot columns by source row.
    std::vector<unsigned int> pivotRows(tileCount * tileSize * tileSize);
    std::vector<unsigned int> pivotColumns(numNodes * tileSize);
    std::vector<unsigned int> pivotRowMinima(tileCount * tileSize);

    #pragma omp parallel num_threads(threads)
    {
        for (size_t pivotTile = 0; pivotTile < tileCount; ++pivotTile) {
            const size_t firstPivot = pivotTile * tileSize;
            const size_t pivotCount = std::min(tileSize, numNodes - firstPivot);

            #pragma omp single
            {
                for (size_t k = 0; k < pivotCount; ++k) {
                    const size_t pivot = firstPivot + k;
                    unsigned int* const saved = pivotRows.data() + pivot * tileSize;
                    pivotRowMinima[pivot] =
                        savePivotRow(saved, distances + pivot * numNodes + firstPivot, pivotCount);
                    for (size_t i = firstPivot; i < firstPivot + pivotCount; ++i) {
                        const unsigned int toPivot = distances[i * numNodes + pivot];
                        pivotColumns[i * tileSize + k] = toPivot;
                        const size_t offset = i * numNodes + firstPivot;
                        relaxRow(distances + offset, paths + offset, saved, pivotCount,
                                 toPivot, static_cast<unsigned int>(pivot));
                    }
                }
            }

            // The tiles intersecting the pivot tile's rows or columns are independent.
            #pragma omp for schedule(static)
            for (size_t panel = 0; panel < 2 * otherTiles; ++panel) {
                const size_t panelTile = panel / 2;
                const size_t tile = panelTile + (panelTile >= pivotTile);
                const size_t first = tile * tileSize;
                const size_t count = std::min(tileSize, numNodes - first);
                if (panel % 2 == 0) {
                    for (size_t k = 0; k < pivotCount; ++k) {
                        const size_t pivot = firstPivot + k;
                        unsigned int* const saved = pivotRows.data() + (first + k) * tileSize;
                        pivotRowMinima[first + k] =
                            savePivotRow(saved, distances + pivot * numNodes + first, count);
                        for (size_t i = firstPivot; i < firstPivot + pivotCount; ++i) {
                            const size_t offset = i * numNodes + first;
                            relaxRow(distances + offset, paths + offset, saved, count,
                                     pivotColumns[i * tileSize + k], static_cast<unsigned int>(pivot));
                        }
                    }
                } else {
                    for (size_t k = 0; k < pivotCount; ++k) {
                        const size_t pivot = firstPivot + k;
                        const unsigned int* const saved = pivotRows.data() + (firstPivot + k) * tileSize;
                        for (size_t i = first; i < first + count; ++i) {
                            const unsigned int toPivot = distances[i * numNodes + pivot];
                            pivotColumns[i * tileSize + k] = toPivot;
                            const size_t offset = i * numNodes + firstPivot;
                            relaxRow(distances + offset, paths + offset, saved, pivotCount,
                                     toPivot, static_cast<unsigned int>(pivot));
                        }
                    }
                }
            }

            // The remaining tiles use only the saved pivots and their own entries.
            #pragma omp for collapse(2) schedule(static)
            for (size_t rowBlock = 0; rowBlock < otherTiles; ++rowBlock) {
                for (size_t colBlock = 0; colBlock < otherTiles; ++colBlock) {
                    const size_t rowTile = rowBlock + (rowBlock >= pivotTile);
                    const size_t colTile = colBlock + (colBlock >= pivotTile);
                    const size_t firstRow = rowTile * tileSize;
                    const size_t lastRow = std::min(firstRow + tileSize, numNodes);
                    const size_t firstCol = colTile * tileSize;
                    const size_t width = std::min(tileSize, numNodes - firstCol);
                    const unsigned int* const rows = pivotRows.data() + firstCol * tileSize;
                    const unsigned int* const minima = pivotRowMinima.data() + firstCol;
                    for (size_t i = firstRow; i < lastRow; ++i) {
                        const size_t offset = i * numNodes + firstCol;
                        unsigned int* const row = distances + offset;
                        unsigned int maximum = 0;
                        #pragma omp simd reduction(max:maximum)
                        for (size_t j = 0; j < width; ++j) {
                            maximum = std::max(maximum, row[j]);
                        }

                        // The benchmark's edge weights cannot overflow a candidate sum.
                        // Distances only decrease, so maximum remains an upper bound.
                        const unsigned int* const column = pivotColumns.data() + i * tileSize;
                        for (size_t k = 0; k < pivotCount; ++k) {
                            const unsigned int toPivot = column[k];
                            if (toPivot + minima[k] < maximum) {
                                relaxRow(row, paths + offset, rows + k * tileSize, width,
                                         toPivot, static_cast<unsigned int>(firstPivot + k));
                            }
                        }
                    }
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
