#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <omp.h>
#include <vector>

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

static inline void relaxRow(unsigned int* row, unsigned int* intermediates,
                            const unsigned int* pivotRow, size_t width,
                            unsigned int toPivot, unsigned int pivot) {
    #pragma omp simd
    for (size_t j = 0; j < width; ++j) {
        const unsigned int candidate = toPivot + pivotRow[j];
        if (candidate < row[j]) {
            row[j] = candidate;
            intermediates[j] = pivot;
        }
    }
}

static inline void relaxRowPair(unsigned int* row, unsigned int* intermediates,
                                const unsigned int* firstRow, const unsigned int* secondRow,
                                size_t width, unsigned int toFirst, unsigned int toSecond,
                                unsigned int first, unsigned int second) {
    // Both candidates use saved distances. The first pivot wins ties.
    #pragma omp simd
    for (size_t j = 0; j < width; ++j) {
        const unsigned int firstDistance = toFirst + firstRow[j];
        const unsigned int secondDistance = toSecond + secondRow[j];
        const unsigned int candidate = std::min(firstDistance, secondDistance);
        if (candidate < row[j]) {
            row[j] = candidate;
            intermediates[j] = secondDistance < firstDistance ? second : first;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    constexpr size_t tileSize = 64;  // One candidate bit per pivot.
    const size_t tiles = (numNodes + tileSize - 1) / tileSize;
    const size_t panelTiles = tiles > 0 ? tiles - 1 : 0;
    const int threads = static_cast<int>(std::min<size_t>(
        omp_get_max_threads(), std::max<size_t>(1, 2 * panelTiles)));

    // Record each pivot row and column at the point where the serial algorithm
    // uses them. Later improvements to those rows and columns must not change
    // which pivot first reaches a shortest distance in another tile.
    // Pack the saved rows by destination tile for contiguous access in the
    // innermost tile updates, even when the full matrix has a large stride.
    std::vector<unsigned int> pivotRows(tiles * tileSize * tileSize);
    std::vector<unsigned int> pivotRowMin(tiles * tileSize);
    std::vector<unsigned int> pivotColumns(tileSize * numNodes);

    #pragma omp parallel num_threads(threads)
    {
        for (size_t block = 0; block < tiles; ++block) {
            const size_t begin = block * tileSize;
            const size_t end = std::min(begin + tileSize, numNodes);
            const size_t width = end - begin;
            auto* diagonalRows = pivotRows.data() + block * tileSize * tileSize;

            // A zero diagonal keeps the active row and column unchanged by
            // their own pivot, so each can be saved before its relaxation.
            #pragma omp single
            {
                for (size_t k = begin; k < end; ++k) {
                    const size_t pivot = k - begin;
                    auto* savedRow = diagonalRows + pivot * tileSize;
                    std::copy_n(dist.data() + k * numNodes + begin, width, savedRow);
                    for (size_t i = begin; i < end; ++i) {
                        const size_t row = i * numNodes;
                        const unsigned int toPivot = dist[row + k];
                        pivotColumns[i * tileSize + pivot] = toPivot;
                        relaxRow(dist.data() + row + begin, path.data() + row + begin,
                                 savedRow, width, toPivot, k);
                    }
                }
            }

            // The pivot row and column tiles depend only on the diagonal tile
            // and their own earlier updates within this block of pivots.
            #pragma omp for schedule(static)
            for (size_t panel = 0; panel < 2 * panelTiles; ++panel) {
                const size_t other = panel / 2;
                const size_t tile = other + (other >= block);
                const size_t first = tile * tileSize;
                const size_t last = std::min(first + tileSize, numNodes);
                if (panel % 2 == 0) {
                    auto* savedRows = pivotRows.data() + tile * tileSize * tileSize;
                    auto* lowerBounds = pivotRowMin.data() + tile * tileSize;
                    for (size_t k = begin; k < end; ++k) {
                        const size_t pivot = k - begin;
                        auto* savedRow = savedRows + pivot * tileSize;
                        const auto* row = dist.data() + k * numNodes + first;
                        unsigned int lowerBound = std::numeric_limits<unsigned int>::max();
                        #pragma omp simd reduction(min:lowerBound)
                        for (size_t j = 0; j < last - first; ++j) {
                            savedRow[j] = row[j];
                            lowerBound = std::min(lowerBound, row[j]);
                        }
                        lowerBounds[pivot] = lowerBound;
                        for (size_t i = begin; i < end; ++i) {
                            const size_t offset = i * numNodes + first;
                            relaxRow(dist.data() + offset, path.data() + offset, savedRow,
                                     last - first, pivotColumns[i * tileSize + pivot], k);
                        }
                    }
                } else {
                    for (size_t i = first; i < last; ++i) {
                        const size_t offset = i * numNodes + begin;
                        for (size_t k = begin; k < end; ++k) {
                            const size_t pivot = k - begin;
                            const unsigned int toPivot = dist[i * numNodes + k];
                            pivotColumns[i * tileSize + pivot] = toPivot;
                            relaxRow(dist.data() + offset, path.data() + offset,
                                     diagonalRows + pivot * tileSize, width, toPivot, k);
                        }
                    }
                }
            }

            // Every remaining tile can now replay the same pivot inputs
            // independently. The barrier at the end completes this block.
            #pragma omp for collapse(2) schedule(static)
            for (size_t rowTile = 0; rowTile < panelTiles; ++rowTile) {
                for (size_t colTile = 0; colTile < panelTiles; ++colTile) {
                    const size_t firstRow = (rowTile + (rowTile >= block)) * tileSize;
                    const size_t tile = colTile + (colTile >= block);
                    const size_t firstCol = tile * tileSize;
                    const size_t lastRow = std::min(firstRow + tileSize, numNodes);
                    const size_t cols = std::min(tileSize, numNodes - firstCol);
                    const auto* savedRows = pivotRows.data() + tile * tileSize * tileSize;
                    const auto* lowerBounds = pivotRowMin.data() + tile * tileSize;
                    for (size_t i = firstRow; i < lastRow; ++i) {
                        const size_t offset = i * numNodes + firstCol;
                        auto* row = dist.data() + offset;
                        auto* intermediates = path.data() + offset;
                        const auto* toPivots = pivotColumns.data() + i * tileSize;

                        // The upper bound stays valid as distances decrease.
                        // Sums of benchmark distances fit in unsigned int.
                        unsigned int upperBound = 0;
                        #pragma omp simd reduction(max:upperBound)
                        for (size_t j = 0; j < cols; ++j) {
                            upperBound = std::max(upperBound, row[j]);
                        }
                        std::uint64_t candidates = 0;
                        #pragma omp simd reduction(|:candidates)
                        for (size_t pivot = 0; pivot < width; ++pivot) {
                            const bool possible = toPivots[pivot] + lowerBounds[pivot] < upperBound;
                            candidates |= std::uint64_t{possible} << pivot;
                        }
                        while (candidates != 0) {
                            const size_t first = std::countr_zero(candidates);
                            candidates &= candidates - 1;
                            if (candidates == 0) {
                                relaxRow(row, intermediates, savedRows + first * tileSize,
                                         cols, toPivots[first], begin + first);
                                break;
                            }
                            const size_t second = std::countr_zero(candidates);
                            candidates &= candidates - 1;
                            relaxRowPair(row, intermediates, savedRows + first * tileSize,
                                         savedRows + second * tileSize, cols, toPivots[first],
                                         toPivots[second], begin + first, begin + second);
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
