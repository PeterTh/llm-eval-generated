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
constexpr size_t TILE_SIZE = 32;

size_t tileCount(size_t numNodes) {
    return (numNodes + TILE_SIZE - 1) / TILE_SIZE;
}

int threadCount(size_t numNodes) {
    // Small matrices have too little work per tile to fill a large team.
    const auto maxThreads = static_cast<size_t>(omp_get_max_threads());
    return static_cast<int>(std::clamp(tileCount(numNodes), size_t{1}, maxThreads));
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
    #pragma omp parallel for schedule(static) num_threads(threadCount(numNodes))
    for (size_t j = 0; j < numNodes; ++j) {
        std::fill_n(path.data() + j * numNodes, numNodes, static_cast<unsigned int>(j));
    }
}

// Each tile consumes the distances to and from k as they were at the start
// of iteration k. The tiles intersecting the pivot block also record them.
// The zero diagonal means iteration k cannot improve its own row or column.
template <bool SaveRows, bool SaveColumns>
void updateTile(unsigned int* dist, unsigned int* path, size_t n,
                size_t rowBegin, size_t colBegin, size_t pivotBegin, size_t pivotEnd,
                unsigned int* pivotRows, unsigned int* pivotColumns, unsigned int* pivotRowMinima) {
    const size_t height = std::min(TILE_SIZE, n - rowBegin);
    const size_t width = std::min(TILE_SIZE, n - colBegin);
    // Keep all snapshots used by a tile contiguous, even for a large matrix.
    auto* rowHistory = pivotRows + colBegin * TILE_SIZE;
    auto* columnHistory = pivotColumns + rowBegin * TILE_SIZE;
    auto* rowMinHistory = pivotRowMinima + colBegin;
    unsigned int rowMax[TILE_SIZE];
    if constexpr (!SaveRows && !SaveColumns) {
        for (size_t i = 0; i < height; ++i) {
            const auto* row = dist + (rowBegin + i) * n + colBegin;
            unsigned int maxDistance = 0;
            #pragma omp simd reduction(max:maxDistance)
            for (size_t j = 0; j < width; ++j) {
                maxDistance = std::max(maxDistance, row[j]);
            }
            rowMax[i] = maxDistance;
        }
    }

    for (size_t k = pivotBegin; k < pivotEnd; ++k) {
        const size_t step = k - pivotBegin;
        auto* fromPivot = rowHistory + step * TILE_SIZE;
        auto* toPivot = columnHistory + step * TILE_SIZE;
        if constexpr (SaveRows) {
            const auto* pivotRow = dist + k * n + colBegin;
            unsigned int minDistance = pivotRow[0];
            #pragma omp simd reduction(min:minDistance)
            for (size_t j = 0; j < width; ++j) {
                fromPivot[j] = pivotRow[j];
                minDistance = std::min(minDistance, pivotRow[j]);
            }
            rowMinHistory[step] = minDistance;
        }
        if constexpr (SaveColumns) {
            for (size_t i = 0; i < height; ++i) {
                toPivot[i] = dist[(rowBegin + i) * n + k];
            }
        }
        for (size_t i = 0; i < height; ++i) {
            const unsigned int via = toPivot[i];
            if constexpr (!SaveRows && !SaveColumns) {
                // Benchmark sums fit in unsigned int, and distances only decrease.
                // Skip the row if even its smallest candidate is at least its initial maximum.
                if (via + rowMinHistory[step] >= rowMax[i]) {
                    continue;
                }
            }
            const size_t offset = (rowBegin + i) * n + colBegin;
            auto* distanceRow = dist + offset;
            auto* pathRow = path + offset;
            #pragma omp simd
            for (size_t j = 0; j < width; ++j) {
                const unsigned int candidate = via + fromPivot[j];
                const unsigned int previous = distanceRow[j];
                pathRow[j] = candidate < previous ? static_cast<unsigned int>(k) : pathRow[j];
                distanceRow[j] = std::min(candidate, previous);
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const size_t tiles = tileCount(numNodes);
    std::vector<unsigned int> pivotRows(tiles * TILE_SIZE * TILE_SIZE);
    std::vector<unsigned int> pivotColumns(pivotRows.size());
    std::vector<unsigned int> pivotRowMinima(tiles * TILE_SIZE);
    auto* distances = dist.data();
    auto* paths = path.data();
    auto* rows = pivotRows.data();
    auto* columns = pivotColumns.data();
    auto* minima = pivotRowMinima.data();

    #pragma omp parallel num_threads(threadCount(numNodes))
    {
        for (size_t pivot = 0; pivot < tiles; ++pivot) {
            const size_t begin = pivot * TILE_SIZE;
            const size_t end = std::min(begin + TILE_SIZE, numNodes);

            #pragma omp single
            {
                updateTile<true, true>(distances, paths, numNodes, begin, begin,
                                       begin, end, rows, columns, minima);
            }

            // Off-diagonal panels depend only on themselves and the pivot tile.
            #pragma omp for schedule(static)
            for (size_t panel = 0; panel < 2 * (tiles - 1); ++panel) {
                const size_t tile = panel / 2;
                const size_t offset = (tile + (tile >= pivot)) * TILE_SIZE;
                if (panel % 2 == 0) {
                    updateTile<true, false>(distances, paths, numNodes, begin, offset,
                                            begin, end, rows, columns, minima);
                } else {
                    updateTile<false, true>(distances, paths, numNodes, offset, begin,
                                            begin, end, rows, columns, minima);
                }
            }

            // Replay pivots in order with the recorded distances so strict
            // improvements record exactly the same k as in the original loops.
            #pragma omp for collapse(2) schedule(static)
            for (size_t row = 0; row < tiles - 1; ++row) {
                for (size_t col = 0; col < tiles - 1; ++col) {
                    const size_t rowBegin = (row + (row >= pivot)) * TILE_SIZE;
                    const size_t colBegin = (col + (col >= pivot)) * TILE_SIZE;
                    updateTile<false, false>(distances, paths, numNodes, rowBegin, colBegin,
                                             begin, end, rows, columns, minima);
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
    double gflops = ops / std::chrono::duration<double>(end - start).count() / 1e9;
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
