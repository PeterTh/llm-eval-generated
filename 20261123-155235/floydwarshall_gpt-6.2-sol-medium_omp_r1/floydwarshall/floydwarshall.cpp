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
    for (size_t j = 0; j < numNodes; ++j) {
        std::fill_n(path.data() + j * numNodes, numNodes, static_cast<unsigned int>(j));
    }
}

namespace {

constexpr size_t TILE_SIZE = 32;

inline void relaxRow(unsigned int* distances, unsigned int* paths,
                     const unsigned int* pivotRow, size_t width,
                     unsigned int toPivot, unsigned int pivot) {
    #pragma omp simd
    for (size_t j = 0; j < width; ++j) {
        const unsigned int candidate = toPivot + pivotRow[j];
        if (candidate < distances[j]) {
            distances[j] = candidate;
            paths[j] = pivot;
        }
    }
}

// Advance a tile on the pivot rows or columns, saving its operands before
// each pivot so the other tiles see exactly the same values in the same order.
template<bool saveRows, bool saveColumns>
void advancePanel(unsigned int* dist, unsigned int* path, size_t n,
                  unsigned int* pivotRows, unsigned int* pivotColumns,
                  unsigned int* pivotMinima,
                  size_t firstPivot, size_t pivotCount, size_t rowStart, size_t colStart) {
    const size_t rowEnd = std::min(rowStart + TILE_SIZE, n);
    const size_t width = std::min(TILE_SIZE, n - colStart);
    auto* tileRows = pivotRows + colStart * TILE_SIZE;
    auto* tileMinima = pivotMinima + colStart;
    for (size_t p = 0; p < pivotCount; ++p) {
        const size_t k = firstPivot + p;
        auto* pivotRow = tileRows + p * width;
        if constexpr (saveRows) {
            const auto* source = dist + k * n + colStart;
            unsigned int minimum = INF;
            #pragma omp simd reduction(min:minimum)
            for (size_t j = 0; j < width; ++j) {
                pivotRow[j] = source[j];
                minimum = std::min(minimum, source[j]);
            }
            tileMinima[p] = minimum;
        }
        if constexpr (saveColumns) {
            for (size_t i = rowStart; i < rowEnd; ++i) {
                pivotColumns[i * TILE_SIZE + p] = dist[i * n + k];
            }
        }
        for (size_t i = rowStart; i < rowEnd; ++i) {
            const size_t offset = i * n + colStart;
            relaxRow(dist + offset, path + offset, pivotRow, width,
                     pivotColumns[i * TILE_SIZE + p], k);
        }
    }
}

void advanceTile(unsigned int* dist, unsigned int* path, size_t n,
                 const unsigned int* pivotRows, const unsigned int* pivotColumns,
                 const unsigned int* pivotMinima,
                 size_t firstPivot, size_t pivotCount, size_t rowStart, size_t colStart) {
    const size_t rowEnd = std::min(rowStart + TILE_SIZE, n);
    const size_t width = std::min(TILE_SIZE, n - colStart);
    const auto* tileRows = pivotRows + colStart * TILE_SIZE;
    const auto* tileMinima = pivotMinima + colStart;
    for (size_t i = rowStart; i < rowEnd; ++i) {
        auto* distances = dist + i * n + colStart;
        auto* paths = path + i * n + colStart;
        unsigned int upperBound = 0;
        #pragma omp simd reduction(max:upperBound)
        for (size_t j = 0; j < width; ++j) {
            upperBound = std::max(upperBound, distances[j]);
        }

        // Distances only decrease, so skip the row if even the smallest
        // candidate cannot improve any of its entries.
        for (size_t p = 0; p < pivotCount; ++p) {
            const unsigned int toPivot = pivotColumns[i * TILE_SIZE + p];
            if (toPivot + tileMinima[p] < upperBound) {
                relaxRow(distances, paths, tileRows + p * width,
                         width, toPivot, firstPivot + p);
            }
        }
    }
}

} // namespace

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    const size_t tileCount = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const size_t outerTiles = tileCount > 0 ? tileCount - 1 : 0;
    // Use roughly one tile's worth of rows per worker to limit barrier overhead.
    const int threads = std::min<size_t>(omp_get_max_threads(), std::max<size_t>(1, tileCount));
    // Group row snapshots by destination tile and column snapshots by source row.
    std::vector<unsigned int> pivotRows(TILE_SIZE * numNodes);
    std::vector<unsigned int> pivotColumns(TILE_SIZE * numNodes);
    std::vector<unsigned int> pivotMinima(TILE_SIZE * tileCount);
    auto* distances = dist.data();
    auto* paths = path.data();
    auto* rows = pivotRows.data();
    auto* columns = pivotColumns.data();
    auto* minima = pivotMinima.data();

    // Each pivot's row and column stay fixed during its own iteration because
    // its diagonal entry is zero. Barriers publish those operands in stages.
    #pragma omp parallel num_threads(threads)
    {
        for (size_t block = 0; block < tileCount; ++block) {
            const size_t firstPivot = block * TILE_SIZE;
            const size_t pivotCount = std::min(TILE_SIZE, numNodes - firstPivot);

            // The diagonal tile supplies the operands shared by both panels.
            #pragma omp single
            advancePanel<true, true>(distances, paths, numNodes, rows, columns, minima,
                                     firstPivot, pivotCount, firstPivot, firstPivot);

            // Each panel tile supplies one slice of the pivot rows or columns.
            #pragma omp for schedule(static)
            for (size_t panel = 0; panel < 2 * outerTiles; ++panel) {
                const size_t index = panel / 2;
                const size_t start = (index + (index >= block)) * TILE_SIZE;
                if (panel % 2 == 0) {
                    advancePanel<true, false>(distances, paths, numNodes, rows, columns, minima,
                                              firstPivot, pivotCount, firstPivot, start);
                } else {
                    advancePanel<false, true>(distances, paths, numNodes, rows, columns, minima,
                                              firstPivot, pivotCount, start, firstPivot);
                }
            }

            // All other tiles can now use the saved operands independently.
            #pragma omp for collapse(2) schedule(static)
            for (size_t i = 0; i < outerTiles; ++i) {
                for (size_t j = 0; j < outerTiles; ++j) {
                    const size_t rowStart = (i + (i >= block)) * TILE_SIZE;
                    const size_t colStart = (j + (j >= block)) * TILE_SIZE;
                    advanceTile(distances, paths, numNodes, rows, columns, minima,
                                firstPivot, pivotCount, rowStart, colStart);
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
