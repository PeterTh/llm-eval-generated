#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <omp.h>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr size_t TILE_SIZE = 64;

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
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Update one cache-sized tile using all intermediate vertices in [kBegin, kEnd).
// dist and path are separate allocations, so restrict lets the compiler generate
// an efficient masked SIMD update for the innermost loop.
inline void updateTile(unsigned int* __restrict dist,
                       unsigned int* __restrict path,
                       const size_t numNodes,
                       const size_t iBegin,
                       const size_t iEnd,
                       const size_t jBegin,
                       const size_t jEnd,
                       const size_t kBegin,
                       const size_t kEnd) noexcept {
    for (size_t k = kBegin; k < kEnd; ++k) {
        const unsigned int* const rowK = dist + k * numNodes;

        for (size_t i = iBegin; i < iEnd; ++i) {
            unsigned int* const rowI = dist + i * numNodes;
            unsigned int* const pathRowI = path + i * numNodes;
            const unsigned int distIK = rowI[k];

            #pragma omp simd
            for (size_t j = jBegin; j < jEnd; ++j) {
                const unsigned int newDist = distIK + rowK[j];
                if (newDist < rowI[j]) {
                    rowI[j] = newDist;
                    pathRowI[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    // Blocked Floyd-Warshall has three dependency phases per diagonal tile:
    // close the diagonal tile, update its row and column, then update all
    // remaining tiles. A single persistent OpenMP team processes every phase.
    const size_t numTiles = (numNodes + TILE_SIZE - 1) / TILE_SIZE;
    const size_t offDiagonalTiles = numTiles - 1;
    const size_t phaseTwoTasks = 2 * offDiagonalTiles;
    const size_t phaseThreeTasks = offDiagonalTiles * offDiagonalTiles;
    // Threads beyond the largest phase would only wait at every barrier.
    const size_t maxUsefulThreads =
        std::max<size_t>(1, std::max(phaseTwoTasks, phaseThreeTasks));
    const int numThreads = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(omp_get_max_threads()), maxUsefulThreads));
    unsigned int* const distData = dist.data();
    unsigned int* const pathData = path.data();

    #pragma omp parallel num_threads(numThreads) default(none) \
            shared(distData, pathData, numNodes, numTiles, phaseTwoTasks)
    {
        for (size_t blockK = 0; blockK < numTiles; ++blockK) {
            const size_t kBegin = blockK * TILE_SIZE;
            const size_t kEnd = std::min(kBegin + TILE_SIZE, numNodes);

            // Phase 1: only the diagonal tile can be updated here.
            #pragma omp single
            updateTile(distData, pathData, numNodes,
                       kBegin, kEnd, kBegin, kEnd, kBegin, kEnd);

            // Phase 2: tiles in the pivot row and pivot column are independent.
            #pragma omp for schedule(static)
            for (size_t task = 0; task < phaseTwoTasks; ++task) {
                const bool rowTile = task < numTiles - 1;
                size_t block = rowTile ? task : task - (numTiles - 1);
                if (block >= blockK) {
                    ++block;
                }

                const size_t blockBegin = block * TILE_SIZE;
                const size_t blockEnd = std::min(blockBegin + TILE_SIZE, numNodes);
                if (rowTile) {
                    updateTile(distData, pathData, numNodes,
                               kBegin, kEnd, blockBegin, blockEnd, kBegin, kEnd);
                } else {
                    updateTile(distData, pathData, numNodes,
                               blockBegin, blockEnd, kBegin, kEnd, kBegin, kEnd);
                }
            }

            // Phase 3: no remaining tile shares output with another tile.
            #pragma omp for collapse(2) schedule(static)
            for (size_t blockI = 0; blockI < numTiles; ++blockI) {
                for (size_t blockJ = 0; blockJ < numTiles; ++blockJ) {
                    if (blockI == blockK || blockJ == blockK) {
                        continue;
                    }

                    const size_t iBegin = blockI * TILE_SIZE;
                    const size_t iEnd = std::min(iBegin + TILE_SIZE, numNodes);
                    const size_t jBegin = blockJ * TILE_SIZE;
                    const size_t jEnd = std::min(jBegin + TILE_SIZE, numNodes);
                    updateTile(distData, pathData, numNodes,
                               iBegin, iEnd, jBegin, jEnd, kBegin, kEnd);
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
