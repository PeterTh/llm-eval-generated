#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Count distinct physical cores. Running one thread per physical core avoids
// SMT oversubscription, which otherwise degrades this memory-bound kernel.
int physicalCoreCount() {
    std::ifstream file("/proc/cpuinfo");
    if (!file) {
        return 0;
    }

    std::set<std::pair<int, int>> cores;
    std::string line;
    int physicalId = -1;
    int coreId = -1;

    auto record = [&]() {
        if (physicalId >= 0 && coreId >= 0) {
            cores.emplace(physicalId, coreId);
        }
        physicalId = -1;
        coreId = -1;
    };

    while (std::getline(file, line)) {
        if (line.empty()) {
            record();
        } else if (line.compare(0, 11, "physical id") == 0) {
            physicalId = atoi(line.c_str() + line.find(':') + 1);
        } else if (line.compare(0, 7, "core id") == 0) {
            coreId = atoi(line.c_str() + line.find(':') + 1);
        }
    }
    record();

    return static_cast<int>(cores.size());
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const size_t total = numNodes * numNodes;

    // First-touch the matrix in parallel so pages are distributed evenly across
    // NUMA nodes before the deterministic serial PRNG fill below.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < total; ++i) {
        dist[i] = 0;
    }

    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < total; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    // Final value of every cell is its column index, so each column can be
    // filled independently.
    #pragma omp parallel for schedule(static)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = static_cast<unsigned int>(j);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    unsigned int* const distBase = dist.data();
    unsigned int* const pathBase = path.data();

    // Classic Floyd-Warshall algorithm. The k loop is sequential because each
    // iteration depends on the previous one, but the independent i (column)
    // loop is distributed across threads. The innermost j loop is contiguous
    // in the column-major layout and is vectorized.
    #pragma omp parallel proc_bind(spread)
    {
        for (size_t k = 0; k < numNodes; ++k) {
            // Column k: dist[j][k] for all j (contiguous).
            const unsigned int* __restrict colK = distBase + idx2(0, k, numNodes);

            #pragma omp for schedule(static)
            for (size_t i = 0; i < numNodes; ++i) {
                // Column k is never modified during this iteration, so i == k
                // performs no updates; skipping it also avoids aliasing colI
                // with colK.
                if (i == k) {
                    continue;
                }

                // Column i: dist[j][i] for all j (contiguous).
                unsigned int* __restrict colI = distBase + idx2(0, i, numNodes);
                unsigned int* __restrict pathI = pathBase + idx2(0, i, numNodes);
                const unsigned int distIK = colI[k];

                #pragma omp simd
                for (size_t j = 0; j < numNodes; ++j) {
                    const unsigned int newDist = distIK + colK[j];
                    if (newDist < colI[j]) {
                        colI[j] = newDist;
                        pathI[j] = static_cast<unsigned int>(k);
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
    
    // Avoid oversubscribing SMT threads for this memory-bound kernel.
    const int cores = physicalCoreCount();
    if (cores > 0 && cores < omp_get_max_threads()) {
        omp_set_num_threads(cores);
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
