#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mpi.h>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t firstColumn, const unsigned int rangeMin,
                              const unsigned int rangeMax) {
    // rand_r advances the glibc LCG three times for each returned value.  Advancing
    // directly to this rank's contiguous column range avoids a serial initializer.
    const auto advanceLcg = [](unsigned int state, size_t steps) {
        uint64_t multiplier = 1103515245U;
        uint64_t increment = 12345U;
        uint64_t accumulatedMultiplier = 1;
        uint64_t accumulatedIncrement = 0;
        while (steps != 0) {
            if (steps & 1U) {
                accumulatedIncrement = accumulatedIncrement * multiplier + increment;
                accumulatedMultiplier *= multiplier;
            }
            increment = increment * (multiplier + 1U);
            multiplier *= multiplier;
            steps >>= 1U;
        }
        return static_cast<unsigned int>(accumulatedMultiplier * state + accumulatedIncrement);
    };

    // The original flattened layout is a contiguous sequence of complete columns.
    unsigned int seed = advanceLcg(42U, 3U * firstColumn * numNodes);
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < dist.size(); ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Local column c represents global destination firstColumn + c.
    for (size_t c = 0; c < dist.size() / numNodes; ++c) {
        const size_t globalColumn = firstColumn + c;
        dist[c * numNodes + globalColumn] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t firstColumn) {
    const size_t localColumns = path.size() / numNodes;
    for (size_t c = 0; c < localColumns; ++c) {
        const size_t j = firstColumn + c;
        for (size_t i = 0; i < numNodes; ++i) {
            path[c * numNodes + i] = j;
        }
        path[c * numNodes + j] = j;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes, const size_t firstColumn,
                   const size_t localColumns, const int rank, const int processCount) {
    std::vector<unsigned int> distanceToK(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        // Columns use [floor(r*n/p), floor((r+1)*n/p)); invert that
        // partitioning to find the rank which owns global column k.
        const int owner = static_cast<int>(((k + 1) * static_cast<size_t>(processCount) - 1) / numNodes);
        if (rank == owner) {
            const size_t localK = k - firstColumn;
            std::memcpy(distanceToK.data(), dist.data() + localK * numNodes,
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(distanceToK.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);

        // A local column is contiguous, unlike the original i-major traversal.
        for (size_t localJ = 0; localJ < localColumns; ++localJ) {
            unsigned int* const distanceColumn = dist.data() + localJ * numNodes;
            unsigned int* const pathColumn = path.data() + localJ * numNodes;
            const unsigned int distanceFromK = distanceColumn[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distanceToK[j] + distanceFromK;
                if (newDist < distanceColumn[j]) {
                    distanceColumn[j] = newDist;
                    pathColumn[j] = k;
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
    MPI_Init(&argc, &argv);
    int rank = 0;
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes == 0 || numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        if (rank == 0) printf("Number of nodes is out of range\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", processCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t firstColumn = (numNodes * static_cast<size_t>(rank)) / processCount;
    const size_t finalColumn = (numNodes * static_cast<size_t>(rank + 1)) / processCount;
    const size_t localColumns = finalColumn - firstColumn;
    
    // Allocate matrices
    std::vector<unsigned int> dist(localColumns * numNodes);
    std::vector<unsigned int> path(localColumns * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, firstColumn, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, firstColumn);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, firstColumn, localColumns, rank, processCount);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxElapsed * 1000.0);
        const double gflops = maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    // Print results for external validation (integer hash-based)
    int result = 0;
    if (printResults || validate) {
        std::vector<int> counts;
        std::vector<int> displacements;
        std::vector<unsigned int> globalDist;
        if (rank == 0) {
            counts.resize(processCount);
            displacements.resize(processCount);
            for (int r = 0; r < processCount; ++r) {
                const size_t begin = (numNodes * static_cast<size_t>(r)) / processCount;
                const size_t end = (numNodes * static_cast<size_t>(r + 1)) / processCount;
                counts[r] = static_cast<int>((end - begin) * numNodes);
                displacements[r] = static_cast<int>(begin * numNodes);
            }
            globalDist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                    rank == 0 ? globalDist.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displacements.data() : nullptr,
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            if (printResults) print_results_int(globalDist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                if (validateResult(globalDist, numNodes)) printf("Validation: PASSED\n");
                else { printf("Validation: FAILED\n"); result = 1; }
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
