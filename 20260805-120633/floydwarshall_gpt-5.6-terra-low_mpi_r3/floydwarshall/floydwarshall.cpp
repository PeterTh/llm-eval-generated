#include <algorithm>
#include <chrono>
#include <cmath>
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

int ownerOfRow(const size_t row, const size_t numNodes, const int numRanks) {
    const size_t baseRows = numNodes / static_cast<size_t>(numRanks);
    const size_t extraRows = numNodes % static_cast<size_t>(numRanks);
    const size_t largerBlockRows = (baseRows + 1) * extraRows;
    if (row < largerBlockRows) {
        return static_cast<int>(row / (baseRows + 1));
    }
    return static_cast<int>(extraRows + (row - largerBlockRows) / baseRows);
}

// The flattened layout stores a complete destination row contiguously.  Assigning
// destination rows to ranks therefore makes every local update contiguous in i.
void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const size_t localRows,
                      const int rank,
                      const int numRanks) {
    std::vector<unsigned int> kthRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, numRanks);
        if (rank == owner) {
            const size_t firstLocalRow = (numNodes / static_cast<size_t>(numRanks)) *
                                         static_cast<size_t>(rank) +
                std::min(static_cast<size_t>(rank), numNodes % static_cast<size_t>(numRanks));
            std::memcpy(kthRow.data(), localDist.data() + (k - firstLocalRow) * numNodes,
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(kthRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t localJ = 0; localJ < localRows; ++localJ) {
            unsigned int* const distRow = localDist.data() + localJ * numNodes;
            unsigned int* const pathRow = localPath.data() + localJ * numNodes;
            const unsigned int distKJ = distRow[k];
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDist = kthRow[i] + distKJ;
                if (newDist < distRow[i]) {
                    distRow[i] = newDist;
                    pathRow[i] = static_cast<unsigned int>(k);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

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
        numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) / numNodes) {
        if (rank == 0) printf("Number of nodes is out of range for MPI matrix distribution\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate matrices
    const size_t baseRows = numNodes / static_cast<size_t>(numRanks);
    const size_t extraRows = numNodes % static_cast<size_t>(numRanks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);
    std::vector<int> counts(numRanks), displacements(numRanks);
    for (int r = 0; r < numRanks; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows ? 1 : 0);
        counts[r] = static_cast<int>(rows * numNodes);
        displacements[r] = r == 0 ? 0 : displacements[r - 1] + counts[r - 1];
    }
    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);
    std::vector<unsigned int> dist;
    std::vector<unsigned int> path;
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshallMPI(localDist, localPath, numNodes, localRows, rank, numRanks);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long long localDuration = static_cast<long long>(duration.count());
    long long globalDuration = 0;
    MPI_Allreduce(&localDuration, &globalDuration, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Computation time: %lld ms\n", globalDuration);
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (globalDuration / 1000.0) / 1e9;
    if (rank == 0) printf("Performance: %.3f GOPS\n", gflops);

    if (printResults || validate) {
        if (rank == 0) {
            dist.resize(numNodes * numNodes);
            path.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? dist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
        MPI_Gatherv(localPath.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? path.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(dist, "DistanceMatrix");
    }
    
    // Validation
    int result = 0;
    if (validate && rank == 0) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            result = 0;
        } else {
            printf("Validation: FAILED\n");
            result = 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
