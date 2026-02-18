#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

inline constexpr size_t rowsForRank(const int rank, const size_t base, const size_t rem) noexcept {
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

inline constexpr size_t startRowForRank(const int rank, const size_t base, const size_t rem) noexcept {
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

inline constexpr int ownerForRow(const size_t row, const size_t base, const size_t rem) noexcept {
    const size_t threshold = (base + 1) * rem;
    if (row < threshold) {
        return static_cast<int>(row / (base + 1));
    }
    if (base == 0) {
        return 0;
    }
    return static_cast<int>(rem + (row - threshold) / base);
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

void floydWarshall(std::vector<unsigned int>& distLocal, 
                   std::vector<unsigned int>& pathLocal, 
                   const size_t numNodes,
                   const size_t localRows,
                   const size_t localStart,
                   const int rank,
                   const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    std::vector<unsigned int> rowK(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerForRow(k, base, rem);
        if (owner == rank && localRows > 0) {
            const size_t localK = k - localStart;
            std::memcpy(rowK.data(),
                        distLocal.data() + localK * numNodes,
                        numNodes * sizeof(unsigned int));
        }

        MPI_Bcast(rowK.data(),
                  static_cast<int>(numNodes),
                  MPI_UNSIGNED,
                  owner,
                  MPI_COMM_WORLD);

        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* distRow = distLocal.data() + i * numNodes;
            unsigned int* pathRow = pathLocal.data() + i * numNodes;
            const unsigned int distIK = distRow[k];

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rowK[j];
                if (newDist < distRow[j]) {
                    distRow[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;
    
    // Parse command line arguments
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                parseStatus = 2;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                parseStatus = 1;
                break;
            }
        }
    }

    uint64_t numNodes64 = static_cast<uint64_t>(numNodes);
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&parseStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numNodes64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    numNodes = static_cast<size_t>(numNodes64);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t localRows = rowsForRank(rank, base, rem);
    const size_t localStart = startRowForRank(rank, base, rem);

    std::vector<int> counts(size);
    std::vector<int> displs(size);
    for (int r = 0; r < size; ++r) {
        const size_t rows = rowsForRank(r, base, rem);
        counts[r] = static_cast<int>(rows * numNodes);
        displs[r] = static_cast<int>(startRowForRank(r, base, rem) * numNodes);
    }

    // Allocate local matrices
    std::vector<unsigned int> distLocal(localRows * numNodes);
    std::vector<unsigned int> pathLocal(localRows * numNodes);
    unsigned int* distLocalPtr = distLocal.empty() ? nullptr : distLocal.data();
    unsigned int* pathLocalPtr = pathLocal.empty() ? nullptr : pathLocal.data();
    
    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> distFull;
    std::vector<unsigned int> pathFull;
    if (rank == 0) {
        distFull.resize(numNodes * numNodes);
        pathFull.resize(numNodes * numNodes);
        initializeDistanceMatrix(distFull, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(pathFull, numNodes);
    }

    MPI_Scatterv(rank == 0 ? distFull.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_UNSIGNED,
                 distLocalPtr,
                 counts[rank],
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? pathFull.data() : nullptr,
                 counts.data(),
                 displs.data(),
                 MPI_UNSIGNED,
                 pathLocalPtr,
                 counts[rank],
                 MPI_UNSIGNED,
                 0,
                 MPI_COMM_WORLD);

    if (rank == 0) {
        std::vector<unsigned int>().swap(distFull);
        std::vector<unsigned int>().swap(pathFull);
    }
    
    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(distLocal, pathLocal, numNodes, localRows, localStart, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localElapsed = end - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        const long long durationMs = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);
    }
    
    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    if (rank == 0) {
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    const bool needGather = printResults || validate;
    std::vector<unsigned int> distFullResult;
    if (needGather) {
        if (rank == 0) {
            distFullResult.resize(numNodes * numNodes);
        }
        MPI_Gatherv(distLocalPtr,
                    counts[rank],
                    MPI_UNSIGNED,
                    rank == 0 ? distFullResult.data() : nullptr,
                    counts.data(),
                    displs.data(),
                    MPI_UNSIGNED,
                    0,
                    MPI_COMM_WORLD);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(distFullResult, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
        }
        int validInt = 1;
        if (rank == 0) {
            validInt = validateResult(distFullResult, numNodes) ? 1 : 0;
            if (validInt) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&validInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Finalize();
        return validInt ? 0 : 1;
    }
    
    MPI_Finalize();
    return 0;
}
