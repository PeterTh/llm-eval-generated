#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

inline size_t blockStart(const int rank, const size_t numNodes, const int numRanks) {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), rem);
}

inline size_t blockCount(const int rank, const size_t numNodes, const int numRanks) {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    return base + (static_cast<size_t>(rank) < rem ? 1 : 0);
}

inline int blockOwner(const size_t row, const size_t numNodes, const int numRanks) {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem = numNodes % static_cast<size_t>(numRanks);
    if (base == 0) {
        return static_cast<int>(row);
    }
    const size_t boundary = (base + 1) * rem;
    if (row < boundary) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - boundary) / base);
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t startRow, const size_t localRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t skip = startRow * numNodes;
    for (size_t i = 0; i < skip; ++i) {
        (void)rand_r(&seed);
    }

    const size_t elems = localRows * numNodes;
    for (size_t i = 0; i < elems; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < localRows; ++i) {
        dist[i * numNodes + (startRow + i)] = 0;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   const size_t numNodes,
                   const size_t startRow,
                   const size_t localRows,
                   MPI_Comm comm,
                   const int rank,
                   const int numRanks) {
    std::vector<unsigned int> kRow(numNodes);

    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = blockOwner(k, numNodes, numRanks);
        if (rank == owner) {
            std::memcpy(kRow.data(), dist.data() + (k - startRow) * numNodes,
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(kRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);

        const unsigned int* __restrict__ rowK = kRow.data();

        // For each local source node r
        for (size_t r = 0; r < localRows; ++r) {
            unsigned int* __restrict__ dr = dist.data() + r * numNodes;
            const unsigned int distRK = dr[k];

            // For each destination node c
            for (size_t c = 0; c < numNodes; ++c) {
                const unsigned int newDist = distRK + rowK[c];
                if (newDist < dr[c]) {
                    dr[c] = newDist;
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
    bool showHelp = false;
    bool badOption = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            badOption = true;
        }
    }

    if (showHelp) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 0;
    }

    if (badOption) {
        if (rank == 0) {
            printf("Unknown option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const size_t startRow = blockStart(rank, numNodes, numRanks);
    const size_t localRows = blockCount(rank, numNodes, numRanks);

    // Allocate matrix
    std::vector<unsigned int> dist(localRows * numNodes);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, startRow, localRows, 1, MAX_DISTANCE);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    floydWarshall(dist, numNodes, startRow, localRows,
                  MPI_COMM_WORLD, rank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    const double endTime = MPI_Wtime();

    const long durationMs = static_cast<long>((endTime - startTime) * 1000.0);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    const bool needFullMatrix = printResults || validate;
    std::vector<unsigned int> fullDist;

    if (needFullMatrix) {
        std::vector<int> recvCounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            recvCounts[r] = static_cast<int>(blockCount(r, numNodes, numRanks) * numNodes);
            displs[r] = static_cast<int>(blockStart(r, numNodes, numRanks) * numNodes);
        }

        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
        }

        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr,
                    recvCounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation
    if (validate) {
        bool valid = false;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(fullDist, numNodes);
        }

        int validFlag = valid ? 1 : 0;
        MPI_Bcast(&validFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (validFlag) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return validFlag ? 0 : 1;
    }

    MPI_Finalize();
    return 0;
}
