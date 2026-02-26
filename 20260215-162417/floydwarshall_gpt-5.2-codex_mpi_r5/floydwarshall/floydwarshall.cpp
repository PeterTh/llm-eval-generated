#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
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

struct RowDistribution {
    size_t startRow;
    size_t localRows;
};

inline RowDistribution getRowDistribution(const size_t numNodes, const int rank, const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    RowDistribution dist{base * r + std::min(r, rem), base + (r < rem ? 1U : 0U)};
    return dist;
}

inline int ownerOfRow(const size_t row, const size_t numNodes, const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    if (base == 0) {
        return static_cast<int>(row);
    }
    const size_t pivot = (base + 1) * rem;
    if (row < pivot) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - pivot) / base);
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

void initializePathMatrixLocal(std::vector<unsigned int>& path, const size_t numNodes,
                               const size_t startRow, const size_t localRows) {
    for (size_t local = 0; local < localRows; ++local) {
        const size_t i = startRow + local;
        unsigned int* row = path.data() + local * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = static_cast<unsigned int>(i);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    // Classic Floyd-Warshall algorithm
    // For each intermediate node k
    for (size_t k = 0; k < numNodes; ++k) {
        // For each source node i
        for (size_t i = 0; i < numNodes; ++i) {
            // For each destination node j
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                const unsigned int newDist = distIK + distKJ;
                
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
}

void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const size_t startRow,
                      const size_t localRows,
                      const int size) {
    std::vector<unsigned int> rowK(numNodes);
    const int nCount = static_cast<int>(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, size);
        unsigned int* rowKPtr = rowK.data();
        if (localRows > 0 && k >= startRow && k < startRow + localRows) {
            const size_t localIndex = k - startRow;
            rowKPtr = localDist.data() + localIndex * numNodes;
        }

        MPI_Bcast(rowKPtr, nCount, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t local = 0; local < localRows; ++local) {
            unsigned int* rowI = localDist.data() + local * numNodes;
            unsigned int* pathRowI = localPath.data() + local * numNodes;
            const unsigned int distIK = rowI[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = rowI[j];
                const unsigned int newDist = distIK + rowKPtr[j];
                if (newDist < distIJ) {
                    rowI[j] = newDist;
                    pathRowI[j] = static_cast<unsigned int>(k);
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
    int exitCode = 0;
    int showHelp = 0;
    
    // Parse command line arguments on rank 0
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
                showHelp = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&showHelp, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (showHelp || exitCode != 0) {
        MPI_Finalize();
        return exitCode;
    }

    uint64_t numNodes64 = rank == 0 ? static_cast<uint64_t>(numNodes) : 0;
    int validateInt = validate ? 1 : 0;
    int printResultsInt = printResults ? 1 : 0;
    MPI_Bcast(&numNodes64, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(numNodes64);
    validate = validateInt != 0;
    printResults = printResultsInt != 0;

    const RowDistribution distInfo = getRowDistribution(numNodes, rank, size);
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);

    int errorFlag = 0;
    std::vector<int> counts;
    std::vector<int> displs;
    if (rank == 0) {
        if (numNodes > static_cast<size_t>(INT_MAX)) {
            printf("Error: number of nodes is too large for MPI counts\n");
            errorFlag = 1;
        }
        if (!errorFlag) {
            counts.resize(size);
            displs.resize(size);
            size_t offset = 0;
            for (int r = 0; r < size; ++r) {
                const size_t rows = base + (static_cast<size_t>(r) < rem ? 1U : 0U);
                const size_t elems = rows * numNodes;
                if (elems > static_cast<size_t>(INT_MAX) || offset > static_cast<size_t>(INT_MAX)) {
                    errorFlag = 1;
                    break;
                }
                counts[r] = static_cast<int>(elems);
                displs[r] = static_cast<int>(offset);
                offset += elems;
            }
            if (errorFlag) {
                printf("Error: matrix size exceeds MPI limits\n");
            }
        }
    }

    MPI_Bcast(&errorFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (errorFlag) {
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    std::vector<unsigned int> localDist(distInfo.localRows * numNodes);
    std::vector<unsigned int> localPath(distInfo.localRows * numNodes);

    if (rank == 0) {
        std::vector<unsigned int> distGlobal(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(distGlobal, numNodes, 1, MAX_DISTANCE);
        MPI_Scatterv(distGlobal.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                     localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0,
                     MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED, localDist.data(),
                     static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    initializePathMatrixLocal(localPath, numNodes, distInfo.startRow, distInfo.localRows);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallMPI(localDist, localPath, numNodes, distInfo.startRow, distInfo.localRows, size);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const double localTime = end - start;
    double maxTime = 0.0;
    MPI_Reduce(&localTime, &maxTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long ms = static_cast<long>(maxTime * 1000.0);
        printf("Computation time: %ld ms\n", ms);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = maxTime > 0.0 ? ops / maxTime / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> distGlobal;
    if (printResults || validate) {
        if (rank == 0) {
            distGlobal.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                    rank == 0 ? distGlobal.data() : nullptr,
                    rank == 0 ? counts.data() : nullptr, rank == 0 ? displs.data() : nullptr,
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    int finalExit = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(distGlobal, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(distGlobal, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
                finalExit = 0;
            } else {
                printf("Validation: FAILED\n");
                finalExit = 1;
            }
        }
    }

    MPI_Bcast(&finalExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return finalExit;
}
