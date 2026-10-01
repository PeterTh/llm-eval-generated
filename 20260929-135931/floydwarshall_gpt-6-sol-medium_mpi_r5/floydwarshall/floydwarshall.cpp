#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>

#include <mpi.h>

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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t firstRow,
                   const std::vector<int>& rowCounts,
                   const std::vector<int>& rowOffsets, const int rank,
                   MPI_Comm comm) {
    std::vector<unsigned int> pivot(numNodes);
    const size_t localRows = numNodes == 0 ? 0 : dist.size() / numNodes;
    int owner = 0;

    for (size_t k = 0; k < numNodes; ++k) {
        while (k >= static_cast<size_t>(rowOffsets[owner] + rowCounts[owner])) {
            ++owner;
        }

        // Rows stay local. Only the row whose source is k is needed elsewhere.
        unsigned int* pivotRow = rank == owner
            ? dist.data() + (k - firstRow) * numNodes : pivot.data();
        MPI_Bcast(pivotRow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);

        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* row = dist.data() + i * numNodes;
            unsigned int* pathRow = path.data() + i * numNodes;
            const unsigned int distIK = row[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivotRow[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
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
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (worldRank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // MPI's vector collectives use int counts and displacements.
    if (numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes > static_cast<size_t>(INT_MAX) / std::max<size_t>(numNodes, 1)) {
        if (worldRank == 0) fprintf(stderr, "Number of nodes is too large for MPI matrix transfers\n");
        MPI_Finalize();
        return 1;
    }

    const int activeRanks = static_cast<int>(std::min<size_t>(std::max<size_t>(numNodes, 1), worldSize));
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    int result = 0;
    if (comm != MPI_COMM_NULL) {
        const int rank = worldRank;
        std::vector<int> rowCounts(activeRanks), rowOffsets(activeRanks);
        std::vector<int> counts(activeRanks), displacements(activeRanks);
        for (int p = 0; p < activeRanks; ++p) {
            rowOffsets[p] = static_cast<int>((numNodes * p) / activeRanks);
            rowCounts[p] = static_cast<int>((numNodes * (p + 1)) / activeRanks) - rowOffsets[p];
            counts[p] = rowCounts[p] * static_cast<int>(numNodes);
            displacements[p] = rowOffsets[p] * static_cast<int>(numNodes);
        }

        std::vector<unsigned int> dist;
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
            dist.resize(numNodes * numNodes);
            initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        }

        const size_t firstRow = static_cast<size_t>(rowOffsets[rank]);
        const size_t localRows = static_cast<size_t>(rowCounts[rank]);
        std::vector<unsigned int> localDist(localRows * numNodes);
        std::vector<unsigned int> localPath(localRows * numNodes);
        for (size_t i = 0; i < localRows; ++i) {
            std::fill_n(localPath.data() + i * numNodes, numNodes,
                        static_cast<unsigned int>(firstRow + i));
        }
        MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(),
                     displacements.data(), MPI_UNSIGNED, localDist.data(),
                     counts[rank], MPI_UNSIGNED, 0, comm);
        if (rank == 0) {
            if (!printResults && !validate) std::vector<unsigned int>().swap(dist);
            printf("Computing shortest paths...\n");
        }

        const double start = MPI_Wtime();
        floydWarshall(localDist, localPath, numNodes, firstRow,
                      rowCounts, rowOffsets, rank, comm);
        const double localSeconds = MPI_Wtime() - start;
        double seconds = 0;
        MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

        if (printResults || validate) {
            MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED,
                        rank == 0 ? dist.data() : nullptr, counts.data(),
                        displacements.data(), MPI_UNSIGNED, 0, comm);
        }

        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", ops / seconds / 1e9);
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                result = validateResult(dist, numNodes) ? 0 : 1;
                printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
            }
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
