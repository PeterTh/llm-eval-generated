#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
                   const std::vector<int>& rowOffsets, const int rank) {
    std::vector<unsigned int> pivot(numNodes);
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    int owner = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        while (k >= static_cast<size_t>(rowOffsets[owner] + rowCounts[owner])) {
            ++owner;
        }
        unsigned int* const pivotRow = rank == owner
            ? dist.data() + (k - firstRow) * numNodes : pivot.data();
        MPI_Bcast(pivotRow, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // The pivot row is unchanged because all edge weights are nonnegative
        // and its diagonal entry is zero.
        for (size_t localRow = 0; localRow < localRows; ++localRow) {
            if (firstRow + localRow == k) continue;
            unsigned int* const row = dist.data() + localRow * numNodes;
            unsigned int* const pathRow = path.data() + localRow * numNodes;
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
    if (numNodes != 0 &&
        numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) / numNodes) {
        if (rank == 0) fprintf(stderr, "Number of nodes is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const size_t baseRows = numNodes / static_cast<size_t>(ranks);
    const size_t extraRows = numNodes % static_cast<size_t>(ranks);
    std::vector<int> rowCounts(ranks), rowOffsets(ranks);
    std::vector<int> elementCounts(ranks), elementOffsets(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < extraRows);
        rowCounts[r] = static_cast<int>(rows);
        rowOffsets[r] = static_cast<int>(offset);
        elementCounts[r] = static_cast<int>(rows * numNodes);
        elementOffsets[r] = static_cast<int>(offset * numNodes);
        offset += rows;
    }
    const size_t firstRow = static_cast<size_t>(rowOffsets[rank]);
    const size_t localRows = static_cast<size_t>(rowCounts[rank]);
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    std::vector<unsigned int> fullDist;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, elementCounts.data(),
                 elementOffsets.data(), MPI_UNSIGNED, dist.data(), elementCounts[rank],
                 MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    for (size_t i = 0; i < localRows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes,
                    static_cast<unsigned int>(firstRow + i));
    }
    if (rank == 0) {
        std::vector<unsigned int>().swap(fullDist);
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, firstRow, rowCounts, rowOffsets, rank);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / seconds / 1e9);
    }

    if (printResults || validate) {
        if (rank == 0) fullDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), elementCounts[rank], MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, elementCounts.data(),
                    elementOffsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    int result = 0;
    if (rank == 0) {
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
