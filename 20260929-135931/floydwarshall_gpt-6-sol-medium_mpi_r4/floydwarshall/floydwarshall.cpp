#include <algorithm>
#include <climits>
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

size_t firstRow(const int rank, const size_t numNodes, const int ranks) {
    const size_t base = numNodes / static_cast<size_t>(ranks);
    const size_t extra = numNodes % static_cast<size_t>(ranks);
    return static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), extra);
}

size_t rowCount(const int rank, const size_t numNodes, const int ranks) {
    return numNodes / static_cast<size_t>(ranks) +
           (static_cast<size_t>(rank) < numNodes % static_cast<size_t>(ranks));
}

int ownerOfRow(const size_t row, const size_t numNodes, const int ranks) {
    const size_t base = numNodes / static_cast<size_t>(ranks);
    const size_t extra = numNodes % static_cast<size_t>(ranks);
    const size_t longRows = extra * (base + 1);
    return static_cast<int>(row < longRows ? row / (base + 1)
                                            : extra + (row - longRows) / base);
}

// Generate rows in the original rand_r order without storing the full graph on rank 0.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t localRows, const int rank, const int ranks) {
    if (rank == 0) {
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        std::vector<unsigned int> row(numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                row[j] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
            }
            row[i] = 0;
            const int owner = ownerOfRow(i, numNodes, ranks);
            if (owner == 0) {
                std::copy(row.begin(), row.end(), dist.begin() + i * numNodes);
            } else {
                MPI_Send(row.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, 0, MPI_COMM_WORLD);
            }
        }
    } else {
        for (size_t i = 0; i < localRows; ++i) {
            MPI_Recv(dist.data() + i * numNodes, static_cast<int>(numNodes), MPI_UNSIGNED,
                     0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t startRow, const int rank, const int ranks) {
    const size_t localRows = numNodes ? dist.size() / numNodes : 0;
    std::vector<unsigned int> remotePivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfRow(k, numNodes, ranks);
        unsigned int* pivot = owner == rank
            ? dist.data() + (k - startRow) * numNodes : remotePivot.data();
        // Row k is unchanged during iteration k because dist[k][k] is zero.
        MPI_Bcast(pivot, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t i = 0; i < localRows; ++i) {
            unsigned int* row = dist.data() + i * numNodes;
            unsigned int* pathRow = path.data() + i * numNodes;
            const unsigned int viaK = row[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int candidate = viaK + pivot[j];
                if (candidate < row[j]) {
                    row[j] = candidate;
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
    int ranks = 1;
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

    // MPI counts and displacements are int; this also excludes impractical graphs.
    if (numNodes > static_cast<size_t>(INT_MAX) / std::max(numNodes, size_t{1})) {
        if (rank == 0) fprintf(stderr, "Graph is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    
    const size_t startRow = firstRow(rank, numNodes, ranks);
    const size_t localRows = rowCount(rank, numNodes, ranks);
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    
    initializeDistanceMatrix(dist, numNodes, localRows, rank, ranks);
    for (size_t i = 0; i < localRows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes,
                    static_cast<unsigned int>(startRow + i));
    }
    
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, startRow, rank, ranks);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000.0));
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }

    std::vector<unsigned int> result;
    if (printResults || validate) {
        std::vector<int> counts(ranks), offsets(ranks);
        for (int r = 0; r < ranks; ++r) {
            counts[r] = static_cast<int>(rowCount(r, numNodes, ranks) * numNodes);
            offsets[r] = static_cast<int>(firstRow(r, numNodes, ranks) * numNodes);
        }
        if (rank == 0) result.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, result.data(),
                    counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    
    if (rank == 0 && printResults) print_results_int(result, "DistanceMatrix");
    
    int valid = 1;
    if (rank == 0 && validate) {
        printf("Validating result...\n");
        valid = validateResult(result, numNodes) ? 1 : 0;
        printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
