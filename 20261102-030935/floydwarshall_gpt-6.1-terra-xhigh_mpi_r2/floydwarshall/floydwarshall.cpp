#include <algorithm>
#include <climits>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Each rank owns a consecutive range of source rows.  The original flat
// layout stores each of those rows contiguously.
size_t rowCount(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}

size_t rowStart(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

// Generate the original rand_r sequence once, in its original order.  Stream
// it to the row owners so rank zero need not hold the complete graph.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              int rank, int ranks, MPI_Comm comm) {
    constexpr size_t chunkSize = 65536;
    if (rank == 0) {
        std::vector<unsigned int> chunk(chunkSize);
        unsigned int seed = 42;
        size_t globalIndex = 0;
        for (int dest = 0; dest < ranks; ++dest) {
            const size_t count = rowCount(n, dest, ranks) * n;
            for (size_t offset = 0; offset < count;) {
                const size_t length = std::min(chunkSize, count - offset);
                for (size_t i = 0; i < length; ++i, ++globalIndex) {
                    const unsigned int value = 1 + static_cast<unsigned int>(
                        200.0 * rand_r(&seed) / static_cast<double>(RAND_MAX));
                    chunk[i] = globalIndex / n == globalIndex % n ? 0 : value;
                }
                if (dest == 0) {
                    std::copy_n(chunk.data(), length, dist.data() + offset);
                } else {
                    MPI_Send(chunk.data(), static_cast<int>(length), MPI_UNSIGNED,
                             dest, 0, comm);
                }
                offset += length;
            }
        }
    } else {
        for (size_t offset = 0; offset < dist.size();) {
            const int length = static_cast<int>(std::min(chunkSize, dist.size() - offset));
            MPI_Recv(dist.data() + offset, length, MPI_UNSIGNED, 0, 0, comm,
                     MPI_STATUS_IGNORE);
            offset += length;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, size_t n,
                   int rank, int ranks, MPI_Comm comm) {
    const size_t localRows = rowCount(n, rank, ranks);
    const size_t firstRow = rowStart(n, rank, ranks);
    std::vector<unsigned int> pivot(n);
    int owner = 0;
    size_t ownerEnd = rowCount(n, 0, ranks);
    for (size_t k = 0; k < n; ++k) {
        if (k == ownerEnd) {
            ++owner;
            ownerEnd += rowCount(n, owner, ranks);
        }
        unsigned int* pivotRow = rank == owner
            ? dist.data() + (k - firstRow) * n : pivot.data();
        MPI_Bcast(pivotRow, static_cast<int>(n), MPI_UNSIGNED, owner, comm);
        for (size_t i = 0; i < localRows; ++i) {
            if (firstRow + i == k) continue;
            unsigned int* row = dist.data() + i * n;
            const unsigned int distanceToPivot = row[k];
            for (size_t j = 0; j < n; ++j) {
                row[j] = std::min(row[j], distanceToPivot + pivotRow[j]);
            }
        }
    }
}

// Gather only for the optional result reporting and validation paths.
std::vector<unsigned int> gatherDistanceMatrix(const std::vector<unsigned int>& local,
                                                size_t n, int rank, int ranks,
                                                MPI_Comm comm) {
    std::vector<unsigned int> result;
    if (rank == 0) {
        result.resize(n * n);
        std::copy(local.begin(), local.end(), result.begin());
        for (int src = 1; src < ranks; ++src) {
            const size_t count = rowCount(n, src, ranks) * n;
            const size_t start = rowStart(n, src, ranks) * n;
            for (size_t offset = 0; offset < count;) {
                const int length = static_cast<int>(std::min<size_t>(INT_MAX, count - offset));
                MPI_Recv(result.data() + start + offset, length, MPI_UNSIGNED,
                         src, 1, comm, MPI_STATUS_IGNORE);
                offset += length;
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const int length = static_cast<int>(std::min<size_t>(INT_MAX, local.size() - offset));
            MPI_Send(local.data() + offset, length, MPI_UNSIGNED, 0, 1, comm);
            offset += length;
        }
    }
    return result;
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
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            const unsigned long long value = strtoull(arg, &end, 10);
            if (arg[0] == '-' || end == arg || *end != '\0' ||
                value > static_cast<unsigned long long>(INT_MAX) ||
                (value != 0 && value > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / value)) {
                if (worldRank == 0) printf("Invalid number of nodes: %s\n", arg);
                parseStatus = 1;
                break;
            }
            numNodes = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            parseStatus = 2;
            break;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    // Extra processes cannot contribute rows; exclude them from the per-pivot broadcasts.
    const int activeRanks = numNodes == 0 ? 1 :
        static_cast<int>(std::min(numNodes, static_cast<size_t>(worldSize)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeRanks) {
        MPI_Finalize();
        return 0;
    }
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(rowCount(numNodes, rank, ranks) * numNodes);
    initializeDistanceMatrix(dist, numNodes, rank, ranks, comm);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    floydWarshall(dist, numNodes, rank, ranks, comm);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        const long long milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = maxElapsed > 0.0 ? ops / maxElapsed / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int status = 0;
    if (printResults || validate) {
        std::vector<unsigned int> result = gatherDistanceMatrix(dist, numNodes, rank, ranks, comm);
        if (rank == 0) {
            if (printResults) print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(result, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
