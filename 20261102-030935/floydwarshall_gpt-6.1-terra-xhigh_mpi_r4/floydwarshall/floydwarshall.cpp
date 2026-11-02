#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous row blocks preserve the original flattened matrix order.
size_t rowCount(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}

size_t firstRow(size_t n, int rank, int ranks) {
    return static_cast<size_t>(rank) * (n / ranks) +
           std::min(static_cast<size_t>(rank), n % ranks);
}

// Generate the original rand_r stream once, distributing it in bounded chunks.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              int rank, int ranks, MPI_Comm comm) {
    constexpr size_t chunkSize = 1 << 20;
    if (rank == 0) {
        unsigned int seed = 42;
        auto generate = [&seed](unsigned int* data, size_t count) {
            for (size_t i = 0; i < count; ++i) {
                data[i] = 1 + static_cast<unsigned int>(
                    static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / (double)RAND_MAX);
            }
        };
        generate(dist.data(), dist.size());
        if (ranks > 1) {
            std::vector<unsigned int> buffer(std::min(chunkSize, rowCount(n, 1, ranks) * n));
            for (int dest = 1; dest < ranks; ++dest) {
                size_t remaining = rowCount(n, dest, ranks) * n;
                while (remaining != 0) {
                    const size_t count = std::min(remaining, buffer.size());
                    generate(buffer.data(), count);
                    MPI_Send(buffer.data(), static_cast<int>(count), MPI_UNSIGNED, dest, 0, comm);
                    remaining -= count;
                }
            }
        }
    } else {
        for (size_t offset = 0; offset < dist.size();) {
            const size_t count = std::min(chunkSize, dist.size() - offset);
            MPI_Recv(dist.data() + offset, static_cast<int>(count), MPI_UNSIGNED,
                     0, 0, comm, MPI_STATUS_IGNORE);
            offset += count;
        }
    }
    const size_t first = firstRow(n, rank, ranks);
    for (size_t i = 0; i < rowCount(n, rank, ranks); ++i) {
        dist[i * n + first + i] = 0;
    }
}

void floydWarshall(std::vector<unsigned int>& dist, size_t n,
                   int rank, int ranks, MPI_Comm comm) {
    const size_t localRows = rowCount(n, rank, ranks);
    const size_t first = firstRow(n, rank, ranks);
    std::vector<unsigned int> pivot(n);
    int owner = 0;
    size_t ownerEnd = rowCount(n, 0, ranks);
    for (size_t k = 0; k < n; ++k) {
        if (k == ownerEnd) {
            ++owner;
            ownerEnd += rowCount(n, owner, ranks);
        }
        unsigned int* pivotRow = rank == owner ? dist.data() + (k - first) * n : pivot.data();
        // MPI counts are int, so split very long rows if necessary.
        for (size_t offset = 0; offset < n;) {
            const int count = static_cast<int>(std::min(n - offset,
                static_cast<size_t>(std::numeric_limits<int>::max())));
            MPI_Bcast(pivotRow + offset, count, MPI_UNSIGNED, owner, comm);
            offset += count;
        }
        for (size_t i = 0; i < localRows; ++i) {
            if (first + i == k) continue; // A zero diagonal leaves the pivot row unchanged.
            unsigned int* row = dist.data() + i * n;
            const unsigned int distIK = row[k];
            for (size_t j = 0; j < n; ++j) {
                row[j] = std::min(row[j], distIK + pivotRow[j]);
            }
        }
    }
}

void gatherDistanceMatrix(const std::vector<unsigned int>& local,
                          std::vector<unsigned int>& full, size_t n,
                          int rank, int ranks, MPI_Comm comm) {
    constexpr size_t chunkSize = 1 << 20;
    if (rank == 0) {
        full.resize(n * n);
        std::copy(local.begin(), local.end(), full.begin());
        for (int src = 1; src < ranks; ++src) {
            const size_t start = firstRow(n, src, ranks) * n;
            const size_t count = rowCount(n, src, ranks) * n;
            for (size_t offset = 0; offset < count;) {
                const size_t chunk = std::min(chunkSize, count - offset);
                MPI_Recv(full.data() + start + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                         src, 1, comm, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const size_t chunk = std::min(chunkSize, local.size() - offset);
            MPI_Send(local.data() + offset, static_cast<int>(chunk), MPI_UNSIGNED, 0, 1, comm);
            offset += chunk;
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

    if (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        if (worldRank == 0) fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize();
        return 1;
    }

    // Ranks without rows do not participate in the per-pivot collectives.
    const int activeRanks = numNodes < static_cast<size_t>(worldSize)
        ? static_cast<int>(std::max(numNodes, size_t{1})) : worldSize;
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, worldRank < activeRanks ? 0 : MPI_UNDEFINED,
                   worldRank, &comm);
    if (worldRank >= activeRanks) {
        MPI_Finalize();
        return 0;
    }
    const int rank = worldRank;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(rowCount(numNodes, rank, activeRanks) * numNodes);
    initializeDistanceMatrix(dist, numNodes, rank, activeRanks, comm);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    floydWarshall(dist, numNodes, rank, activeRanks, comm);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (rank == 0) {
        const auto milliseconds = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / maxElapsed / 1e9);
    }

    std::vector<unsigned int> full;
    if (validate || printResults) {
        gatherDistanceMatrix(dist, full, numNodes, rank, activeRanks, comm);
    }
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results_int(full, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(full, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, comm);
    MPI_Comm_free(&comm);
    MPI_Finalize();
    return status;
}
