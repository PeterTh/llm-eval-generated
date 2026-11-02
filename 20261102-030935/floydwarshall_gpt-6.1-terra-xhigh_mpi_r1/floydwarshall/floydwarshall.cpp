#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Rows are divided into contiguous, nearly equal blocks, in rank order.
size_t rowCount(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}

size_t firstRow(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

// MPI counts are int, so transfer large blocks in bounded pieces.
constexpr size_t TRANSFER_SIZE = 1024 * 1024;

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              int rank, int ranks) {
    if (rank == 0) {
        unsigned int seed = 42;
        std::vector<unsigned int> buffer;
        if (ranks > 1) buffer.resize(std::min(TRANSFER_SIZE, n * rowCount(n, 1, ranks)));
        for (int r = 0; r < ranks; ++r) {
            const size_t begin = firstRow(n, r, ranks) * n;
            const size_t count = rowCount(n, r, ranks) * n;
            for (size_t offset = 0; offset < count;) {
                const size_t chunk = std::min(TRANSFER_SIZE, count - offset);
                unsigned int* target = r == 0 ? dist.data() + offset : buffer.data();
                for (size_t i = 0; i < chunk; ++i) {
                    const size_t index = begin + offset + i;
                    const unsigned int value = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / (double)RAND_MAX);
                    target[i] = index / n == index % n ? 0 : value;
                }
                if (r != 0)
                    MPI_Send(target, static_cast<int>(chunk), MPI_UNSIGNED, r, 0, MPI_COMM_WORLD);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < dist.size();) {
            const size_t chunk = std::min(TRANSFER_SIZE, dist.size() - offset);
            MPI_Recv(dist.data() + offset, static_cast<int>(chunk), MPI_UNSIGNED, 0, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += chunk;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   size_t first, size_t rows, int rank, int ranks) {
    std::vector<unsigned int> pivot(n);
    int owner = 0;
    size_t ownerEnd = rowCount(n, 0, ranks);
    for (size_t k = 0; k < n; ++k) {
        while (k >= ownerEnd) {
            ++owner;
            ownerEnd += rowCount(n, owner, ranks);
        }
        unsigned int* pivotRow = rank == owner ? dist.data() + (k - first) * n : pivot.data();
        for (size_t offset = 0; offset < n;) {
            const int count = static_cast<int>(std::min(n - offset, static_cast<size_t>(INT_MAX)));
            MPI_Bcast(pivotRow + offset, count, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            offset += count;
        }
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * n;
            unsigned int* pathRow = path.data() + i * n;
            const unsigned int distIK = row[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int newDist = distIK + pivotRow[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(const std::vector<unsigned int>& local,
                                          size_t n, int rank, int ranks) {
    std::vector<unsigned int> full;
    if (rank == 0) {
        full.resize(n * n);
        std::copy(local.begin(), local.end(), full.begin());
        for (int r = 1; r < ranks; ++r) {
            const size_t begin = firstRow(n, r, ranks) * n;
            const size_t count = rowCount(n, r, ranks) * n;
            for (size_t offset = 0; offset < count;) {
                const size_t chunk = std::min(TRANSFER_SIZE, count - offset);
                MPI_Recv(full.data() + begin + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                         r, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const size_t chunk = std::min(TRANSFER_SIZE, local.size() - offset);
            MPI_Send(local.data() + offset, static_cast<int>(chunk), MPI_UNSIGNED, 0, 1,
                     MPI_COMM_WORLD);
            offset += chunk;
        }
    }
    return full;
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    const size_t first = firstRow(numNodes, rank, ranks);
    const size_t rows = rowCount(numNodes, rank, ranks);
    std::vector<unsigned int> dist(rows * numNodes);
    std::vector<unsigned int> path(rows * numNodes);
    initializeDistanceMatrix(dist, numNodes, rank, ranks);
    for (size_t i = 0; i < rows; ++i)
        std::fill_n(path.data() + i * numNodes, numNodes, static_cast<unsigned int>(first + i));

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, first, rows, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / (milliseconds / 1000.0) / 1e9);
    }

    int status = 0;
    if (printResults || validate) {
        std::vector<unsigned int> full = gatherDistances(dist, numNodes, rank, ranks);
        if (rank == 0) {
            if (printResults) print_results_int(full, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
