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

// Consecutive blocks of logical source rows are stored on each rank.
size_t rowCount(size_t n, int processes, int rank) {
    return n / processes + (static_cast<size_t>(rank) < n % processes);
}

size_t firstRow(size_t n, int processes, int rank) {
    return (n / processes) * rank + std::min(static_cast<size_t>(rank), n % processes);
}

// MPI point-to-point counts are int, while a block can contain more elements.
void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    const size_t limit = static_cast<size_t>(std::numeric_limits<int>::max());
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, limit));
        if (send) {
            MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        } else {
            MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        data += chunk;
        count -= chunk;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& local, size_t n,
                              int rank, int processes) {
    if (rank != 0) {
        transfer(local.data(), local.size(), 0, 0, false);
        return;
    }

    // Generate the original random sequence once, distributing it in row order.
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    std::vector<unsigned int> buffer;
    for (int target = 0; target < processes; ++target) {
        const size_t first = firstRow(n, processes, target);
        const size_t rows = rowCount(n, processes, target);
        if (target != 0) buffer.resize(rows * n);
        unsigned int* data = target == 0 ? local.data() : buffer.data();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const unsigned int value = 1 + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
                data[i * n + j] = i + first == j ? 0 : value;
            }
        }
        if (target != 0) transfer(data, rows * n, target, 0, true);
    }
}

void floydWarshall(std::vector<unsigned int>& dist, size_t n, int rank, int processes) {
    const size_t rows = rowCount(n, processes, rank);
    const size_t first = firstRow(n, processes, rank);
    std::vector<unsigned int> receivedPivot(n);
    int owner = 0;
    size_t ownerEnd = rowCount(n, processes, owner);

    for (size_t k = 0; k < n; ++k) {
        while (k >= ownerEnd) {
            ++owner;
            ownerEnd += rowCount(n, processes, owner);
        }
        unsigned int* pivot = rank == owner
            ? dist.data() + (k - first) * n : receivedPivot.data();
        // n is parsed as an int, so the pivot row fits in an MPI count.
        MPI_Bcast(pivot, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * n;
            const unsigned int toPivot = row[k];
            // The pivot row is unchanged because its diagonal element is zero.
            if (toPivot == 0) continue;
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = toPivot + pivot[j];
                if (candidate < row[j]) row[j] = candidate;
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
    int rank, processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool done = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            done = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            done = true;
            break;
        }
    }

    if (!done) {
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }

        const size_t rows = rowCount(numNodes, processes, rank);
        std::vector<unsigned int> dist(rows * numNodes);
        initializeDistanceMatrix(dist, numNodes, rank, processes);

        if (rank == 0) printf("Computing shortest paths...\n");
        MPI_Barrier(MPI_COMM_WORLD);
        const double start = MPI_Wtime();
        floydWarshall(dist, numNodes, rank, processes);
        const double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
            printf("Computation time: %ld ms\n", milliseconds);
            double ops = (double)numNodes * numNodes * numNodes;
            double gflops = ops / (milliseconds / 1000.0) / 1e9;
            printf("Performance: %.3f GOPS\n", gflops);
        }

        if (printResults || validate) {
            std::vector<unsigned int> all;
            if (rank == 0) {
                all.resize(numNodes * numNodes);
                std::copy(dist.begin(), dist.end(), all.begin());
                for (int source = 1; source < processes; ++source) {
                    const size_t count = rowCount(numNodes, processes, source) * numNodes;
                    if (count != 0) {
                        transfer(all.data() + firstRow(numNodes, processes, source) * numNodes,
                                 count, source, 1, false);
                    }
                }
                if (printResults) print_results_int(all, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(all, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    if (!valid) exitCode = 1;
                }
            } else {
                transfer(dist.data(), dist.size(), 0, 1, true);
            }
        }
    }
    MPI_Finalize();
    return exitCode;
}
