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

// Transfer large rows and blocks without exceeding MPI's int count limit.
void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void broadcastRow(unsigned int* row, size_t count, int owner) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(row, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        row += chunk;
        count -= chunk;
    }
}

size_t rowStart(int rank, size_t n, int ranks) {
    return static_cast<size_t>(rank) * (n / ranks) +
           std::min(static_cast<size_t>(rank), n % ranks);
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

    const size_t first = rowStart(rank, numNodes, ranks);
    const size_t rows = rowStart(rank + 1, numNodes, ranks) - first;
    std::vector<unsigned int> dist(rows * numNodes);
    std::vector<unsigned int> path(rows * numNodes);
    std::vector<unsigned int> pivot(numNodes);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        // Generate blocks in global storage order, preserving rand_r's sequence.
        unsigned int seed = 42;
        std::vector<unsigned int> block;
        for (int p = 0; p < ranks; ++p) {
            const size_t start = rowStart(p, numNodes, ranks);
            const size_t count = (rowStart(p + 1, numNodes, ranks) - start) * numNodes;
            unsigned int* out;
            if (p == 0) out = dist.data();
            else { block.resize(count); out = block.data(); }
            for (size_t i = 0; i < count; ++i) {
                out[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
            }
            for (size_t i = 0; i < count / (numNodes ? numNodes : 1); ++i)
                out[i * numNodes + start + i] = 0;
            if (p != 0) transfer(out, count, p, 0, true);
        }
    } else {
        transfer(dist.data(), dist.size(), 0, 0, false);
    }
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < numNodes; ++j)
            path[i * numNodes + j] = static_cast<unsigned int>(j);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();
    int owner = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        while (k >= rowStart(owner + 1, numNodes, ranks)) ++owner;
        if (rank == owner)
            std::copy_n(dist.data() + (k - first) * numNodes, numNodes, pivot.data());
        broadcastRow(pivot.data(), numNodes, owner);
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * numNodes;
            unsigned int* pathRow = path.data() + i * numNodes;
            const unsigned int ik = row[k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int candidate = ik + pivot[j];
                if (candidate < row[j]) {
                    row[j] = candidate;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
    const double elapsed = MPI_Wtime() - startTime;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / (milliseconds / 1000.0) / 1e9);
    }

    std::vector<unsigned int> result;
    if (validate || printResults) {
        if (rank == 0) {
            result.resize(numNodes * numNodes);
            std::copy(dist.begin(), dist.end(), result.begin());
            for (int p = 1; p < ranks; ++p) {
                const size_t start = rowStart(p, numNodes, ranks);
                const size_t count = (rowStart(p + 1, numNodes, ranks) - start) * numNodes;
                if (count) transfer(result.data() + start * numNodes, count, p, 1, false);
            }
        } else {
            transfer(dist.data(), dist.size(), 0, 1, true);
        }
    }
    int status = 0;
    if (rank == 0) {
        if (printResults) print_results_int(result, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(result, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
