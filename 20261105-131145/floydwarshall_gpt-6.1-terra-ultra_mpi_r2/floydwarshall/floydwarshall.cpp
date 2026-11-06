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

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous row blocks, with at most one extra row on each early rank.
size_t rowCount(size_t n, int ranks, int rank) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}

size_t firstRow(size_t n, int ranks, int rank) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

// MPI counts are int; use chunks so large local matrices are also supported.
void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send)
            MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD);
        else
            MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void broadcast(unsigned int* data, size_t count, int root) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_UNSIGNED, root, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

void updateRows(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                size_t n, size_t first, size_t rows, size_t k, const unsigned int* pivot) {
    for (size_t i = first; i < first + rows; ++i) {
        unsigned int* const d = dist.data() + i * n;
        unsigned int* const p = path.data() + i * n;
        const unsigned int dik = d[k];
        for (size_t j = 0; j < n; ++j) {
            const unsigned int candidate = dik + pivot[j];
            if (candidate < d[j]) {
                d[j] = candidate;
                p[j] = static_cast<unsigned int>(k);
            }
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, int rank, int ranks) {
    // Each owner advances only the pivot block first, recording the pivot row
    // at each step. All ranks then apply those snapshots to their other rows.
    // This reduces the number of broadcasts without changing the recurrence.
    constexpr size_t batchSize = 32;
    const size_t localRows = rowCount(n, ranks, rank);
    const size_t localFirst = firstRow(n, ranks, rank);
    std::vector<unsigned int> pivots(std::min(n, batchSize) * n);
    for (int owner = 0; owner < ranks; ++owner) {
        const size_t begin = firstRow(n, ranks, owner);
        const size_t end = begin + rowCount(n, ranks, owner);
        for (size_t start = begin; start < end; start += batchSize) {
            const size_t count = std::min(batchSize, end - start);
            if (rank == owner) {
                for (size_t b = 0; b < count; ++b) {
                    const size_t k = start + b;
                    unsigned int* const pivot = pivots.data() + b * n;
                    std::copy_n(dist.data() + (k - localFirst) * n, n, pivot);
                    updateRows(dist, path, n, start - localFirst, count, k, pivot);
                }
            }
            broadcast(pivots.data(), count * n, owner);
            for (size_t b = 0; b < count; ++b) {
                const unsigned int* const pivot = pivots.data() + b * n;
                if (rank == owner) {
                    const size_t blockFirst = start - localFirst;
                    updateRows(dist, path, n, 0, blockFirst, start + b, pivot);
                    updateRows(dist, path, n, blockFirst + count,
                               localRows - blockFirst - count, start + b, pivot);
                } else {
                    updateRows(dist, path, n, 0, localRows, start + b, pivot);
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

    const size_t localRows = rowCount(numNodes, ranks, rank);
    const size_t localFirst = firstRow(numNodes, ranks, rank);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    for (size_t i = 0; i < localRows; ++i)
        std::fill_n(path.data() + i * numNodes, numNodes, static_cast<unsigned int>(localFirst + i));

    // Generate exactly the original serial rand_r stream, then distribute rows.
    if (rank == 0) {
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        for (int target = 0; target < ranks; ++target) {
            const size_t rows = rowCount(numNodes, ranks, target);
            const size_t first = firstRow(numNodes, ranks, target);
            std::vector<unsigned int> buffer;
            if (target != 0) buffer.resize(rows * numNodes);
            unsigned int* data = target == 0 ? dist.data() : buffer.data();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < numNodes; ++j) {
                    unsigned int value = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
                    data[i * numNodes + j] = (first + i == j) ? 0 : value;
                }
            }
            if (target != 0) transfer(data, rows * numNodes, target, true);
        }
    } else {
        transfer(dist.data(), dist.size(), 0, false);
    }

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            std::copy(dist.begin(), dist.end(), fullDist.begin());
            for (int source = 1; source < ranks; ++source) {
                const size_t count = rowCount(numNodes, ranks, source) * numNodes;
                if (count) transfer(fullDist.data() + firstRow(numNodes, ranks, source) * numNodes,
                                    count, source, false);
            }
        } else {
            transfer(dist.data(), dist.size(), 0, true);
        }
    }

    int status = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / (milliseconds / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) status = 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
