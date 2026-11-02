#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Rows are contiguous: the original idx2(j, i, n) addresses row i, column j.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Balanced, contiguous row ranges keep both computation and final output ordered.
size_t rowCount(size_t n, int ranks, int rank) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}

size_t firstRow(size_t n, int ranks, int rank) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

// MPI's basic count argument is an int. Split large contiguous transfers.
void sendValues(const unsigned int* values, size_t count, int peer, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(values, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        values += chunk;
        count -= chunk;
    }
}

void receiveValues(unsigned int* values, size_t count, int peer, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(values, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        values += chunk;
        count -= chunk;
    }
}

void initializeMatrices(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                        size_t n, int rank, int ranks) {
    const size_t start = firstRow(n, ranks, rank);
    const size_t rows = rowCount(n, ranks, rank);

    // Generate the same rand_r stream in the same order as the original program.
    // Only a single rank generates it, and at most one rank's rows are buffered.
    if (rank == 0) {
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        std::vector<unsigned int> buffer;
        for (int target = 0; target < ranks; ++target) {
            const size_t targetStart = firstRow(n, ranks, target);
            const size_t count = rowCount(n, ranks, target) * n;
            if (target != 0) buffer.resize(count);
            unsigned int* values = target == 0 ? dist.data() : buffer.data();
            for (size_t index = 0; index < count; ++index) {
                values[index] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
            }
            for (size_t i = 0; i < rowCount(n, ranks, target); ++i) {
                values[i * n + targetStart + i] = 0;
            }
            if (target != 0) sendValues(values, count, target, 0);
        }
    } else {
        receiveValues(dist.data(), dist.size(), 0, 0);
    }
    for (size_t i = 0; i < rows; ++i) {
        std::fill_n(path.data() + i * n, n, static_cast<unsigned int>(start + i));
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, int rank, int ranks) {
    const size_t start = firstRow(n, ranks, rank);
    const size_t rows = rowCount(n, ranks, rank);
    std::vector<unsigned int> pivot(n);
    int owner = 0;
    size_t ownerEnd = rowCount(n, ranks, 0);

    for (size_t k = 0; k < n; ++k) {
        while (k >= ownerEnd) {
            ++owner;
            ownerEnd += rowCount(n, ranks, owner);
        }
        unsigned int* pivotRow = rank == owner ? dist.data() + (k - start) * n : pivot.data();
        for (size_t offset = 0; offset < n;) {
            const int chunk = static_cast<int>(std::min(n - offset, static_cast<size_t>(INT_MAX)));
            MPI_Bcast(pivotRow + offset, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            offset += chunk;
        }

        for (size_t i = 0; i < rows; ++i) {
            // The pivot row is unchanged because its diagonal is zero.
            if (start + i == k) continue;
            unsigned int* __restrict__ row = dist.data() + i * n;
            unsigned int* __restrict__ pathRow = path.data() + i * n;
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

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(n, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, n)];
                const unsigned int distIK = dist[idx2(k, i, n)];
                const unsigned int distKJ = dist[idx2(j, k, n)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
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
    if (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes) {
        if (rank == 0) fprintf(stderr, "Number of nodes is too large\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    const size_t rows = rowCount(numNodes, ranks, rank);
    std::vector<unsigned int> dist(rows * numNodes);
    std::vector<unsigned int> path(rows * numNodes);
    initializeMatrices(dist, path, numNodes, rank, ranks);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, numNodes, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / (milliseconds / 1000.0) / 1e9);
    }

    int valid = 1;
    if (printResults || validate) {
        std::vector<unsigned int> result;
        if (rank == 0) {
            result.resize(numNodes * numNodes);
            std::copy(dist.begin(), dist.end(), result.begin());
            for (int source = 1; source < ranks; ++source) {
                const size_t count = rowCount(numNodes, ranks, source) * numNodes;
                if (count != 0) receiveValues(result.data() + firstRow(numNodes, ranks, source) * numNodes,
                                              count, source, 1);
            }
            if (printResults) print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                valid = validateResult(result, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        } else {
            sendValues(dist.data(), dist.size(), 0, 1);
        }
    }
    if (validate) MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
