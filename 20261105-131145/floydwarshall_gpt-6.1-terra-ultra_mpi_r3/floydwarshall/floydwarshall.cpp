#include <algorithm>
#include <climits>
#include <cmath>
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

// Consecutive source rows are assigned to each rank, including empty blocks
// when there are more ranks than vertices.
size_t rowCount(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks ? 1 : 0);
}

size_t rowStart(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

// MPI counts are int, whereas matrix sizes use size_t.
void sendElements(const unsigned int* data, size_t count, int dest, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Send(data, chunk, MPI_UNSIGNED, dest, tag, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

void receiveElements(unsigned int* data, size_t count, int source, int tag) {
    while (count != 0) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Recv(data, chunk, MPI_UNSIGNED, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              int rank, int ranks) {
    if (rank != 0) {
        receiveElements(dist.data(), dist.size(), 0, 0);
        return;
    }

    // Generate in the same order and with the same seed as the original.
    unsigned int seed = 42;
    std::vector<unsigned int> buffer;
    for (int owner = 0; owner < ranks; ++owner) {
        const size_t first = rowStart(n, owner, ranks);
        const size_t rows = rowCount(n, owner, ranks);
        if (owner != 0) buffer.resize(rows * n);
        unsigned int* block = owner == 0 ? dist.data() : buffer.data();
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < n; ++j) {
                const unsigned int value = 1 + static_cast<unsigned int>(
                    static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
                block[i * n + j] = first + i == j ? 0 : value;
            }
        }
        if (owner != 0) sendElements(block, rows * n, owner, 0);
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   int rank, int ranks) {
    const size_t first = rowStart(n, rank, ranks);
    const size_t rows = rowCount(n, rank, ranks);
    std::vector<unsigned int> pivot(n);

    for (int owner = 0; owner < ranks; ++owner) {
        const size_t begin = rowStart(n, owner, ranks);
        const size_t end = begin + rowCount(n, owner, ranks);
        for (size_t k = begin; k < end; ++k) {
            unsigned int* pivotRow = rank == owner
                ? dist.data() + (k - first) * n : pivot.data();
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

    const size_t first = rowStart(numNodes, rank, ranks);
    const size_t rows = rowCount(numNodes, rank, ranks);
    std::vector<unsigned int> dist(rows * numNodes);
    std::vector<unsigned int> path(rows * numNodes);
    initializeDistanceMatrix(dist, numNodes, rank, ranks);
    for (size_t i = 0; i < rows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes,
                    static_cast<unsigned int>(first + i));
    }

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
        const double gflops = ops / (milliseconds / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            std::copy(dist.begin(), dist.end(), fullDist.begin());
            for (int owner = 1; owner < ranks; ++owner) {
                const size_t count = rowCount(numNodes, owner, ranks) * numNodes;
                if (count != 0) {
                    receiveElements(fullDist.data() + rowStart(numNodes, owner, ranks) * numNodes,
                                    count, owner, 1);
                }
            }
        } else {
            sendElements(dist.data(), dist.size(), 0, 1);
        }
    }

    int status = 0;
    if (rank == 0) {
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
