#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
#include <limits>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Each rank owns consecutive source rows, matching the original storage order.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              size_t firstRow) {
    unsigned int seed = 42;
    for (size_t i = 0; i < firstRow * n; ++i) rand_r(&seed);
    for (size_t i = 0; i < dist.size(); ++i) {
        dist[i] = 1 + static_cast<unsigned int>(
            static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
    }
    for (size_t i = 0; i < dist.size() / n; ++i)
        dist[i * n + firstRow + i] = 0;
}

void floydWarshall(std::vector<unsigned int>& dist, size_t n,
                   size_t firstRow, size_t rows, int rank, int ranks) {
    std::vector<unsigned int> pivot(n);
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    for (size_t k = 0; k < n; ++k) {
        const size_t largeRows = (base + 1) * extra;
        const int owner = static_cast<int>(k < largeRows ? k / (base + 1)
                          : extra + (k - largeRows) / base);
        if (rank == owner)
            std::copy_n(dist.data() + (k - firstRow) * n, n, pivot.data());
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t i = 0; i < rows; ++i) {
            unsigned int* row = dist.data() + i * n;
            const unsigned int ik = row[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = ik + pivot[j];
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
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    int status = 0;
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
    if (numNodes > INT_MAX || (numNodes && numNodes > std::numeric_limits<size_t>::max() / numNodes)) {
        if (rank == 0) fprintf(stderr, "Number of nodes is too large\n");
        MPI_Finalize();
        return 1;
    }
    const size_t base = numNodes / static_cast<size_t>(ranks);
    const size_t extra = numNodes % static_cast<size_t>(ranks);
    const size_t rows = base + (static_cast<size_t>(rank) < extra);
    const size_t firstRow = static_cast<size_t>(rank) * base +
                            std::min(static_cast<size_t>(rank), extra);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(rows * numNodes);
    if (rows) initializeDistanceMatrix(dist, numNodes, firstRow);
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, numNodes, firstRow, rows, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const long milliseconds = static_cast<long>(maxElapsed * 1000);
        printf("Computation time: %ld ms\n", milliseconds);
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / maxElapsed / 1e9);
    }
    if (validate || printResults) {
        std::vector<unsigned int> full;
        if (rank == 0) full.resize(numNodes * numNodes);
        // Chunk transfers to support matrices larger than MPI's int count limit.
        for (int source = 0; source < ranks; ++source) {
            const size_t sourceRows = base + (static_cast<size_t>(source) < extra);
            const size_t sourceFirst = static_cast<size_t>(source) * base +
                                       std::min(static_cast<size_t>(source), extra);
            const size_t count = sourceRows * numNodes;
            if (source == 0 && rank == 0) {
                std::copy(dist.begin(), dist.end(), full.begin());
            } else if (rank == 0 || rank == source) {
                for (size_t offset = 0; offset < count;) {
                    const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
                    if (rank == source)
                        MPI_Send(dist.data() + offset, chunk, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD);
                    else
                        MPI_Recv(full.data() + sourceFirst * numNodes + offset, chunk, MPI_UNSIGNED,
                                 source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    offset += static_cast<size_t>(chunk);
                }
            }
        }
        if (rank == 0) {
            if (printResults) print_results_int(full, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
