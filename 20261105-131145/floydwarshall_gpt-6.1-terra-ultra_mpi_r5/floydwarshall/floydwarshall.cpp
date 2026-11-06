#include <algorithm>
#include <mpi.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Contiguous source-row blocks preserve the original flattened matrix order.
size_t rowCount(size_t n, int ranks, int rank) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks);
}

size_t firstRow(size_t n, int ranks, int rank) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

// MPI counts are int, so large contiguous transfers are split into chunks.
constexpr size_t TRANSFER_CHUNK = 1024 * 1024;

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              int rank, int ranks) {
    if (rank == 0) {
        unsigned int seed = 42;
        std::vector<unsigned int> buffer(std::min(TRANSFER_CHUNK, n * n));
        for (int dest = 0; dest < ranks; ++dest) {
            const size_t count = rowCount(n, ranks, dest) * n;
            size_t offset = 0;
            while (offset < count) {
                const size_t chunk = std::min(TRANSFER_CHUNK, count - offset);
                unsigned int* values = dest == 0 ? dist.data() + offset : buffer.data();
                for (size_t i = 0; i < chunk; ++i) {
                    values[i] = 1 + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                }
                if (dest != 0) {
                    MPI_Send(values, static_cast<int>(chunk), MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
                }
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < dist.size();) {
            const size_t chunk = std::min(TRANSFER_CHUNK, dist.size() - offset);
            MPI_Recv(dist.data() + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                     0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += chunk;
        }
    }
    const size_t first = firstRow(n, ranks, rank);
    for (size_t i = 0; i < rowCount(n, ranks, rank); ++i) {
        dist[i * n + first + i] = 0;
    }
}

// Close a small panel on its owner, then broadcast its pivot rows together.
// All weights are nonnegative and diagonal entries are zero, so the closed
// panel gives the same shortest distances as the original per-pivot updates.
void floydWarshall(std::vector<unsigned int>& dist, size_t n, int rank, int ranks) {
    const size_t localRows = rowCount(n, ranks, rank);
    const size_t localFirst = firstRow(n, ranks, rank);
    const size_t panelSize = std::max<size_t>(1, std::min<size_t>(32, (n / ranks) / 4));
    std::vector<unsigned int> panel(std::min(panelSize, n) * n);

    for (int owner = 0; owner < ranks; ++owner) {
        const size_t ownerFirst = firstRow(n, ranks, owner);
        const size_t ownerEnd = ownerFirst + rowCount(n, ranks, owner);
        for (size_t begin = ownerFirst; begin < ownerEnd; begin += panelSize) {
            const size_t rows = std::min(panelSize, ownerEnd - begin);
            unsigned int* pivots = rank == owner
                ? dist.data() + (begin - localFirst) * n : panel.data();
            if (rank == owner) {
                for (size_t k = 0; k < rows; ++k) {
                    const unsigned int* pivot = pivots + k * n;
                    for (size_t i = 0; i < rows; ++i) {
                        unsigned int* row = pivots + i * n;
                        const unsigned int dik = row[begin + k];
                        for (size_t j = 0; j < n; ++j) {
                            row[j] = std::min(row[j], dik + pivot[j]);
                        }
                    }
                }
            }
            for (size_t offset = 0; offset < rows * n;) {
                const size_t chunk = std::min(TRANSFER_CHUNK, rows * n - offset);
                MPI_Bcast(pivots + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                          owner, MPI_COMM_WORLD);
                offset += chunk;
            }
            for (size_t i = 0; i < localRows; ++i) {
                if (rank == owner && localFirst + i >= begin && localFirst + i < begin + rows)
                    continue;
                unsigned int* row = dist.data() + i * n;
                for (size_t k = 0; k < rows; ++k) {
                    const unsigned int dik = row[begin + k];
                    const unsigned int* pivot = pivots + k * n;
                    for (size_t j = 0; j < n; ++j) {
                        row[j] = std::min(row[j], dik + pivot[j]);
                    }
                }
            }
        }
    }
}

void gatherDistances(const std::vector<unsigned int>& local, std::vector<unsigned int>& full,
                     size_t n, int rank, int ranks) {
    if (rank == 0) {
        std::copy(local.begin(), local.end(), full.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t count = rowCount(n, ranks, source) * n;
            const size_t first = firstRow(n, ranks, source) * n;
            for (size_t offset = 0; offset < count;) {
                const size_t chunk = std::min(TRANSFER_CHUNK, count - offset);
                MPI_Recv(full.data() + first + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                         source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const size_t chunk = std::min(TRANSFER_CHUNK, local.size() - offset);
            MPI_Send(local.data() + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                     0, 1, MPI_COMM_WORLD);
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
    std::vector<unsigned int> dist(rowCount(numNodes, ranks, rank) * numNodes);
    initializeDistanceMatrix(dist, numNodes, rank, ranks);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, numNodes, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long duration = static_cast<long>(maxElapsed * 1000);
        printf("Computation time: %ld ms\n", duration);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gflops = ops / maxElapsed / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> full;
    if (validate || printResults) {
        if (rank == 0) full.resize(numNodes * numNodes);
        gatherDistances(dist, full, numNodes, rank, ranks);
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
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
