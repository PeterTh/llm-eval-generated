#include <algorithm>
#include <chrono>
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

// Rows are divided into contiguous, nearly equal ranges. This is also the
// order in which the original program generated and printed the matrix.
static size_t rowCount(size_t n, int rank, int ranks) {
    return n / ranks + (static_cast<size_t>(rank) < n % ranks ? 1 : 0);
}

static size_t rowStart(size_t n, int rank, int ranks) {
    return (n / ranks) * rank + std::min(static_cast<size_t>(rank), n % ranks);
}

// Keep individual MPI counts within int range, including for large matrices.
constexpr size_t transferChunk = 1u << 20;

static void initializeDistanceMatrix(std::vector<unsigned int>& local, size_t n,
                                     int rank, int ranks) {
    if (rank == 0) {
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        std::vector<unsigned int> buffer;
        for (int dest = 0; dest < ranks; ++dest) {
            const size_t count = rowCount(n, dest, ranks) * n;
            if (dest != 0) buffer.resize(std::min(count, transferChunk));
            for (size_t offset = 0; offset < count;) {
                const size_t chunk = std::min(count - offset, transferChunk);
                unsigned int* out = dest == 0 ? local.data() + offset : buffer.data();
                for (size_t j = 0; j < chunk; ++j) {
                    out[j] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / (double)RAND_MAX);
                }
                if (dest != 0)
                    MPI_Send(out, static_cast<int>(chunk), MPI_UNSIGNED, dest, 0, MPI_COMM_WORLD);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const size_t chunk = std::min(local.size() - offset, transferChunk);
            MPI_Recv(local.data() + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                     0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += chunk;
        }
    }
    const size_t first = rowStart(n, rank, ranks);
    for (size_t i = 0; i < rowCount(n, rank, ranks); ++i)
        local[i * n + first + i] = 0;
}

static void floydWarshall(std::vector<unsigned int>& dist, size_t n,
                          int rank, int ranks) {
    const size_t localRows = rowCount(n, rank, ranks);
    const size_t firstRow = rowStart(n, rank, ranks);
    std::vector<unsigned int> pivot(n);

    for (int owner = 0; owner < ranks; ++owner) {
        const size_t first = rowStart(n, owner, ranks);
        const size_t last = first + rowCount(n, owner, ranks);
        for (size_t k = first; k < last; ++k) {
            unsigned int* rowK = rank == owner ? dist.data() + (k - firstRow) * n : pivot.data();
            for (size_t offset = 0; offset < n;) {
                const size_t chunk = std::min(n - offset, transferChunk);
                MPI_Bcast(rowK + offset, static_cast<int>(chunk), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
                offset += chunk;
            }

            // The diagonal is zero and all edge weights are nonnegative, so
            // both the pivot row and dist[i,k] are unchanged during this step.
            for (size_t i = 0; i < localRows; ++i) {
                unsigned int* __restrict__ row = dist.data() + i * n;
                const unsigned int dik = row[k];
                const unsigned int* __restrict__ pivotRow = rowK;
                // The pivot row cannot improve itself. Skipping it also keeps
                // the two restrict-qualified pointers disjoint.
                if (firstRow + i == k) continue;
                for (size_t j = 0; j < n; ++j) {
                    const unsigned int candidate = dik + pivotRow[j];
                    if (candidate < row[j]) row[j] = candidate;
                }
            }
        }
    }
}

static std::vector<unsigned int> gatherDistanceMatrix(const std::vector<unsigned int>& local,
                                                       size_t n, int rank, int ranks) {
    std::vector<unsigned int> full;
    if (rank == 0) {
        full.resize(n * n);
        std::copy(local.begin(), local.end(), full.begin());
        for (int source = 1; source < ranks; ++source) {
            const size_t count = rowCount(n, source, ranks) * n;
            const size_t start = rowStart(n, source, ranks) * n;
            for (size_t offset = 0; offset < count;) {
                const size_t chunk = std::min(count - offset, transferChunk);
                MPI_Recv(full.data() + start + offset, static_cast<int>(chunk), MPI_UNSIGNED,
                         source, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                offset += chunk;
            }
        }
    } else {
        for (size_t offset = 0; offset < local.size();) {
            const size_t chunk = std::min(local.size() - offset, transferChunk);
            MPI_Send(local.data() + offset, static_cast<int>(chunk), MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD);
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
    int parseStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 2;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
            break;
        }
    }
    if (parseStatus) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(rowCount(numNodes, rank, ranks) * numNodes);
    initializeDistanceMatrix(dist, numNodes, rank, ranks);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, numNodes, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long long duration = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", duration);
        const double ops = (double)numNodes * numNodes * numNodes;
        const double gflops = ops / (duration / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int result = 0;
    if (printResults || validate) {
        std::vector<unsigned int> full = gatherDistanceMatrix(dist, numNodes, rank, ranks);
        if (rank == 0) {
            if (printResults) print_results_int(full, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                result = valid ? 0 : 1;
            }
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
