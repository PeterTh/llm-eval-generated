#include <mpi.h>

#include <algorithm>
#include <chrono>
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

// Row-block decomposition: rank r owns matrix rows [rowStart, rowEnd).
// Element (i,j) of the global matrix lives at dist[idx2(j, i, n)] == i*n + j,
// i.e. rows are contiguous in memory, so a block of rows is a contiguous slab.
struct Decomposition {
    int rank = 0;
    int size = 1;
    size_t rowStart = 0;
    size_t rowEnd = 0;
    size_t numLocalRows() const noexcept { return rowEnd - rowStart; }
};

static Decomposition makeDecomposition(const size_t numNodes, const int rank, const int size) {
    Decomposition d;
    d.rank = rank;
    d.size = size;
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    d.rowStart = r * base + std::min(r, rem);
    d.rowEnd = d.rowStart + base + (r < rem ? 1 : 0);
    return d;
}

static inline int ownerOfRow(const size_t row, const size_t numNodes, const int size) {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t boundary = rem * (base + 1);
    if (row < boundary) return static_cast<int>(row / (base + 1));
    return static_cast<int>(rem + (row - boundary) / base);
}

// Reproduces the exact sequential rand_r stream of the original code, but only
// stores the rows owned by this rank.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const Decomposition& dec) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t localBegin = dec.rowStart * numNodes;
    const size_t localEnd = dec.rowEnd * numNodes;

    // Advance the generator past all elements preceding our block.
    for (size_t i = 0; i < localBegin; ++i) {
        (void)rand_r(&seed);
    }
    for (size_t i = localBegin; i < localEnd; ++i) {
        dist[i - localBegin] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = dec.rowStart; i < dec.rowEnd; ++i) {
        dist[(i - dec.rowStart) * numNodes + i] = 0;
    }
}

// Original fills path[i][j] = j for all i (the final state of the original
// initialization loop); only the locally owned rows are stored.
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const Decomposition& dec) {
    for (size_t i = dec.rowStart; i < dec.rowEnd; ++i) {
        unsigned int* row = path.data() + (i - dec.rowStart) * numNodes;
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = static_cast<unsigned int>(j);
        }
    }
}

// Relax one local row against pivot row k.
static inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                            const unsigned int* __restrict pivotRow, const unsigned int distIK,
                            const unsigned int k, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int distIJ = distRow[j];
        const unsigned int newDist = distIK + pivotRow[j];
        const bool better = newDist < distIJ;
        distRow[j] = better ? newDist : distIJ;
        pathRow[j] = better ? k : pathRow[j];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const Decomposition& dec) {
    // Two pivot-row buffers so the broadcast of row k+1 overlaps the
    // relaxation against row k.
    std::vector<unsigned int> pivotBuf[2];
    pivotBuf[0].resize(numNodes);
    pivotBuf[1].resize(numNodes);

    auto localRowPtr = [&](const size_t globalRow) {
        return dist.data() + (globalRow - dec.rowStart) * numNodes;
    };
    auto localPathPtr = [&](const size_t globalRow) {
        return path.data() + (globalRow - dec.rowStart) * numNodes;
    };

    if (numNodes == 0) return;

    // Broadcast initial pivot row 0.
    {
        const int owner = ownerOfRow(0, numNodes, dec.size);
        if (dec.rank == owner) {
            std::memcpy(pivotBuf[0].data(), localRowPtr(0), numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(pivotBuf[0].data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const unsigned int* pivot = pivotBuf[k & 1].data();
        unsigned int* nextPivot = pivotBuf[(k + 1) & 1].data();
        const unsigned int kk = static_cast<unsigned int>(k);

        MPI_Request req = MPI_REQUEST_NULL;
        const bool hasNext = (k + 1 < numNodes);
        const size_t nextRow = k + 1;
        bool nextIsMine = false;

        if (hasNext) {
            const int owner = ownerOfRow(nextRow, numNodes, dec.size);
            nextIsMine = (dec.rank == owner);
            if (nextIsMine) {
                // Relax row k+1 first so it can be shipped immediately.
                unsigned int* dRow = localRowPtr(nextRow);
                unsigned int* pRow = localPathPtr(nextRow);
                relaxRow(dRow, pRow, pivot, dRow[k], kk, numNodes);
                std::memcpy(nextPivot, dRow, numNodes * sizeof(unsigned int));
            }
            MPI_Ibcast(nextPivot, static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD, &req);
        }

        // Relax remaining local rows. Row k itself is unchanged by iteration k
        // (dist[k][k] == 0), and row k+1 was already handled if it is ours.
        for (size_t i = dec.rowStart; i < dec.rowEnd; ++i) {
            if (i == k) continue;
            if (nextIsMine && i == nextRow) continue;
            unsigned int* dRow = localRowPtr(i);
            relaxRow(dRow, localPathPtr(i), pivot, dRow[k], kk, numNodes);
        }

        if (hasNext) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
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

// Gather the distributed row blocks into a full matrix on rank 0.
static void gatherMatrix(const std::vector<unsigned int>& local, std::vector<unsigned int>& full,
                         const size_t numNodes, const Decomposition& dec) {
    // Use a contiguous row datatype so counts are in rows (avoids int overflow).
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    std::vector<int> counts(dec.size), displs(dec.size);
    for (int r = 0; r < dec.size; ++r) {
        const Decomposition d = makeDecomposition(numNodes, r, dec.size);
        counts[r] = static_cast<int>(d.numLocalRows());
        displs[r] = static_cast<int>(d.rowStart);
    }
    if (dec.rank == 0) {
        full.resize(numNodes * numNodes);
    }
    MPI_Gatherv(local.data(), static_cast<int>(dec.numLocalRows()), rowType,
                dec.rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), rowType,
                0, MPI_COMM_WORLD);
    MPI_Type_free(&rowType);
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
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
        printf("MPI ranks: %d\n", size);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Decomposition dec = makeDecomposition(numNodes, rank, size);

    // Allocate local row blocks
    std::vector<unsigned int> dist(dec.numLocalRows() * numNodes);
    std::vector<unsigned int> path(dec.numLocalRows() * numNodes);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, dec);
    initializePathMatrix(path, numNodes, dec);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, dec);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localMs = static_cast<long>(duration.count());
    long maxMs = localMs;
    MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int exitCode = 0;
    if (printResults || validate) {
        std::vector<unsigned int> fullDist;
        gatherMatrix(dist, fullDist, numNodes, dec);

        if (rank == 0) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullDist, numNodes);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
