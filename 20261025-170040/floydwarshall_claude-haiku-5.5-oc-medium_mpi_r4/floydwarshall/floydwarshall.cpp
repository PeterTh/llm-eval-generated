#include <mpi.h>

#include <algorithm>
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

// Block row distribution: rank p owns global rows [rowBegin(p), rowBegin(p + 1)).
inline size_t rowBegin(const size_t p, const size_t numProcs, const size_t numNodes) noexcept {
    return numNodes * p / numProcs;
}

// Generates the same random matrix as the sequential version (identical rand_r stream on
// every rank), but stores only the rows [rowLo, rowHi) owned by this rank.
void initializeLocalDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                                   const size_t rowLo, const size_t rowHi,
                                   const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t total = numNodes * numNodes;
    const size_t base = rowLo * numNodes;
    const size_t end = rowHi * numNodes;

    for (size_t g = 0; g < total; ++g) {
        const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (g >= base && g < end) {
            dist[g - base] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowLo; i < rowHi; ++i) {
        dist[(i - rowLo) * numNodes + i] = 0;
    }
}

// Path matrix entry (row r, column c) is initialized to r for all c.
void initializeLocalPathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                               const size_t rowLo, const size_t rowHi) {
    for (size_t i = rowLo; i < rowHi; ++i) {
        std::fill_n(path.begin() + (i - rowLo) * numNodes, numNodes, static_cast<unsigned int>(i));
    }
}

// Relaxes one locally owned row i against intermediate node k:
//   dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]), path[i][j] = k on improvement.
// rowK is a private copy of row k, which is invariant during step k.
static inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                            const unsigned int* __restrict rowK, const unsigned int distIK,
                            const unsigned int k, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const unsigned int oldDist = distRow[j];
        const bool better = newDist < oldDist;
        distRow[j] = better ? newDist : oldDist;
        pathRow[j] = better ? k : pathRow[j];
    }
}

// Distributed Floyd-Warshall over row blocks.
// At step k, row k (and column k) is unchanged by the step, so the owner of row k
// broadcasts it and every rank relaxes its own rows against that copy. The row for step
// k+1 is relaxed first by its owner and sent with a non-blocking broadcast, which overlaps
// communication with the local update of the remaining rows.
void floydWarshallMPI(std::vector<unsigned int>& localDist,
                      std::vector<unsigned int>& localPath,
                      const size_t numNodes,
                      const size_t rowLo,
                      const size_t rowHi,
                      const std::vector<int>& ownerOf,
                      const MPI_Datatype rowType,
                      const int rank) {
    if (numNodes == 0) {
        return;
    }

    const size_t localRows = rowHi - rowLo;
    std::vector<unsigned int> rowBuf[2] = {std::vector<unsigned int>(numNodes),
                                           std::vector<unsigned int>(numNodes)};

    // Row 0 for step 0.
    if (ownerOf[0] == rank) {
        std::copy_n(localDist.begin(), numNodes, rowBuf[0].begin());
    }
    MPI_Bcast(rowBuf[0].data(), 1, rowType, ownerOf[0], MPI_COMM_WORLD);

    for (size_t k = 0; k < numNodes; ++k) {
        const unsigned int* cur = rowBuf[k & 1].data();
        unsigned int* nxt = rowBuf[(k + 1) & 1].data();
        const bool hasNext = (k + 1) < numNodes;
        const int nextOwner = hasNext ? ownerOf[k + 1] : -1;

        size_t relaxedLocal = localRows;  // sentinel: no row already relaxed this step
        MPI_Request req = MPI_REQUEST_NULL;

        if (hasNext) {
            if (nextOwner == rank) {
                const size_t li = (k + 1) - rowLo;
                unsigned int* distRow = localDist.data() + li * numNodes;
                unsigned int* pathRow = localPath.data() + li * numNodes;
                relaxRow(distRow, pathRow, cur, distRow[k], static_cast<unsigned int>(k), numNodes);
                std::copy_n(distRow, numNodes, nxt);
                relaxedLocal = li;
            }
            MPI_Ibcast(nxt, 1, rowType, nextOwner, MPI_COMM_WORLD, &req);
        }

        for (size_t li = 0; li < localRows; ++li) {
            if (li == relaxedLocal) {
                continue;
            }
            unsigned int* distRow = localDist.data() + li * numNodes;
            unsigned int* pathRow = localPath.data() + li * numNodes;
            relaxRow(distRow, pathRow, cur, distRow[k], static_cast<unsigned int>(k), numNodes);
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

    int rank = 0;
    int numProcs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
    const bool root = (rank == 0);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank, so all ranks agree on the exit path)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) {
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI processes: %d\n", numProcs);
    }

    const size_t P = static_cast<size_t>(numProcs);
    const size_t R = static_cast<size_t>(rank);
    const size_t rowLo = rowBegin(R, P, numNodes);
    const size_t rowHi = rowBegin(R + 1, P, numNodes);
    const size_t localRows = rowHi - rowLo;

    // Row datatype: one row of numNodes unsigned ints, used for broadcasts and gathers
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    // Owner rank of each global row
    std::vector<int> ownerOf(numNodes);
    for (int p = 0; p < numProcs; ++p) {
        const size_t pb = rowBegin(static_cast<size_t>(p), P, numNodes);
        const size_t pe = rowBegin(static_cast<size_t>(p) + 1, P, numNodes);
        for (size_t r = pb; r < pe; ++r) {
            ownerOf[r] = p;
        }
    }

    // Allocate local blocks of the matrices
    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);

    // Initialize
    if (root) {
        printf("Initializing graph...\n");
    }
    initializeLocalDistanceMatrix(localDist, numNodes, rowLo, rowHi, 1, MAX_DISTANCE);
    initializeLocalPathMatrix(localPath, numNodes, rowLo, rowHi);

    // Run Floyd-Warshall
    if (root) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshallMPI(localDist, localPath, numNodes, rowLo, rowHi, ownerOf, rowType, rank);

    const double localElapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&localElapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (root) {
        const long long durationMs = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        const double ops = (double)numNodes * numNodes * numNodes;
        const double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int exitCode = 0;

    // Results and validation need the full matrix on the root
    if (printResults || validate) {
        std::vector<unsigned int> fullDist;
        std::vector<int> counts(numProcs);
        std::vector<int> displs(numProcs);
        for (int p = 0; p < numProcs; ++p) {
            const size_t pb = rowBegin(static_cast<size_t>(p), P, numNodes);
            const size_t pe = rowBegin(static_cast<size_t>(p) + 1, P, numNodes);
            counts[p] = static_cast<int>(pe - pb);
            displs[p] = static_cast<int>(pb);
        }
        if (root) {
            fullDist.resize(numNodes * numNodes);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(localRows), rowType,
                    fullDist.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);

        if (printResults && root) {
            print_results_int(fullDist, "DistanceMatrix");
        }

        if (validate) {
            int valid = 0;
            if (root) {
                printf("Validating result...\n");
                valid = validateResult(fullDist, numNodes) ? 1 : 0;
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
            MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
            exitCode = valid ? 0 : 1;
        }
    }

    MPI_Type_free(&rowType);
    MPI_Finalize();
    return exitCode;
}
