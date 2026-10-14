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

// Row-block distribution of the matrix rows over the MPI ranks.
// Rank r owns rows [start[r], start[r + 1]); the first (numNodes % numRanks) ranks get one extra row.
struct RowPartition {
    std::vector<size_t> start;

    RowPartition(const size_t numNodes, const int numRanks) : start(numRanks + 1) {
        const size_t base = numNodes / numRanks;
        const size_t extra = numNodes % numRanks;
        start[0] = 0;
        for (int r = 0; r < numRanks; ++r) {
            start[r + 1] = start[r] + base + (static_cast<size_t>(r) < extra ? 1 : 0);
        }
    }

    // Rank that owns the given global row (row must be < numNodes)
    int owner(const size_t row) const {
        const auto it = std::upper_bound(start.begin(), start.end(), row);
        return static_cast<int>(it - start.begin()) - 1;
    }
};

// Fills the local rows [rowBegin, rowEnd) of the distance matrix.
// The random values are drawn in the same global order as the sequential version,
// so the generated graph is identical for any number of ranks.
void initializeDistanceMatrix(std::vector<unsigned int>& localDist, const size_t numNodes,
                              const size_t rowBegin, const size_t rowEnd,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t row = 0; row < numNodes; ++row) {
        const bool isLocal = row >= rowBegin && row < rowEnd;
        unsigned int* out = isLocal ? &localDist[(row - rowBegin) * numNodes] : nullptr;
        for (size_t col = 0; col < numNodes; ++col) {
            const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (out) {
                out[col] = value;
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t row = rowBegin; row < rowEnd; ++row) {
        localDist[(row - rowBegin) * numNodes + row] = 0;
    }
}

// Every path entry starts out as its row index (path[i][j] = i)
void initializePathMatrix(std::vector<unsigned int>& localPath, const size_t numNodes,
                          const size_t rowBegin, const size_t rowEnd) {
    for (size_t row = rowBegin; row < rowEnd; ++row) {
        std::fill_n(&localPath[(row - rowBegin) * numNodes], numNodes, static_cast<unsigned int>(row));
    }
}

// Relaxes one row i through intermediate node k: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j]).
// dist[i][k] cannot change during iteration k (dist[k][k] == 0), so reading it once is exact.
inline void relaxRow(unsigned int* distRow, unsigned int* pathRow, const unsigned int* pivotRow,
                     const size_t numNodes, const size_t k) {
    const unsigned int distIK = distRow[k];
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + pivotRow[j];
        const bool better = newDist < distRow[j];
        distRow[j] = better ? newDist : distRow[j];
        pathRow[j] = better ? static_cast<unsigned int>(k) : pathRow[j];
    }
}

// Distributed Floyd-Warshall over row blocks.
// Iteration k needs row k (the pivot row) on every rank. The owner of row k+1 relaxes that
// row first and starts a non-blocking broadcast of it, so the broadcast overlaps with the
// relaxation of the remaining local rows for iteration k.
void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const size_t numNodes,
                   const RowPartition& part,
                   const int rank,
                   MPI_Datatype rowType) {
    if (numNodes == 0) {
        return;
    }

    const size_t rowBegin = part.start[rank];
    const size_t numLocal = part.start[rank + 1] - rowBegin;

    // Two pivot buffers: pivotBuf[(k & 1) * numNodes] holds row k while the other one receives row k+1
    std::vector<unsigned int> pivotBuf(2 * numNodes);

    if (part.owner(0) == rank) {
        std::copy_n(localDist.begin(), numNodes, pivotBuf.begin());
    }
    MPI_Bcast(pivotBuf.data(), 1, rowType, part.owner(0), MPI_COMM_WORLD);

    for (size_t k = 0; k < numNodes; ++k) {
        unsigned int* pivot = pivotBuf.data() + (k & 1) * numNodes;
        unsigned int* next = pivotBuf.data() + ((k + 1) & 1) * numNodes;
        const bool hasNext = k + 1 < numNodes;
        MPI_Request request = MPI_REQUEST_NULL;

        if (hasNext) {
            const int nextOwner = part.owner(k + 1);
            if (nextOwner == rank) {
                const size_t li = k + 1 - rowBegin;
                relaxRow(&localDist[li * numNodes], &localPath[li * numNodes], pivot, numNodes, k);
                std::copy_n(&localDist[li * numNodes], numNodes, next);
            }
            MPI_Ibcast(next, 1, rowType, nextOwner, MPI_COMM_WORLD, &request);
        }

        for (size_t li = 0; li < numLocal; ++li) {
            // Row k+1 was already relaxed above (only its owner has it locally)
            if (hasNext && rowBegin + li == k + 1) {
                continue;
            }
            relaxRow(&localDist[li * numNodes], &localPath[li * numNodes], pivot, numNodes, k);
        }

        if (hasNext) {
            MPI_Wait(&request, MPI_STATUS_IGNORE);
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
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int exitCode = 0;
    bool exitEarly = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            exitEarly = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            exitEarly = true;
            break;
        }
    }

    if (exitEarly) {
        MPI_Finalize();
        return exitCode;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Each rank owns a block of rows of the distance and path matrices
    const RowPartition part(numNodes, numRanks);
    const size_t rowBegin = part.start[rank];
    const size_t rowEnd = part.start[rank + 1];
    const size_t numLocal = rowEnd - rowBegin;

    // One matrix row as an MPI datatype, so counts are in rows and cannot overflow int
    MPI_Datatype rowType;
    MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    // Allocate local matrices
    std::vector<unsigned int> localDist(numLocal * numNodes);
    std::vector<unsigned int> localPath(numLocal * numNodes);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(localDist, numNodes, rowBegin, rowEnd, 1, MAX_DISTANCE);
    initializePathMatrix(localPath, numNodes, rowBegin, rowEnd);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    floydWarshall(localDist, localPath, numNodes, part, rank, rowType);

    const double elapsed = MPI_Wtime() - start;

    // The slowest rank determines the runtime
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    // Gather the full distance matrix on rank 0 only when it is needed for output or validation
    std::vector<unsigned int> dist;
    if (printResults || validate) {
        if (rank == 0) {
            dist.resize(numNodes * numNodes);
        }
        std::vector<int> counts(numRanks);
        std::vector<int> displs(numRanks);
        for (int r = 0; r < numRanks; ++r) {
            counts[r] = static_cast<int>(part.start[r + 1] - part.start[r]);
            displs[r] = static_cast<int>(part.start[r]);
        }
        MPI_Gatherv(localDist.data(), static_cast<int>(numLocal), rowType,
                    dist.data(), counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const auto ms = static_cast<long>(maxElapsed * 1000.0);
        printf("Computation time: %ld ms\n", ms);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (ms / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    // Make the exit status identical on all ranks
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Type_free(&rowType);
    MPI_Finalize();
    return exitCode;
}
