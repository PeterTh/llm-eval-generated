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

// Row decomposition: with idx2(j, i, n) == i * n + j, the matrix is stored row-major
// with respect to the source node i, so each row of the distance/path matrix is
// contiguous and rows can be distributed across MPI ranks without any transposition.
struct RowPartition {
    size_t base;   // minimum number of rows per rank
    size_t rem;    // number of ranks holding one additional row
    int numRanks;

    RowPartition(const size_t numNodes, const int nRanks) noexcept
        : base(numNodes / static_cast<size_t>(nRanks)),
          rem(numNodes % static_cast<size_t>(nRanks)),
          numRanks(nRanks) {}

    size_t start(const int r) const noexcept {
        const size_t rr = static_cast<size_t>(r);
        return rr * base + std::min(rr, rem);
    }
    size_t count(const int r) const noexcept {
        return base + (static_cast<size_t>(r) < rem ? 1u : 0u);
    }
    // Rank owning row `row`
    int owner(const size_t row) const noexcept {
        const size_t split = rem * (base + 1);
        if (row < split) {
            return static_cast<int>(row / (base + 1));
        }
        return static_cast<int>(rem + (row - split) / base);
    }
};

// Relax one row i against the broadcast pivot row k.
// Branch-free so that the compiler can vectorize the update of both matrices.
static inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                            const unsigned int* __restrict rowK, const unsigned int distIK,
                            const unsigned int k, const size_t numNodes) noexcept {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const unsigned int oldDist = distRow[j];
        const bool better = newDist < oldDist;
        distRow[j] = better ? newDist : oldDist;
        pathRow[j] = better ? k : pathRow[j];
    }
}

// Generate the same pseudo-random sequence as the serial version, but keep only the
// locally owned rows. The generator is strictly sequential, so every rank replays the
// stream up to the end of its own block (cheap compared to the O(n^3) solve).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t rowStart, const size_t numLocalRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t firstFlat = rowStart * numNodes;
    const size_t lastFlat = firstFlat + numLocalRows * numNodes;

    for (size_t i = 0; i < firstFlat; ++i) {
        (void)rand_r(&seed);
    }
    for (size_t i = firstFlat; i < lastFlat; ++i) {
        dist[i - firstFlat] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numLocalRows; ++i) {
        dist[i * numNodes + (rowStart + i)] = 0;
    }
}

// The serial initialization assigns path[i][j] = i for all entries (both of its writes
// reduce to "row index"), so each rank can fill its own rows directly.
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowStart, const size_t numLocalRows) {
    for (size_t i = 0; i < numLocalRows; ++i) {
        std::fill_n(path.data() + i * numNodes, numNodes, static_cast<unsigned int>(rowStart + i));
    }
}

// Distributed Floyd-Warshall: for step k the owner of row k broadcasts it, every rank
// relaxes its own rows against it. The owner of row k+1 updates that row first and the
// broadcast for step k+1 is started immediately (non-blocking), so communication overlaps
// with the relaxation of the remaining local rows.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const RowPartition& part,
                   const int rank,
                   MPI_Comm comm) {
    const size_t rowStart = part.start(rank);
    const size_t numLocalRows = part.count(rank);
    const size_t rowEnd = rowStart + numLocalRows;

    std::vector<unsigned int> bufA(numNodes), bufB(numNodes);
    unsigned int* curRow = bufA.data();
    unsigned int* nextRow = bufB.data();

    unsigned int* const distData = dist.data();
    unsigned int* const pathData = path.data();

    const int count = static_cast<int>(numNodes);
    MPI_Request req = MPI_REQUEST_NULL;

    // Prime the pipeline with row 0.
    {
        const int ownerK = part.owner(0);
        if (rank == ownerK) {
            // The owner of row 0 is the rank whose block starts at row 0.
            std::memcpy(curRow, distData, numNodes * sizeof(unsigned int));
        }
        MPI_Ibcast(curRow, count, MPI_UNSIGNED, ownerK, comm, &req);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);

        const size_t kNext = k + 1;
        if (kNext < numNodes) {
            const int ownerNext = part.owner(kNext);
            if (rank == ownerNext) {
                unsigned int* const dRow = distData + (kNext - rowStart) * numNodes;
                relaxRow(dRow, pathData + (kNext - rowStart) * numNodes, curRow, dRow[k],
                         static_cast<unsigned int>(k), numNodes);
                std::memcpy(nextRow, dRow, numNodes * sizeof(unsigned int));
            }
            MPI_Ibcast(nextRow, count, MPI_UNSIGNED, ownerNext, comm, &req);
        }

        // Row k itself is invariant under step k (dist[k][k] == 0), row k+1 is already done.
        for (size_t i = rowStart; i < rowEnd; ++i) {
            if (i == k || i == kNext) continue;
            const size_t local = i - rowStart;
            unsigned int* const dRow = distData + local * numNodes;
            relaxRow(dRow, pathData + local * numNodes, curRow, dRow[k],
                     static_cast<unsigned int>(k), numNodes);
        }

        std::swap(curRow, nextRow);
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

    if (numNodes == 0) {
        if (rank == 0) printf("Number of nodes must be positive\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    const RowPartition part(numNodes, numRanks);
    const size_t rowStart = part.start(rank);
    const size_t numLocalRows = part.count(rank);

    // Allocate local row blocks of the matrices
    std::vector<unsigned int> dist(numLocalRows * numNodes);
    std::vector<unsigned int> path(numLocalRows * numNodes);

    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, rowStart, numLocalRows, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, rowStart, numLocalRows);

    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, part, rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Collect the full distance matrix on rank 0 only if it is actually needed
    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        std::vector<int> counts, displs;
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int r = 0; r < numRanks; ++r) {
                counts[r] = static_cast<int>(part.count(r) * numNodes);
                displs[r] = static_cast<int>(part.start(r) * numNodes);
            }
        }
        MPI_Gatherv(dist.data(), static_cast<int>(numLocalRows * numNodes), MPI_UNSIGNED,
                    fullDist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    // Print results for external validation (integer hash-based)
    if (printResults && rank == 0) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
