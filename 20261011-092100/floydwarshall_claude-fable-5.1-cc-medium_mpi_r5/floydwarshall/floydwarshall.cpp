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

// ---------------------------------------------------------------------------
// MPI row-block decomposition helpers
//
// The matrices are stored such that element (src=i, dst=j) is located at
// idx2(j, i, n) = i * n + j, i.e. every source row is a contiguous block of
// n elements. Rank r owns the contiguous row range [rowStart[r], rowStart[r+1]).
// ---------------------------------------------------------------------------
struct Decomposition {
    int rank = 0;
    int size = 1;
    size_t numNodes = 0;
    std::vector<size_t> rowStart;  // size + 1 entries

    size_t localStart() const { return rowStart[rank]; }
    size_t localEnd() const { return rowStart[rank + 1]; }
    size_t localRows() const { return localEnd() - localStart(); }

    int ownerOf(const size_t row) const {
        // rowStart is monotonically non-decreasing; find r with rowStart[r] <= row < rowStart[r+1]
        const auto it = std::upper_bound(rowStart.begin(), rowStart.end(), row);
        return static_cast<int>(it - rowStart.begin()) - 1;
    }
};

Decomposition makeDecomposition(const size_t numNodes) {
    Decomposition d;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.size);
    d.numNodes = numNodes;
    d.rowStart.resize(d.size + 1);
    const size_t base = numNodes / d.size;
    const size_t rem = numNodes % d.size;
    d.rowStart[0] = 0;
    for (int r = 0; r < d.size; ++r) {
        d.rowStart[r + 1] = d.rowStart[r] + base + (static_cast<size_t>(r) < rem ? 1 : 0);
    }
    return d;
}

// Each rank reproduces the exact sequential rand_r stream (seed 42) and keeps
// only the values belonging to its own rows. This yields a bit-identical
// matrix to the original single-process initialization.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const Decomposition& dec,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    const size_t numNodes = dec.numNodes;
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t firstIdx = dec.localStart() * numNodes;
    const size_t lastIdx = dec.localEnd() * numNodes;

    // Skip over the random values belonging to rows of lower ranks
    for (size_t i = 0; i < firstIdx; ++i) {
        (void)rand_r(&seed);
    }
    for (size_t i = firstIdx; i < lastIdx; ++i) {
        dist[i - firstIdx] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = dec.localStart(); i < dec.localEnd(); ++i) {
        dist[idx2(i, i - dec.localStart(), numNodes)] = 0;
    }
}

// Original semantics: for every j, path[idx2(i, j)] = j (row j <- j) and
// path[idx2(j, i)] = i (column j <- row index). Both writes store the row
// index at every location, so the final matrix has path[row][col] = row.
void initializePathMatrix(std::vector<unsigned int>& path, const Decomposition& dec) {
    const size_t numNodes = dec.numNodes;
    const size_t localRows = dec.localRows();
    for (size_t r = 0; r < localRows; ++r) {
        unsigned int* row = path.data() + r * numNodes;
        const unsigned int rowIdx = static_cast<unsigned int>(dec.localStart() + r);
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = rowIdx;
        }
    }
}

// Relax one local row against the broadcast pivot row k.
static inline void relaxRow(unsigned int* __restrict distRow,
                            unsigned int* __restrict pathRow,
                            const unsigned int* __restrict rowK,
                            const unsigned int distIK,
                            const unsigned int k,
                            const size_t numNodes) noexcept {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const bool better = newDist < distRow[j];
        distRow[j] = better ? newDist : distRow[j];
        pathRow[j] = better ? k : pathRow[j];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const Decomposition& dec) {
    const size_t numNodes = dec.numNodes;
    const size_t localStart = dec.localStart();
    const size_t localRows = dec.localRows();

    // Two pivot-row buffers so that the broadcast of row k+1 can be
    // overlapped with the relaxation of iteration k.
    std::vector<unsigned int> pivotBuf[2];
    pivotBuf[0].resize(numNodes);
    pivotBuf[1].resize(numNodes);

    // Precompute pivot owners once.
    std::vector<int> owner(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        owner[k] = dec.ownerOf(k);
    }

    MPI_Request req = MPI_REQUEST_NULL;

    // Kick off broadcast of pivot row 0.
    {
        unsigned int* buf = pivotBuf[0].data();
        if (owner[0] == dec.rank) {
            std::memcpy(buf, dist.data() + (0 - localStart) * numNodes, numNodes * sizeof(unsigned int));
        }
        MPI_Ibcast(buf, static_cast<int>(numNodes), MPI_UNSIGNED, owner[0], MPI_COMM_WORLD, &req);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const int cur = static_cast<int>(k & 1);
        const int nxt = cur ^ 1;
        const unsigned int* rowK = pivotBuf[cur].data();

        MPI_Wait(&req, MPI_STATUS_IGNORE);

        const unsigned int kk = static_cast<unsigned int>(k);
        const size_t kNext = k + 1;
        const bool hasNext = kNext < numNodes;
        const bool ownsNext = hasNext && owner[kNext] == dec.rank;

        // If this rank owns the next pivot row, relax it first and start its
        // broadcast so that communication overlaps with the remaining rows.
        size_t skipLocal = static_cast<size_t>(-1);
        if (ownsNext) {
            const size_t r = kNext - localStart;
            unsigned int* distRow = dist.data() + r * numNodes;
            unsigned int* pathRow = path.data() + r * numNodes;
            relaxRow(distRow, pathRow, rowK, distRow[k], kk, numNodes);
            std::memcpy(pivotBuf[nxt].data(), distRow, numNodes * sizeof(unsigned int));
            skipLocal = r;
        }
        if (hasNext) {
            MPI_Ibcast(pivotBuf[nxt].data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                       owner[kNext], MPI_COMM_WORLD, &req);
        }

        for (size_t r = 0; r < localRows; ++r) {
            if (r == skipLocal) continue;
            unsigned int* distRow = dist.data() + r * numNodes;
            unsigned int* pathRow = path.data() + r * numNodes;
            relaxRow(distRow, pathRow, rowK, distRow[k], kk, numNodes);
        }
    }

    if (req != MPI_REQUEST_NULL) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
    }
}

// Gather the distributed distance matrix onto rank 0 (full matrix only allocated there).
void gatherMatrix(const std::vector<unsigned int>& local, std::vector<unsigned int>& full,
                  const Decomposition& dec) {
    std::vector<int> counts(dec.size), displs(dec.size);
    for (int r = 0; r < dec.size; ++r) {
        counts[r] = static_cast<int>((dec.rowStart[r + 1] - dec.rowStart[r]) * dec.numNodes);
        displs[r] = static_cast<int>(dec.rowStart[r] * dec.numNodes);
    }
    if (dec.rank == 0) {
        full.resize(dec.numNodes * dec.numNodes);
    }
    MPI_Gatherv(local.data(), static_cast<int>(dec.localRows() * dec.numNodes), MPI_UNSIGNED,
                dec.rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
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
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const bool isRoot = (rank == 0);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    bool earlyExit = false;
    
    // Parse command line arguments (all ranks see identical argv)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (isRoot) printUsage(argv[0]);
            earlyExit = true;
            exitCode = 0;
            break;
        } else {
            if (isRoot) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            earlyExit = true;
            exitCode = 1;
            break;
        }
    }
    if (earlyExit) {
        MPI_Finalize();
        return exitCode;
    }
    
    if (isRoot) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const Decomposition dec = makeDecomposition(numNodes);
    
    // Allocate local row blocks
    std::vector<unsigned int> dist(dec.localRows() * numNodes);
    std::vector<unsigned int> path(dec.localRows() * numNodes);
    
    // Initialize
    if (isRoot) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, dec, 1, MAX_DISTANCE);
    initializePathMatrix(path, dec);
    
    // Run Floyd-Warshall
    if (isRoot) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double tStart = MPI_Wtime();
    
    floydWarshall(dist, path, dec);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double tEnd = MPI_Wtime();
    const long durationMs = static_cast<long>((tEnd - tStart) * 1000.0);
    
    if (isRoot) {
        printf("Computation time: %ld ms\n", durationMs);
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        gatherMatrix(dist, fullDist, dec);
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults && isRoot) {
        print_results_int(fullDist, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        int valid = 1;
        if (isRoot) {
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
    
    MPI_Finalize();
    return exitCode;
}
