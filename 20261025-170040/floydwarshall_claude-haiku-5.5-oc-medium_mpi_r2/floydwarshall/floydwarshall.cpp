// Only the C MPI API is used; skip the (deprecated) C++ bindings
#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
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

// Upper bound on the number of Floyd-Warshall steps processed per block
constexpr size_t MAX_BLOCK = 64;
// Column tile width for the bulk update (keeps the pivot-row tile cache-resident)
constexpr size_t COL_TILE = 512;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Block row distribution: rank p owns global rows [begin(p), end(p))
struct RowPartition {
    size_t n;
    size_t P;
    size_t begin(size_t p) const { return p * n / P; }
    size_t end(size_t p) const { return (p + 1) * n / P; }
};

// Fills the locally owned rows [rowLo, rowHi) of the distance matrix.
// The full random sequence is replayed on every rank so that the values are
// identical to the serial generation order; only the owned rows are stored.
void initializeLocalDistance(std::vector<unsigned int>& dist, const size_t numNodes,
                             const size_t rowLo, const size_t rowHi,
                             const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t startIdx = rowLo * numNodes;
    const size_t endIdx = rowHi * numNodes;

    for (size_t i = 0; i < endIdx; ++i) {
        const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= startIdx) {
            dist[i - startIdx] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t r = rowLo; r < rowHi; ++r) {
        dist[(r - rowLo) * numNodes + r] = 0;
    }
}

// path[r][c] = r for all r, c (same content as the serial initialization)
void initializeLocalPath(std::vector<unsigned int>& path, const size_t numNodes,
                         const size_t rowLo, const size_t rowHi) {
    for (size_t r = rowLo; r < rowHi; ++r) {
        std::fill_n(path.begin() + (r - rowLo) * numNodes, numNodes, static_cast<unsigned int>(r));
    }
}

// dst[j] = min(dst[j], dik + src[j]) for j in [0, len)
inline void relaxDist(unsigned int* __restrict dst, const unsigned int* __restrict src,
                      const unsigned int dik, const size_t len) {
    for (size_t j = 0; j < len; ++j) {
        const unsigned int nd = dik + src[j];
        dst[j] = nd < dst[j] ? nd : dst[j];
    }
}

// Same as relaxDist, additionally recording the intermediate node k on improvement
inline void relaxDistPath(unsigned int* __restrict dst, unsigned int* __restrict pth,
                          const unsigned int* __restrict src, const unsigned int dik,
                          const unsigned int k, const size_t len) {
    for (size_t j = 0; j < len; ++j) {
        const unsigned int nd = dik + src[j];
        const unsigned int cur = dst[j];
        const bool lt = nd < cur;
        dst[j] = lt ? nd : cur;
        pth[j] = lt ? k : pth[j];
    }
}

// Distributed blocked Floyd-Warshall.
//
// Steps k are processed in blocks [k0, k0 + bn). For each block:
//   1. The start-of-block state of the pivot rows is broadcast from their owners.
//   2. Every rank replays the pivot rows through the block (cheap, B^2 * n / 2 work)
//      to obtain the row snapshots snap[kk] = row (k0+kk) as it stands at step k0+kk.
//   3. For each local row, the B block-column entries are propagated through the
//      block to obtain the distance D[i][k0+kk] needed at each step.
//   4. The local rows are updated for all B steps, tile by tile over columns, so the
//      snapshot tile and the row segments stay in cache across the steps.
// This performs exactly the same relaxations, in a dependency-equivalent order, as
// the classic k-i-j loop, so the resulting matrices are identical.
void floydWarshallDistributed(std::vector<unsigned int>& dist,
                              std::vector<unsigned int>& path,
                              const size_t numNodes,
                              const RowPartition& part,
                              const size_t rank,
                              MPI_Comm comm) {
    const size_t n = numNodes;
    const size_t P = part.P;
    const size_t rowLo = part.begin(rank);
    const size_t rowHi = part.end(rank);
    const size_t nloc = rowHi - rowLo;

    if (n == 0) {
        return;
    }

    // Block size: keeps the redundant pivot-panel work below ~1/16 of the total
    const size_t blockSize = std::max<size_t>(1, std::min<size_t>(MAX_BLOCK, n / (8 * P)));

    std::vector<unsigned int> pivot(blockSize * n);  // start-of-block pivot rows, then advanced in place
    std::vector<unsigned int> snap(blockSize * n);   // snap[kk] = row (k0+kk) at step k0+kk
    std::vector<unsigned int> dk(std::max<size_t>(1, nloc * blockSize));  // per local row, D[i][k0+kk] at step kk
    std::vector<unsigned int> chain(blockSize);

    for (size_t k0 = 0; k0 < n; k0 += blockSize) {
        const size_t bn = std::min(blockSize, n - k0);
        const size_t k1 = k0 + bn;

        // 1. Broadcast the pivot rows of this block from their owners
        for (size_t p = 0; p < P; ++p) {
            const size_t s = std::max(k0, part.begin(p));
            const size_t e = std::min(k1, part.end(p));
            if (s >= e) {
                continue;
            }
            unsigned int* buf = pivot.data() + (s - k0) * n;
            const int cnt = static_cast<int>((e - s) * n);
            if (p == rank) {
                std::memcpy(buf, dist.data() + (s - rowLo) * n, sizeof(unsigned int) * (e - s) * n);
            }
            MPI_Bcast(buf, cnt, MPI_UNSIGNED, static_cast<int>(p), comm);
        }

        // 2. Replay the pivot block to obtain step-state snapshots
        for (size_t kk = 0; kk < bn; ++kk) {
            const size_t k = k0 + kk;
            unsigned int* snapK = snap.data() + kk * n;
            std::memcpy(snapK, pivot.data() + kk * n, sizeof(unsigned int) * n);
            for (size_t rr = kk + 1; rr < bn; ++rr) {
                unsigned int* pr = pivot.data() + rr * n;
                relaxDist(pr, snapK, pr[k], n);
            }
        }

        // 3. Per local row: distances to block intermediates at each step
        for (size_t li = 0; li < nloc; ++li) {
            const unsigned int* row = dist.data() + li * n;
            unsigned int* d = dk.data() + li * bn;
            for (size_t kk = 0; kk < bn; ++kk) {
                chain[kk] = row[k0 + kk];
            }
            for (size_t kk = 0; kk < bn; ++kk) {
                d[kk] = chain[kk];
                const unsigned int dik = chain[kk];
                const unsigned int* snapK = snap.data() + kk * n + k0;
                for (size_t kk2 = kk + 1; kk2 < bn; ++kk2) {
                    const unsigned int nd = dik + snapK[kk2];
                    chain[kk2] = nd < chain[kk2] ? nd : chain[kk2];
                }
            }
        }

        // 4. Bulk update of local rows over all steps of the block, tiled over columns
        for (size_t j0 = 0; j0 < n; j0 += COL_TILE) {
            const size_t len = std::min(COL_TILE, n - j0);
            for (size_t li = 0; li < nloc; ++li) {
                unsigned int* row = dist.data() + li * n + j0;
                unsigned int* prow = path.data() + li * n + j0;
                const unsigned int* d = dk.data() + li * bn;
                for (size_t kk = 0; kk < bn; ++kk) {
                    relaxDistPath(row, prow, snap.data() + kk * n + j0, d[kk],
                                  static_cast<unsigned int>(k0 + kk), len);
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

int runBenchmark(int argc, char** argv, const size_t rank, const size_t size, MPI_Comm comm) {
    const bool root = (rank == 0);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on all ranks, so all ranks agree on exit)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (root) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowPartition part{numNodes, size};
    const size_t rowLo = part.begin(rank);
    const size_t rowHi = part.end(rank);
    const size_t nloc = rowHi - rowLo;

    // Allocate local row blocks of the matrices
    std::vector<unsigned int> dist(nloc * numNodes);
    std::vector<unsigned int> path(nloc * numNodes);

    // Initialize
    if (root) printf("Initializing graph...\n");
    initializeLocalDistance(dist, numNodes, rowLo, rowHi, 1, MAX_DISTANCE);
    initializeLocalPath(path, numNodes, rowLo, rowHi);

    // Run Floyd-Warshall
    if (root) printf("Computing shortest paths...\n");
    MPI_Barrier(comm);
    const double t0 = MPI_Wtime();

    floydWarshallDistributed(dist, path, numNodes, part, rank, comm);

    const double elapsed = MPI_Wtime() - t0;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    if (root) {
        const auto durationMs = static_cast<long long>(maxElapsed * 1000.0);
        printf("Computation time: %lld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather the full distance matrix on the root when it is needed for output/validation
    int validationStatus = 0;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (size_t p = 0; p < size; ++p) {
            counts[p] = static_cast<int>((part.end(p) - part.begin(p)) * numNodes);
            displs[p] = static_cast<int>(part.begin(p) * numNodes);
        }
        std::vector<unsigned int> full;
        if (root) full.resize(numNodes * numNodes);

        MPI_Gatherv(dist.data(), static_cast<int>(nloc * numNodes), MPI_UNSIGNED,
                    full.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, comm);

        if (root) {
            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(full, "DistanceMatrix");
            }

            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(full, numNodes);

                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    validationStatus = 1;
                }
            }
        }

        if (validate) {
            MPI_Bcast(&validationStatus, 1, MPI_INT, 0, comm);
        }
    }

    return validationStatus;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runBenchmark(argc, argv, static_cast<size_t>(rank), static_cast<size_t>(size),
                                MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}
