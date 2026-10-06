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

// Block-cyclic row distribution: global row block g (rows [g*B, min((g+1)*B, n)))
// is owned by rank g % nprocs and stored as local block g / nprocs.
struct Distribution {
    size_t n;
    size_t B;          // rows per block
    size_t nb;         // number of global blocks
    int rank;
    int nprocs;
    size_t localRows;  // number of rows stored on this rank

    int owner(size_t g) const { return static_cast<int>(g % nprocs); }
    size_t blockStart(size_t g) const { return g * B; }
    size_t blockRows(size_t g) const { return std::min(B, n - g * B); }
    // Local row offset of a block owned by this rank
    size_t localOffset(size_t g) const { return (g / nprocs) * B; }
};

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const Distribution& d,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t n = d.n;

    // Replay the sequential random stream, keeping only rows owned by this rank
    for (size_t g = 0; g < d.nb; ++g) {
        const size_t count = d.blockRows(g) * n;
        if (d.owner(g) == d.rank) {
            unsigned int* out = dist.data() + d.localOffset(g) * n;
            for (size_t e = 0; e < count; ++e) {
                out[e] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            }
            // Set diagonal to zero (distance from node to itself is 0)
            for (size_t r = 0; r < d.blockRows(g); ++r) {
                out[r * n + d.blockStart(g) + r] = 0;
            }
        } else {
            for (size_t e = 0; e < count; ++e) {
                (void)rand_r(&seed);
            }
        }
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const Distribution& d) {
    // Equivalent to the original initialization: path[i][j] = i
    const size_t n = d.n;
    for (size_t g = d.rank; g < d.nb; g += d.nprocs) {
        const size_t off = d.localOffset(g);
        for (size_t r = 0; r < d.blockRows(g); ++r) {
            const unsigned int i = static_cast<unsigned int>(d.blockStart(g) + r);
            std::fill_n(path.data() + (off + r) * n, n, i);
        }
    }
}

// Relax one row i against pivot row k: dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
static inline void relaxRow(unsigned int* __restrict dRow, unsigned int* __restrict pRow,
                            const unsigned int* __restrict rowK, const unsigned int dik,
                            const unsigned int k, const size_t n) noexcept {
    for (size_t j = 0; j < n; ++j) {
        const unsigned int newDist = dik + rowK[j];
        const unsigned int cur = dRow[j];
        const bool better = newDist < cur;
        dRow[j] = better ? newDist : cur;
        pRow[j] = better ? k : pRow[j];
    }
}

// Apply pivots k0..k0+nk-1 (pivot rows in buf) in order to local rows [row0, row0+rows).
// Per element this is exactly the classic k-ordered update.
static void applyPanel(unsigned int* dist, unsigned int* path, const unsigned int* buf,
                       const size_t k0, const size_t nk, const size_t row0,
                       const size_t rows, const size_t n, MPI_Request* pending) {
    for (size_t r = row0; r < row0 + rows; ++r) {
        unsigned int* dRow = dist + r * n;
        unsigned int* pRow = path + r * n;
        for (size_t kk = 0; kk < nk; ++kk) {
            const size_t k = k0 + kk;
            relaxRow(dRow, pRow, buf + kk * n, dRow[k], static_cast<unsigned int>(k), n);
        }
        if (pending && *pending != MPI_REQUEST_NULL) {
            int flag;
            MPI_Test(pending, &flag, MPI_STATUS_IGNORE);  // drive progress of the broadcast
        }
    }
}

// Owner of block g runs the classic algorithm over the pivots of block g restricted to
// the block's own rows, recording each pivot row as of its iteration into buf.
static void computePanel(unsigned int* dist, unsigned int* path, unsigned int* buf,
                         const Distribution& d, const size_t g) {
    const size_t n = d.n;
    const size_t k0 = d.blockStart(g);
    const size_t nk = d.blockRows(g);
    const size_t off = d.localOffset(g);
    for (size_t kk = 0; kk < nk; ++kk) {
        const size_t k = k0 + kk;
        unsigned int* rowK = buf + kk * n;
        std::memcpy(rowK, dist + (off + kk) * n, n * sizeof(unsigned int));
        for (size_t r = 0; r < nk; ++r) {
            unsigned int* dRow = dist + (off + r) * n;
            relaxRow(dRow, path + (off + r) * n, rowK, dRow[k], static_cast<unsigned int>(k), n);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const Distribution& d) {
    // Floyd-Warshall over a block-cyclic row distribution, processing pivots in blocks.
    // Row k is not modified during iteration k (non-negative weights, zero diagonal),
    // so each pivot block is computed by its owner and broadcast. The next pivot block
    // is computed early and its broadcast overlaps with the local updates.
    const size_t n = d.n;
    const size_t B = d.B;
    std::vector<unsigned int> bufs[2] = {std::vector<unsigned int>(B * n),
                                         std::vector<unsigned int>(B * n)};
    MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    auto startBcast = [&](size_t g) {
        unsigned int* buf = bufs[g & 1].data();
        if (d.owner(g) == d.rank) {
            computePanel(dist.data(), path.data(), buf, d, g);
        }
        MPI_Ibcast(buf, static_cast<int>(d.blockRows(g) * n), MPI_UNSIGNED, d.owner(g),
                   MPI_COMM_WORLD, &reqs[g & 1]);
    };

    if (d.nb > 0) startBcast(0);

    for (size_t g = 0; g < d.nb; ++g) {
        MPI_Wait(&reqs[g & 1], MPI_STATUS_IGNORE);
        const unsigned int* buf = bufs[g & 1].data();
        const size_t k0 = d.blockStart(g);
        const size_t nk = d.blockRows(g);
        const bool haveNext = g + 1 < d.nb;
        const bool ownNext = haveNext && d.owner(g + 1) == d.rank;

        if (haveNext) {
            if (ownNext) {
                applyPanel(dist.data(), path.data(), buf, k0, nk, d.localOffset(g + 1),
                           d.blockRows(g + 1), n, nullptr);
            }
            startBcast(g + 1);
        }

        MPI_Request* pending = haveNext ? &reqs[(g + 1) & 1] : nullptr;
        for (size_t h = d.rank; h < d.nb; h += d.nprocs) {
            if (h == g || (ownNext && h == g + 1)) continue;
            applyPanel(dist.data(), path.data(), buf, k0, nk, d.localOffset(h),
                       d.blockRows(h), n, pending);
        }
    }
}

// Collect the distributed matrix on rank 0 (returned non-empty on rank 0 only)
std::vector<unsigned int> gatherMatrix(const std::vector<unsigned int>& local,
                                       const Distribution& d) {
    const size_t n = d.n;
    std::vector<unsigned int> full;
    if (d.rank == 0) full.resize(n * n);
    for (size_t g = 0; g < d.nb; ++g) {
        const int src = d.owner(g);
        const size_t count = d.blockRows(g) * n;
        if (d.rank == 0) {
            unsigned int* dst = full.data() + d.blockStart(g) * n;
            if (src == 0) {
                std::memcpy(dst, local.data() + d.localOffset(g) * n, count * sizeof(unsigned int));
            } else {
                MPI_Recv(dst, static_cast<int>(count), MPI_UNSIGNED, src, 0, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
            }
        } else if (src == d.rank) {
            MPI_Send(local.data() + d.localOffset(g) * n, static_cast<int>(count), MPI_UNSIGNED,
                     0, 0, MPI_COMM_WORLD);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    const bool root = rank == 0;

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
            if (root) printUsage(argv[0]);
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
    }

    // Pivot block size: large enough for cache reuse of each row, small enough to give
    // several blocks per rank (load balance) and a short serial panel critical path.
    Distribution d;
    d.n = numNodes;
    d.rank = rank;
    d.nprocs = nprocs;
    d.B = std::clamp<size_t>(numNodes / (4 * static_cast<size_t>(nprocs)), 1, 64);
    d.nb = (numNodes + d.B - 1) / d.B;
    d.localRows = 0;
    for (size_t g = rank; g < d.nb; g += nprocs) d.localRows += d.blockRows(g);

    // Allocate local row blocks
    std::vector<unsigned int> dist(d.localRows * numNodes);
    std::vector<unsigned int> path(d.localRows * numNodes);

    // Initialize
    if (root) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, d, 1, MAX_DISTANCE);
    initializePathMatrix(path, d);

    // Run Floyd-Warshall
    if (root) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, d);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) fullDist = gatherMatrix(dist, d);

    // Print results for external validation (integer hash-based)
    if (printResults && root) {
        print_results_int(fullDist, "DistanceMatrix");
    }

    int exitCode = 0;
    // Validation
    if (validate) {
        if (root) {
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
