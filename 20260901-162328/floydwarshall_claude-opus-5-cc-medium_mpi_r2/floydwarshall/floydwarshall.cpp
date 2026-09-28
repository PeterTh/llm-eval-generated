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

// The flat layout above means that idx2(j, i, n) == i * n + j, i.e. the matrix is
// stored row-major with `i` selecting the row and `j` the (contiguous) column.
// The MPI decomposition therefore distributes complete rows across ranks: rank r
// owns rows [rowBegin(r), rowBegin(r+1)). Every Floyd-Warshall step k needs the
// full row k, which the owning rank broadcasts to everyone.

// Block row distribution: first (n % size) ranks get one extra row.
inline size_t rowBegin(const size_t numNodes, const int rank, const int size) noexcept {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

inline int rowOwner(const size_t row, const size_t numNodes, const int size) noexcept {
    const size_t base = numNodes / static_cast<size_t>(size);
    const size_t rem = numNodes % static_cast<size_t>(size);
    if (base == 0) {
        return static_cast<int>(row);
    }
    if (row < rem * (base + 1)) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(rem + (row - rem * (base + 1)) / base);
}

// Generates exactly the same sequence as the serial code, but keeps only the rows
// owned by this rank (the RNG is inherently sequential, so the cheap O(n^2) stream
// is replayed everywhere while only O(n^2 / p) values are stored).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t firstRow, const size_t localRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t begin = firstRow * numNodes;
    const size_t end = begin + localRows * numNodes;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        const unsigned int value = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= begin && i < end) {
            dist[i - begin] = value;
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < localRows; ++i) {
        dist[i * numNodes + (firstRow + i)] = 0;
    }
}

// The serial initialization assigns path[idx2(i, j, n)] = j for every (i, j),
// i.e. every entry of a row equals that row's index.
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t firstRow, const size_t localRows) {
    for (size_t i = 0; i < localRows; ++i) {
        std::fill_n(path.begin() + i * numNodes, numNodes,
                    static_cast<unsigned int>(firstRow + i));
    }
}

// Updates one owned row against the broadcast pivot row k.
static inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                            const unsigned int* __restrict kRow, const unsigned int distIK,
                            const unsigned int k, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + kRow[j];
        const bool better = newDist < distRow[j];
        // Branch-free so the compiler can vectorize both updates with a blend.
        distRow[j] = better ? newDist : distRow[j];
        pathRow[j] = better ? k : pathRow[j];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t firstRow, const size_t localRows,
                   const int size, MPI_Comm comm) {
    // Distributed Floyd-Warshall. For step k the owner of row k broadcasts it; all
    // ranks then relax their own rows against it. The pivot row for step k+1 is
    // produced (and broadcast non-blockingly) before the bulk of step k's work, so
    // the broadcast overlaps with computation.
    if (numNodes == 0) {
        return;
    }

    std::vector<unsigned int> kRowBuf(2 * numNodes);
    unsigned int* buf[2] = {kRowBuf.data(), kRowBuf.data() + numNodes};
    MPI_Request req[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};

    const size_t lastRow = firstRow + localRows;
    const int intN = static_cast<int>(numNodes);

    // Prime the pipeline with row 0.
    {
        const int owner = rowOwner(0, numNodes, size);
        if (0 >= firstRow && 0 < lastRow) {
            std::copy_n(dist.data(), numNodes, buf[0]);
        }
        MPI_Ibcast(buf[0], intN, MPI_UNSIGNED, owner, comm, &req[0]);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const int cur = static_cast<int>(k & 1);
        const int nxt = cur ^ 1;
        MPI_Wait(&req[cur], MPI_STATUS_IGNORE);
        const unsigned int* __restrict kRow = buf[cur];
        const unsigned int ku = static_cast<unsigned int>(k);

        // Produce and publish the next pivot row first, so its broadcast can
        // progress while the remaining rows are relaxed.
        const size_t next = k + 1;
        bool ownsNext = false;
        if (next < numNodes) {
            ownsNext = (next >= firstRow && next < lastRow);
            if (ownsNext) {
                const size_t local = next - firstRow;
                unsigned int* distRow = dist.data() + local * numNodes;
                relaxRow(distRow, path.data() + local * numNodes, kRow, distRow[k], ku, numNodes);
                std::copy_n(distRow, numNodes, buf[nxt]);
            }
            MPI_Ibcast(buf[nxt], intN, MPI_UNSIGNED, rowOwner(next, numNodes, size),
                       comm, &req[nxt]);
        }

        for (size_t i = 0; i < localRows; ++i) {
            if (ownsNext && firstRow + i == next) {
                continue;  // already relaxed above
            }
            unsigned int* distRow = dist.data() + i * numNodes;
            relaxRow(distRow, path.data() + i * numNodes, kRow, distRow[k], ku, numNodes);
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
    int size = 1;
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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

    // With more ranks than rows some ranks have no work; split them off so the
    // collectives below run on exactly the participating ranks.
    MPI_Comm comm = MPI_COMM_WORLD;
    if (numNodes > 0 && static_cast<size_t>(size) > numNodes) {
        const int active = static_cast<int>(numNodes);
        MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
        if (rank >= active) {
            MPI_Finalize();
            return 0;
        }
        size = active;
    }

    const size_t firstRow = rowBegin(numNodes, rank, size);
    const size_t localRows = rowBegin(numNodes, rank + 1, size) - firstRow;

    // Allocate local row blocks
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(dist, numNodes, firstRow, localRows, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, firstRow, localRows);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(comm);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, firstRow, localRows, size, comm);

    MPI_Barrier(comm);
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

    // Result printing and validation operate on the full matrix, so collect it on
    // rank 0 only when actually requested.
    if (printResults || validate) {
        std::vector<unsigned int> full;
        std::vector<int> counts, displs;
        if (rank == 0) {
            full.resize(numNodes * numNodes);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t b = rowBegin(numNodes, r, size);
                counts[r] = static_cast<int>((rowBegin(numNodes, r + 1, size) - b) * numNodes);
                displs[r] = static_cast<int>(b * numNodes);
            }
        }
        MPI_Gatherv(dist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                    rank == 0 ? full.data() : nullptr, counts.data(), displs.data(),
                    MPI_UNSIGNED, 0, comm);

        int failed = 0;
        if (rank == 0) {
            if (printResults) {
                print_results_int(full, "DistanceMatrix");
            }
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(full, numNodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                failed = valid ? 0 : 1;
            }
        }
        MPI_Bcast(&failed, 1, MPI_INT, 0, comm);
        MPI_Finalize();
        return failed;
    }

    MPI_Finalize();
    return 0;
}
