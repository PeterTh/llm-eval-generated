#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cstdint>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Each rank stores a contiguous block of rows [rowStart, rowStart+rowCount).
// Row-major: local[(i-rowStart)*n + j] == dist[idx2(j, i, n)].
struct Partition {
    size_t n;
    int p;
    size_t base, rem;
    Partition(size_t n_, int p_) : n(n_), p(p_), base(n_ / p_), rem(n_ % p_) {}
    size_t start(int r) const { return r * base + std::min<size_t>(r, rem); }
    size_t count(int r) const { return base + (static_cast<size_t>(r) < rem ? 1 : 0); }
    int owner(size_t row) const {
        const size_t big = rem * (base + 1);
        if (row < big) return static_cast<int>(row / (base + 1));
        return static_cast<int>(rem + (row - big) / base);
    }
};

void initializeDistanceMatrix(std::vector<unsigned int>& local, const size_t numNodes,
                              const size_t rowStart, const size_t rowCount,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t lo = rowStart * numNodes;
    const size_t hi = (rowStart + rowCount) * numNodes;

    // The RNG stream is sequential: every rank walks it but keeps only its rows.
    for (size_t i = 0; i < hi; ++i) {
        const int r = rand_r(&seed);
        if (i >= lo)
            local[i - lo] = rangeMin + (unsigned int)(range * r / (double)RAND_MAX);
    }

    for (size_t i = rowStart; i < rowStart + rowCount; ++i) {
        local[(i - rowStart) * numNodes + i] = 0;
    }
}

// Original initialization yields path[i][j] = i for all entries.
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowStart, const size_t rowCount) {
    for (size_t i = 0; i < rowCount; ++i) {
        unsigned int* row = &path[i * numNodes];
        for (size_t j = 0; j < numNodes; ++j) row[j] = static_cast<unsigned int>(rowStart + i);
    }
}

static inline void relaxRow(unsigned int* __restrict d, unsigned int* __restrict pth,
                            const unsigned int* __restrict rowK, const unsigned int dik,
                            const unsigned int k, const size_t n) {
    for (size_t j = 0; j < n; ++j) {
        const unsigned int nd = dik + rowK[j];
        const bool better = nd < d[j];
        d[j] = better ? nd : d[j];
        pth[j] = better ? k : pth[j];
    }
}

void floydWarshall(std::vector<unsigned int>& local,
                   std::vector<unsigned int>& path,
                   const size_t n, const Partition& part, const int rank, MPI_Comm comm) {
    const size_t rowStart = part.start(rank);
    const size_t rowCount = part.count(rank);
    std::vector<unsigned int> bufs[2] = {std::vector<unsigned int>(n), std::vector<unsigned int>(n)};
    MPI_Request req;

    auto post = [&](size_t k, std::vector<unsigned int>& buf) {
        const int root = part.owner(k);
        if (root == rank)
            std::memcpy(buf.data(), &local[(k - rowStart) * n], n * sizeof(unsigned int));
        MPI_Ibcast(buf.data(), static_cast<int>(n), MPI_UNSIGNED, root, comm, &req);
    };

    if (n == 0) return;
    post(0, bufs[0]);

    for (size_t k = 0; k < n; ++k) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        const unsigned int* rowK = bufs[k & 1].data();
        const size_t next = k + 1;
        size_t skip = SIZE_MAX;  // local row already handled

        if (next < n) {
            std::vector<unsigned int>& nb = bufs[next & 1];
            if (part.owner(next) == rank) {
                // Update row k+1 first so it can be broadcast early.
                const size_t li = next - rowStart;
                relaxRow(&local[li * n], &path[li * n], rowK, local[li * n + k],
                         static_cast<unsigned int>(k), n);
                skip = li;
            }
            post(next, nb);
        }

        for (size_t li = 0; li < rowCount; ++li) {
            if (li == skip) continue;
            relaxRow(&local[li * n], &path[li * n], rowK, local[li * n + k],
                     static_cast<unsigned int>(k), n);
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    }

    const Partition part(numNodes, size);
    const size_t rowStart = part.start(rank);
    const size_t rowCount = part.count(rank);

    std::vector<unsigned int> local(rowCount * numNodes);
    std::vector<unsigned int> path(rowCount * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(local, numNodes, rowStart, rowCount, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, rowStart, rowCount);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(local, path, numNodes, part, rank, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int result = 0;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(part.count(r) * numNodes);
            displs[r] = static_cast<int>(part.start(r) * numNodes);
        }
        std::vector<unsigned int> dist;
        if (rank == 0) dist.resize(numNodes * numNodes);
        MPI_Gatherv(local.data(), counts[rank], MPI_UNSIGNED, dist.data(), counts.data(),
                    displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (printResults) {
                print_results_int(dist, "DistanceMatrix");
            }
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(dist, numNodes);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    result = 1;
                }
            }
        }
    }

    MPI_Finalize();
    return result;
}
