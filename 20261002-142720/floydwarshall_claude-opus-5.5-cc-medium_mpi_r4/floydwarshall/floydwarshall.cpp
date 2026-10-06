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

// Block row distribution: rank r owns rows [rowStart(r), rowStart(r+1))
struct RowDist {
    size_t n;
    int size;
    size_t base, rem;
    RowDist(size_t n_, int size_) : n(n_), size(size_), base(n_ / size_), rem(n_ % size_) {}
    size_t start(int r) const { return r * base + std::min<size_t>(r, rem); }
    size_t count(int r) const { return base + (static_cast<size_t>(r) < rem ? 1 : 0); }
    int owner(size_t row) const {
        const size_t split = rem * (base + 1);
        if (row < split) return static_cast<int>(row / (base + 1));
        return static_cast<int>(rem + (row - split) / base);
    }
};

// Initialize the locally owned rows [r0, r0+nr) with exactly the same values
// as the sequential generator (the rand_r stream is advanced past earlier rows).
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t r0, const size_t nr) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t first = r0 * numNodes;
    for (size_t i = 0; i < first; ++i) {
        (void)rand_r(&seed);
    }
    for (size_t i = 0; i < nr * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < nr; ++i) {
        dist[i * numNodes + (r0 + i)] = 0;
    }
}

// Equivalent to the sequential initialization: element (row r, col c) ends up
// as r, i.e. path[idx2(c, r, n)] = r.
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t r0, const size_t nr) {
    for (size_t i = 0; i < nr; ++i) {
        unsigned int* row = path.data() + i * numNodes;
        const unsigned int r = static_cast<unsigned int>(r0 + i);
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = r;
        }
    }
}

static inline void relaxRow(unsigned int* __restrict distI, unsigned int* __restrict pathI,
                            const unsigned int* __restrict rowK, const size_t n,
                            const unsigned int k) {
    const unsigned int distIK = distI[k];
#pragma GCC ivdep
    for (size_t j = 0; j < n; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const unsigned int d = distI[j];
        const bool better = newDist < d;
        distI[j] = better ? newDist : d;
        pathI[j] = better ? k : pathI[j];
    }
}

// Distributed Floyd-Warshall: each rank owns a block of rows. At step k the
// owner of row k broadcasts it. Row k+1 is relaxed first by its owner and its
// broadcast is overlapped with the remaining computation of step k.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const RowDist& rd, const int rank) {
    if (numNodes == 0) return;
    const size_t n = numNodes;
    const size_t r0 = rd.start(rank);
    const size_t nr = rd.count(rank);
    const int count = static_cast<int>(n);

    std::vector<unsigned int> rowBuf[2] = {std::vector<unsigned int>(n),
                                           std::vector<unsigned int>(n)};

    // Row 0
    {
        const int root = rd.owner(0);
        if (root == rank) {
            std::memcpy(rowBuf[0].data(), dist.data() + (0 - r0) * n, n * sizeof(unsigned int));
        }
        MPI_Bcast(rowBuf[0].data(), count, MPI_UNSIGNED, root, MPI_COMM_WORLD);
    }

    for (size_t k = 0; k < n; ++k) {
        const unsigned int* rowK = rowBuf[k & 1].data();
        const unsigned int kk = static_cast<unsigned int>(k);
        MPI_Request req = MPI_REQUEST_NULL;
        size_t skipRow = static_cast<size_t>(-1);

        if (k + 1 < n) {
            const size_t next = k + 1;
            const int root = rd.owner(next);
            unsigned int* nextBuf = rowBuf[next & 1].data();
            if (root == rank) {
                const size_t li = next - r0;
                relaxRow(dist.data() + li * n, path.data() + li * n, rowK, n, kk);
                std::memcpy(nextBuf, dist.data() + li * n, n * sizeof(unsigned int));
                skipRow = li;
            }
            MPI_Ibcast(nextBuf, count, MPI_UNSIGNED, root, MPI_COMM_WORLD, &req);
        }

        for (size_t i = 0; i < nr; ++i) {
            if (i == skipRow) continue;
            relaxRow(dist.data() + i * n, path.data() + i * n, rowK, n, kk);
            if (req != MPI_REQUEST_NULL && (i & 15) == 15) {
                int flag;
                MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            }
        }

        if (req != MPI_REQUEST_NULL) {
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
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const RowDist rd(numNodes, size);
    const size_t r0 = rd.start(rank);
    const size_t nr = rd.count(rank);
    
    // Allocate local row blocks
    std::vector<unsigned int> dist(nr * numNodes);
    std::vector<unsigned int> path(nr * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, r0, nr);
    initializePathMatrix(path, numNodes, r0, nr);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, rd, rank);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather full distance matrix on rank 0 (only needed for output/validation)
    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = static_cast<int>(rd.count(r) * numNodes);
            displs[r] = static_cast<int>(rd.start(r) * numNodes);
        }
        if (rank == 0) fullDist.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), static_cast<int>(nr * numNodes), MPI_UNSIGNED,
                    fullDist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
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
                ret = 1;
            }
        }
    }
    MPI_Bcast(&ret, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    MPI_Finalize();
    return ret;
}
