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

// Rows [rowStart, rowStart+numRows) are owned by each rank (block distribution).
// Layout is row-major: element (i,j) lives at i*n + j (identical to idx2(j,i,n)).
struct Partition {
    size_t n;
    int size;
    size_t base, rem;
    Partition(size_t n_, int size_) : n(n_), size(size_), base(n_ / size_), rem(n_ % size_) {}
    size_t start(int r) const { return r * base + std::min<size_t>(r, rem); }
    size_t count(int r) const { return base + (static_cast<size_t>(r) < rem ? 1 : 0); }
    int owner(size_t row) const {
        const size_t big = rem * (base + 1);
        if (row < big) return static_cast<int>(row / (base + 1));
        return static_cast<int>(rem + (row - big) / base);
    }
};

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const size_t rowStart, const size_t numRows,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t lo = rowStart * numNodes;
    const size_t hi = (rowStart + numRows) * numNodes;

    // The random stream is sequential; every rank advances it but keeps only its rows.
    for (size_t i = 0; i < hi; ++i) {
        const unsigned int v = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= lo) dist[i - lo] = v;
    }

    for (size_t r = 0; r < numRows; ++r) {
        dist[r * numNodes + rowStart + r] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowStart, const size_t numRows) {
    for (size_t r = 0; r < numRows; ++r)
        for (size_t j = 0; j < numNodes; ++j)
            path[r * numNodes + j] = static_cast<unsigned int>(rowStart + r);
}

static inline void relaxRow(unsigned int* __restrict drow, unsigned int* __restrict prow,
                            const unsigned int* __restrict rowK, const size_t k, const size_t n) {
    const unsigned int dik = drow[k];
    const unsigned int kk = static_cast<unsigned int>(k);
    for (size_t j = 0; j < n; ++j) {
        const unsigned int nd = dik + rowK[j];
        const bool better = nd < drow[j];
        drow[j] = better ? nd : drow[j];
        prow[j] = better ? kk : prow[j];
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t n, const Partition& part, const int rank) {
    const size_t rowStart = part.start(rank);
    const size_t numRows = part.count(rank);
    std::vector<unsigned int> bufA(n), bufB(n);
    unsigned int* cur = bufA.data();
    unsigned int* nxt = bufB.data();
    MPI_Request req = MPI_REQUEST_NULL;

    if (n == 0) return;
    if (part.owner(0) == rank) memcpy(cur, &dist[(0 - rowStart) * n], n * sizeof(unsigned int));
    MPI_Bcast(cur, (int)n, MPI_UNSIGNED, part.owner(0), MPI_COMM_WORLD);

    for (size_t k = 0; k < n; ++k) {
        size_t skip = (size_t)-1;
        if (k + 1 < n) {
            // Lookahead: owner of row k+1 relaxes it first and starts its broadcast,
            // overlapping communication with the remaining local computation.
            const int own = part.owner(k + 1);
            if (own == rank) {
                const size_t lr = k + 1 - rowStart;
                relaxRow(&dist[lr * n], &path[lr * n], cur, k, n);
                memcpy(nxt, &dist[lr * n], n * sizeof(unsigned int));
                skip = lr;
            }
            MPI_Ibcast(nxt, (int)n, MPI_UNSIGNED, own, MPI_COMM_WORLD, &req);
        }
        for (size_t r = 0; r < numRows; ++r) {
            if (r == skip) continue;
            relaxRow(&dist[r * n], &path[r * n], cur, k, n);
        }
        if (k + 1 < n) {
            MPI_Wait(&req, MPI_STATUS_IGNORE);
            std::swap(cur, nxt);
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
    int rank, size;
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
        printf("MPI ranks: %d\n", size);
    }

    Partition part(numNodes, size);
    const size_t rowStart = part.start(rank);
    const size_t numRows = part.count(rank);
    std::vector<unsigned int> dist(numRows * numNodes);
    std::vector<unsigned int> path(numRows * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, rowStart, numRows, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes, rowStart, numRows);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, part, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather full distance matrix on rank 0 (only needed for output/validation)
    std::vector<unsigned int> full;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (int)(part.count(r) * numNodes);
            displs[r] = (int)(part.start(r) * numNodes);
        }
        if (rank == 0) full.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, full.data(), counts.data(),
                    displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    if (rank != 0) {
        MPI_Finalize();
        return 0;
    }
    
    // Print results for external validation (integer hash-based)
    if (printResults) {
        print_results_int(full, "DistanceMatrix");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(full, numNodes);
        
        if (valid) {
            printf("Validation: PASSED\n");
            MPI_Finalize();
            return 0;
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
