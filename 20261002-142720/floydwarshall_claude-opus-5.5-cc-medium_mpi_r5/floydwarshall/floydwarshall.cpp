#include <algorithm>
#include <cstdint>
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
static inline size_t rowStart(const size_t numNodes, const int rank, const int size) noexcept {
    const size_t base = numNodes / size;
    const size_t rem = numNodes % size;
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

static inline int rowOwner(const size_t row, const size_t numNodes, const int size) noexcept {
    const size_t base = numNodes / size;
    const size_t rem = numNodes % size;
    const size_t split = rem * (base + 1);
    if (row < split) return static_cast<int>(row / (base + 1));
    return static_cast<int>(rem + (row - split) / base);
}

// Initializes the locally owned rows [r0, r1) of the distance matrix.
// The random sequence is generated identically to the serial version.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t r0, const size_t r1) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t lo = r0 * numNodes;
    const size_t hi = r1 * numNodes;

    for (size_t i = 0; i < hi; ++i) {
        const unsigned int v = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= lo) dist[i - lo] = v;
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = r0; i < r1; ++i) {
        dist[(i - r0) * numNodes + i] = 0;
    }
}

// Initializes the locally owned rows [r0, r1) of the path matrix:
// every entry of row i ends up as i (equivalent to the serial initialization).
void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t r0, const size_t r1) {
    for (size_t i = r0; i < r1; ++i) {
        unsigned int* row = &path[(i - r0) * numNodes];
        for (size_t j = 0; j < numNodes; ++j) {
            row[j] = static_cast<unsigned int>(i);
        }
    }
}

static inline void relaxRow(unsigned int* __restrict__ d, unsigned int* __restrict__ p,
                            const unsigned int* __restrict__ rk, const size_t k,
                            const size_t numNodes) noexcept {
    const unsigned int dik = d[k];
    const unsigned int kk = static_cast<unsigned int>(k);
#pragma GCC ivdep
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = dik + rk[j];
        const unsigned int old = d[j];
        const bool better = newDist < old;
        d[j] = better ? newDist : old;
        p[j] = better ? kk : p[j];
    }
}

// Distributed Floyd-Warshall with 1D row-block decomposition.
// At step k, row k is broadcast from its owner; the owner of row k+1 updates
// that row first and starts its broadcast (non-blocking) to overlap
// communication with the remaining computation of step k.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes, const size_t r0, const size_t r1,
                   const int rank, const int size) {
    if (numNodes == 0) return;
    const size_t localRows = r1 - r0;
    std::vector<unsigned int> rowBuf[2] = {std::vector<unsigned int>(numNodes),
                                           std::vector<unsigned int>(numNodes)};
    MPI_Request req = MPI_REQUEST_NULL;

    // Row 0 broadcast
    {
        const int owner = rowOwner(0, numNodes, size);
        if (owner == rank) {
            std::memcpy(rowBuf[0].data(), &dist[(0 - r0) * numNodes], numNodes * sizeof(unsigned int));
        }
        MPI_Ibcast(rowBuf[0].data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD, &req);
    }

    constexpr size_t TEST_INTERVAL = 16;

    for (size_t k = 0; k < numNodes; ++k) {
        MPI_Wait(&req, MPI_STATUS_IGNORE);
        const unsigned int* rk = rowBuf[k & 1].data();

        size_t skipRow = SIZE_MAX;
        if (k + 1 < numNodes) {
            const size_t next = k + 1;
            const int owner = rowOwner(next, numNodes, size);
            unsigned int* nb = rowBuf[next & 1].data();
            if (owner == rank) {
                const size_t li = next - r0;
                relaxRow(&dist[li * numNodes], &path[li * numNodes], rk, k, numNodes);
                std::memcpy(nb, &dist[li * numNodes], numNodes * sizeof(unsigned int));
                skipRow = li;
            }
            MPI_Ibcast(nb, (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD, &req);
        }

        for (size_t li = 0; li < localRows; ++li) {
            if (li == skipRow) continue;
            relaxRow(&dist[li * numNodes], &path[li * numNodes], rk, k, numNodes);
            if (req != MPI_REQUEST_NULL && (li % TEST_INTERVAL) == TEST_INTERVAL - 1) {
                int flag;
                MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Wait(&req, MPI_STATUS_IGNORE);
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
    
    const size_t r0 = rowStart(numNodes, rank, size);
    const size_t r1 = rowStart(numNodes, rank + 1, size);
    const size_t localRows = r1 - r0;

    // Allocate local row blocks
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, r0, r1);
    initializePathMatrix(path, numNodes, r0, r1);
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshall(dist, path, numNodes, r0, r1, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather full distance matrix on rank 0 if needed
    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        std::vector<int> counts, displs;
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
            counts.resize(size);
            displs.resize(size);
            for (int r = 0; r < size; ++r) {
                const size_t a = rowStart(numNodes, r, size);
                const size_t b = rowStart(numNodes, r + 1, size);
                counts[r] = (int)((b - a) * numNodes);
                displs[r] = (int)(a * numNodes);
            }
        }
        MPI_Gatherv(dist.data(), (int)(localRows * numNodes), MPI_UNSIGNED,
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
