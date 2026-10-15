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

// Rows [rowBegin, rowEnd) of the global matrices are owned by this rank.
// Local storage is row-major: local row r corresponds to global row rowBegin + r.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t rowBegin, const size_t rowEnd) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // Every rank replays the same random sequence, keeping only its own rows.
    const size_t first = rowBegin * numNodes;
    const size_t last = rowEnd * numNodes;
    for (size_t i = 0; i < first; ++i) rand_r(&seed);
    for (size_t i = first; i < last; ++i) {
        dist[i - first] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        dist[(i - rowBegin) * numNodes + i] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowBegin, const size_t rowEnd) {
    for (size_t i = rowBegin; i < rowEnd; ++i) {
        unsigned int* row = &path[(i - rowBegin) * numNodes];
        for (size_t j = 0; j < numNodes; ++j) row[j] = (unsigned int)i;
    }
}

static inline void relaxRow(unsigned int* __restrict distRow, unsigned int* __restrict pathRow,
                            const unsigned int* __restrict rowK, const unsigned int k,
                            const size_t numNodes) {
    const unsigned int distIK = distRow[k];
    for (size_t j = 0; j < numNodes; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const bool better = newDist < distRow[j];
        distRow[j] = better ? newDist : distRow[j];
        pathRow[j] = better ? k : pathRow[j];
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const std::vector<int>& rowStarts, const int rank) {
    const size_t rowBegin = rowStarts[rank];
    const size_t localRows = rowStarts[rank + 1] - rowStarts[rank];
    const int nodes = (int)numNodes;

    auto ownerOf = [&](size_t row) {
        return (int)(std::upper_bound(rowStarts.begin(), rowStarts.end(), (int)row) -
                     rowStarts.begin()) - 1;
    };

    std::vector<unsigned int> bufStore[2] = {std::vector<unsigned int>(numNodes),
                                             std::vector<unsigned int>(numNodes)};

    // Row 0 is broadcast up front; afterwards row k+1 is broadcast (non-blocking)
    // while the remaining rows are being relaxed with row k.
    {
        const int owner = ownerOf(0);
        if (owner == rank) memcpy(bufStore[0].data(), &dist[0], numNodes * sizeof(unsigned int));
        MPI_Bcast(bufStore[0].data(), nodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
    }

    for (size_t k = 0; k < numNodes; ++k) {
        const unsigned int* rowK = bufStore[k & 1].data();
        MPI_Request req = MPI_REQUEST_NULL;
        size_t skipRow = (size_t)-1;  // local row already handled

        if (k + 1 < numNodes) {
            unsigned int* next = bufStore[(k + 1) & 1].data();
            const int owner = ownerOf(k + 1);
            if (owner == rank) {
                const size_t lr = k + 1 - rowBegin;
                relaxRow(&dist[lr * numNodes], &path[lr * numNodes], rowK, (unsigned int)k, numNodes);
                memcpy(next, &dist[lr * numNodes], numNodes * sizeof(unsigned int));
                skipRow = lr;
            }
            MPI_Ibcast(next, nodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD, &req);
        }

        int flag = 0;
        for (size_t r = 0; r < localRows; ++r) {
            if (r == skipRow) continue;
            relaxRow(&dist[r * numNodes], &path[r * numNodes], rowK, (unsigned int)k, numNodes);
            if (req != MPI_REQUEST_NULL && !flag && (r & 3) == 3) {
                MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
            }
        }
        if (req != MPI_REQUEST_NULL && !flag) MPI_Wait(&req, MPI_STATUS_IGNORE);
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

    // Block row distribution
    std::vector<int> rowStarts(size + 1);
    for (int r = 0; r <= size; ++r) {
        rowStarts[r] = (int)((numNodes / size) * r + std::min<size_t>(r, numNodes % size));
    }
    const size_t rowBegin = rowStarts[rank];
    const size_t rowEnd = rowStarts[rank + 1];
    const size_t localRows = rowEnd - rowBegin;

    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, rowBegin, rowEnd);
    initializePathMatrix(path, numNodes, rowBegin, rowEnd);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rowStarts, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    int status = 0;
    if (printResults || validate) {
        std::vector<unsigned int> full;
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (rowStarts[r + 1] - rowStarts[r]) * (int)numNodes;
            displs[r] = rowStarts[r] * (int)numNodes;
        }
        if (rank == 0) full.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, full.data(), counts.data(),
                    displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        if (rank == 0) {
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
                    status = 1;
                }
            }
        }
    }

    MPI_Finalize();
    return status;
}
