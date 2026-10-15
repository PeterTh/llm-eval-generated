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

// Rows [rowBegin, rowEnd) of the matrix are stored locally (row-major, row length n).
// The full random sequence is generated on every rank (O(n^2), negligible vs. O(n^3))
// so the matrix is identical to the serial version.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t rowBegin, const size_t rowEnd) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    const size_t total = numNodes * numNodes;
    const size_t lo = rowBegin * numNodes, hi = rowEnd * numNodes;
    for (size_t i = 0; i < total; ++i) {
        const unsigned int v = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        if (i >= lo && i < hi) dist[i - lo] = v;
    }

    for (size_t i = rowBegin; i < rowEnd; ++i) {
        dist[(i - rowBegin) * numNodes + i] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const size_t rowBegin, const size_t rowEnd) {
    for (size_t r = rowBegin; r < rowEnd; ++r) {
        unsigned int* row = &path[(r - rowBegin) * numNodes];
        for (size_t c = 0; c < numNodes; ++c) row[c] = (unsigned int)r;
    }
}

static inline void relaxRow(unsigned int* __restrict di, unsigned int* __restrict pi,
                            const unsigned int* __restrict rowK, const unsigned int distIK,
                            const unsigned int k, const size_t n) {
    for (size_t j = 0; j < n; ++j) {
        const unsigned int newDist = distIK + rowK[j];
        const unsigned int d = di[j];
        const bool less = newDist < d;
        di[j] = less ? newDist : d;
        pi[j] = less ? k : pi[j];
    }
}

// Row-block distributed Floyd-Warshall. Row k is broadcast from its owner each
// iteration; the broadcast of row k+1 is overlapped with the update of the other rows.
void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes,
                   const std::vector<int>& rowStart, int rank, int size) {
    const size_t n = numNodes;
    const size_t rowBegin = rowStart[rank], rowEnd = rowStart[rank + 1];
    const size_t myRows = rowEnd - rowBegin;
    if (n == 0) return;

    auto owner = [&](size_t row) {
        int o = (int)(std::upper_bound(rowStart.begin(), rowStart.end(), (int)row) - rowStart.begin()) - 1;
        return std::min(o, size - 1);
    };

    std::vector<unsigned int> buf[2] = {std::vector<unsigned int>(n), std::vector<unsigned int>(n)};
    MPI_Request req = MPI_REQUEST_NULL;

    // Row 0
    {
        const int o = owner(0);
        if (o == rank) std::memcpy(buf[0].data(), &dist[0 - rowBegin], n * sizeof(unsigned int));
        MPI_Bcast(buf[0].data(), (int)n, MPI_UNSIGNED, o, MPI_COMM_WORLD);
    }

    for (size_t k = 0; k < n; ++k) {
        const unsigned int* rowK = buf[k & 1].data();
        unsigned int* next = buf[(k + 1) & 1].data();
        const bool hasNext = (k + 1 < n);
        const int nextOwner = hasNext ? owner(k + 1) : -1;
        const size_t nk = k + 1;
        const bool ownNext = hasNext && nextOwner == rank;

        // Update the row needed next first, so its broadcast can start early.
        if (ownNext) {
            const size_t li = nk - rowBegin;
            relaxRow(&dist[li * n], &path[li * n], rowK, dist[li * n + k], (unsigned int)k, n);
            std::memcpy(next, &dist[li * n], n * sizeof(unsigned int));
        }
        if (hasNext) {
            MPI_Ibcast(next, (int)n, MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD, &req);
        }

        for (size_t li = 0; li < myRows; ++li) {
            if (ownNext && li == nk - rowBegin) continue;
            relaxRow(&dist[li * n], &path[li * n], rowK, dist[li * n + k], (unsigned int)k, n);
        }

        if (hasNext) MPI_Wait(&req, MPI_STATUS_IGNORE);
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
    }

    // Block row distribution
    std::vector<int> rowStart(size + 1);
    for (int r = 0; r <= size; ++r) {
        rowStart[r] = (int)(numNodes / size * r + std::min<size_t>(r, numNodes % size));
    }
    const size_t rowBegin = rowStart[rank], rowEnd = rowStart[rank + 1];

    std::vector<unsigned int> dist((rowEnd - rowBegin) * numNodes);
    std::vector<unsigned int> path((rowEnd - rowBegin) * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, rowBegin, rowEnd);
    initializePathMatrix(path, numNodes, rowBegin, rowEnd);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes, rowStart, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Gather the full distance matrix on rank 0 (only needed for output/validation)
    std::vector<unsigned int> full;
    if (printResults || validate) {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            counts[r] = (rowStart[r + 1] - rowStart[r]) * (int)numNodes;
            displs[r] = rowStart[r] * (int)numNodes;
        }
        if (rank == 0) full.resize(numNodes * numNodes);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, full.data(), counts.data(),
                    displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    int ret = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
                ret = 1;
            }
        }
    }

    MPI_Finalize();
    return ret;
}
