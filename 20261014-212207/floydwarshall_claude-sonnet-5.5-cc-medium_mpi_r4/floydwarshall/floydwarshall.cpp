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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Distributed Floyd-Warshall: block-row distribution. Each rank owns rows
// [rowStart, rowStart+localRows). Row k is broadcast from its owner at each step;
// the broadcast of row k+1 is overlapped with the update of the remaining rows.
void floydWarshallMPI(std::vector<unsigned int>& dist,
                      std::vector<unsigned int>& path,
                      const size_t numNodes,
                      const std::vector<int>& rowCounts,
                      const std::vector<int>& rowDispls,
                      const int rank) {
    const size_t n = numNodes;
    const size_t localRows = rowCounts[rank];
    const size_t rowStart = rowDispls[rank];
    const int nprocs = (int)rowCounts.size();

    auto ownerOf = [&](size_t k) {
        int lo = 0, hi = nprocs - 1;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if ((size_t)rowDispls[mid] <= k) lo = mid; else hi = mid - 1;
        }
        return lo;
    };

    auto updateRow = [&](size_t li, size_t k, const unsigned int* __restrict rowK) {
        unsigned int* __restrict d = &dist[li * n];
        unsigned int* __restrict p = &path[li * n];
        const unsigned int distIK = d[k];
        const unsigned int kk = (unsigned int)k;
        for (size_t j = 0; j < n; ++j) {
            const unsigned int newDist = distIK + rowK[j];
            const bool less = newDist < d[j];
            d[j] = less ? newDist : d[j];
            p[j] = less ? kk : p[j];
        }
    };

    std::vector<unsigned int> buf[2] = {std::vector<unsigned int>(n), std::vector<unsigned int>(n)};
    if (n == 0) return;

    int owner = ownerOf(0);
    if (rank == owner) memcpy(buf[0].data(), &dist[(0 - rowStart) * n], n * sizeof(unsigned int));
    MPI_Bcast(buf[0].data(), (int)n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

    for (size_t k = 0; k < n; ++k) {
        const unsigned int* rowK = buf[k & 1].data();
        MPI_Request req = MPI_REQUEST_NULL;
        size_t skip = (size_t)-1;  // local row already updated
        if (k + 1 < n) {
            const int nextOwner = ownerOf(k + 1);
            unsigned int* nb = buf[(k + 1) & 1].data();
            if (rank == nextOwner) {
                skip = k + 1 - rowStart;
                updateRow(skip, k, rowK);
                memcpy(nb, &dist[skip * n], n * sizeof(unsigned int));
            }
            MPI_Ibcast(nb, (int)n, MPI_UNSIGNED, nextOwner, MPI_COMM_WORLD, &req);
        }
        for (size_t li = 0; li < localRows; ++li) {
            if (li != skip) updateRow(li, k, rowK);
        }
        if (req != MPI_REQUEST_NULL) MPI_Wait(&req, MPI_STATUS_IGNORE);
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
    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
        printf("MPI ranks: %d\n", nprocs);
    }

    // Block-row distribution
    std::vector<int> rowCounts(nprocs), rowDispls(nprocs);
    for (int r = 0, off = 0; r < nprocs; ++r) {
        rowCounts[r] = (int)(numNodes / nprocs + ((size_t)r < numNodes % nprocs ? 1 : 0));
        rowDispls[r] = off;
        off += rowCounts[r];
    }
    const size_t localRows = rowCounts[rank];

    MPI_Datatype rowType;
    MPI_Type_contiguous((int)numNodes, MPI_UNSIGNED, &rowType);
    MPI_Type_commit(&rowType);

    std::vector<unsigned int> full;
    std::vector<unsigned int> dist(localRows * numNodes);
    std::vector<unsigned int> path(localRows * numNodes);

    // Initialize (RNG stream is sequential, so rank 0 generates and scatters)
    if (rank == 0) {
        printf("Initializing graph...\n");
        full.resize(numNodes * numNodes);
        initializeDistanceMatrix(full, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(full.data(), rowCounts.data(), rowDispls.data(), rowType,
                 dist.data(), (int)localRows, rowType, 0, MPI_COMM_WORLD);
    // path[i][j] = i (global row index)
    for (size_t li = 0; li < localRows; ++li) {
        const unsigned int gi = (unsigned int)(rowDispls[rank] + li);
        for (size_t j = 0; j < numNodes; ++j) path[li * numNodes + j] = gi;
    }

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallMPI(dist, path, numNodes, rowCounts, rowDispls, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    MPI_Gatherv(dist.data(), (int)localRows, rowType,
                full.data(), rowCounts.data(), rowDispls.data(), rowType, 0, MPI_COMM_WORLD);
    MPI_Type_free(&rowType);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", (long)duration.count());

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(full, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(full, numNodes)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Finalize();
    return status;
}
