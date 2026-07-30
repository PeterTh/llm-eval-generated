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

// Index calculation for flattened column-major 2D array:
//   idx2(row, col, n) = col * n + row
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Local index: position of (local_col_idx, row) in the local buffer
inline constexpr size_t local_idx(const size_t l, const size_t i, const size_t n) noexcept {
    return l * n + i;
}

// ---------------------------------------------------------------------------
// Full matrix initialization (rank 0 only)
// ---------------------------------------------------------------------------
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
    
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// ---------------------------------------------------------------------------
// Local path initialization: each process initialises its own columns.
// Equivalent semantics to the serial version: path[row][col] = col.
// ---------------------------------------------------------------------------
void initializePathMatrixLocal(std::vector<unsigned int>& local_path, const size_t numNodes,
                               const size_t localCols, const size_t startCol) {
    for (size_t l = 0; l < localCols; ++l) {
        const unsigned int j = static_cast<unsigned int>(startCol + l);
        unsigned int* col = &local_path[local_idx(l, 0, numNodes)];
        for (size_t i = 0; i < numNodes; ++i) {
            col[i] = j;
        }
    }
}

// ---------------------------------------------------------------------------
// MPI-parallel Floyd-Warshall
//
// The distance/path matrices are block-column distributed. Each rank
// owns a contiguous set of columns.  At iteration k the rank that owns
// column k broadcasts it to every other rank; each rank then updates its
// local columns using the received column k.
// ---------------------------------------------------------------------------
void floydWarshallMPI(std::vector<unsigned int>& local_dist,
                      std::vector<unsigned int>& local_path,
                      const size_t numNodes,
                      const size_t localCols,
                      const std::vector<int>& colOwner,
                      const std::vector<size_t>& colLocalIdx,
                      const int rank) {
    std::vector<unsigned int> col_k(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = colOwner[k];

        // Owner copies column k into the broadcast buffer; then everyone
        // participates in the broadcast.
        if (rank == owner) {
            const size_t lk = colLocalIdx[k];
            std::memcpy(col_k.data(), &local_dist[local_idx(lk, 0, numNodes)],
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(col_k.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  owner, MPI_COMM_WORLD);

        // Update every locally owned column using the received column k.
        for (size_t l = 0; l < localCols; ++l) {
            const unsigned int distKJ = local_dist[local_idx(l, k, numNodes)];
            unsigned int*       distCol = &local_dist[local_idx(l, 0, numNodes)];
            unsigned int*       pathCol = &local_path[local_idx(l, 0, numNodes)];

            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDist = col_k[i] + distKJ;
                if (newDist < distCol[i]) {
                    distCol[i] = newDist;
                    pathCol[i] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Validation (operates on the full, gathered matrix on rank 0)
// ---------------------------------------------------------------------------
bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    const size_t sample = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t kk = 0; kk < numNodes; ++kk) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(kk, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, kk, numNodes)];

                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, kk);
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

// ---------------------------------------------------------------------------
// Helpers to compute the block-column distribution
// ---------------------------------------------------------------------------
static void computeDistribution(const size_t numNodes, const int numRanks,
                                std::vector<size_t>& colsPerRank,
                                std::vector<size_t>& startCols) {
    const size_t base = numNodes / static_cast<size_t>(numRanks);
    const size_t rem  = numNodes % static_cast<size_t>(numRanks);
    colsPerRank.resize(static_cast<size_t>(numRanks));
    startCols.resize(static_cast<size_t>(numRanks));
    size_t col = 0;
    for (int r = 0; r < numRanks; ++r) {
        colsPerRank[static_cast<size_t>(r)] = base + (static_cast<size_t>(r) < rem ? 1 : 0);
        startCols[static_cast<size_t>(r)] = col;
        col += colsPerRank[static_cast<size_t>(r)];
    }
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int numRanks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // ---- parse arguments on rank 0 then broadcast ----
    size_t numNodes = 512;
    int validate = 0;
    int printResults = 0;
    int exitFlag = 0;   // non-zero → all ranks exit

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = static_cast<size_t>(atoi(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                exitFlag = 1;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitFlag = 2;
            }
        }
    }

    // Broadcast parsed values so every rank knows whether to exit
    MPI_Bcast(&exitFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitFlag) {
        MPI_Finalize();
        return (exitFlag == 2) ? 1 : 0;
    }

    unsigned long long nn = static_cast<unsigned long long>(numNodes);
    MPI_Bcast(&nn, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    numNodes = static_cast<size_t>(nn);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    const bool doValidate   = (validate != 0);
    const bool doPrintRes   = (printResults != 0);

    // ---- determine this rank's column slice ----
    std::vector<size_t> colsPerRank, startCols;
    computeDistribution(numNodes, numRanks, colsPerRank, startCols);

    const size_t localCols = colsPerRank[static_cast<size_t>(rank)];
    const size_t startCol  = startCols[static_cast<size_t>(rank)];

    // ---- precompute column → owning rank & local index  ----
    std::vector<int>  colOwner(static_cast<size_t>(numNodes));
    std::vector<size_t> colLocalIdx(static_cast<size_t>(numNodes));
    for (int r = 0; r < numRanks; ++r) {
        const size_t sc = startCols[static_cast<size_t>(r)];
        const size_t nc = colsPerRank[static_cast<size_t>(r)];
        for (size_t l = 0; l < nc; ++l) {
            colOwner[sc + l]  = r;
            colLocalIdx[sc + l] = l;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", numRanks);
        printf("Validation: %s\n", doValidate ? "enabled" : "disabled");
    }

    // ---- allocate local matrices ----
    std::vector<unsigned int> local_dist(numNodes * localCols);
    std::vector<unsigned int> local_path(numNodes * localCols);

    // ---- initialise distance matrix on rank 0 and scatter ----
    {
        std::vector<int> sendcounts(static_cast<size_t>(numRanks));
        std::vector<int> displs(static_cast<size_t>(numRanks));
        for (int r = 0; r < numRanks; ++r) {
            const size_t cols = colsPerRank[static_cast<size_t>(r)];
            const size_t disp = startCols[static_cast<size_t>(r)];
            sendcounts[static_cast<size_t>(r)] = static_cast<int>(numNodes * cols);
            displs[static_cast<size_t>(r)]    = static_cast<int>(numNodes * disp);
        }

        if (rank == 0) {
            std::vector<unsigned int> full_dist(numNodes * numNodes);
            initializeDistanceMatrix(full_dist, numNodes, 1, MAX_DISTANCE);

            MPI_Scatterv(full_dist.data(), sendcounts.data(), displs.data(), MPI_UNSIGNED,
                         local_dist.data(), static_cast<int>(numNodes * localCols), MPI_UNSIGNED,
                         0, MPI_COMM_WORLD);
        } else {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED,
                         local_dist.data(), static_cast<int>(numNodes * localCols), MPI_UNSIGNED,
                         0, MPI_COMM_WORLD);
        }
    }

    // ---- initialise path matrix (locally) ----
    initializePathMatrixLocal(local_path, numNodes, localCols, startCol);

    if (rank == 0) {
        printf("Initializing graph...\n");
        printf("Computing shortest paths...\n");
    }

    // ---- timed parallel Floyd-Warshall ----
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    floydWarshallMPI(local_dist, local_path, numNodes, localCols,
                     colOwner, colLocalIdx, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) {
        const auto duration =
            std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());

        const double ops = static_cast<double>(numNodes) *
                           static_cast<double>(numNodes) *
                           static_cast<double>(numNodes);
        const double gflops = ops / (static_cast<double>(duration.count()) / 1000.0) / 1.0e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // ---- gather results for validation / output ----
    if (doValidate || doPrintRes) {
        std::vector<unsigned int> full_dist;
        std::vector<unsigned int> full_path;

        std::vector<int> recvcounts(static_cast<size_t>(numRanks));
        std::vector<int> displs(static_cast<size_t>(numRanks));
        for (int r = 0; r < numRanks; ++r) {
            const size_t cols = colsPerRank[static_cast<size_t>(r)];
            const size_t disp = startCols[static_cast<size_t>(r)];
            recvcounts[static_cast<size_t>(r)] = static_cast<int>(numNodes * cols);
            displs[static_cast<size_t>(r)]    = static_cast<int>(numNodes * disp);
        }

        if (rank == 0) {
            full_dist.resize(numNodes * numNodes);
            full_path.resize(numNodes * numNodes);
        }

        MPI_Gatherv(local_dist.data(), static_cast<int>(numNodes * localCols), MPI_UNSIGNED,
                    rank == 0 ? full_dist.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        MPI_Gatherv(local_path.data(), static_cast<int>(numNodes * localCols), MPI_UNSIGNED,
                    rank == 0 ? full_path.data() : nullptr,
                    recvcounts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        if (rank == 0) {
            if (doPrintRes) {
                print_results_int(full_dist, "DistanceMatrix");
            }
            if (doValidate) {
                printf("Validating result...\n");
                if (validateResult(full_dist, numNodes)) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                    MPI_Finalize();
                    return 1;
                }
            }
        }
    }

    MPI_Finalize();
    return 0;
}
