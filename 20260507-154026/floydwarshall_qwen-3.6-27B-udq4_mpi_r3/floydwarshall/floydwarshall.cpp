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

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

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
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    size_t numNodes = 512;
    int validate = 0;
    int printResults = 0;

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
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Number of MPI processes: %d\n", numProcs);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // ---- Block decomposition: distribute rows across processes ----
    size_t baseRows = numNodes / numProcs;
    size_t extraRows = numNodes % numProcs;

    size_t myFirstRow = 0;
    for (int p = 0; p < rank; ++p) {
        myFirstRow += baseRows + (static_cast<size_t>(p) < extraRows ? 1 : 0);
    }
    size_t myNumRows = baseRows + (static_cast<size_t>(rank) < extraRows ? 1 : 0);

    // Precompute which process owns each row (for broadcast root selection)
    std::vector<int> procOfRow(numNodes);
    {
        size_t cum = 0;
        for (int p = 0; p < numProcs; ++p) {
            size_t rows = baseRows + (static_cast<size_t>(p) < extraRows ? 1 : 0);
            for (size_t r = 0; r < rows; ++r) {
                procOfRow[cum + r] = p;
            }
            cum += rows;
        }
    }

    // Gather counts / displacements (row-major blocks)
    std::vector<int> counts(numProcs);
    std::vector<int> displs(numProcs);
    {
        size_t cum = 0;
        for (int p = 0; p < numProcs; ++p) {
            size_t rows = baseRows + (static_cast<size_t>(p) < extraRows ? 1 : 0);
            counts[p] = static_cast<int>(rows * numNodes);
            displs[p] = static_cast<int>(cum);
            cum += rows * numNodes;
        }
    }

    // Local storage (row-major: [myNumRows][numNodes])
    std::vector<unsigned int> localDist(myNumRows * numNodes);
    std::vector<unsigned int> localPath(myNumRows * numNodes);

    if (rank == 0) printf("Initializing graph...\n");

    // Initialize distance matrix: each process advances the PRNG to its block.
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE - 1) + 1.0; // 200.0
    for (size_t s = 0; s < myFirstRow * numNodes; ++s) rand_r(&seed);
    for (size_t i = 0; i < myNumRows; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            localDist[i * numNodes + j] = 1u + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
        }
        localDist[i * numNodes + (myFirstRow + i)] = 0; // diagonal = 0
    }

    // Initialize path matrix: path[i][j] = i for all i, j.
    for (size_t i = 0; i < myNumRows; ++i) {
        unsigned int gr = static_cast<unsigned int>(myFirstRow + i);
        for (size_t j = 0; j < numNodes; ++j) localPath[i * numNodes + j] = gr;
    }

    // Buffer for broadcasting row k each iteration
    std::vector<unsigned int> rowKDist(numNodes);

    if (rank == 0) printf("Computing shortest paths...\n");

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // ---- Parallel Floyd-Warshall (1D row decomposition) ----
    for (size_t k = 0; k < numNodes; ++k) {
        int rootProc = procOfRow[k];

        // Process owning row k copies it into the broadcast buffer
        if (rank == rootProc) {
            size_t lr = k - myFirstRow;
            for (size_t j = 0; j < numNodes; ++j)
                rowKDist[j] = localDist[lr * numNodes + j];
        }

        // Broadcast row k to every process
        MPI_Bcast(rowKDist.data(), static_cast<int>(numNodes), MPI_UNSIGNED,
                  rootProc, MPI_COMM_WORLD);

        // Each process updates its local rows using the received row k
        for (size_t i = 0; i < myNumRows; ++i) {
            const unsigned int distIK = localDist[i * numNodes + k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rowKDist[j];
                if (newDist < localDist[i * numNodes + j]) {
                    localDist[i * numNodes + j] = newDist;
                    localPath[i * numNodes + j] = static_cast<unsigned int>(k);
                }
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    MPI_Barrier(MPI_COMM_WORLD);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localDuration = static_cast<long>(duration.count());
    long maxDuration = 0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    // ---- Gather distance matrix back to rank 0 ----
    if (rank == 0) {
        std::vector<unsigned int> globalDist(numNodes * numNodes);
        MPI_Gatherv(localDist.data(), static_cast<int>(myNumRows * numNodes), MPI_UNSIGNED,
                    globalDist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        printf("Computation time: %ld ms\n", maxDuration);

        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        double gflops = ops / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(globalDist, "DistanceMatrix");
        }

        if (validate) {
            printf("Validating result...\n");
            if (validateResult(globalDist, numNodes)) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                MPI_Finalize();
                return 1;
            }
        }
    } else {
        MPI_Gatherv(localDist.data(), static_cast<int>(myNumRows * numNodes), MPI_UNSIGNED,
                    nullptr, nullptr, nullptr, MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
