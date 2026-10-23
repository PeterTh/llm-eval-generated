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

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

static inline int rowOwner(const size_t k, const size_t base,
                           const size_t remainder) {
    const size_t cut = remainder * (base + 1);
    if (k < cut) {
        return static_cast<int>(k / (base + 1));
    }
    return static_cast<int>(remainder + (k - cut) / base);
}

void initializeDistanceMatrix(std::vector<unsigned int>& localDist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const size_t rowStart, const size_t rowCount) {
    if (rowCount == 0) {
        return;
    }

    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t col = 0; col < numNodes; ++col) {
        for (size_t row = 0; row < numNodes; ++row) {
            const unsigned int value =
                rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
            if (row >= rowStart && row < rowStart + rowCount) {
                localDist[(row - rowStart) * numNodes + col] = value;
            }
        }
    }

    for (size_t r = 0; r < rowCount; ++r) {
        const size_t row = rowStart + r;
        localDist[r * numNodes + row] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& localPath, const size_t numNodes,
                          const size_t rowCount) {
    for (size_t r = 0; r < rowCount; ++r) {
        for (size_t j = 0; j < numNodes; ++j) {
            localPath[r * numNodes + j] = static_cast<unsigned int>(j);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const size_t numNodes,
                   const size_t rowStart,
                   const size_t rowCount,
                   const int* rowCounts,
                   const int* rowDispls,
                   const int rank,
                   const int size) {
    std::vector<unsigned int> rowK(numNodes);
    std::vector<unsigned int> colK(numNodes);
    std::vector<unsigned int> colSend(rowCount);

    const size_t base = numNodes / size;
    const size_t remainder = numNodes % size;
    unsigned int* const distBase = localDist.data();
    unsigned int* const pathBase = localPath.data();

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner(k, base, remainder);

        if (rank == owner) {
            std::memcpy(rowK.data(),
                        distBase + (k - rowStart) * numNodes,
                        numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner,
                  MPI_COMM_WORLD);

        for (size_t r = 0; r < rowCount; ++r) {
            colSend[r] = distBase[r * numNodes + k];
        }
        MPI_Allgatherv(colSend.data(), static_cast<int>(rowCount), MPI_UNSIGNED,
                       colK.data(), rowCounts, rowDispls, MPI_UNSIGNED,
                       MPI_COMM_WORLD);

        const unsigned int* const colPtr = colK.data() + rowStart;
        for (size_t r = 0; r < rowCount; ++r) {
            const unsigned int dik = colPtr[r];
            unsigned int* __restrict__ drow = distBase + r * numNodes;
            unsigned int* __restrict__ prow = pathBase + r * numNodes;
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = dik + rowK[j];
                if (newDist < drow[j]) {
                    drow[j] = newDist;
                    prow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    const size_t limit = std::min(numNodes, static_cast<size_t>(10));
    for (size_t i = 0; i < limit; ++i) {
        for (size_t j = 0; j < limit; ++j) {
            const unsigned int distIJ = dist[i * numNodes + j];
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];

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

    const size_t base = numNodes / size;
    const size_t remainder = numNodes % size;

    std::vector<int> rowCounts(size);
    std::vector<int> rowDispls(size);
    size_t offset = 0;
    for (int r = 0; r < size; ++r) {
        const size_t count = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        rowCounts[r] = static_cast<int>(count);
        rowDispls[r] = static_cast<int>(offset);
        offset += count;
    }

    const size_t rowStart = static_cast<size_t>(rowDispls[rank]);
    const size_t rowCount = static_cast<size_t>(rowCounts[rank]);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    std::vector<unsigned int> localDist(rowCount * numNodes);
    std::vector<unsigned int> localPath(rowCount * numNodes);

    initializeDistanceMatrix(localDist, numNodes, 1, MAX_DISTANCE, rowStart, rowCount);
    initializePathMatrix(localPath, numNodes, rowCount);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(localDist, localPath, numNodes, rowStart, rowCount,
                  rowCounts.data(), rowDispls.data(), rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    std::vector<unsigned int> fullDist;
    if (printResults || validate) {
        if (rank == 0) {
            fullDist.resize(numNodes * numNodes);
        }

        std::vector<int> gatherCounts(size);
        std::vector<int> gatherDispls(size);
        for (int r = 0; r < size; ++r) {
            gatherCounts[r] = rowCounts[r] * static_cast<int>(numNodes);
            gatherDispls[r] = rowDispls[r] * static_cast<int>(numNodes);
        }

        MPI_Gatherv(localDist.data(), static_cast<int>(rowCount * numNodes), MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr,
                    gatherCounts.data(), gatherDispls.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        std::vector<unsigned int> output(numNodes * numNodes);
        for (size_t row = 0; row < numNodes; ++row) {
            for (size_t col = 0; col < numNodes; ++col) {
                output[idx2(row, col, numNodes)] = fullDist[row * numNodes + col];
            }
        }
        print_results_int(output, "DistanceMatrix");
    }

    if (validate) {
        bool valid = true;
        if (rank == 0) {
            printf("Validating result...\n");
            valid = validateResult(fullDist, numNodes);
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }

        MPI_Finalize();
        return (rank == 0 && !valid) ? 1 : 0;
    }

    MPI_Finalize();
    return 0;
}
