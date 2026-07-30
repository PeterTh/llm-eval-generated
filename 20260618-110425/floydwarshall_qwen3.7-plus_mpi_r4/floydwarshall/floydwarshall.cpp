#include <mpi.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array: dist[i][j] stored at j*n+i
// This means dist[i][j] = array[i*n + j] -- effectively row-major for [i][j] access
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// Compute local row range for a given rank
inline void getLocalRange(size_t numNodes, int size, int rank,
                          size_t& startRow, size_t& localRows) {
    size_t rowsPerProc = numNodes / size;
    startRow = static_cast<size_t>(rank) * rowsPerProc;
    localRows = (rank == size - 1) ? (numNodes - startRow) : rowsPerProc;
}

void initializeDistanceMatrix(std::vector<unsigned int>& localDist,
                              const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              int rank, int size) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    // All processes generate the same random sequence for reproducibility
    // but only keep their local rows
    size_t startRow, localRows;
    getLocalRange(numNodes, size, rank, startRow, localRows);

    localDist.resize(localRows * numNodes);

    // Generate full matrix on rank 0 and scatter
    if (rank == 0) {
        // Generate full matrix in the original layout
        std::vector<unsigned int> fullDist(numNodes * numNodes);
        for (size_t i = 0; i < numNodes * numNodes; ++i) {
            fullDist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }

        // Set diagonal to zero
        for (size_t i = 0; i < numNodes; ++i) {
            fullDist[idx2(i, i, numNodes)] = 0;
        }

        // Scatter row by row: each process gets contiguous rows
        // The original layout: idx2(i,j,n) = j*n+i, so element [i][j] is at position j*n+i
        // For row i, elements are at positions: i, n+i, 2n+i, ..., (n-1)*n+i
        // These are NOT contiguous in the original layout, so we need to pack them
        std::vector<unsigned int> packedFull(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                packedFull[i * numNodes + j] = fullDist[idx2(i, j, numNodes)];
            }
        }

        // Compute send counts and displacements
        std::vector<int> sendCounts(size);
        std::vector<int> sendDispls(size);
        for (int p = 0; p < size; ++p) {
            size_t pStart, pRows;
            getLocalRange(numNodes, size, p, pStart, pRows);
            sendCounts[p] = static_cast<int>(pRows * numNodes);
            sendDispls[p] = static_cast<int>(pStart * numNodes);
        }

        MPI_Scatterv(packedFull.data(), sendCounts.data(), sendDispls.data(), MPI_UNSIGNED,
                     localDist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED,
                     localDist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                     0, MPI_COMM_WORLD);
    }
}

void initializePathMatrix(std::vector<unsigned int>& localPath, const size_t numNodes,
                          size_t startRow, size_t localRows) {
    localPath.resize(localRows * numNodes);
    // Original: path[idx2(i,j,n)] = j for initialization (path[i][j] = j)
    // We store locally as localPath[li * numNodes + j] = j
    for (size_t li = 0; li < localRows; ++li) {
        for (size_t j = 0; j < numNodes; ++j) {
            localPath[li * numNodes + j] = j;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath,
                   const size_t numNodes,
                   int rank, int size,
                   size_t startRow, size_t localRows) {
    std::vector<unsigned int> rowK(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        // Determine owner of row k
        size_t ownerStart, ownerRows;
        int ownerRank;
        {
            size_t rowsPerProc = numNodes / size;
            ownerRank = static_cast<int>(k / rowsPerProc);
            if (ownerRank >= size) ownerRank = size - 1;
        }

        // Owner copies row k into rowK buffer
        if (rank == ownerRank) {
            size_t localK = k - startRow;
            for (size_t j = 0; j < numNodes; ++j) {
                rowK[j] = localDist[localK * numNodes + j];
            }
        }

        // Broadcast row k from owner
        MPI_Bcast(rowK.data(), static_cast<int>(numNodes), MPI_UNSIGNED, ownerRank, MPI_COMM_WORLD);

        // Each process updates its local rows
        for (size_t li = 0; li < localRows; ++li) {
            const unsigned int distIK = localDist[li * numNodes + k];
            const size_t rowBase = li * numNodes;

            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + rowK[j];
                if (newDist < localDist[rowBase + j]) {
                    localDist[rowBase + j] = newDist;
                    localPath[rowBase + j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }

    // 2. Triangle inequality check
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

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments on rank 0
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numNodes = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
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

    // Broadcast parameters
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    int validateInt = validate ? 1 : 0;
    int printInt = printResults ? 1 : 0;
    MPI_Bcast(&validateInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printInt, 1, MPI_INT, 0, MPI_COMM_WORLD);
    validate = validateInt != 0;
    printResults = printInt != 0;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Compute local range
    size_t startRow, localRows;
    getLocalRange(numNodes, size, rank, startRow, localRows);

    // Allocate local matrices
    std::vector<unsigned int> localDist;
    std::vector<unsigned int> localPath;

    // Initialize
    if (rank == 0) {
        printf("Initializing graph...\n");
    }
    initializeDistanceMatrix(localDist, numNodes, 1, MAX_DISTANCE, rank, size);
    initializePathMatrix(localPath, numNodes, startRow, localRows);

    // Run Floyd-Warshall
    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(localDist, localPath, numNodes, rank, size, startRow, localRows);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // Get max time across all processes
    double localDuration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    double maxDuration;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.0f ms\n", maxDuration);
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (maxDuration / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather results to rank 0 in packed row-major format
    std::vector<unsigned int> fullDistPacked;
    std::vector<int> recvCounts(size);
    std::vector<int> recvDispls(size);

    if (rank == 0) {
        fullDistPacked.resize(numNodes * numNodes);
        for (int p = 0; p < size; ++p) {
            size_t pStart, pRows;
            getLocalRange(numNodes, size, p, pStart, pRows);
            recvCounts[p] = static_cast<int>(pRows * numNodes);
            recvDispls[p] = static_cast<int>(pStart * numNodes);
        }
    }

    MPI_Gatherv(localDist.data(), static_cast<int>(localRows * numNodes), MPI_UNSIGNED,
                fullDistPacked.data(), recvCounts.data(), recvDispls.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    // On rank 0, convert packed row-major back to original column-major layout for output
    if (rank == 0) {
        std::vector<unsigned int> dist(numNodes * numNodes);
        for (size_t i = 0; i < numNodes; ++i) {
            for (size_t j = 0; j < numNodes; ++j) {
                dist[idx2(i, j, numNodes)] = fullDistPacked[i * numNodes + j];
            }
        }

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
            }
        }
    }

    MPI_Finalize();
    return 0;
}
