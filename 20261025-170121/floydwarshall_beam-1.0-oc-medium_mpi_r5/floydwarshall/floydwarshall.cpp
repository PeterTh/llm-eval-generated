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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }

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

void rowRange(int rank, int size, size_t numNodes, size_t& start, size_t& end) {
    size_t rowsPerProc = numNodes / size;
    size_t remainder = numNodes % size;
    start = (size_t)rank * rowsPerProc + std::min((size_t)rank, remainder);
    end = start + rowsPerProc + (rank < (int)remainder ? 1 : 0);
}

int ownerOf(size_t row, int size, size_t numNodes) {
    size_t start, end;
    for (int p = 0; p < size; ++p) {
        rowRange(p, size, numNodes, start, end);
        if (row >= start && row < end) return p;
    }
    return size - 1;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
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

    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        dist.assign(numNodes * numNodes, 0);
        path.assign(numNodes * numNodes, 0);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
    }

    size_t startRow, endRow;
    rowRange(rank, size, numNodes, startRow, endRow);
    size_t localRows = endRow - startRow;

    std::vector<int> counts(size), displs(size);
    size_t acc = 0;
    for (int p = 0; p < size; ++p) {
        size_t s, e;
        rowRange(p, size, numNodes, s, e);
        counts[p] = (int)((e - s) * numNodes);
        displs[p] = (int)(acc * numNodes);
        acc += (e - s);
    }

    std::vector<unsigned int> localDist(localRows * numNodes);
    std::vector<unsigned int> localPath(localRows * numNodes);

    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), (int)(localRows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? path.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), (int)(localRows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned int> rowK(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = ownerOf(k, size, numNodes);
        if (rank == owner) {
            size_t localIdx = k - startRow;
            std::copy(localDist.begin() + localIdx * numNodes,
                      localDist.begin() + (localIdx + 1) * numNodes,
                      rowK.begin());
        }
        MPI_Bcast(rowK.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t li = 0; li < localRows; ++li) {
            unsigned int distIK = localDist[li * numNodes + k];
            unsigned int* rowI = &localDist[li * numNodes];
            unsigned int* pathI = &localPath[li * numNodes];
            for (size_t j = 0; j < numNodes; ++j) {
                unsigned int distIJ = rowI[j];
                unsigned int distKJ = rowK[j];
                unsigned int newDist;
                if (distIK >= INF || distKJ >= INF) newDist = INF;
                else newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    rowI[j] = newDist;
                    pathI[j] = (unsigned int)k;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    if (rank == 0) dist.assign(numNodes * numNodes, 0);
    MPI_Gatherv(localDist.data(), (int)(localRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

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
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }

    MPI_Finalize();
    return 0;
}
