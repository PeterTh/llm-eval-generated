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

void floydWarshallMPI(std::vector<unsigned int>& distLocal,
                      std::vector<unsigned int>& pathLocal,
                      const size_t numNodes,
                      int rank, int size) {
    size_t base = (numNodes * (size_t)rank) / size;
    size_t end = (numNodes * (size_t)(rank + 1)) / size;
    size_t localRows = end - base;

    std::vector<unsigned int> rowK(numNodes);
    std::vector<unsigned int> pathRowK(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        int owner = -1;
        for (int r = 0; r < size; ++r) {
            size_t b = (numNodes * (size_t)r) / size;
            size_t e = (numNodes * (size_t)(r + 1)) / size;
            if (k >= b && k < e) { owner = r; break; }
        }

        if (rank == owner) {
            size_t li = k - base;
            for (size_t j = 0; j < numNodes; ++j) {
                rowK[j] = distLocal[li * numNodes + j];
                pathRowK[j] = pathLocal[li * numNodes + j];
            }
        }
        MPI_Bcast(rowK.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        MPI_Bcast(pathRowK.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t li = 0; li < localRows; ++li) {
            unsigned int distIK = distLocal[li * numNodes + k];
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = distLocal[li * numNodes + j];
                const unsigned int distKJ = rowK[j];
                const unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    distLocal[li * numNodes + j] = newDist;
                    pathLocal[li * numNodes + j] = (unsigned int)k;
                }
            }
        }
    }
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
            }
        }
    }
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    size_t base = (numNodes * (size_t)rank) / size;
    size_t end = (numNodes * (size_t)(rank + 1)) / size;
    size_t localRows = end - base;

    std::vector<unsigned int> distLocal(localRows * numNodes);
    std::vector<unsigned int> pathLocal(localRows * numNodes);

    // Initialize full matrices on rank 0 then scatter rows
    std::vector<unsigned int> distFull, pathFull;
    if (rank == 0) {
        distFull.resize(numNodes * numNodes);
        pathFull.resize(numNodes * numNodes);
        initializeDistanceMatrix(distFull, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(pathFull, numNodes);
    }

    // Scatter by source rows (each row numNodes elements)
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        size_t b = (numNodes * (size_t)r) / size;
        size_t e = (numNodes * (size_t)(r + 1)) / size;
        counts[r] = (int)((e - b) * numNodes);
        displs[r] = (int)(b * numNodes);
    }
    MPI_Scatterv(rank == 0 ? distFull.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 distLocal.data(), (int)(localRows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? pathFull.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 pathLocal.data(), (int)(localRows * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshallMPI(distLocal, pathLocal, numNodes, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endt = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endt - start);

    // Gather back
    std::vector<unsigned int> distResult, pathResult;
    if (rank == 0) {
        distResult.resize(numNodes * numNodes);
        pathResult.resize(numNodes * numNodes);
    }
    MPI_Gatherv(distLocal.data(), (int)(localRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? distResult.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(distResult, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(distResult, numNodes);
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
