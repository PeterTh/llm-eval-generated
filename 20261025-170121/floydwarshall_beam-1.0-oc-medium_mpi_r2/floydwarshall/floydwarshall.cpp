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
    MPI_Bcast(&numNodes, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Block distribution by destination index (first idx)
    size_t js = (rank * numNodes) / size;
    size_t je = ((rank + 1) * numNodes) / size;
    size_t nj = je - js;

    std::vector<unsigned int> distFull, pathFull;
    if (rank == 0) {
        distFull.assign(numNodes * numNodes, 0);
        pathFull.assign(numNodes * numNodes, 0);
        initializeDistanceMatrix(distFull, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(pathFull, numNodes);
    }

    std::vector<unsigned int> localDist(nj * numNodes);
    std::vector<unsigned int> localPath(nj * numNodes);

    // scatter dist
    {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            size_t rjs = (r * numNodes) / size;
            size_t rje = ((r + 1) * numNodes) / size;
            counts[r] = (int)((rje - rjs) * numNodes);
            displs[r] = (int)(rjs * numNodes);
        }
        MPI_Scatterv(distFull.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                     localDist.data(), (int)(nj * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Scatterv(pathFull.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                     localPath.data(), (int)(nj * numNodes), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    std::vector<unsigned int> rowK(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        int owner = (int)((k * size) / numNodes); // approx inverse? need owner of block containing k
        // compute owner properly
        owner = 0;
        for (int r = 0; r < size; ++r) {
            size_t rjs = (r * numNodes) / size;
            size_t rje = ((r + 1) * numNodes) / size;
            if (k >= rjs && k < rje) { owner = r; break; }
        }
        if (rank == owner) {
            // local block for dest k: index (k-js)*numNodes .. +numNodes
            size_t li = k - js;
            for (size_t i = 0; i < numNodes; ++i) rowK[i] = localDist[li * numNodes + i];
        }
        MPI_Bcast(rowK.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t li = 0; li < nj; ++li) {
            unsigned int distKJ = localDist[li * numNodes + k]; // dist[j][k]
            for (size_t i = 0; i < numNodes; ++i) {
                unsigned int distIJ = localDist[li * numNodes + i];
                unsigned int distIK = rowK[i]; // dist[k][i]
                unsigned int newDist = distIK + distKJ;
                if (newDist < distIJ) {
                    localDist[li * numNodes + i] = newDist;
                    localPath[li * numNodes + i] = (unsigned int)k;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();

    // gather
    {
        std::vector<int> counts(size), displs(size);
        for (int r = 0; r < size; ++r) {
            size_t rjs = (r * numNodes) / size;
            size_t rje = ((r + 1) * numNodes) / size;
            counts[r] = (int)((rje - rjs) * numNodes);
            displs[r] = (int)(rjs * numNodes);
        }
        MPI_Gatherv(localDist.data(), (int)(nj * numNodes), MPI_UNSIGNED,
                    distFull.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        MPI_Gatherv(localPath.data(), (int)(nj * numNodes), MPI_UNSIGNED,
                    pathFull.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        if (printResults) {
            print_results_int(distFull, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(distFull, numNodes);
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
