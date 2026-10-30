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

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const size_t base = numNodes / size;
    const size_t rem = numNodes % size;
    const size_t start = rank * base + std::min((size_t)rank, rem);
    const size_t end = start + base + (rank < (int)rem ? 1 : 0);
    const size_t owned = end - start;

    std::vector<unsigned int> rowk_dist(numNodes);
    std::vector<unsigned int> rowk_path(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        int owner = 0;
        size_t acc = 0;
        for (int r = 0; r < size; ++r) {
            size_t rows = base + (r < (int)rem ? 1 : 0);
            if (k >= acc && k < acc + rows) { owner = r; break; }
            acc += rows;
        }
        if (rank == owner) {
            // row k is contiguous starting at k*numNodes
            std::memcpy(rowk_dist.data(), &dist[k * numNodes], numNodes * sizeof(unsigned int));
            std::memcpy(rowk_path.data(), &path[k * numNodes], numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(rowk_dist.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        MPI_Bcast(rowk_path.data(), (int)numNodes, MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t i = start; i < end; ++i) {
            unsigned int dik = dist[idx2(k, i, numNodes)]; // = dist[i*numNodes + k]? check
            // dist[idx2(k,i)] = i*numNodes + k
            // already computed above? idx2(k,i)=i*n+k yes
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distKJ = rowk_dist[j];
                const unsigned int newDist = dik + distKJ;
                if (newDist < distIJ) {
                    dist[idx2(j, i, numNodes)] = newDist;
                    path[idx2(j, i, numNodes)] = k;
                }
            }
        }
    }
    // suppress unused
    (void)owned;
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

    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }

    // Gather rows to root
    int base = (int)(numNodes / size);
    int rem = (int)(numNodes % size);
    // counts per rank in elements
    std::vector<int> counts(size), displs(size);
    size_t acc = 0;
    for (int r = 0; r < size; ++r) {
        int rows = base + (r < rem ? 1 : 0);
        counts[r] = rows * (int)numNodes;
        displs[r] = (int)(acc * numNodes);
        acc += rows;
    }
    // root needs full dist/path; non-root send their owned rows
    // Determine this rank's start row
    size_t rstart = rank * base + std::min((size_t)rank, (size_t)rem);
    unsigned int* sendbuf = &dist[rstart * numNodes];
    std::vector<unsigned int> fullDist, fullPath;
    if (rank == 0) { fullDist.resize(numNodes*numNodes); fullPath.resize(numNodes*numNodes); }
    MPI_Gatherv(sendbuf, counts[rank], MPI_UNSIGNED, fullDist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(&path[rstart * numNodes], counts[rank], MPI_UNSIGNED, fullPath.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        dist = fullDist;
        path = fullPath;
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
