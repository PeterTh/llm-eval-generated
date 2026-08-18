#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The original benchmark uses dist[destination * n + source].  The distributed
// representation below stores complete source rows contiguously instead:
// localDist[sourceInBlock * n + destination].

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    }
    for (size_t i = 0; i < numNodes; ++i) {
        // This is the original column-major diagonal, and is also the row-major one.
        dist[i * numNodes + i] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t localRows,
                          const size_t numNodes) {
    for (size_t source = 0; source < localRows; ++source) {
        for (size_t destination = 0; destination < numNodes; ++destination) {
            path[source * numNodes + destination] = static_cast<unsigned int>(destination);
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                  const size_t numNodes, const size_t firstSource, const int rank,
                  const std::vector<int>& rowsPerRank, const std::vector<int>& ownerPerVertex) {
    std::vector<unsigned int> pivot(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerPerVertex[k];

        if (rank == owner) {
            const size_t pivotLocal = k - firstSource;
            std::copy_n(dist.data() + pivotLocal * numNodes, numNodes, pivot.data());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        const size_t localRows = rowsPerRank[rank];
        for (size_t source = 0; source < localRows; ++source) {
            const unsigned int sourceToK = dist[source * numNodes + k];
            unsigned int* row = dist.data() + source * numNodes;
            unsigned int* pathRow = path.data() + source * numNodes;
            for (size_t destination = 0; destination < numNodes; ++destination) {
                const unsigned int candidate = sourceToK + pivot[destination];
                if (candidate < row[destination]) {
                    row[destination] = candidate;
                    pathRow[destination] = static_cast<unsigned int>(k);
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
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[i * numNodes + j];
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                           i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n", progName);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, worldSize = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    size_t numNodes = 512;
    bool validate = false, printResults = false, parseError = false, showHelp = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) showHelp = true;
        else parseError = true;
    }
    if (showHelp || parseError) {
        if (rank == 0) {
            if (parseError) printf("Unknown option\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    if (numNodes > static_cast<size_t>(std::numeric_limits<int>::max()) ||
        (numNodes != 0 && numNodes > std::numeric_limits<size_t>::max() / numNodes)) {
        if (rank == 0) printf("Number of nodes is too large\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<int> rowsPerRank(worldSize), firstPerRank(worldSize);
    const size_t base = numNodes / static_cast<size_t>(worldSize);
    const size_t remainder = numNodes % static_cast<size_t>(worldSize);
    size_t first = 0;
    for (int r = 0; r < worldSize; ++r) {
        firstPerRank[r] = static_cast<int>(first);
        rowsPerRank[r] = static_cast<int>(base + (static_cast<size_t>(r) < remainder));
        first += static_cast<size_t>(rowsPerRank[r]);
    }
    const size_t localRows = static_cast<size_t>(rowsPerRank[rank]);
    const size_t localElements = localRows * numNodes;
    if (localElements > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) printf("A local matrix block is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    std::vector<int> ownerPerVertex(numNodes);
    for (int r = 0; r < worldSize; ++r)
        for (int s = 0; s < rowsPerRank[r]; ++s)
            ownerPerVertex[static_cast<size_t>(firstPerRank[r] + s)] = r;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
               numNodes, validate ? "enabled" : "disabled");
    }
    std::vector<unsigned int> localDist(localElements), localPath(localElements);
    std::vector<unsigned int> globalDist, packed;
    if (rank == 0) {
        printf("Initializing graph...\n");
        globalDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(globalDist, numNodes, 1, MAX_DISTANCE);
        packed.resize(numNodes * numNodes);
        for (int r = 0; r < worldSize; ++r)
            for (int s = 0; s < rowsPerRank[r]; ++s)
                for (size_t d = 0; d < numNodes; ++d)
                    packed[(static_cast<size_t>(firstPerRank[r]) + s) * numNodes + d] =
                        globalDist[d * numNodes + static_cast<size_t>(firstPerRank[r]) + s];
    }
    std::vector<int> counts(worldSize), displacements(worldSize);
    for (int r = 0; r < worldSize; ++r) {
        counts[r] = static_cast<int>(static_cast<size_t>(rowsPerRank[r]) * numNodes);
        displacements[r] = static_cast<int>(static_cast<size_t>(firstPerRank[r]) * numNodes);
    }
    MPI_Scatterv(rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    initializePathMatrix(localPath, localRows, numNodes);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing shortest paths...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    floydWarshall(localDist, localPath, numNodes, static_cast<size_t>(firstPerRank[rank]), rank,
                  rowsPerRank, ownerPerVertex);
    const auto end = std::chrono::high_resolution_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    double maximumElapsed = 0.0;
    MPI_Reduce(&elapsed, &maximumElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maximumElapsed * 1000.0));
        printf("Performance: %.3f GOPS\n", (maximumElapsed > 0.0 ?
               static_cast<double>(numNodes) * numNodes * numNodes / maximumElapsed / 1e9 : 0.0));
    }

    if (validate || printResults) {
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? packed.data() : nullptr,
                    counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (size_t source = 0; source < numNodes; ++source)
                for (size_t destination = 0; destination < numNodes; ++destination)
                    globalDist[destination * numNodes + source] = packed[source * numNodes + destination];
            if (printResults) print_results_int(globalDist, "DistanceMatrix");
            if (validate) printf("Validating result...\n");
        }
    }
    int valid = 1;
    if (validate && rank == 0) valid = validateResult(globalDist, numNodes) ? 1 : 0;
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (validate && rank == 0) printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
    MPI_Finalize();
    return validate && !valid ? 1 : 0;
}
