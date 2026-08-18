#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

struct ColumnPartition {
    size_t first;
    size_t count;
};

ColumnPartition columnsForRank(const size_t n, const int ranks, const int rank) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t rankSize = static_cast<size_t>(rank);
    return {rankSize * base + std::min(rankSize, extra), base + (rankSize < extra ? 1 : 0)};
}

int ownerOfColumn(const size_t column, const size_t n, const int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t extra = n % static_cast<size_t>(ranks);
    const size_t largerBlockColumns = (base + 1) * extra;
    if (column < largerBlockColumns) {
        return static_cast<int>(column / (base + 1));
    }
    return static_cast<int>(extra + (column - largerBlockColumns) / base);
}

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

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

void floydWarshallDistributed(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                              const size_t numNodes, const ColumnPartition localColumns,
                              const int ranks, const int rank, std::vector<unsigned int>& pivot) {
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = ownerOfColumn(k, numNodes, ranks);
        if (rank == owner) {
            const size_t localK = k - localColumns.first;
            std::memcpy(pivot.data(), dist.data() + localK * numNodes, numNodes * sizeof(unsigned int));
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        // Each rank owns contiguous complete destination columns.  Keeping j outermost
        // gives contiguous writes and reuses dist[k][j] throughout its source-column scan.
        for (size_t localJ = 0; localJ < localColumns.count; ++localJ) {
            unsigned int* const distanceColumn = dist.data() + localJ * numNodes;
            unsigned int* const pathColumn = path.data() + localJ * numNodes;
            const unsigned int distanceKJ = distanceColumn[k];
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int newDistance = pivot[i] + distanceKJ;
                if (newDistance < distanceColumn[i]) {
                    distanceColumn[i] = newDistance;
                    pathColumn[i] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(i, j, numNodes)];
                const unsigned int distIK = dist[idx2(i, k, numNodes)];
                const unsigned int distKJ = dist[idx2(k, j, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0;
    int ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(std::atoi(argv[++i]));
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes > std::numeric_limits<size_t>::max() / numNodes ||
        numNodes * numNodes > static_cast<size_t>(INT_MAX)) {
        if (rank == 0) std::printf("Number of nodes is outside the supported MPI count range\n");
        MPI_Finalize();
        return 1;
    }

    const ColumnPartition localColumns = columnsForRank(numNodes, ranks, rank);
    const size_t localElements = localColumns.count * numNodes;
    std::vector<unsigned int> localDist(localElements);
    std::vector<unsigned int> localPath(localElements);
    std::vector<unsigned int> pivot(numNodes);

    std::vector<int> counts;
    std::vector<int> displacements;
    std::vector<unsigned int> fullDist;
    std::vector<unsigned int> fullPath;
    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\n", numNodes);
        std::printf("MPI ranks: %d\n", ranks);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
        counts.resize(ranks);
        displacements.resize(ranks);
        for (int process = 0; process < ranks; ++process) {
            const ColumnPartition part = columnsForRank(numNodes, ranks, process);
            counts[process] = static_cast<int>(part.count * numNodes);
            displacements[process] = static_cast<int>(part.first * numNodes);
        }
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, numNodes);
    }

    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displacements.data() : nullptr, MPI_UNSIGNED, localDist.data(),
                 static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? fullPath.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                 rank == 0 ? displacements.data() : nullptr, MPI_UNSIGNED, localPath.data(),
                 static_cast<int>(localElements), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing shortest paths...\n");
    const auto start = std::chrono::steady_clock::now();
    floydWarshallDistributed(localDist, localPath, numNodes, localColumns, ranks, rank, pivot);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (printResults || validate) {
        if (rank == 0) fullDist.resize(numNodes * numNodes);
        MPI_Gatherv(localDist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, rank == 0 ? counts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        std::printf("Performance: %.3f GOPS\n", ops / elapsedSeconds / 1e9);
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
