#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

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
        dist[i] = rangeMin +
                  static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

// Consecutive rows give each process contiguous memory and make the pivot row
// a single, bandwidth-efficient MPI broadcast.
struct RowPartition {
    size_t firstRow;
    size_t rowCount;
    std::vector<int> counts;
    std::vector<int> displacements;
};

RowPartition makePartition(const size_t n, const int rank, const int processCount) {
    const size_t base = n / static_cast<size_t>(processCount);
    const size_t extra = n % static_cast<size_t>(processCount);
    RowPartition part{static_cast<size_t>(rank) * base +
                          std::min(static_cast<size_t>(rank), extra),
                      base + (static_cast<size_t>(rank) < extra),
                      std::vector<int>(processCount), std::vector<int>(processCount)};

    size_t displacement = 0;
    for (int p = 0; p < processCount; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < extra);
        part.counts[p] = static_cast<int>(rows * n);
        part.displacements[p] = static_cast<int>(displacement);
        displacement += rows * n;
    }
    return part;
}

int rowOwner(const size_t row, const size_t n, const int processCount) noexcept {
    const size_t base = n / static_cast<size_t>(processCount);
    const size_t extra = n % static_cast<size_t>(processCount);
    const size_t largeRows = (base + 1) * extra;
    if (row < largeRows) {
        return static_cast<int>(row / (base + 1));
    }
    return static_cast<int>(extra + (row - largeRows) / base);
}

void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   const size_t numNodes, const RowPartition& part,
                   const int rank, const int processCount) {
    std::vector<unsigned int> pivotRow(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = rowOwner(k, numNodes, processCount);
        if (rank == owner) {
            const size_t localPivot = k - part.firstRow;
            std::copy_n(dist.data() + localPivot * numNodes, numNodes, pivotRow.data());
        }
        MPI_Bcast(pivotRow.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner,
                  MPI_COMM_WORLD);

        const unsigned int* __restrict__ const pivot = pivotRow.data();
        for (size_t localRow = 0; localRow < part.rowCount; ++localRow) {
            unsigned int* __restrict__ const row = dist.data() + localRow * numNodes;
            unsigned int* __restrict__ const pathRow = path.data() + localRow * numNodes;
            const unsigned int distIK = row[k];

            // The simple contiguous loop is intentionally exposed to compiler
            // vectorization; it is the dominant O(n^3 / P) work.
            for (size_t j = 0; j < numNodes; ++j) {
                const unsigned int newDist = distIK + pivot[j];
                if (newDist < row[j]) {
                    row[j] = newDist;
                    pathRow[j] = static_cast<unsigned int>(k);
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
    int processCount = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            parseStatus = 2;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 1;
        }
    }

    // MPI's standard collectives use int element counts. Reject an unsupported
    // size explicitly instead of silently truncating a count.
    if (numNodes > static_cast<size_t>(INT_MAX) ||
        (numNodes != 0 && numNodes > static_cast<size_t>(INT_MAX) / numNodes)) {
        if (rank == 0) fprintf(stderr, "Matrix is too large for MPI collective counts\n");
        parseStatus = 1;
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 2 ? 0 : 1;
    }

    const RowPartition part = makePartition(numNodes, rank, processCount);
    std::vector<unsigned int> localDist(part.rowCount * numNodes);
    std::vector<unsigned int> localPath(part.rowCount * numNodes);
    std::vector<unsigned int> fullDist;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("MPI processes: %d\n", processCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        fullDist.resize(numNodes * numNodes);
        initializeDistanceMatrix(fullDist, numNodes, 1, MAX_DISTANCE);
    }

    MPI_Scatterv(rank == 0 ? fullDist.data() : nullptr, part.counts.data(),
                 part.displacements.data(), MPI_UNSIGNED, localDist.data(),
                 static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    fullDist.clear();
    fullDist.shrink_to_fit();

    for (size_t localRow = 0; localRow < part.rowCount; ++localRow) {
        const unsigned int initialPath = static_cast<unsigned int>(part.firstRow + localRow);
        std::fill_n(localPath.data() + localRow * numNodes, numNodes, initialPath);
    }

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(localDist, localPath, numNodes, part, rank, processCount);
    const double localElapsed = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localElapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        const double gops = elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0;
        printf("Performance: %.3f GOPS\n", gops);
    }

    if (validate || printResults) {
        if (rank == 0) fullDist.resize(numNodes * numNodes);
        MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                    rank == 0 ? fullDist.data() : nullptr, part.counts.data(),
                    part.displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }

    int exitCode = 0;
    if (rank == 0) {
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(fullDist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
