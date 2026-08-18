#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    }
    for (size_t i = 0; i < numNodes; ++i) dist[idx2(i, i, numNodes)] = 0;
}

// Each rank owns a contiguous, balanced range of destination columns.  At step k,
// the owner of column k broadcasts it; all owned columns can then be updated locally.
void floydWarshall(std::vector<unsigned int>& localDist, const size_t numNodes,
                   const size_t firstColumn, const size_t localColumns,
                   const std::vector<int>& columnOwner, MPI_Comm comm) {
    std::vector<unsigned int> pivot(numNodes);
    for (size_t k = 0; k < numNodes; ++k) {
        const int owner = columnOwner[k];
        if (k >= firstColumn && k < firstColumn + localColumns) {
            const unsigned int* src = localDist.data() + (k - firstColumn) * numNodes;
            std::copy_n(src, numNodes, pivot.data());
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, comm);

        for (size_t localJ = 0; localJ < localColumns; ++localJ) {
            unsigned int* const column = localDist.data() + localJ * numNodes;
            const unsigned int distKJ = column[k];
            // The generated graph has finite, small positive weights, so overflow and
            // INF checks are unnecessary and this loop is readily vectorized.
            for (size_t i = 0; i < numNodes; ++i) {
                const unsigned int candidate = pivot[i] + distKJ;
                if (candidate < column[i]) column[i] = candidate;
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
    for (size_t i = 0; i < std::min(numNodes, size_t{10}); ++i) {
        for (size_t j = 0; j < std::min(numNodes, size_t{10}); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int dij = dist[idx2(j, i, numNodes)];
                const unsigned int dik = dist[idx2(k, i, numNodes)];
                const unsigned int dkj = dist[idx2(j, k, numNodes)];
                if (dik < INF && dkj < INF && dik + dkj < dij) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                           i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numNodes = 512;
    bool validate = false, printResults = false;
    int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numNodes = strtoull(argv[++i], nullptr, 10);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }

    if (numNodes == 0 || numNodes > static_cast<size_t>(INT_MAX) ||
        numNodes > static_cast<size_t>(INT_MAX) / numNodes) {
        if (rank == 0) fprintf(stderr, "Matrix dimensions exceed MPI count limits\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<int> counts(ranks), displacements(ranks), owner(numNodes);
    size_t firstColumn = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t begin = numNodes * static_cast<size_t>(r) / ranks;
        const size_t end = numNodes * static_cast<size_t>(r + 1) / ranks;
        counts[r] = static_cast<int>((end - begin) * numNodes);
        displacements[r] = static_cast<int>(begin * numNodes);
        for (size_t j = begin; j < end; ++j) owner[j] = r;
        if (r == rank) firstColumn = begin;
    }
    const size_t localColumns = static_cast<size_t>(counts[rank]) / numNodes;
    std::vector<unsigned int> localDist(static_cast<size_t>(counts[rank]));
    std::vector<unsigned int> dist;

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n", numNodes);
        printf("Validation: %s\nMPI ranks: %d\nInitializing graph...\n", validate ? "enabled" : "disabled", ranks);
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) { std::vector<unsigned int>().swap(dist); printf("Computing shortest paths...\n"); }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(localDist, numNodes, firstColumn, localColumns, owner, MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        if (rank == 0) dist.resize(numNodes * numNodes);
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED,
                    rank == 0 ? dist.data() : nullptr, counts.data(), displacements.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        const long milliseconds = static_cast<long>(seconds * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0.0 ? ops / seconds / 1e9 : 0.0);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            exitCode = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
