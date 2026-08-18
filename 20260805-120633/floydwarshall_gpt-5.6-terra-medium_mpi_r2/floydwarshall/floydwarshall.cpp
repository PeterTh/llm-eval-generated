#include <algorithm>
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

// The externally visible matrix format used by the original benchmark.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
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
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                if (distIK < INF && distKJ < INF && distIK + distKJ < distIJ) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t numNodes = 512;
    bool validate = false, printResults = false;
    int argumentError = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            argumentError = 1;
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
        }
    }
    if (argumentError) { MPI_Finalize(); return 1; }

    // MPI collectives use int counts.  This also prevents size_t multiplication overflow.
    const bool tooLarge = numNodes > static_cast<size_t>(INT_MAX) ||
        (numNodes != 0 && numNodes > static_cast<size_t>(INT_MAX) / numNodes);
    if (tooLarge) {
        if (rank == 0) printf("Number of nodes is too large for this MPI implementation\n");
        MPI_Finalize();
        return 1;
    }
    const size_t elements = numNodes * numNodes;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n", numNodes);
        printf("Validation: %s\nInitializing graph...\n", validate ? "enabled" : "disabled");
    }

    std::vector<int> counts(ranks), displacements(ranks), rowStarts(ranks + 1);
    for (int process = 0; process < ranks; ++process) {
        const size_t begin = numNodes * static_cast<size_t>(process) / ranks;
        const size_t end = numNodes * static_cast<size_t>(process + 1) / ranks;
        rowStarts[process] = static_cast<int>(begin);
        counts[process] = static_cast<int>((end - begin) * numNodes);
        displacements[process] = static_cast<int>(begin * numNodes);
    }
    rowStarts[ranks] = static_cast<int>(numNodes);
    const size_t localRows = numNodes * static_cast<size_t>(rank + 1) / ranks -
                             numNodes * static_cast<size_t>(rank) / ranks;
    std::vector<unsigned int> localDist(localRows * numNodes), localPath(localRows * numNodes);

    // Generate in original column-major RNG order, but write directly into the
    // contiguous [source][destination] layout required by Scatterv.
    std::vector<unsigned int> packedDist, packedPath;
    if (rank == 0) {
        packedDist.resize(elements);
        packedPath.resize(elements);
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        for (size_t flatIndex = 0; flatIndex < elements; ++flatIndex) {
            const size_t source = flatIndex / numNodes;
            const size_t destination = flatIndex % numNodes;
            packedDist[source * numNodes + destination] =
                1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
            // The original path initializer leaves every entry equal to its source.
            packedPath[source * numNodes + destination] = static_cast<unsigned int>(source);
        }
        for (size_t node = 0; node < numNodes; ++node)
            packedDist[node * numNodes + node] = 0;
    }
    MPI_Scatterv(rank == 0 ? packedDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? packedPath.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing shortest paths...\n");
    std::vector<unsigned int> pivot(numNodes);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    int owner = 0;
    for (size_t k = 0; k < numNodes; ++k) {
        while (owner + 1 < ranks && k >= static_cast<size_t>(rowStarts[owner + 1])) ++owner;
        const int pivotOffset = static_cast<int>((k - static_cast<size_t>(rowStarts[owner])) * numNodes);
        if (rank == owner)
            std::copy_n(localDist.data() + pivotOffset, numNodes, pivot.data());
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, owner, MPI_COMM_WORLD);

        for (size_t localSource = 0; localSource < localRows; ++localSource) {
            unsigned int* const distanceRow = localDist.data() + localSource * numNodes;
            unsigned int* const pathRow = localPath.data() + localSource * numNodes;
            const unsigned int distanceToPivot = distanceRow[k];
            for (size_t destination = 0; destination < numNodes; ++destination) {
                const unsigned int candidate = distanceToPivot + pivot[destination];
                if (candidate < distanceRow[destination]) {
                    distanceRow[destination] = candidate;
                    pathRow[destination] = static_cast<unsigned int>(k);
                }
            }
        }
    }
    const double localSeconds = MPI_Wtime() - start;
    double elapsedSeconds = 0.0;
    MPI_Reduce(&localSeconds, &elapsedSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<unsigned int> finalDist;
    if (validate || printResults) {
        std::vector<unsigned int> gathered;
        if (rank == 0) gathered.resize(elements);
        MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? gathered.data() : nullptr,
                    counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            finalDist.resize(elements);
            for (size_t source = 0; source < numNodes; ++source)
                for (size_t destination = 0; destination < numNodes; ++destination)
                    finalDist[idx2(destination, source, numNodes)] = gathered[source * numNodes + destination];
        }
    }

    int exitCode = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(elapsedSeconds * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / elapsedSeconds / 1e9);
        if (printResults) print_results_int(finalDist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            if (validateResult(finalDist, numNodes)) printf("Validation: PASSED\n");
            else { printf("Validation: FAILED\n"); exitCode = 1; }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
