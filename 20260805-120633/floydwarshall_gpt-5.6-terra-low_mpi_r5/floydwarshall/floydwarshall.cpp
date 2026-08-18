#include <algorithm>
#include <chrono>
#include <climits>
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

static void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes) {
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = 1U + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < numNodes; ++i)
        dist[idx2(i, i, numNodes)] = 0;
}

static bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(numNodes, size_t{10}); ++i)
        for (size_t j = 0; j < std::min(numNodes, size_t{10}); ++j)
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int ij = dist[idx2(j, i, numNodes)];
                const unsigned int ik = dist[idx2(k, i, numNodes)];
                const unsigned int kj = dist[idx2(j, k, numNodes)];
                if (ik < INF && kj < INF && ik + kj < ij) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

static void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    unsigned long long nodeCount = 512;
    int validate = 0, printResults = 0, exitCode = 0;
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc)
                nodeCount = std::strtoull(argv[++i], nullptr, 10);
            else if (std::strcmp(argv[i], "-v") == 0) validate = 1;
            else if (std::strcmp(argv[i], "-r") == 0) printResults = 1;
            else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); exitCode = 2; break; }
            else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); exitCode = 1; break; }
        }
        if (nodeCount == 0 || nodeCount > 46340ULL) {
            std::printf("Number of nodes must be between 1 and 46340 for this MPI implementation\n");
            exitCode = 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (exitCode) { MPI_Finalize(); return exitCode == 2 ? 0 : 1; }
    MPI_Bcast(&nodeCount, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);
    const size_t n = static_cast<size_t>(nodeCount);

    // Rows are sources i, with destinations j contiguous: local[i][j] == dist[j,i].
    std::vector<int> counts(ranks), displacements(ranks);
    const size_t base = n / static_cast<size_t>(ranks), remainder = n % static_cast<size_t>(ranks);
    size_t localRows = 0, firstRow = 0;
    for (int p = 0; p < ranks; ++p) {
        const size_t rows = base + (static_cast<size_t>(p) < remainder);
        counts[p] = static_cast<int>(rows * n);
        displacements[p] = static_cast<int>(firstRow * n);
        if (p == rank) { localRows = rows; firstRow = displacements[p] / n; }
        firstRow += rows;
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n",
                    n, validate ? "enabled" : "disabled");
    }
    std::vector<unsigned int> localDist(localRows * n);
    std::vector<unsigned int> initialPacked;
    if (rank == 0) {
        std::vector<unsigned int> initial(n * n);
        initializeDistanceMatrix(initial, n);
        initialPacked.resize(n * n);
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                initialPacked[i * n + j] = initial[idx2(j, i, n)];
    }
    MPI_Scatterv(rank == 0 ? initialPacked.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    std::vector<unsigned int>().swap(initialPacked);
    std::vector<unsigned int> pivotRow(n);
    if (rank == 0) std::printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k / (base + 1)) < static_cast<int>(remainder)
                              ? static_cast<int>(k / (base + 1))
                              : static_cast<int>(remainder + (k - remainder * (base + 1)) / base);
        if (rank == owner)
            std::memcpy(pivotRow.data(), localDist.data() + (k - static_cast<size_t>(displacements[rank] / n)) * n,
                        n * sizeof(unsigned int));
        MPI_Bcast(pivotRow.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t localI = 0; localI < localRows; ++localI) {
            unsigned int* const row = localDist.data() + localI * n;
            const unsigned int dik = row[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = dik + pivotRow[j];
                if (candidate < row[j]) row[j] = candidate;
            }
        }
    }
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<unsigned int> packed, dist;
    if (validate || printResults) {
        if (rank == 0) packed.resize(n * n);
        MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED,
                    rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            dist.resize(n * n);
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j)
                    dist[idx2(j, i, n)] = packed[i * n + j];
        }
    }
    int result = 0;
    if (rank == 0) {
        const long milliseconds = static_cast<long>(seconds * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        const double gops = seconds > 0.0 ? static_cast<double>(n) * n * n / seconds / 1e9 : 0.0;
        std::printf("Performance: %.3f GOPS\n", gops);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            result = validateResult(dist, n) ? 0 : 1;
            std::printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
