#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Matrices retain the benchmark's column-major indexing: [destination][source].
inline constexpr size_t idx2(const size_t destination, const size_t source,
                             const size_t n) noexcept {
    return source * n + destination;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t n) {
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE) + 1.0 - 1.0;
    for (size_t i = 0; i < n * n; ++i) {
        dist[i] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    }
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    const size_t sample = std::min(n, size_t(10));
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int ik = dist[idx2(k, i, n)];
                const unsigned int kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < dist[idx2(j, i, n)]) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                                i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t n = 512;
    bool validate = false, printResults = false, parseError = false, help = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (*argv[i] == '\0' || *end != '\0' || value == 0 ||
                value > std::numeric_limits<size_t>::max()) parseError = true;
            else n = static_cast<size_t>(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) help = true;
        else parseError = true;
    }
    if (help || parseError) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return parseError ? 1 : 0;
    }
    if (n > std::numeric_limits<size_t>::max() / n ||
        n * n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) std::printf("Number of nodes is too large for this MPI implementation\n");
        MPI_Finalize();
        return 1;
    }

    const size_t base = n / static_cast<size_t>(world);
    const size_t remainder = n % static_cast<size_t>(world);
    const size_t localRows = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t localElements = localRows * n;
    std::vector<unsigned int> dist(localElements), path(localElements);
    std::vector<unsigned int> pivotRow(n);

    std::vector<unsigned int> global, packed;
    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n",
                    n);
        std::printf("Validation: %s\n", validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
        global.resize(n * n);
        initializeDistanceMatrix(global, n);
        packed.resize(n * n);
        size_t out = 0;
        for (int r = 0; r < world; ++r) {
            const size_t rows = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
            const size_t start = static_cast<size_t>(r) * base +
                                 std::min(static_cast<size_t>(r), remainder);
            for (size_t source = 0; source < n; ++source)
                for (size_t destination = start; destination < start + rows; ++destination)
                    packed[out++] = global[idx2(destination, source, n)];
        }
    }

    std::vector<int> counts(world), displacements(world);
    for (int r = 0; r < world; ++r) {
        const size_t rows = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        const size_t start = static_cast<size_t>(r) * base +
                             std::min(static_cast<size_t>(r), remainder);
        counts[r] = static_cast<int>(rows * n);
        displacements[r] = static_cast<int>(start * n); // packed is rank-ordered
    }
    MPI_Scatterv(rank == 0 ? packed.data() : nullptr, counts.data(), displacements.data(),
                 MPI_UNSIGNED, dist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                 0, MPI_COMM_WORLD);
    for (size_t source = 0; source < n; ++source)
        for (size_t local = 0; local < localRows; ++local)
            path[idx2(local, source, localRows)] = static_cast<unsigned int>(source);
    global.clear();
    global.shrink_to_fit();
    packed.clear();
    packed.shrink_to_fit();

    if (rank == 0) std::printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();
    for (size_t k = 0; k < n; ++k) {
        const int owner = remainder == 0
                              ? static_cast<int>(k / base)
                              : (k < (base + 1) * remainder
                                     ? static_cast<int>(k / (base + 1))
                                     : static_cast<int>(remainder + (k - (base + 1) * remainder) / base));
        const size_t ownerStart = static_cast<size_t>(owner) * base +
                                  std::min(static_cast<size_t>(owner), remainder);
        if (rank == owner) {
            const size_t localK = k - ownerStart;
            for (size_t source = 0; source < n; ++source)
                pivotRow[source] = dist[idx2(localK, source, localRows)];
        }
        MPI_Bcast(pivotRow.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t source = 0; source < n; ++source) {
            const unsigned int sourceToK = pivotRow[source];
            for (size_t local = 0; local < localRows; ++local) {
                const unsigned int candidate = dist[idx2(local, k, localRows)] + sourceToK;
                unsigned int& current = dist[idx2(local, source, localRows)];
                if (candidate < current) {
                    current = candidate;
                    path[idx2(local, source, localRows)] = static_cast<unsigned int>(k);
                }
            }
        }
    }
    const double localSeconds = MPI_Wtime() - startTime;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double gops = seconds > 0.0
                                ? n * static_cast<double>(n) * n / seconds / 1e9
                                : 0.0;
        std::printf("Computation time: %.0f ms\nPerformance: %.3f GOPS\n",
                    seconds * 1000.0, gops);
    }

    int valid = 1;
    const bool needResult = validate || printResults;
    if (needResult) {
        std::vector<unsigned int> gathered;
        if (rank == 0) gathered.resize(n * n);
        MPI_Gatherv(dist.data(), static_cast<int>(localElements), MPI_UNSIGNED,
                    rank == 0 ? gathered.data() : nullptr, counts.data(), displacements.data(),
                    MPI_UNSIGNED, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            global.assign(n * n, 0);
            size_t in = 0;
            for (int r = 0; r < world; ++r) {
                const size_t rows = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
                const size_t start = static_cast<size_t>(r) * base +
                                     std::min(static_cast<size_t>(r), remainder);
                for (size_t source = 0; source < n; ++source)
                    for (size_t destination = start; destination < start + rows; ++destination)
                        global[idx2(destination, source, n)] = gathered[in++];
            }
            if (printResults) print_results_int(global, "DistanceMatrix");
            if (validate) {
                std::printf("Validating result...\n");
                valid = validateResult(global, n) ? 1 : 0;
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            }
        }
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
