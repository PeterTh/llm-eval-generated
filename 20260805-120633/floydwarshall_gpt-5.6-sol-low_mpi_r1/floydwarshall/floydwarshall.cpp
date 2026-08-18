#include <algorithm>
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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t n,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

// Rows are divided into contiguous, nearly equal ranges.  Keeping complete rows
// local makes the O(n^3/p) update cache friendly and requires one broadcast per k.
void floydWarshall(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                   size_t n, size_t firstRow, const std::vector<int>& rowCounts,
                   const std::vector<int>& rowOffsets, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    std::vector<unsigned int> pivot(n);
    int owner = 0;

    for (size_t k = 0; k < n; ++k) {
        while (owner + 1 < static_cast<int>(rowCounts.size()) &&
               k >= static_cast<size_t>(rowOffsets[owner] + rowCounts[owner]))
            ++owner;

        if (rank == owner) {
            const size_t localK = k - firstRow;
            std::memcpy(pivot.data(), dist.data() + localK * n, n * sizeof(unsigned int));
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, comm);

        const size_t localRows = static_cast<size_t>(rowCounts[rank]);
        for (size_t li = 0; li < localRows; ++li) {
            unsigned int* const row = dist.data() + li * n;
            unsigned int* const pathRow = path.data() + li * n;
            const unsigned int dik = row[k];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = dik + pivot[j];
                if (candidate < row[j]) {
                    row[j] = candidate;
                    pathRow[j] = static_cast<unsigned int>(k);
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t{10}); ++i)
        for (size_t j = 0; j < std::min(n, size_t{10}); ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int dij = dist[idx2(j, i, n)];
                const unsigned int dik = dist[idx2(k, i, n)];
                const unsigned int dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF && dik + dkj < dij) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (!end || *end || value == 0) status = 1;
            else n = static_cast<size_t>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) status = 2;
        else status = 1;
    }
    if (status) {
        if (rank == 0) {
            if (status == 1) printf("Invalid command line\n");
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return status == 2 ? 0 : 1;
    }
    if (n > static_cast<size_t>(INT_MAX) || n > static_cast<size_t>(INT_MAX) / n) {
        if (rank == 0) fprintf(stderr, "Matrix is too large for MPI collective counts\n");
        MPI_Finalize();
        return 1;
    }

    std::vector<int> rowCounts(ranks), rowOffsets(ranks), counts(ranks), displs(ranks);
    const size_t base = n / static_cast<size_t>(ranks), remainder = n % static_cast<size_t>(ranks);
    size_t offset = 0;
    for (int r = 0; r < ranks; ++r) {
        rowOffsets[r] = static_cast<int>(offset);
        rowCounts[r] = static_cast<int>(base + (static_cast<size_t>(r) < remainder));
        counts[r] = static_cast<int>(static_cast<size_t>(rowCounts[r]) * n);
        displs[r] = static_cast<int>(offset * n);
        offset += static_cast<size_t>(rowCounts[r]);
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n", n);
        printf("MPI ranks: %d\nValidation: %s\nInitializing graph...\n", ranks,
               validate ? "enabled" : "disabled");
    }
    std::vector<unsigned int> full;
    if (rank == 0) {
        full.resize(n * n);
        initializeDistanceMatrix(full, n, 1, MAX_DISTANCE);
    }
    const size_t localElements = static_cast<size_t>(counts[rank]);
    std::vector<unsigned int> dist(localElements), path(localElements);
    MPI_Scatterv(rank == 0 ? full.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    full.clear();
    full.shrink_to_fit();
    const size_t firstRow = static_cast<size_t>(rowOffsets[rank]);
    for (size_t li = 0; li < static_cast<size_t>(rowCounts[rank]); ++li)
        for (size_t j = 0; j < n; ++j) path[li * n + j] = static_cast<unsigned int>(j);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    floydWarshall(dist, path, n, firstRow, rowCounts, rowOffsets, MPI_COMM_WORLD);
    const double localTime = MPI_Wtime() - start;
    double elapsed = 0.0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (validate || printResults) {
        if (rank == 0) full.resize(n * n);
        MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? full.data() : nullptr,
                    counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    }
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", elapsed * 1000.0);
        const double ops = static_cast<double>(n) * static_cast<double>(n) * static_cast<double>(n);
        printf("Performance: %.3f GOPS\n", elapsed > 0.0 ? ops / elapsed / 1e9 : 0.0);
        if (printResults) print_results_int(full, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(full, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
