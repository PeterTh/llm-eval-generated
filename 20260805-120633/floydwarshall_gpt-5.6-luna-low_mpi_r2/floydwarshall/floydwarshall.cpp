#include <algorithm>
#include <chrono>
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

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    for (size_t j = 0; j < n; ++j)
        for (size_t i = 0; i < n; ++i) {
            path[idx2(i, j, n)] = static_cast<unsigned int>(j);
            path[idx2(j, i, n)] = static_cast<unsigned int>(i);
        }
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of nodes (default: 512)\n  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, world = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0) { if (rank == 0) std::printf("Number of nodes must be positive\n"); MPI_Finalize(); return 1; }

    // Balance columns, with at most one extra column per rank.
    const size_t base = n / static_cast<size_t>(world), extra = n % static_cast<size_t>(world);
    const size_t localN = base + (static_cast<size_t>(rank) < extra ? 1 : 0);
    std::vector<int> counts(world), displs(world);
    for (int r = 0; r < world; ++r) {
        const size_t cols = base + (static_cast<size_t>(r) < extra ? 1 : 0);
        counts[r] = static_cast<int>(cols * n);
        displs[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), extra)) * n);
    }

    std::vector<unsigned int> globalDist, globalPath;
    if (rank == 0) {
        globalDist.resize(n * n); globalPath.resize(n * n);
        initializeDistanceMatrix(globalDist, n, 1, MAX_DISTANCE);
        initializePathMatrix(globalPath, n);
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n", n, validate ? "enabled" : "disabled");
    }
    std::vector<unsigned int> dist(localN * n), path(localN * n), pivot(n);
    MPI_Scatterv(rank == 0 ? globalDist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? globalPath.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 path.data(), static_cast<int>(path.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    globalDist.clear(); globalPath.clear();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) std::printf("Computing shortest paths...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    for (size_t k = 0; k < n; ++k) {
        int owner = static_cast<int>(std::min(k / (base + 1), extra));
        if (k >= (base + 1) * extra) owner = static_cast<int>(extra + (k - (base + 1) * extra) / base);
        const size_t offset = k - (static_cast<size_t>(owner) * base + std::min(static_cast<size_t>(owner), extra));
        if (rank == owner) std::copy_n(dist.data() + offset * n, n, pivot.data());
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t i = 0; i < localN; ++i) {
            const unsigned int dik = dist[i * n + k];
            unsigned int* col = dist.data() + i * n;
            unsigned int* pcol = path.data() + i * n;
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = dik + pivot[j];
                if (candidate < col[j]) { col[j] = candidate; pcol[j] = static_cast<unsigned int>(k); }
            }
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration<double>(end - start).count(), maxSeconds = 0;
    MPI_Reduce(&seconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) globalDist.resize(n * n);
    MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED, rank == 0 ? globalDist.data() : nullptr,
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        const double gops = maxSeconds > 0.0 ? n * n * n / maxSeconds / 1e9 : 0.0;
        std::printf("Computation time: %.0f ms\nPerformance: %.3f GOPS\n", maxSeconds * 1000.0, gops);
        if (printResults) print_results_int(globalDist, "DistanceMatrix");
        if (validate) {
            bool ok = true;
            for (size_t i = 0; i < n; ++i) ok &= globalDist[idx2(i, i, n)] == 0;
            for (size_t i = 0; i < std::min(n, size_t(10)); ++i) for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
                for (size_t k = 0; k < n; ++k) if (globalDist[idx2(k, i, n)] < INF && globalDist[idx2(j, k, n)] < INF && globalDist[idx2(k, i, n)] + globalDist[idx2(j, k, n)] < globalDist[idx2(j, i, n)]) ok = false;
            std::printf("Validating result...\nValidation: %s\n", ok ? "PASSED" : "FAILED");
            MPI_Finalize(); return ok ? 0 : 1;
        }
    }
    MPI_Finalize(); return 0;
}
