#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <vector>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept { return j * n + i; }

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n, unsigned int lo, unsigned int hi) {
    unsigned int seed = 42;
    const double range = static_cast<double>(hi - lo) + 1.0;
    for (size_t x = 0; x < n * n; ++x)
        dist[x] = lo + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    for (size_t j = 0; j < n; ++j) for (size_t i = 0; i < n; ++i) path[idx2(i, j, n)] = j;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) if (dist[idx2(i, i, n)] != 0) {
        printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i); return false;
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int a = dist[idx2(k, i, n)], b = dist[idx2(j, k, n)];
                if (a < INF && b < INF && a + b < dist[idx2(j, i, n)]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k); return false;
                }
            }
    return true;
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of nodes in the graph (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, printResults = false; int bad = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = static_cast<size_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } bad = 1; }
    }
    if (n == 0 || n > static_cast<size_t>(INT_MAX) || n*n > static_cast<size_t>(INT_MAX)) bad = 1;
    int anyBad; MPI_Allreduce(&bad, &anyBad, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (anyBad) { MPI_Finalize(); return 1; }
    if (rank == 0) { printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nInitializing graph...\n", n, validate ? "enabled" : "disabled"); }

    // Source-node slabs are distributed in contiguous blocks; matrix entries within each slab
    // retain the original destination-major layout.
    const size_t base = n / ranks, rem = n % ranks;
    const size_t first = rank * base + std::min<size_t>(rank, rem);
    const size_t count = base + (static_cast<size_t>(rank) < rem);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t c = base + (static_cast<size_t>(r) < rem);
        const size_t f = static_cast<size_t>(r) * base + std::min<size_t>(r, rem);
        counts[r] = static_cast<int>(c * n); displs[r] = static_cast<int>(f * n);
    }
    std::vector<unsigned int> fullDist, fullPath, packedDist, packedPath;
    if (rank == 0) {
        fullDist.resize(n*n); fullPath.resize(n*n); initializeDistanceMatrix(fullDist, n, 1, MAX_DISTANCE); initializePathMatrix(fullPath, n);
        packedDist.resize(n*n); packedPath.resize(n*n);
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) {
            packedDist[i*n+j] = fullDist[idx2(j,i,n)]; packedPath[i*n+j] = fullPath[idx2(j,i,n)];
        }
    }
    std::vector<unsigned int> dist(count*n), path(count*n), pivot(n);
    MPI_Scatterv(rank == 0 ? packedDist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 dist.data(), static_cast<int>(count*n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? packedPath.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 path.data(), static_cast<int>(count*n), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k < rem * (base + 1)
            ? k / (base + 1) : rem + (k - rem * (base + 1)) / base);
        if (rank == owner) {
            const size_t local = k - first;
            for (size_t j = 0; j < n; ++j) pivot[j] = dist[j*count + local];
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        for (size_t i = 0; i < count; ++i) {
            const unsigned int dik = dist[k*count+i];
            for (size_t j = 0; j < n; ++j) {
                const unsigned int candidate = dik + pivot[j];
                unsigned int& current = dist[j*count+i];
                if (candidate < current) { current = candidate; path[j*count+i] = static_cast<unsigned int>(k); }
            }
        }
    }
    const double elapsed = MPI_Wtime() - start;
    std::vector<unsigned int> gatheredDist, gatheredPath;
    if (rank == 0) { gatheredDist.resize(n*n); gatheredPath.resize(n*n); }
    MPI_Gatherv(dist.data(), static_cast<int>(count*n), MPI_UNSIGNED, rank == 0 ? packedDist.data() : nullptr,
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(path.data(), static_cast<int>(count*n), MPI_UNSIGNED, rank == 0 ? packedPath.data() : nullptr,
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) fullDist[idx2(j,i,n)] = packedDist[i*n+j];
        printf("Computation time: %ld ms\n", static_cast<long>(elapsed * 1000.0));
        const double ops = static_cast<double>(n)*n*n;
        printf("Performance: %.3f GOPS\n", ops / elapsed / 1e9);
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) { printf("Validating result...\n"); bad = !validateResult(fullDist,n); printf("Validation: %s\n", bad ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&bad, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return bad;
}
