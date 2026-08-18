#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t e_ = (call);                                                       \
    if (e_ != cudaSuccess) {                                                       \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                cudaGetErrorString(e_));                                           \
        MPI_Abort(MPI_COMM_WORLD, 2);                                               \
    }                                                                              \
} while (0)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int lo, unsigned int hi) {
    unsigned int seed = 42;
    const double range = static_cast<double>(hi - lo) + 1.0;
    for (size_t p = 0; p < n * n; ++p)
        dist[p] = lo + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j) path[static_cast<size_t>(i) * n + j] = static_cast<unsigned int>(j);
}

__global__ void relaxRows(unsigned int* __restrict__ dist,
                          unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ pivot,
                          size_t n, size_t localRows, unsigned int k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= localRows || j >= n) return;
    const size_t p = i * n + j;
    const unsigned int candidate = dist[i * n + k] + pivot[j];
    if (candidate < dist[p]) {
        dist[p] = candidate;
        path[p] = k;
    }
}

static void decomposition(size_t n, int ranks, std::vector<int>& rows,
                          std::vector<int>& counts, std::vector<int>& displs) {
    rows.resize(ranks); counts.resize(ranks); displs.resize(ranks);
    size_t off = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t nr = n / ranks + (static_cast<size_t>(r) < n % ranks);
        if (nr * n > static_cast<size_t>(std::numeric_limits<int>::max())) {
            fprintf(stderr, "Matrix is too large for MPI counts\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        rows[r] = static_cast<int>(nr);
        counts[r] = static_cast<int>(nr * n);
        displs[r] = static_cast<int>(off * n);
        off += nr;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int bad = 0;
#pragma omp parallel for reduction(|:bad) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        bad |= dist[idx2(i, i, n)] != 0;
    if (bad) return false;
    const size_t sample = std::min(n, size_t{10});
#pragma omp parallel for collapse(2) reduction(|:bad) schedule(static)
    for (long long i = 0; i < static_cast<long long>(sample); ++i)
        for (long long j = 0; j < static_cast<long long>(sample); ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int dik = dist[idx2(k, static_cast<size_t>(i), n)];
                const unsigned int dkj = dist[idx2(static_cast<size_t>(j), k, n)];
                if (dik < INF && dkj < INF && dik + dkj < dist[idx2(static_cast<size_t>(j), static_cast<size_t>(i), n)]) bad = 1;
            }
    return !bad;
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\n  -n <num>  Number of nodes (default: 512)\n"
           "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, printResults = false;
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "-n") && a + 1 < argc) n = strtoull(argv[++a], nullptr, 10);
        else if (!strcmp(argv[a], "-v")) validate = true;
        else if (!strcmp(argv[a], "-r")) printResults = true;
        else if (!strcmp(argv[a], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[a]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (!rank) fprintf(stderr, "Node count must be in [1, INT_MAX]\n");
        MPI_Finalize(); return 1;
    }

    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) fprintf(stderr, "A CUDA device is required\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    int localRank = 0; MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    MPI_Comm_rank(local, &localRank); MPI_Comm_free(&local);
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    std::vector<int> rows, counts, displs; decomposition(n, ranks, rows, counts, displs);
    const size_t localRows = static_cast<size_t>(rows[rank]);
    std::vector<unsigned int> dist, path;
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n"
               "MPI ranks: %d, OpenMP threads/rank: %d\nValidation: %s\nInitializing graph...\n",
               n, ranks, omp_get_max_threads(), validate ? "enabled" : "disabled");
        dist.resize(n * n); path.resize(n * n);
        initializeDistanceMatrix(dist, n, 1, MAX_DISTANCE); initializePathMatrix(path, n);
    }
    std::vector<unsigned int> localDist(localRows * n), localPath(localRows * n);
    MPI_Scatterv(rank ? nullptr : dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, std::max<size_t>(1, localDist.size()) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max<size_t>(1, localPath.size()) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPivot, n * sizeof(unsigned int)));
    if (!localDist.empty()) {
        CUDA_CHECK(cudaMemcpy(dDist, localDist.data(), localDist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, localPath.data(), localPath.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if (!rank) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD); const double begin = MPI_Wtime();
    const dim3 block(32, 8);
    size_t first = 0;
    for (int owner = 0; owner < ranks; ++owner) {
        const size_t last = first + static_cast<size_t>(rows[owner]);
        for (size_t k = first; k < last; ++k) {
            unsigned int* pivot = rank == owner ? dDist + (k - first) * n : dPivot;
            CUDA_CHECK(cudaDeviceSynchronize());
            MPI_Bcast(pivot, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            if (localRows) {
                const dim3 grid(static_cast<unsigned>((n + block.x - 1) / block.x),
                                static_cast<unsigned>((localRows + block.y - 1) / block.y));
                relaxRows<<<grid, block>>>(dDist, dPath, pivot, n, localRows, static_cast<unsigned int>(k));
                CUDA_CHECK(cudaGetLastError());
            }
        }
        first = last;
    }
    CUDA_CHECK(cudaDeviceSynchronize()); MPI_Barrier(MPI_COMM_WORLD);
    const double localSeconds = MPI_Wtime() - begin; double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!localDist.empty()) CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, localDist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank ? nullptr : dist.data(),
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dDist));

    int result = 0;
    if (!rank) {
        printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", seconds * 1e3,
               static_cast<double>(n) * n * n / seconds / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) { printf("Validating result...\n"); const bool ok = validateResult(dist, n);
            printf("Validation: %s\n", ok ? "PASSED" : "FAILED"); result = ok ? 0 : 1; }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return result;
}
