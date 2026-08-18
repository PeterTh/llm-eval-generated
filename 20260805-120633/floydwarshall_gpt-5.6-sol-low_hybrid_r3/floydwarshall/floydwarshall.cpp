#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

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

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t cuda_status_ = (call);                                             \
    if (cuda_status_ != cudaSuccess) {                                             \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                     cudaGetErrorString(cuda_status_));                            \
        MPI_Abort(MPI_COMM_WORLD, 2);                                              \
    }                                                                               \
} while (0)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int rangeMin, unsigned int rangeMax) {
    // Keep the original rand_r stream so result hashes remain exactly compatible.
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t p = 0; p < n * n; ++p)
        dist[p] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / RAND_MAX);
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
#pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (size_t j = 0; j < n; ++j) path[static_cast<size_t>(i) * n + j] = static_cast<unsigned int>(i);
}

__global__ void floydWarshallStep(unsigned int* __restrict__ dist,
                                  unsigned int* __restrict__ path,
                                  const unsigned int* __restrict__ pivot,
                                  size_t n, size_t localRows, unsigned int k) {
    const size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = localRows * n;
    if (p >= total) return;
    const size_t j = p % n;
    const size_t localI = p / n;
    const unsigned int candidate = dist[localI * n + k] + pivot[j];
    if (candidate < dist[p]) {
        dist[p] = candidate;
        path[p] = k;
    }
}

static void decomposition(size_t n, int ranks, std::vector<int>& counts,
                          std::vector<int>& displs, std::vector<size_t>& rowStarts) {
    counts.resize(ranks); displs.resize(ranks); rowStarts.resize(ranks + 1);
    const size_t base = n / static_cast<size_t>(ranks), extra = n % static_cast<size_t>(ranks);
    size_t row = 0;
    for (int r = 0; r < ranks; ++r) {
        rowStarts[r] = row;
        const size_t rows = base + (static_cast<size_t>(r) < extra);
        const size_t elements = rows * n;
        if (elements > INT_MAX || row * n > INT_MAX) {
            std::fprintf(stderr, "Matrix is too large for MPI collectives\n");
            MPI_Abort(MPI_COMM_WORLD, 3);
        }
        counts[r] = static_cast<int>(elements);
        displs[r] = static_cast<int>(row * n);
        row += rows;
    }
    rowStarts[ranks] = row;
}

void floydWarshallHybrid(std::vector<unsigned int>& localDist,
                         std::vector<unsigned int>& localPath, size_t n,
                         size_t firstRow, const std::vector<size_t>& rowStarts,
                         int rank, int ranks) {
    const size_t elements = localDist.size();
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, std::max<size_t>(elements, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max<size_t>(elements, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPivot, n * sizeof(unsigned int)));
    if (elements) {
        CUDA_CHECK(cudaMemcpy(dDist, localDist.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, localPath.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    unsigned int* pivot = nullptr;
    CUDA_CHECK(cudaMallocHost(&pivot, n * sizeof(unsigned int)));
    const int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);

    int owner = 0;
    for (size_t k = 0; k < n; ++k) {
        while (owner + 1 < ranks && k >= rowStarts[owner + 1]) ++owner;
        if (rank == owner) {
            const size_t localK = k - firstRow;
            CUDA_CHECK(cudaMemcpy(pivot, dDist + localK * n, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(pivot, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dPivot, pivot, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (elements) {
            floydWarshallStep<<<blocks, threads>>>(dDist, dPath, dPivot, n, elements / n,
                                                   static_cast<unsigned int>(k));
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (elements) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), dPath, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFreeHost(pivot)); CUDA_CHECK(cudaFree(dPivot));
    CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dDist));
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int failed = 0;
#pragma omp parallel for reduction(|:failed) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        failed |= (dist[idx2(i, i, n)] != 0);
    if (failed) { std::printf("Validation failed: a diagonal element is not zero\n"); return false; }
    const size_t sample = std::min(n, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) reduction(|:failed) schedule(static)
    for (long long i = 0; i < static_cast<long long>(sample); ++i)
        for (long long j = 0; j < static_cast<long long>(sample); ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int ik = dist[idx2(k, i, n)], kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < dist[idx2(j, i, n)]) failed = 1;
            }
    if (failed) std::printf("Validation failed: triangle inequality violated\n");
    return !failed;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of nodes (default: 512)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", p);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks; MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    // Select and initialize the node-local GPU before the measured region.  This
    // also prevents every rank from accidentally targeting device zero.
    MPI_Comm shared;
    int localRank = 0, deviceCount = 0;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    MPI_Comm_rank(shared, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr)); // eagerly create the CUDA context
    MPI_Comm_free(&shared);
    size_t n = 512; bool validate = false, printResults = false; int exitCode = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > UINT_MAX || n > INT_MAX) {
        if (!rank) std::fprintf(stderr, "Node count must be between 1 and %u\n", UINT_MAX);
        MPI_Finalize(); return 1;
    }
    if (!rank) std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
                           n, validate ? "enabled" : "disabled");

    std::vector<int> counts, displs; std::vector<size_t> starts;
    decomposition(n, ranks, counts, displs, starts);
    std::vector<unsigned int> dist, path;
    if (!rank) {
        dist.resize(n * n); path.resize(n * n);
        std::printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, n, 1, MAX_DISTANCE); initializePathMatrix(path, n);
    }
    std::vector<unsigned int> localDist(counts[rank]), localPath(counts[rank]);
    MPI_Scatterv(rank ? nullptr : dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (!rank) std::printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    floydWarshallHybrid(localDist, localPath, n, starts[rank], starts, rank, ranks);
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = MPI_Wtime() - start; double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank ? nullptr : dist.data(),
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), counts[rank], MPI_UNSIGNED, rank ? nullptr : path.data(),
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", seconds * 1000.0,
                    static_cast<double>(n) * n * n / seconds / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n"); const bool ok = validateResult(dist, n);
            std::printf("Validation: %s\n", ok ? "PASSED" : "FAILED"); exitCode = ok ? 0 : 1;
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return exitCode;
}
