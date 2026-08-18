#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

#define CUDA_CHECK(call) do {                                                    \
    cudaError_t e_ = (call);                                                     \
    if (e_ != cudaSuccess) {                                                     \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(e_));                                    \
        MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                            \
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
    for (long long row = 0; row < static_cast<long long>(n); ++row)
        for (size_t col = 0; col < n; ++col)
            // This is the final value produced by the original symmetric writes.
            path[static_cast<size_t>(row) * n + col] =
                static_cast<unsigned int>(std::max(static_cast<size_t>(row), col));
}

__global__ void relaxKernel(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ pivot,
                            size_t elements, size_t n, size_t k) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= elements) return;
    const size_t localRow = p / n;
    const size_t j = p - localRow * n;
    const unsigned int candidate = dist[localRow * n + k] + pivot[j];
    if (candidate < dist[p]) {
        dist[p] = candidate;
        path[p] = static_cast<unsigned int>(k);
    }
}

static int ownerOfRow(size_t row, size_t n, int ranks) {
    const size_t q = n / static_cast<size_t>(ranks), rem = n % static_cast<size_t>(ranks);
    if (row < (q + 1) * rem) return static_cast<int>(row / (q + 1));
    return static_cast<int>(rem + (row - (q + 1) * rem) / q);
}

void floydWarshall(std::vector<unsigned int>& localDist,
                   std::vector<unsigned int>& localPath, size_t n,
                   size_t firstRow, int rank, int ranks) {
    int localRank = 0;
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    MPI_Comm_rank(localComm, &localRank);
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) std::fprintf(stderr, "This benchmark requires CUDA devices.\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    const size_t elements = localDist.size();
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPivot, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dDist, localDist.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, localPath.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));

    std::vector<unsigned int> pivot(n);
    constexpr int threads = 256;
    const int blocks = static_cast<int>((elements + threads - 1) / threads);
    for (size_t k = 0; k < n; ++k) {
        const int owner = ownerOfRow(k, n, ranks);
        if (rank == owner) {
            const size_t localK = k - firstRow;
            CUDA_CHECK(cudaMemcpy(pivot.data(), dDist + localK * n,
                                  n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dPivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (elements) relaxKernel<<<blocks, threads>>>(dDist, dPath, dPivot, elements, n, k);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(localPath.data(), dPath, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dDist));
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    for (size_t i = 0; i < std::min(n, size_t{10}); ++i)
        for (size_t j = 0; j < std::min(n, size_t{10}); ++j)
            for (size_t k = 0; k < n; ++k) {
                const auto dij = dist[idx2(j, i, n)], dik = dist[idx2(k, i, n)], dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF && dik + dkj < dij) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(INT_MAX) || n > static_cast<size_t>(INT_MAX) / n) {
        if (!rank) std::fprintf(stderr, "Matrix size is invalid or exceeds MPI count limits.\n");
        MPI_Finalize(); return 1;
    }
    if (static_cast<size_t>(ranks) > n) {
        if (!rank) std::fprintf(stderr, "Number of MPI ranks must not exceed number of nodes.\n");
        MPI_Finalize(); return 1;
    }
    if (!rank) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n", n);
        std::printf("Validation: %s\nInitializing graph...\n", validate ? "enabled" : "disabled");
    }
    std::vector<int> counts(ranks), offsets(ranks);
    size_t rowOffset = 0;
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = n / ranks + (static_cast<size_t>(r) < n % ranks);
        counts[r] = static_cast<int>(rows * n); offsets[r] = static_cast<int>(rowOffset * n); rowOffset += rows;
    }
    const size_t localRows = static_cast<size_t>(counts[rank]) / n;
    const size_t firstRow = static_cast<size_t>(offsets[rank]) / n;
    std::vector<unsigned int> dist, path;
    if (!rank) { dist.resize(n * n); path.resize(n * n); initializeDistanceMatrix(dist, n, 1, MAX_DISTANCE); initializePathMatrix(path, n); }
    std::vector<unsigned int> localDist(counts[rank]), localPath(counts[rank]);
    MPI_Scatterv(rank ? nullptr : dist.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : path.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (!rank) std::printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    floydWarshall(localDist, localPath, n, firstRow, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD); const double elapsed = MPI_Wtime() - start;
    MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank ? nullptr : dist.data(),
                counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), counts[rank], MPI_UNSIGNED, rank ? nullptr : path.data(),
                counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int result = 0;
    if (!rank) {
        const long ms = static_cast<long>(elapsed * 1000.0);
        std::printf("Computation time: %ld ms\nPerformance: %.3f GOPS\n", ms,
                    static_cast<double>(n) * n * n / elapsed / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) { std::printf("Validating result...\n"); result = validateResult(dist, n) ? 0 : 1; std::printf("Validation: %s\n", result ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return result;
}
