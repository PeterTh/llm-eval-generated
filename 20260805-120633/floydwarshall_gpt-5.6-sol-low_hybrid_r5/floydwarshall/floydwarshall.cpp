#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

#define CUDA_CHECK(call) do {                                                     \
    cudaError_t e_ = (call);                                                      \
    if (e_ != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                     cudaGetErrorString(e_));                                     \
        MPI_Abort(MPI_COMM_WORLD, 2);                                             \
    }                                                                             \
} while (0)

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int lo, unsigned int hi) {
    // Keep the generator order identical to the original benchmark.
    unsigned int seed = 42;
    const double range = static_cast<double>(hi - lo) + 1.0;
    for (size_t p = 0; p < n * n; ++p)
        dist[p] = lo + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        for (long long j = 0; j < static_cast<long long>(n); ++j)
            path[idx2(j, i, n)] = static_cast<unsigned int>(j);
}

__global__ void update_rows(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ pivot,
                            size_t n, size_t rows, size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (i >= rows || j >= n) return;
    const size_t p = i * n + j;
    const unsigned int candidate = dist[i * n + k] + pivot[j];
    if (candidate < dist[p]) {
        dist[p] = candidate;
        path[p] = static_cast<unsigned int>(k);
    }
}

static int owner_of(size_t row, size_t n, int ranks) {
    const size_t q = n / static_cast<size_t>(ranks), r = n % static_cast<size_t>(ranks);
    if (row < (q + 1) * r) return static_cast<int>(row / (q + 1));
    return static_cast<int>(r + (row - (q + 1) * r) / q);
}

void distributedFloydWarshall(std::vector<unsigned int>& localDist,
                              std::vector<unsigned int>& localPath,
                              size_t n, size_t firstRow, int rank, int ranks) {
    const size_t rows = n ? localDist.size() / n : 0;
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    unsigned int* pivot = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, std::max<size_t>(1, localDist.size()) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max<size_t>(1, localPath.size()) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPivot, std::max<size_t>(1, n) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&pivot, std::max<size_t>(1, n) * sizeof(unsigned int)));
    if (!localDist.empty()) {
        CUDA_CHECK(cudaMemcpy(dDist, localDist.data(), localDist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, localPath.data(), localPath.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((n + block.x - 1) / block.x),
                    static_cast<unsigned int>((rows + block.y - 1) / block.y));
    for (size_t k = 0; k < n; ++k) {
        const int owner = owner_of(k, n, ranks);
        if (rank == owner) {
            const size_t localK = k - firstRow;
            CUDA_CHECK(cudaMemcpy(pivot, dDist + localK * n, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_Bcast(pivot, static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(dPivot, pivot, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (rows) {
            update_rows<<<grid, block>>>(dDist, dPath, dPivot, n, rows, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (!localDist.empty()) {
        CUDA_CHECK(cudaMemcpy(localDist.data(), dDist, localDist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), dPath, localPath.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaFreeHost(pivot));
    CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dDist));
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        valid &= (dist[idx2(i, i, n)] == 0);
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(std::min<size_t>(n, 10)); ++i)
        for (long long j = 0; j < static_cast<long long>(std::min<size_t>(n, 10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                const auto dij = dist[idx2(j, i, n)], dik = dist[idx2(k, i, n)], dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF) valid &= (dik + dkj >= dij);
            }
    return valid != 0;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num> Number of nodes (default: 512)\n  -v Validate\n  -r Print results\n  -h Show help\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 3);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank); MPI_Comm_free(&localComm);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "A CUDA GPU is required.\n"); MPI_Abort(MPI_COMM_WORLD, 4); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));

    size_t n = 512; bool validate = false, printResults = false; int parseError = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else parseError = 1;
    }
    if (parseError || !n || n > static_cast<size_t>(INT_MAX)) {
        if (!rank) { std::fprintf(stderr, "Invalid arguments (n must be in 1..INT_MAX).\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }
    if (static_cast<size_t>(ranks) > n) {
        if (!rank) std::fprintf(stderr, "Number of MPI ranks must not exceed number of nodes.\n");
        MPI_Finalize(); return 1;
    }

    const size_t base = n / ranks, extra = n % ranks;
    const size_t rows = base + (static_cast<size_t>(rank) < extra), first = rank * base + std::min<size_t>(rank, extra);
    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rr = base + (static_cast<size_t>(r) < extra);
        const size_t ff = r * base + std::min<size_t>(r, extra);
        if (rr * n > INT_MAX || ff * n > INT_MAX) { if (!rank) std::fprintf(stderr, "Matrix exceeds MPI count range.\n"); MPI_Abort(MPI_COMM_WORLD, 5); }
        counts[r] = static_cast<int>(rr * n); displs[r] = static_cast<int>(ff * n);
    }
    std::vector<unsigned int> dist, path;
    if (!rank) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\nInitializing graph...\n", n, validate ? "enabled" : "disabled", ranks, omp_get_max_threads());
        dist.resize(n * n); path.resize(n * n);
        initializeDistanceMatrix(dist, n, 1, MAX_DISTANCE); initializePathMatrix(path, n);
    }
    std::vector<unsigned int> localDist(rows * n), localPath(rows * n);
    MPI_Scatterv(rank ? nullptr : dist.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : path.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) std::printf("Computing shortest paths...\n");
    const double start = MPI_Wtime();
    distributedFloydWarshall(localDist, localPath, n, first, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const double elapsed = MPI_Wtime() - start;
    MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank ? nullptr : dist.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int ok = 0;
    if (!rank) {
        std::printf("Computation time: %ld ms\nPerformance: %.3f GOPS\n", static_cast<long>(elapsed * 1000.0), static_cast<double>(n) * n * n / elapsed / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            std::printf("Validating result...\n");
            ok = validateResult(dist, n) ? 0 : 1;
            std::printf("Validation: %s\n", ok ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize(); return ok;
}
