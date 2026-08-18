#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call) do { \
    const cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error__)); \
    } \
} while (false)

__global__ void floyd_update_k(unsigned int* dist, unsigned int* path,
                               const unsigned int* pivot, const size_t n,
                               const size_t localColumns, const size_t k) {
    __shared__ unsigned int pivotTile[32];
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t column = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (threadIdx.x < 32 && row < n)
        pivotTile[threadIdx.x] = pivot[row];
    __syncthreads();

    if (row < n && column < localColumns) {
        const size_t offset = column * n + row;
        const unsigned int candidate = dist[column * n + k] + pivotTile[threadIdx.x];
        if (candidate < dist[offset]) {
            dist[offset] = candidate;
            path[offset] = static_cast<unsigned int>(k);
        }
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t n,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    // rand_r is stateful, so retain the original sequential stream exactly.
    for (size_t i = 0; i < dist.size(); ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i)
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t n,
                          const size_t firstColumn, const size_t localColumns) {
    #pragma omp parallel for schedule(static)
    for (long long c = 0; c < static_cast<long long>(localColumns); ++c) {
        const size_t globalColumn = firstColumn + static_cast<size_t>(c);
        for (size_t row = 0; row < n; ++row)
            path[c * n + row] = static_cast<unsigned int>(globalColumn);
    }
}

size_t ownerOf(const size_t k, const size_t n, const int ranks) {
    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    const size_t large = (base + 1) * remainder;
    if (k < large) return k / (base + 1);
    return static_cast<size_t>(remainder) + (k - large) / base;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    const size_t sample = std::min(n, static_cast<size_t>(10));
    bool valid = true;
    #pragma omp parallel for collapse(2) schedule(static)
    for (long long ii = 0; ii < static_cast<long long>(sample); ++ii) {
        for (long long jj = 0; jj < static_cast<long long>(sample); ++jj) {
            for (size_t k = 0; k < n; ++k) {
                const size_t i = static_cast<size_t>(ii), j = static_cast<size_t>(jj);
                const unsigned int ik = dist[idx2(k, i, n)];
                const unsigned int kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < dist[idx2(j, i, n)]) {
                    #pragma omp atomic write
                    valid = false;
                    #pragma omp critical
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                }
            }
        }
    }
    return valid;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>  Number of nodes (default: 512)\n  -v        Enable validation\n  -r        Print results for external validation\n  -h        Show this help message\n", name);
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = static_cast<size_t>(strtoull(argv[++i], nullptr, 10));
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || ranks > static_cast<int>(n)) {
        if (rank == 0) fprintf(stderr, "Number of nodes must be at least the MPI rank count.\n");
        MPI_Finalize(); return 1;
    }

    const size_t base = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    const size_t localColumns = base + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstColumn = static_cast<size_t>(rank) * base + std::min(static_cast<size_t>(rank), remainder);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t cols = base + (static_cast<size_t>(r) < remainder ? 1 : 0);
        counts[r] = static_cast<int>(cols * n);
        displacements[r] = static_cast<int>((static_cast<size_t>(r) * base + std::min(static_cast<size_t>(r), remainder)) * n);
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, CUDA: enabled\n", ranks, omp_get_max_threads());
    }
    std::vector<unsigned int> rootDist;
    if (rank == 0) { rootDist.resize(n * n); initializeDistanceMatrix(rootDist, n, 1, MAX_DISTANCE); }
    std::vector<unsigned int> dist(localColumns * n), path(localColumns * n);
    initializePathMatrix(path, n, firstColumn, localColumns);
    MPI_Scatterv(rank == 0 ? rootDist.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA device available on MPI rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, std::max<size_t>(1, dist.size()) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, std::max<size_t>(1, path.size()) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPivot, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), dist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), path.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    std::vector<unsigned int> pivot(n);
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (size_t k = 0; k < n; ++k) {
        const size_t owner = ownerOf(k, n, ranks);
        if (static_cast<size_t>(rank) == owner)
            #pragma omp parallel for schedule(static)
            for (long long row = 0; row < static_cast<long long>(n); ++row)
                pivot[static_cast<size_t>(row)] = dist[(k - firstColumn) * n + static_cast<size_t>(row)];
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, static_cast<int>(owner), MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(dPivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        dim3 block(32, 8);
        dim3 grid((static_cast<unsigned int>(n) + 31u) / 32u,
                  (static_cast<unsigned int>(localColumns) + 7u) / 8u);
        floyd_update_k<<<grid, block, 0, stream>>>(dDist, dPath, dPivot, n, localColumns, k);
        CUDA_CHECK(cudaGetLastError());
        if (k + 1 < n && ownerOf(k + 1, n, ranks) == static_cast<size_t>(rank)) {
            CUDA_CHECK(cudaMemcpyAsync(dist.data() + (k + 1 - firstColumn) * n,
                                       dDist + (k + 1 - firstColumn) * n,
                                       n * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    if (validate || printResults) CUDA_CHECK(cudaMemcpy(dist.data(), dDist, dist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> result;
    if (rank == 0 && (validate || printResults)) result.resize(n * n);
    MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? result.data() : nullptr,
                counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", elapsed * 1000.0,
               static_cast<double>(n) * n * n / elapsed / 1e9);
        if (printResults) print_results_int(result, "DistanceMatrix");
        if (validate) { printf("Validating result...\n"); const bool valid = validateResult(result, n); printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            CUDA_CHECK(cudaFree(dDist)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaStreamDestroy(stream)); MPI_Finalize(); return valid ? 0 : 1; }
    }
    CUDA_CHECK(cudaFree(dDist)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaStreamDestroy(stream));
    MPI_Finalize();
    return 0;
}
