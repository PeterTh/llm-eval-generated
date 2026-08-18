#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// The benchmark's public representation is column-major: element (source,destination)
// is at destination*n + source.  Keeping it avoids changing result hashes.
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

#define CUDA_CHECK(call) do { const cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)
#define MPI_CHECK(call) do { const int e = (call); if (e != MPI_SUCCESS) { \
    std::fprintf(stderr, "MPI failure at %s:%d\n", __FILE__, __LINE__); MPI_Abort(MPI_COMM_WORLD, e); } } while (0)

__global__ void relaxKernel(unsigned int* dist, unsigned int* path,
                            const unsigned int* pivotRow, size_t n,
                            size_t firstSource, size_t sourceCount, size_t k) {
    const size_t i = firstSource + blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= firstSource + sourceCount || j >= n) return;
    const size_t ij = j * n + i;
    const unsigned int candidate = dist[k * n + i] + pivotRow[j];
    if (candidate < dist[ij]) { dist[ij] = candidate; path[ij] = static_cast<unsigned int>(k); }
}

__global__ void packPivotKernel(const unsigned int* dist, unsigned int* row, size_t n, size_t k) {
    const size_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n) row[j] = dist[j * n + k];
}

// Pack assigned source rows into a contiguous row-major MPI message.
__global__ void packRowsKernel(const unsigned int* matrix, unsigned int* packed,
                               size_t n, size_t firstSource, size_t sourceCount) {
    const size_t localI = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t j = blockIdx.y * blockDim.y + threadIdx.y;
    if (localI < sourceCount && j < n)
        packed[localI * n + j] = matrix[j * n + firstSource + localI];
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n) {
    unsigned int seed = 42;
    constexpr double range = static_cast<double>(MAX_DISTANCE);
    // rand_r has a dependence on seed, so generate the exact original sequence serially.
    for (size_t q = 0; q < n * n; ++q)
        dist[q] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(n); ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    #pragma omp parallel for schedule(static)
    for (long long jj = 0; jj < static_cast<long long>(n); ++jj) {
        const size_t j = static_cast<size_t>(jj);
        for (size_t i = 0; i < n; ++i) {
            path[idx2(i, j, n)] = static_cast<unsigned int>(j);
            path[idx2(j, i, n)] = static_cast<unsigned int>(i);
        }
        path[idx2(j, j, n)] = static_cast<unsigned int>(j);
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) if (dist[idx2(i, i, n)] != 0) return false;
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[idx2(k, i, n)] < INF && dist[idx2(j, k, n)] < INF &&
                    dist[idx2(k, i, n)] + dist[idx2(j, k, n)] < dist[idx2(j, i, n)]) return false;
    return true;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of nodes (default: 512)\n  -v        Validate\n  -r        Print results\n  -h        Help\n", p);
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank, ranks; MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank)); MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t n = 512; bool validate = false, printResults = false;
    for (int a = 1; a < argc; ++a) {
        if (!std::strcmp(argv[a], "-n") && a + 1 < argc) n = std::strtoull(argv[++a], nullptr, 10);
        else if (!std::strcmp(argv[a], "-v")) validate = true;
        else if (!std::strcmp(argv[a], "-r")) printResults = true;
        else if (!std::strcmp(argv[a], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > std::numeric_limits<int>::max()) { if (!rank) std::fprintf(stderr, "Invalid node count\n"); MPI_Finalize(); return 1; }
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(rank % devices));

    const size_t first = n * static_cast<size_t>(rank) / ranks;
    const size_t last = n * static_cast<size_t>(rank + 1) / ranks;
    const size_t count = last - first, elements = n * n;
    std::vector<unsigned int> dist(elements), path(elements), pivot(n), packed(count * n);
    initializeDistanceMatrix(dist, n); initializePathMatrix(path, n);
    unsigned int *dDist, *dPath, *dPivot, *dPacked;
    CUDA_CHECK(cudaMalloc(&dDist, elements * sizeof(unsigned int))); CUDA_CHECK(cudaMalloc(&dPath, elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPivot, n * sizeof(unsigned int))); CUDA_CHECK(cudaMalloc(&dPacked, std::max(size_t(1), count * n) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(dDist, dist.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, path.data(), elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    const dim3 threads(32, 8), grid((count + threads.x - 1) / threads.x, (n + threads.y - 1) / threads.y), rowGrid((n + 255) / 256);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD)); const double start = MPI_Wtime();
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>((k * ranks) / n);
        if (rank == owner) { packPivotKernel<<<rowGrid, 256>>>(dDist, dPivot, n, k); CUDA_CHECK(cudaMemcpy(pivot.data(), dPivot, n * sizeof(unsigned int), cudaMemcpyDeviceToHost)); }
        MPI_CHECK(MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(dPivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (count) relaxKernel<<<grid, threads>>>(dDist, dPath, dPivot, n, first, count, k);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    }
    double elapsed = MPI_Wtime() - start, maxElapsed = 0; MPI_CHECK(MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (count) { packRowsKernel<<<grid, threads>>>(dDist, dPacked, n, first, count); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaMemcpy(packed.data(), dPacked, count * n * sizeof(unsigned int), cudaMemcpyDeviceToHost)); }
    std::vector<int> counts(ranks), offsets(ranks); for (int r = 0; r < ranks; ++r) { counts[r] = static_cast<int>((n * size_t(r + 1) / ranks - n * size_t(r) / ranks) * n); offsets[r] = static_cast<int>((n * size_t(r) / ranks) * n); }
    std::vector<unsigned int> gathered; if (!rank) gathered.resize(elements);
    MPI_CHECK(MPI_Gatherv(packed.data(), static_cast<int>(count * n), MPI_UNSIGNED, rank ? nullptr : gathered.data(), counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    if (!rank) {
        #pragma omp parallel for schedule(static)
        for (long long ii = 0; ii < static_cast<long long>(n); ++ii) for (size_t j = 0; j < n; ++j) dist[idx2(ii, j, n)] = gathered[static_cast<size_t>(ii) * n + j];
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nMPI ranks: %d, CUDA devices per rank: 1\nComputation time: %.3f ms\nPerformance: %.3f GOPS\n", n, ranks, maxElapsed * 1000.0, (double(n) * n * n) / maxElapsed / 1e9);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) std::printf("Validation: %s\n", validateResult(dist, n) ? "PASSED" : "FAILED");
    }
    CUDA_CHECK(cudaFree(dPacked)); CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dDist));
    MPI_CHECK(MPI_Finalize()); return 0;
}
