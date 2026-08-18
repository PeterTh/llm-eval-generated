#include <algorithm>
#include <chrono>
#include <cmath>
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

#define CUDA_CHECK(call) do { \
    const cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA failure at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        MPI_Abort(MPI_COMM_WORLD, 2); \
    } \
} while (0)

// The benchmark's external representation is column-major: element (source i,
// destination j) is stored at j*n+i.  A rank owns complete source columns.
__global__ void relaxColumns(unsigned int* dist, unsigned int* path,
                             const unsigned int* pivot, size_t nodes,
                             size_t localColumns, size_t k) {
    const size_t j = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t localI = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (j >= nodes || localI >= localColumns) return;

    const size_t pos = localI * nodes + j;
    const unsigned int candidate = dist[localI * nodes + k] + pivot[j];
    if (candidate < dist[pos]) {
        dist[pos] = candidate;
        path[pos] = static_cast<unsigned int>(k);
    }
}

static void initializeLocal(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                            size_t nodes, size_t firstColumn) {
#pragma omp parallel for schedule(static)
    for (size_t p = 0; p < path.size(); ++p) {
        const size_t localI = p / nodes;
        const size_t i = firstColumn + localI;
        path[p] = static_cast<unsigned int>(i);
    }
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[i * n + i] != 0) { std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i); return false; }
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[i*n+k] < INF && dist[k*n+j] < INF && dist[i*n+k] + dist[k*n+j] < dist[i*n+j]) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k); return false;
                }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t n = 512; bool validate = false, printResults = false;
    for (int a = 1; a < argc; ++a) {
        if (!std::strcmp(argv[a], "-n") && a + 1 < argc) n = std::strtoull(argv[++a], nullptr, 10);
        else if (!std::strcmp(argv[a], "-v")) validate = true;
        else if (!std::strcmp(argv[a], "-r")) printResults = true;
        else if (!std::strcmp(argv[a], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[a]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (!n || n > static_cast<size_t>(std::numeric_limits<int>::max())) { if (!rank) std::fprintf(stderr, "Invalid node count\n"); MPI_Finalize(); return 1; }

    if (n > static_cast<size_t>(std::sqrt(static_cast<double>(std::numeric_limits<int>::max())))) { if (!rank) std::fprintf(stderr, "Node count is too large for MPI collectives\n"); MPI_Finalize(); return 1; }
    MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank; MPI_Comm_rank(local, &localRank);
    int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { if (!rank) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 2); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);

    const size_t first = n * static_cast<size_t>(rank) / ranks;
    const size_t last = n * static_cast<size_t>(rank + 1) / ranks;
    const size_t localCols = last - first, localCount = localCols * n;
    const size_t allocationCount = std::max<size_t>(localCount, 1);
    std::vector<unsigned int> hostDist(localCount), hostPath(localCount), pivot(n);
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        counts[r] = static_cast<int>((n*static_cast<size_t>(r+1)/ranks-n*static_cast<size_t>(r)/ranks)*n);
        offsets[r] = static_cast<int>((n*static_cast<size_t>(r)/ranks)*n);
    }
    std::vector<unsigned int> initial;
    if (!rank) {
        initial.resize(n*n);
        unsigned int seed = 42;
        for (size_t p = 0; p < n*n; ++p)
            initial[p] = 1U + static_cast<unsigned int>(200.0 * rand_r(&seed) / static_cast<double>(RAND_MAX));
        for (size_t i = 0; i < n; ++i) initial[i*n+i] = 0;
    }
    MPI_Scatterv(initial.data(), counts.data(), offsets.data(), MPI_UNSIGNED,
                 hostDist.data(), static_cast<int>(localCount), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    initializeLocal(hostDist, hostPath, n, first);
    unsigned int *deviceDist, *devicePath, *devicePivot;
    CUDA_CHECK(cudaMalloc(&deviceDist, allocationCount * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, allocationCount * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePivot, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(deviceDist, hostDist.data(), localCount*sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, hostPath.data(), localCount*sizeof(unsigned int), cudaMemcpyHostToDevice));

    if (!rank) { std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\nComputing shortest paths...\n", n, validate ? "enabled" : "disabled"); }
    MPI_Barrier(MPI_COMM_WORLD); const double start = MPI_Wtime();
    const dim3 threads(32, 8);
    const dim3 blocks((n + threads.x - 1) / threads.x, std::max<size_t>(1, (localCols + threads.y - 1) / threads.y));
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(((k + 1) * static_cast<size_t>(ranks) - 1) / n);
        if (rank == owner) CUDA_CHECK(cudaMemcpy(pivot.data(), deviceDist + (k - first)*n, n*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(devicePivot, pivot.data(), n*sizeof(unsigned int), cudaMemcpyHostToDevice));
        relaxColumns<<<blocks, threads>>>(deviceDist, devicePath, devicePivot, n, localCols, k);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maxElapsed = 0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(hostDist.data(), deviceDist, localCount*sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> full;
    if (!rank) full.resize(n*n);
    MPI_Gatherv(hostDist.data(), static_cast<int>(localCount), MPI_UNSIGNED, full.data(), counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", maxElapsed*1000.0, static_cast<double>(n)*n*n/maxElapsed/1e9);
        if (printResults) print_results_int(full, "DistanceMatrix");
        if (validate) std::printf("Validation: %s\n", validateResult(full, n) ? "PASSED" : "FAILED");
    }
    CUDA_CHECK(cudaFree(devicePivot)); CUDA_CHECK(cudaFree(devicePath)); CUDA_CHECK(cudaFree(deviceDist));
    MPI_Finalize();
    return 0;
}
