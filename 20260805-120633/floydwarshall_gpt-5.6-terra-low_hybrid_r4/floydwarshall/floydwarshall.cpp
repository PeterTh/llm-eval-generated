#include <algorithm>
#include <chrono>
#include <climits>
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

// The original program stores a matrix entry (source i, destination j) at j*n+i.
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

#define CUDA_CHECK(call) do {                                                        \
    const cudaError_t error = (call);                                                \
    if (error != cudaSuccess) {                                                      \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                     cudaGetErrorString(error));                                     \
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));                          \
    }                                                                                 \
} while (0)

// One MPI rank owns a contiguous set of destination rows.  The pivot row is
// replicated on every GPU, so all entries in locally owned rows are independent.
__global__ void relax_pivot(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ pivot,
                            const size_t entries, const size_t n,
                            const unsigned int k) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (element >= entries) return;

    const size_t i = element % n;
    const unsigned int candidate = pivot[i] + dist[(element / n) * n + k];
    if (candidate < dist[element]) {
        dist[element] = candidate;
        path[element] = k;
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n) {
    unsigned int seed = 42;
    constexpr double range = static_cast<double>(MAX_DISTANCE);
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t n) {
    // Each entry is assigned independently, preserving the original final value.
    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(n); ++j)
        for (size_t i = 0; i < n; ++i)
            path[idx2(i, static_cast<size_t>(j), n)] = static_cast<unsigned int>(j);
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
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
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) n = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (n == 0 || n > static_cast<size_t>(INT_MAX) || n * n > static_cast<size_t>(INT_MAX)) {
        if (!rank) std::fprintf(stderr, "Number of nodes is out of range for MPI counts\n");
        MPI_Finalize(); return 1;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) { if (!rank) std::fprintf(stderr, "No CUDA device is available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const size_t baseRows = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder);
    const size_t localStart = static_cast<size_t>(rank) * baseRows + std::min(static_cast<size_t>(rank), remainder);
    const size_t localEntries = localRows * n;
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < remainder);
        const size_t start = static_cast<size_t>(r) * baseRows + std::min(static_cast<size_t>(r), remainder);
        counts[r] = static_cast<int>(rows * n);
        displacements[r] = static_cast<int>(start * n);
    }

    std::vector<unsigned int> globalDist, globalPath;
    if (!rank) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
        globalDist.resize(n * n); globalPath.resize(n * n);
        initializeDistanceMatrix(globalDist, n);
        initializePathMatrix(globalPath, n);
    }
    std::vector<unsigned int> localDist(localEntries), localPath(localEntries), pivot(n);
    MPI_Scatterv(rank ? nullptr : globalDist.data(), counts.data(), displacements.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localEntries), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank ? nullptr : globalPath.data(), counts.data(), displacements.data(), MPI_UNSIGNED,
                 localPath.data(), static_cast<int>(localEntries), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int *deviceDist, *devicePath, *devicePivot;
    CUDA_CHECK(cudaMalloc(&deviceDist, localEntries * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, localEntries * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePivot, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemcpy(deviceDist, localDist.data(), localEntries * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devicePath, localPath.data(), localEntries * sizeof(unsigned int), cudaMemcpyHostToDevice));

    if (!rank) std::printf("Computing shortest paths with MPI + OpenMP + CUDA (%d ranks)...\n", ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    constexpr int threads = 256;
    for (size_t k = 0; k < n; ++k) {
        const int owner = static_cast<int>(k < (baseRows + 1) * remainder ? k / (baseRows + 1) : remainder + (k - (baseRows + 1) * remainder) / baseRows);
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(pivot.data(), deviceDist + (k - localStart) * n, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        MPI_Bcast(pivot.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(devicePivot, pivot.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (localEntries) {
            const int blocks = static_cast<int>((localEntries + threads - 1) / threads);
            relax_pivot<<<blocks, threads>>>(deviceDist, devicePath, devicePivot, localEntries, n, static_cast<unsigned int>(k));
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaMemcpy(localDist.data(), deviceDist, localEntries * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(localPath.data(), devicePath, localEntries * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    MPI_Gatherv(localDist.data(), static_cast<int>(localEntries), MPI_UNSIGNED, rank ? nullptr : globalDist.data(), counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), static_cast<int>(localEntries), MPI_UNSIGNED, rank ? nullptr : globalPath.data(), counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    const long localMilliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    long milliseconds = 0;
    MPI_Reduce(&localMilliseconds, &milliseconds, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(devicePivot)); CUDA_CHECK(cudaFree(devicePath)); CUDA_CHECK(cudaFree(deviceDist));
    if (!rank) {
        std::printf("Computation time: %ld ms\nPerformance: %.3f GOPS\n", milliseconds,
                    milliseconds ? static_cast<double>(n) * n * n / milliseconds / 1.0e6 : 0.0);
        if (printResults) print_results_int(globalDist, "DistanceMatrix");
        if (validate) std::printf("Validation: %s\n", validateResult(globalDist, n) ? "PASSED" : "FAILED");
    }
    const bool valid = !validate || rank != 0 || validateResult(globalDist, n);
    MPI_Finalize();
    return valid ? 0 : 1;
}
