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
// A moderately large pivot amortizes MPI latency while retaining enough work per GPU kernel.
constexpr int PIVOT_BLOCK = 32;

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] static void fail(const char* where, const char* detail) {
    std::fprintf(stderr, "%s: %s\n", where, detail);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t status, const char* where) {
    if (status != cudaSuccess) fail(where, cudaGetErrorString(status));
}

// The local representation is row-major (source, destination), unlike the public
// representation used by this benchmark.  This makes one source row contiguous on
// the GPU and is converted back before output/validation.
__global__ void relaxRows(unsigned int* __restrict__ distance,
                          unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ pivotRow,
                          int rows, int columns, int k, int firstRow, int rowCount) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = rowCount * columns;
    if (linear >= total) return;

    const int localRow = firstRow + linear / columns;
    const int j = linear - (linear / columns) * columns;
    unsigned int* row = distance + static_cast<size_t>(localRow) * columns;
    const unsigned int candidate = row[k] + pivotRow[j];
    if (candidate < row[j]) {
        row[j] = candidate;
        path[static_cast<size_t>(localRow) * columns + j] = static_cast<unsigned int>(k);
    }
}

static void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n) {
    unsigned int seed = 42;
    const double range = static_cast<double>(MAX_DISTANCE);
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                const auto dij = dist[idx2(j, i, n)];
                const auto dik = dist[idx2(k, i, n)];
                const auto dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF && dik + dkj < dij) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
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
    if (n == 0 || n > static_cast<size_t>(std::numeric_limits<int>::max())) fail("invalid node count", "must fit MPI counts");

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (!deviceCount) fail("CUDA initialization", "no CUDA device available");
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    // Assign whole pivot blocks to ranks.  Thus every pivot block has one owner,
    // which permits one collective broadcast per block rather than per vertex.
    const size_t blockCount = (n + PIVOT_BLOCK - 1) / PIVOT_BLOCK;
    const size_t blocksBase = blockCount / ranks, blocksExtra = blockCount % ranks;
    const size_t localBlocks = blocksBase + (static_cast<size_t>(rank) < blocksExtra);
    const size_t firstBlock = static_cast<size_t>(rank) * blocksBase + std::min(static_cast<size_t>(rank), blocksExtra);
    const size_t rowStart = firstBlock * PIVOT_BLOCK;
    const size_t localRows = rowStart < n ? std::min(n - rowStart, localBlocks * static_cast<size_t>(PIVOT_BLOCK)) : 0;
    if (localRows * n > static_cast<size_t>(std::numeric_limits<int>::max())) fail("matrix too large", "local MPI count overflow");

    std::vector<int> counts(ranks), displs(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rBlocks = blocksBase + (static_cast<size_t>(r) < blocksExtra);
        const size_t rFirstBlock = static_cast<size_t>(r) * blocksBase + std::min(static_cast<size_t>(r), blocksExtra);
        const size_t rStart = rFirstBlock * PIVOT_BLOCK;
        const size_t rows = rStart < n ? std::min(n - rStart, rBlocks * static_cast<size_t>(PIVOT_BLOCK)) : 0;
        counts[r] = static_cast<int>(rows * n);
        displs[r] = r ? displs[r - 1] + counts[r - 1] : 0;
    }
    std::vector<unsigned int> global; // Original benchmark layout, root only.
    std::vector<unsigned int> packed;
    if (!rank) {
        global.resize(n * n);
        initializeDistanceMatrix(global, n);
        packed.resize(n * n);
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            for (size_t j = 0; j < n; ++j) packed[static_cast<size_t>(i) * n + j] = global[idx2(j, static_cast<size_t>(i), n)];
    }
    std::vector<unsigned int> localDist(localRows * n), localPath(localRows * n);
    MPI_Scatterv(rank ? nullptr : packed.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(localRows); ++i)
        for (size_t j = 0; j < n; ++j) localPath[static_cast<size_t>(i) * n + j] = static_cast<unsigned int>(rowStart + i);

    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    cudaCheck(cudaMalloc(&dDist, std::max<size_t>(1, localDist.size()) * sizeof(unsigned int)), "cudaMalloc(distance)");
    cudaCheck(cudaMalloc(&dPath, std::max<size_t>(1, localPath.size()) * sizeof(unsigned int)), "cudaMalloc(path)");
    cudaCheck(cudaMalloc(&dPivot, static_cast<size_t>(PIVOT_BLOCK) * n * sizeof(unsigned int)), "cudaMalloc(pivot)");
    if (!localDist.empty()) {
        cudaCheck(cudaMemcpy(dDist, localDist.data(), localDist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy distance");
        cudaCheck(cudaMemcpy(dPath, localPath.data(), localPath.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy path");
    }

    if (!rank) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nMPI ranks: %d, OpenMP threads/rank: %d\nValidation: %s\nComputing shortest paths...\n", n, ranks, omp_get_max_threads(), validate ? "enabled" : "disabled");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    std::vector<unsigned int> pivot(static_cast<size_t>(PIVOT_BLOCK) * n);
    constexpr int threads = 256;
    for (size_t block = 0; block < n; block += PIVOT_BLOCK) {
        const int blockRows = static_cast<int>(std::min(static_cast<size_t>(PIVOT_BLOCK), n - block));
        int blockOwner = 0;
        size_t ownerFirstBlock = 0;
        for (; blockOwner < ranks; ++blockOwner) {
            const size_t owned = blocksBase + (static_cast<size_t>(blockOwner) < blocksExtra);
            if (block / PIVOT_BLOCK < ownerFirstBlock + owned) break;
            ownerFirstBlock += owned;
        }
        if (rank == blockOwner) {
            const int localPivot = static_cast<int>(block - rowStart);
            // Phase 1: close the diagonal pivot block.  The sequential launches
            // retain the k dependency while each launch exposes all rows to CUDA.
            for (int q = 0; q < blockRows; ++q) {
                const int work = blockRows * static_cast<int>(n);
                relaxRows<<<(work + threads - 1) / threads, threads>>>(dDist, dPath,
                    dDist + static_cast<size_t>(localPivot + q) * n, static_cast<int>(localRows), static_cast<int>(n),
                    static_cast<int>(block + q), localPivot, blockRows);
                cudaCheck(cudaGetLastError(), "pivot block kernel");
            }
            cudaCheck(cudaMemcpy(pivot.data(), dDist + static_cast<size_t>(localPivot) * n,
                                 static_cast<size_t>(blockRows) * n * sizeof(unsigned int), cudaMemcpyDeviceToHost), "download pivot block");
        }
        MPI_Bcast(pivot.data(), blockRows * static_cast<int>(n), MPI_UNSIGNED, blockOwner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(dPivot, pivot.data(), static_cast<size_t>(blockRows) * n * sizeof(unsigned int), cudaMemcpyHostToDevice), "upload pivot block");
        // Phase 2: update all local source rows using the closed pivot block.
        for (int q = 0; q < blockRows; ++q) {
            const int work = static_cast<int>(localRows * n);
            if (work) {
                relaxRows<<<(work + threads - 1) / threads, threads>>>(dDist, dPath, dPivot + static_cast<size_t>(q) * n,
                    static_cast<int>(localRows), static_cast<int>(n), static_cast<int>(block + q), 0, static_cast<int>(localRows));
                cudaCheck(cudaGetLastError(), "distributed relax kernel");
            }
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "CUDA synchronization");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();
    const double localSeconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(localDist.data(), dDist, localDist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "download distance");
    cudaFree(dDist); cudaFree(dPath); cudaFree(dPivot);

    if (!rank) packed.resize(n * n);
    MPI_Gatherv(localDist.data(), static_cast<int>(localDist.size()), MPI_UNSIGNED, rank ? nullptr : packed.data(),
                counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int validResult = 1;
    if (!rank) {
        #pragma omp parallel for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i)
            for (size_t j = 0; j < n; ++j) global[idx2(j, static_cast<size_t>(i), n)] = packed[static_cast<size_t>(i) * n + j];
        const auto milliseconds = static_cast<long long>(seconds * 1000.0);
        std::printf("Computation time: %lld ms\nPerformance: %.3f GOPS\n", static_cast<long long>(milliseconds),
                    seconds > 0.0 ? static_cast<double>(n) * n * n / seconds / 1e9 : 0.0);
        if (printResults) print_results_int(global, "DistanceMatrix");
        if (validate) {
            validResult = validateResult(global, n) ? 1 : 0;
            std::printf("Validation: %s\n", validResult ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&validResult, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return validResult ? 0 : 1;
}
