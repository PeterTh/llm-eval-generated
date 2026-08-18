#include <algorithm>
#include <chrono>
#include <climits>
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

// The public/result layout of this benchmark is column-major.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] static void failMpi(const char* where, int error, MPI_Comm comm) {
    char text[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, text, &length);
    std::fprintf(stderr, "%s: %.*s\n", where, length, text);
    MPI_Abort(comm, error);
    std::abort();
}

static void checkCuda(cudaError_t error, const char* where, MPI_Comm comm) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        MPI_Abort(comm, static_cast<int>(error));
    }
}

#define MPI_CHECK(call) do { const int e_ = (call); if (e_ != MPI_SUCCESS) failMpi(#call, e_, MPI_COMM_WORLD); } while (false)
#define CUDA_CHECK(call) checkCuda((call), #call, MPI_COMM_WORLD)

// localDist and localPath are row-major because ranks own complete source rows.
// One launch is required for each k: Floyd-Warshall's k iterations are a true
// dependency chain, while all (i,j) updates in an iteration are independent.
__global__ void floydStep(unsigned int* __restrict__ localDist,
                          unsigned int* __restrict__ localPath,
                          const unsigned int* __restrict__ pivotRow,
                          const size_t localRows, const size_t n, const size_t k) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t total = localRows * n;
    if (element >= total) return;

    const size_t localI = element / n;
    const size_t j = element - localI * n;
    // d[i,k] and d[k,j] are invariant in iteration k (non-negative weights).
    // Skipping j == k also avoids a needless concurrent read/write of d[i,k].
    if (j == k) return;
    const size_t base = localI * n;
    const unsigned int candidate = localDist[base + k] + pivotRow[j];
    if (candidate < localDist[base + j]) {
        localDist[base + j] = candidate;
        localPath[base + j] = static_cast<unsigned int>(k);
    }
}

static void initializeDistanceMatrix(std::vector<unsigned int>& columnMajor, size_t n) {
    unsigned int seed = 42;
    for (size_t q = 0; q < n * n; ++q) {
        const double range = static_cast<double>(MAX_DISTANCE);
        columnMajor[q] = 1U + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
    for (size_t i = 0; i < n; ++i) columnMajor[idx2(i, i, n)] = 0;
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t{10}); ++i) {
        for (size_t j = 0; j < std::min(n, size_t{10}); ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int dij = dist[idx2(j, i, n)];
                const unsigned int dik = dist[idx2(k, i, n)];
                const unsigned int dkj = dist[idx2(j, k, n)];
                if (dik < INF && dkj < INF && dik + dkj < dij) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n", name);
    std::printf("Options:\n  -n <num>     Number of nodes (default: 512)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_CHECK(MPI_Init(&argc, &argv));
    int rank = 0, ranks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));

    size_t n = 512;
    bool validate = false, printResults = false;
    int argumentError = 0;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const unsigned long long parsed = std::strtoull(argv[++i], &end, 10);
            if (*end || parsed == 0 || parsed > std::numeric_limits<size_t>::max()) argumentError = 1;
            else n = static_cast<size_t>(parsed);
        } else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else argumentError = 1;
    }
    if (argumentError || n > static_cast<size_t>(INT_MAX) || n > static_cast<size_t>(INT_MAX) / n) {
        if (rank == 0) { std::printf("Invalid node count (MPI counts require n*n <= INT_MAX).\n"); printUsage(argv[0]); }
        MPI_Finalize();
        return 1;
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const size_t baseRows = n / static_cast<size_t>(ranks);
    const size_t remainder = n % static_cast<size_t>(ranks);
    const size_t localRows = baseRows + (static_cast<size_t>(rank) < remainder ? 1 : 0);
    const size_t firstRow = static_cast<size_t>(rank) * baseRows + std::min(static_cast<size_t>(rank), remainder);
    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t rows = baseRows + (static_cast<size_t>(r) < remainder ? 1 : 0);
        const size_t first = static_cast<size_t>(r) * baseRows + std::min(static_cast<size_t>(r), remainder);
        counts[r] = static_cast<int>(rows * n);
        displacements[r] = static_cast<int>(first * n);
    }

    std::vector<unsigned int> initialRows;
    if (rank == 0) {
        std::vector<unsigned int> columnMajor(n * n);
        initializeDistanceMatrix(columnMajor, n);
        initialRows.resize(n * n);
        // Preserve the exact old PRNG and external column-major layout.
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) initialRows[i * n + j] = columnMajor[idx2(j, i, n)];
    }
    std::vector<unsigned int> hostDist(localRows * n), hostPath(localRows * n);
    MPI_CHECK(MPI_Scatterv(rank == 0 ? initialRows.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED,
                           hostDist.data(), static_cast<int>(localRows * n), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i)
        for (size_t j = 0; j < n; ++j) hostPath[i * n + j] = static_cast<unsigned int>(firstRow + i);

    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *devicePivot = nullptr, *hostPivot = nullptr;
    const size_t localElements = localRows * n;
    CUDA_CHECK(cudaMalloc(&deviceDist, std::max<size_t>(localElements, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePath, std::max<size_t>(localElements, 1) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&devicePivot, n * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hostPivot, n * sizeof(unsigned int)));
    if (localElements != 0) {
        CUDA_CHECK(cudaMemcpy(deviceDist, hostDist.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, hostPath.data(), localElements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark (MPI + OpenMP + CUDA)\nNumber of nodes: %zu\nMPI ranks: %d\nValidation: %s\nInitializing graph...\nComputing shortest paths...\n", n, ranks, validate ? "enabled" : "disabled");
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto start = std::chrono::steady_clock::now();
    constexpr int threads = 256;
    for (size_t k = 0; k < n; ++k) {
        // Find owner robustly for uneven row partitions; ranks are normally few.
        int pivotOwner = 0;
        while (pivotOwner + 1 < ranks && k >= static_cast<size_t>(displacements[pivotOwner + 1]) / n) ++pivotOwner;
        if (rank == pivotOwner) {
            const size_t localK = k - firstRow;
            CUDA_CHECK(cudaMemcpy(hostPivot, deviceDist + localK * n, n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        MPI_CHECK(MPI_Bcast(hostPivot, static_cast<int>(n), MPI_UNSIGNED, pivotOwner, MPI_COMM_WORLD));
        CUDA_CHECK(cudaMemcpy(devicePivot, hostPivot, n * sizeof(unsigned int), cudaMemcpyHostToDevice));
        if (localElements != 0) {
            floydStep<<<static_cast<unsigned int>((localElements + threads - 1) / threads), threads>>>(deviceDist, devicePath, devicePivot, localRows, n, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const double localSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    double elapsed = 0.0;
    MPI_CHECK(MPI_Reduce(&localSeconds, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (localElements != 0)
        CUDA_CHECK(cudaMemcpy(hostDist.data(), deviceDist, localElements * sizeof(unsigned int), cudaMemcpyDeviceToHost));

    std::vector<unsigned int> gatheredRows, result;
    if (rank == 0) gatheredRows.resize(n * n);
    MPI_CHECK(MPI_Gatherv(hostDist.data(), static_cast<int>(localRows * n), MPI_UNSIGNED,
                          rank == 0 ? gatheredRows.data() : nullptr, counts.data(), displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    int success = 1;
    if (rank == 0) {
        result.resize(n * n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) result[idx2(j, i, n)] = gatheredRows[i * n + j];
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", elapsed * 1000.0,
                    static_cast<double>(n) * n * n / elapsed / 1e9);
        if (printResults) print_results_int(result, "DistanceMatrix");
        if (validate) { std::printf("Validating result...\n"); success = validateResult(result, n) ? 1 : 0; std::printf("Validation: %s\n", success ? "PASSED" : "FAILED"); }
    }
    MPI_CHECK(MPI_Bcast(&success, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaFreeHost(hostPivot));
    CUDA_CHECK(cudaFree(devicePivot)); CUDA_CHECK(cudaFree(devicePath)); CUDA_CHECK(cudaFree(deviceDist));
    MPI_CHECK(MPI_Finalize());
    return success ? 0 : 1;
}
