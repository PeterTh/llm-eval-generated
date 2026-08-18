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

constexpr unsigned int MAX_DISTANCE = 200;

// The public representation is the original destination-major representation.
inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

[[noreturn]] static void fail(const char* what, int rank) {
    std::fprintf(stderr, "rank %d: %s\n", rank, what);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

static void cudaCheck(cudaError_t status, const char* what, int rank) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "rank %d: %s: %s\n", rank, what, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Each rank stores a contiguous range of source rows in conventional row-major
// form: dist[source * n + destination].  The k row is replicated for one step.
__global__ void relaxKernel(unsigned int* dist, unsigned int* path,
                            const unsigned int* kRow, size_t localRows, size_t n, size_t k) {
    const size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t count = localRows * n;
    if (element >= count) return;
    const size_t localI = element / n;
    const size_t j = element - localI * n;
    const unsigned int throughK = dist[localI * n + k] + kRow[j];
    if (throughK < dist[element]) {
        dist[element] = throughK;
        path[element] = static_cast<unsigned int>(k);
    }
}

static void initializeRootMatrix(std::vector<unsigned int>& standard, size_t n) {
    // Preserve the original deterministic rand_r sequence, then transpose it
    // into the internal source-major representation in parallel.
    std::vector<unsigned int> original(n * n);
    unsigned int seed = 42;
    constexpr double range = static_cast<double>(MAX_DISTANCE);
    for (size_t p = 0; p < original.size(); ++p)
        original[p] = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            standard[i * n + j] = (i == j) ? 0 : original[idx2(i, j, n)];
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    const size_t samples = std::min(n, size_t{10});
    for (size_t i = 0; i < samples; ++i)
        for (size_t j = 0; j < samples; ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[idx2(k, i, n)] + dist[idx2(j, k, n)] < dist[idx2(j, i, n)]) return false;
    return true;
}

static void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of nodes (default: 512)\n"
                "  -v        Enable validation\n  -r        Print results\n  -h        Show help\n", name);
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
    if (!n || n > static_cast<size_t>(std::numeric_limits<int>::max()) || n < static_cast<size_t>(ranks)) fail("node count must be at least the MPI rank count and fit MPI counts", rank);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount", rank);
    if (!deviceCount) fail("no CUDA device available", rank);
    cudaCheck(cudaSetDevice(rank % deviceCount), "cudaSetDevice", rank);

    std::vector<int> rows(ranks), counts(ranks), displs(ranks);
    const size_t base = n / ranks, extra = n % ranks;
    for (int r = 0; r < ranks; ++r) {
        rows[r] = static_cast<int>(base + (static_cast<size_t>(r) < extra));
        counts[r] = rows[r] * static_cast<int>(n);
        displs[r] = r ? displs[r - 1] + counts[r - 1] : 0;
    }
    const size_t localRows = static_cast<size_t>(rows[rank]);
    if (n * n > static_cast<size_t>(std::numeric_limits<int>::max())) fail("matrix too large for MPI_Scatterv", rank);

    if (!rank) std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nMPI ranks: %d, OpenMP threads/rank: %d\nValidation: %s\nInitializing graph...\n",
                           n, ranks, omp_get_max_threads(), validate ? "enabled" : "disabled");
    std::vector<unsigned int> rootStandard;
    if (!rank) { rootStandard.resize(n * n); initializeRootMatrix(rootStandard, n); }
    std::vector<unsigned int> hostDist(localRows * n), hostPath(localRows * n), kRow(n);
    MPI_Scatterv(rank ? nullptr : rootStandard.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                 hostDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < localRows; ++i)
        for (size_t j = 0; j < n; ++j) hostPath[i * n + j] = static_cast<unsigned int>(j);

    unsigned int *deviceDist = nullptr, *devicePath = nullptr, *deviceKRow = nullptr;
    cudaCheck(cudaMalloc(&deviceDist, hostDist.size() * sizeof(unsigned int)), "cudaMalloc(dist)", rank);
    cudaCheck(cudaMalloc(&devicePath, hostPath.size() * sizeof(unsigned int)), "cudaMalloc(path)", rank);
    cudaCheck(cudaMalloc(&deviceKRow, n * sizeof(unsigned int)), "cudaMalloc(k row)", rank);
    cudaCheck(cudaMemcpy(deviceDist, hostDist.data(), hostDist.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy dist", rank);
    cudaCheck(cudaMemcpy(devicePath, hostPath.data(), hostPath.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy path", rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) std::printf("Computing shortest paths...\n");
    const auto start = std::chrono::high_resolution_clock::now();
    const int threads = 256;
    for (size_t k = 0; k < n; ++k) {
        int owner = static_cast<int>(k / (base + 1));
        if (static_cast<size_t>(owner) >= extra) owner = static_cast<int>(extra + (k - extra * (base + 1)) / base);
        if (rank == owner) {
            const size_t offset = (k - static_cast<size_t>(displs[rank]) / n) * n;
            cudaCheck(cudaMemcpy(kRow.data(), deviceDist + offset, n * sizeof(unsigned int), cudaMemcpyDeviceToHost), "read k row", rank);
        }
        MPI_Bcast(kRow.data(), static_cast<int>(n), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(deviceKRow, kRow.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice), "upload k row", rank);
        const size_t elements = localRows * n;
        if (elements) relaxKernel<<<static_cast<unsigned>((elements + threads - 1) / threads), threads>>>(deviceDist, devicePath, deviceKRow, localRows, n, k);
        cudaCheck(cudaGetLastError(), "relax kernel", rank);
    }
    cudaCheck(cudaDeviceSynchronize(), "CUDA synchronize", rank);
    const auto end = std::chrono::high_resolution_clock::now();
    double localSeconds = std::chrono::duration<double>(end - start).count(), seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    cudaCheck(cudaMemcpy(hostDist.data(), deviceDist, hostDist.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "download dist", rank);
    cudaCheck(cudaMemcpy(hostPath.data(), devicePath, hostPath.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "download path", rank);

    if (!rank) rootStandard.resize(n * n);
    MPI_Gatherv(hostDist.data(), counts[rank], MPI_UNSIGNED, rank ? nullptr : rootStandard.data(), counts.data(), displs.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    int result = 0;
    if (!rank) {
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", seconds * 1000.0, (double(n) * n * n) / seconds / 1e9);
        std::vector<unsigned int> original(n * n);
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) for (size_t j = 0; j < n; ++j) original[idx2(i, j, n)] = rootStandard[i * n + j];
        if (printResults) print_results_int(original, "DistanceMatrix");
        if (validate) { result = validateResult(original, n) ? 0 : 1; std::printf("Validation: %s\n", result ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    cudaFree(deviceKRow); cudaFree(devicePath); cudaFree(deviceDist);
    MPI_Finalize();
    return result;
}
