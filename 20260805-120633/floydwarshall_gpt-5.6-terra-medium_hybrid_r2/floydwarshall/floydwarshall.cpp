#include <algorithm>
#include <chrono>
#include <climits>
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
constexpr int THREADS = 256;

// The public result keeps the original destination-major layout.  The device
// matrices are source-major so that every MPI-owned source row is contiguous.
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void cudaDie(cudaError_t status, const char* expression, int rank) {
    std::fprintf(stderr, "rank %d: CUDA failure in %s: %s\n", rank, expression,
                 cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, static_cast<int>(status));
    std::abort();
}

#define CUDA_CHECK(expr) do { const cudaError_t _status = (expr); if (_status != cudaSuccess) cudaDie(_status, #expr, rank); } while (0)

__global__ void relaxRows(unsigned int* __restrict__ distance,
                          unsigned int* __restrict__ path,
                          const unsigned int* __restrict__ pivot,
                          const int localRows, const int width, const int pivotColumn) {
    const int linear = blockIdx.x * blockDim.x + threadIdx.x;
    const int elements = localRows * width;
    if (linear >= elements) return;

    const int column = linear % width;
    const int row = linear / width;
    const unsigned int via = distance[row * width + pivotColumn] + pivot[column];
    if (via < distance[linear]) {
        distance[linear] = via;
        path[linear] = static_cast<unsigned int>(pivotColumn);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t nodes) {
    unsigned int seed = 42;
    constexpr double range = static_cast<double>(MAX_DISTANCE);
    for (size_t element = 0; element < nodes * nodes; ++element)
        dist[element] = 1U + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t node = 0; node < nodes; ++node) dist[idx2(node, node, nodes)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t nodes) {
    #pragma omp parallel for schedule(static)
    for (long long source = 0; source < static_cast<long long>(nodes); ++source)
        for (size_t destination = 0; destination < nodes; ++destination)
            path[idx2(static_cast<size_t>(source), destination, nodes)] = static_cast<unsigned int>(destination);
}

bool validateResult(const std::vector<unsigned int>& dist, size_t nodes) {
    for (size_t i = 0; i < nodes; ++i) {
        if (dist[idx2(i, i, nodes)] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(nodes, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(nodes, size_t(10)); ++j)
            for (size_t k = 0; k < nodes; ++k) {
                const unsigned int a = dist[idx2(k, i, nodes)], b = dist[idx2(j, k, nodes)];
                if (a < INF && b < INF && a + b < dist[idx2(j, i, nodes)]) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n"
                "  -v           Enable validation\n  -r           Print results for external validation\n"
                "  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    size_t nodes = 512;
    bool validate = false, printResults = false, badArgument = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "-n") && i + 1 < argc) nodes = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "-v")) validate = true;
        else if (!std::strcmp(argv[i], "-r")) printResults = true;
        else if (!std::strcmp(argv[i], "-h")) { if (rank == 0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else badArgument = true;
    }
    if (badArgument || nodes == 0 || nodes > static_cast<size_t>(INT_MAX) || nodes > static_cast<size_t>(INT_MAX) / nodes) {
        if (rank == 0) { std::printf("Invalid node count or option\n"); printUsage(argv[0]); }
        MPI_Finalize(); return 1;
    }
    const int n = static_cast<int>(nodes);

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) { if (rank == 0) std::fprintf(stderr, "No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    std::vector<int> rows(ranks), counts(ranks), offsets(ranks);
    for (int process = 0; process < ranks; ++process) {
        rows[process] = n / ranks + (process < n % ranks ? 1 : 0);
        counts[process] = rows[process] * n;
    }
    std::partial_sum(counts.begin(), counts.end() - 1, offsets.begin() + 1);
    const int localRows = rows[rank], localOffset = offsets[rank];

    std::vector<unsigned int> resultDistance, resultPath;
    std::vector<unsigned int> packedDistance, packedPath;
    if (rank == 0) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
                    nodes, validate ? "enabled" : "disabled");
        resultDistance.resize(nodes * nodes); resultPath.resize(nodes * nodes);
        initializeDistanceMatrix(resultDistance, nodes); initializePathMatrix(resultPath, nodes);
        packedDistance.resize(nodes * nodes); packedPath.resize(nodes * nodes);
        #pragma omp parallel for collapse(2) schedule(static)
        for (long long source = 0; source < n; ++source)
            for (int destination = 0; destination < n; ++destination) {
                packedDistance[static_cast<size_t>(source) * n + destination] = resultDistance[idx2(source, destination, nodes)];
                packedPath[static_cast<size_t>(source) * n + destination] = resultPath[idx2(source, destination, nodes)];
            }
    }

    std::vector<unsigned int> localDistance(static_cast<size_t>(localRows) * n), localPath(static_cast<size_t>(localRows) * n), pivot(n);
    MPI_Scatterv(rank == 0 ? packedDistance.data() : nullptr, counts.data(), offsets.data(), MPI_UNSIGNED,
                 localDistance.data(), localRows * n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? packedPath.data() : nullptr, counts.data(), offsets.data(), MPI_UNSIGNED,
                 localPath.data(), localRows * n, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    unsigned int *dDistance = nullptr, *dPath = nullptr, *dPivot = nullptr;
    const size_t localAllocation = std::max<size_t>(1, localDistance.size());
    CUDA_CHECK(cudaMalloc(&dDistance, localAllocation * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, localAllocation * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPivot, nodes * sizeof(unsigned int)));
    if (!localDistance.empty()) {
        CUDA_CHECK(cudaMemcpy(dDistance, localDistance.data(), localDistance.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, localPath.data(), localPath.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::steady_clock::now();
    int owner = 0, ownerEnd = rows.empty() ? 0 : rows[0];
    for (int k = 0; k < n; ++k) {
        while (k >= ownerEnd && owner + 1 < ranks) ownerEnd += rows[++owner];
        if (rank == owner)
            CUDA_CHECK(cudaMemcpy(pivot.data(), dDistance + static_cast<size_t>(k - localOffset) * n,
                                  nodes * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        MPI_Bcast(pivot.data(), n, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpyAsync(dPivot, pivot.data(), nodes * sizeof(unsigned int), cudaMemcpyHostToDevice));
        const int elements = localRows * n;
        if (elements) relaxRows<<<(elements + THREADS - 1) / THREADS, THREADS>>>(dDistance, dPath, dPivot, localRows, n, k);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::steady_clock::now();

    if (!localDistance.empty()) {
        CUDA_CHECK(cudaMemcpy(localDistance.data(), dDistance, localDistance.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(localPath.data(), dPath, localPath.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }
    MPI_Gatherv(localDistance.data(), localRows * n, MPI_UNSIGNED, rank == 0 ? packedDistance.data() : nullptr,
                counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Gatherv(localPath.data(), localRows * n, MPI_UNSIGNED, rank == 0 ? packedPath.data() : nullptr,
                counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(dPivot)); CUDA_CHECK(cudaFree(dPath)); CUDA_CHECK(cudaFree(dDistance));
    int status = 0;
    if (rank == 0) {
        #pragma omp parallel for collapse(2) schedule(static)
        for (long long source = 0; source < n; ++source)
            for (int destination = 0; destination < n; ++destination) {
                resultDistance[idx2(source, destination, nodes)] = packedDistance[static_cast<size_t>(source) * n + destination];
                resultPath[idx2(source, destination, nodes)] = packedPath[static_cast<size_t>(source) * n + destination];
            }
        const double seconds = std::chrono::duration<double>(end - start).count();
        std::printf("Computation time: %.3f ms\nPerformance: %.3f GOPS\n", seconds * 1000.0,
                    static_cast<double>(nodes) * nodes * nodes / seconds / 1e9);
        if (printResults) print_results_int(resultDistance, "DistanceMatrix");
        if (validate) { status = validateResult(resultDistance, nodes) ? 0 : 1; std::printf("Validation: %s\n", status ? "FAILED" : "PASSED"); }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
