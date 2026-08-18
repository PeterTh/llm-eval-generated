#include <algorithm>
#include <chrono>
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

inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    // Keep this serial: rand_r's sequence is part of the benchmark's input semantics.
    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);

    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < static_cast<long long>(numNodes); ++i)
        dist[idx2(static_cast<size_t>(i), static_cast<size_t>(i), numNodes)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(numNodes); ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, static_cast<size_t>(j), numNodes)] = static_cast<unsigned int>(j);
            path[idx2(static_cast<size_t>(j), i, numNodes)] = static_cast<unsigned int>(i);
        }
        path[idx2(static_cast<size_t>(j), static_cast<size_t>(j), numNodes)] =
            static_cast<unsigned int>(j);
    }
}

// One block covers a small tile of rows and local source columns. Shared memory
// removes the repeated pivot/source loads while preserving coalesced matrix access.
__global__ void floydStep(unsigned int* dist, unsigned int* path,
                          const unsigned int* pivot, size_t n, size_t localColumns,
                          size_t globalColumnStart, size_t k) {
    constexpr int TILE_X = 32;
    constexpr int TILE_Y = 8;
    __shared__ unsigned int pivotTile[TILE_Y];
    __shared__ unsigned int sourceTile[TILE_X];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t localColumn = static_cast<size_t>(blockIdx.x) * TILE_X + tx;
    const size_t row = static_cast<size_t>(blockIdx.y) * TILE_Y + ty;

    if (tx == 0 && row < n) pivotTile[ty] = pivot[row];
    if (ty == 0 && localColumn < localColumns)
        sourceTile[tx] = dist[localColumn * n + k];
    __syncthreads();

    if (row < n && localColumn < localColumns) {
        const unsigned int oldDistance = dist[localColumn * n + row];
        const unsigned int newDistance = sourceTile[tx] + pivotTile[ty];
        if (newDistance < oldDistance) {
            dist[localColumn * n + row] = newDistance;
            path[localColumn * n + row] = static_cast<unsigned int>(k);
        }
    }
}

__global__ void extractPivot(const unsigned int* dist, unsigned int* pivot,
                             size_t localColumns, size_t localK, size_t n) {
    const size_t row = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row < n) pivot[row] = dist[localK * n + row];
}

static void cudaCheck(const cudaError_t error, const char* operation, int rank) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "MPI rank %d: CUDA error in %s: %s\n", rank, operation,
                     cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, static_cast<int>(error));
    }
}

void floydWarshallHybrid(std::vector<unsigned int>& dist, std::vector<unsigned int>& path,
                         size_t numNodes, size_t globalColumnStart, int rank, int ranks) {
    const size_t localColumns = numNodes == 0 ? 0 : dist.size() / numNodes;
    unsigned int* deviceDist = nullptr;
    unsigned int* devicePath = nullptr;
    unsigned int* devicePivot = nullptr;

    cudaCheck(cudaMalloc(&deviceDist, dist.size() * sizeof(unsigned int)), "cudaMalloc(dist)", rank);
    cudaCheck(cudaMalloc(&devicePath, path.size() * sizeof(unsigned int)), "cudaMalloc(path)", rank);
    cudaCheck(cudaMalloc(&devicePivot, numNodes * sizeof(unsigned int)), "cudaMalloc(pivot)", rank);
    cudaCheck(cudaMemcpy(deviceDist, dist.data(), dist.size() * sizeof(unsigned int),
                         cudaMemcpyHostToDevice), "copy dist to device", rank);
    cudaCheck(cudaMemcpy(devicePath, path.data(), path.size() * sizeof(unsigned int),
                         cudaMemcpyHostToDevice), "copy path to device", rank);

    const dim3 block(32, 8);
    const dim3 grid(static_cast<unsigned int>((localColumns + 31) / 32),
                    static_cast<unsigned int>((numNodes + 7) / 8));
    std::vector<unsigned int> pivot(numNodes);

    for (size_t k = 0; k < numNodes; ++k) {
        // Columns are contiguous and block distributed. The owner can therefore be
        // derived identically by every rank, avoiding an extra collective per k.
        int pivotOwner = 0;
        while (k >= numNodes * static_cast<size_t>(pivotOwner + 1) /
                          static_cast<size_t>(ranks)) ++pivotOwner;
        if (rank == pivotOwner) {
            const size_t localK = k - globalColumnStart;
            extractPivot<<<(static_cast<unsigned int>(numNodes) + 255) / 256, 256>>>
                (deviceDist, devicePivot, localColumns, localK, numNodes);
            cudaCheck(cudaGetLastError(), "launch pivot extraction", rank);
            cudaCheck(cudaMemcpy(pivot.data(), devicePivot, numNodes * sizeof(unsigned int),
                                 cudaMemcpyDeviceToHost), "copy pivot to host", rank);
        }
        MPI_Bcast(pivot.data(), static_cast<int>(numNodes), MPI_UNSIGNED, pivotOwner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(devicePivot, pivot.data(), numNodes * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "copy pivot to device", rank);
        floydStep<<<grid, block>>>(deviceDist, devicePath, devicePivot, numNodes, localColumns,
                                    globalColumnStart, k);
        cudaCheck(cudaGetLastError(), "launch floyd kernel", rank);
    }
    cudaCheck(cudaDeviceSynchronize(), "complete floyd kernels", rank);
    cudaCheck(cudaMemcpy(dist.data(), deviceDist, dist.size() * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost), "copy dist to host", rank);
    cudaCheck(cudaMemcpy(path.data(), devicePath, path.size() * sizeof(unsigned int),
                         cudaMemcpyDeviceToHost), "copy path to host", rank);
    cudaFree(devicePivot);
    cudaFree(devicePath);
    cudaFree(deviceDist);
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t n) {
    for (size_t i = 0; i < n; ++i)
        if (dist[idx2(i, i, n)] != 0) return false;
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                const unsigned int ij = dist[idx2(j, i, n)];
                const unsigned int ik = dist[idx2(k, i, n)];
                const unsigned int kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < ij) return false;
            }
    return true;
}

void printUsage(const char* p) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of nodes (default: 512)\n"
           "  -v           Enable validation\n  -r           Print results\n  -h           Show this help\n", p);
}

int main(int argc, char** argv) {
    size_t n = 512; bool validate = false; bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = static_cast<size_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) { printUsage(argv[0]); return 0; }
        else { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }

    int provided = 0, rank = 0, ranks = 1;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (n == 0) {
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: 0\n"
                   "Validation: %s\nComputation time: 0 ms\nPerformance: 0.000 GOPS\n",
                   validate ? "enabled" : "disabled");
            if (printResults) print_results_int(std::vector<unsigned int>{}, "DistanceMatrix");
            if (validate) printf("Validation: PASSED\n");
        }
        MPI_Finalize();
        return 0;
    }
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "get device count", rank);
    cudaCheck(cudaSetDevice(rank % devices), "select device", rank);

    std::vector<int> counts(ranks), displacements(ranks);
    for (int r = 0; r < ranks; ++r) {
        const size_t first = n * static_cast<size_t>(r) / ranks;
        const size_t last = n * static_cast<size_t>(r + 1) / ranks;
        counts[r] = static_cast<int>((last - first) * n);
        displacements[r] = static_cast<int>(first * n);
    }
    const size_t firstColumn = n * static_cast<size_t>(rank) / ranks;
    const size_t localColumns = n * static_cast<size_t>(rank + 1) / ranks - firstColumn;
    std::vector<unsigned int> fullDist, fullPath;
    if (rank == 0) {
        fullDist.resize(n * n); fullPath.resize(n * n);
        initializeDistanceMatrix(fullDist, n, 1, MAX_DISTANCE);
        initializePathMatrix(fullPath, n);
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n",
               n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist(localColumns * n), path(localColumns * n);
    MPI_Scatterv(fullDist.data(), counts.data(), displacements.data(), MPI_UNSIGNED,
                 dist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Scatterv(fullPath.data(), counts.data(), displacements.data(), MPI_UNSIGNED,
                 path.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("Computing shortest paths...\n");
    const double start = MPI_Wtime();
    floydWarshallHybrid(dist, path, n, firstColumn, rank, ranks);
    const double elapsed = MPI_Wtime() - start;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) fullDist.resize(n * n);
    MPI_Gatherv(dist.data(), counts[rank], MPI_UNSIGNED, fullDist.data(), counts.data(),
                displacements.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
        printf("Performance: %.3f GOPS\n", maxElapsed > 0 ? (double)n*n*n/maxElapsed/1e9 : 0.0);
        if (printResults) print_results_int(fullDist, "DistanceMatrix");
        if (validate) printf("Validation: %s\n", validateResult(fullDist, n) ? "PASSED" : "FAILED");
    }
    MPI_Finalize();
    return 0;
}
