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
constexpr int TILE = 32;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// One CUDA block owns one matrix tile. Strict comparisons retain the original
// intermediate vertex in the path matrix when two paths have equal length.
__global__ void pivotTile(unsigned int* dist, unsigned int* path, int n, int localRow, int round) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int pos = (localRow + y) * n + round * TILE + x;
    unsigned int value = dist[pos];
    unsigned int predecessor = path[pos];
    tile[y][x] = value;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = tile[y][k] + tile[k][x];
        if (candidate < value) { value = candidate; predecessor = round * TILE + k; }
        __syncthreads();
        tile[y][x] = value;
        __syncthreads();
    }
    dist[pos] = value;
    path[pos] = predecessor;
}

__global__ void pivotRow(unsigned int* dist, unsigned int* path, int n, int localRow, int round) {
    const int colTile = blockIdx.x;
    if (colTile == round) return;
    __shared__ unsigned int diagonal[TILE][TILE], tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int pos = (localRow + y) * n + colTile * TILE + x;
    diagonal[y][x] = dist[(localRow + y) * n + round * TILE + x];
    unsigned int value = dist[pos];
    unsigned int predecessor = path[pos];
    tile[y][x] = value;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = diagonal[y][k] + tile[k][x];
        if (candidate < value) { value = candidate; predecessor = round * TILE + k; }
        __syncthreads();
        tile[y][x] = value;
        __syncthreads();
    }
    dist[pos] = value;
    path[pos] = predecessor;
}

__global__ void pivotColumn(unsigned int* dist, unsigned int* path,
                            const unsigned int* pivot, int n, int firstTile, int round) {
    const int rowTile = firstTile + blockIdx.x;
    if (rowTile == round) return;
    __shared__ unsigned int diagonal[TILE][TILE], tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int localRow = blockIdx.x * TILE + y;
    const int pos = localRow * n + round * TILE + x;
    diagonal[y][x] = pivot[y * n + round * TILE + x];
    unsigned int value = dist[pos];
    unsigned int predecessor = path[pos];
    tile[y][x] = value;
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = tile[y][k] + diagonal[k][x];
        if (candidate < value) { value = candidate; predecessor = round * TILE + k; }
        __syncthreads();
        tile[y][x] = value;
        __syncthreads();
    }
    dist[pos] = value;
    path[pos] = predecessor;
}

__global__ void remainingTiles(unsigned int* dist, unsigned int* path,
                               const unsigned int* pivot, int n, int firstTile, int round) {
    const int rowTile = firstTile + blockIdx.y;
    const int colTile = blockIdx.x;
    if (rowTile == round || colTile == round) return;
    __shared__ unsigned int column[TILE][TILE], row[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const int localRow = blockIdx.y * TILE + y;
    const int col = colTile * TILE + x;
    column[y][x] = dist[localRow * n + round * TILE + x];
    row[y][x] = pivot[y * n + col];
    const int pos = localRow * n + col;
    unsigned int value = dist[pos];
    unsigned int predecessor = path[pos];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        const unsigned int candidate = column[y][k] + row[k][x];
        if (candidate < value) { value = candidate; predecessor = round * TILE + k; }
    }
    dist[pos] = value;
    path[pos] = predecessor;
}

__global__ void initializePath(unsigned int* path, int n, int rows) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < rows * n) path[index] = index % n;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t numNodes,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    for (size_t i = 0; i < numNodes; ++i) dist[idx2(i, i, numNodes)] = 0;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    int invalid = 0;
    #pragma omp parallel for collapse(2) reduction(|:invalid) schedule(static)
    for (int i = 0; i < static_cast<int>(std::min(numNodes, size_t(10))); ++i)
        for (int j = 0; j < static_cast<int>(std::min(numNodes, size_t(10))); ++j)
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int dij = dist[idx2(j, i, numNodes)];
                const unsigned int dik = dist[idx2(k, i, numNodes)];
                const unsigned int dkj = dist[idx2(j, k, numNodes)];
                if (dik < INF && dkj < INF && dik + dkj < dij) invalid = 1;
            }
    if (invalid) printf("Validation failed: triangle inequality violated\n");
    return !invalid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false, printResults = false, help = false, badOption = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) numNodes = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) help = true;
        else badOption = true;
    }
    if (help || badOption || numNodes == 0 || numNodes > INT_MAX / TILE) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return badOption || numNodes == 0 || numNodes > INT_MAX / TILE;
    }
    const int n = static_cast<int>(numNodes);
    const int tiles = (n + TILE - 1) / TILE;
    const int padded = tiles * TILE;
    if (static_cast<long long>(padded) * padded > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Matrix too large for MPI counts\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm shared;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared);
    int localRank;
    MPI_Comm_rank(shared, &localRank);
    int devices = 0;
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&shared);
    const int firstTile = rank * tiles / ranks;
    const int endTile = (rank + 1) * tiles / ranks;
    const int localTiles = endTile - firstTile;
    const int localRows = localTiles * TILE;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        offsets[r] = (r * tiles / ranks) * TILE * padded;
        counts[r] = ((r + 1) * tiles / ranks - r * tiles / ranks) * TILE * padded;
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> dist, paddedDist;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        paddedDist.resize(static_cast<size_t>(padded) * padded, INF);
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < n; ++row)
            std::copy_n(dist.data() + static_cast<size_t>(row) * n, n,
                        paddedDist.data() + static_cast<size_t>(row) * padded);
        for (int row = n; row < padded; ++row) paddedDist[static_cast<size_t>(row) * padded + row] = 0;
    }
    std::vector<unsigned int> local(static_cast<size_t>(localRows) * padded);
    MPI_Scatterv(rank == 0 ? paddedDist.data() : nullptr, counts.data(), offsets.data(), MPI_UNSIGNED,
                 local.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    if (rank == 0) paddedDist.clear();
    unsigned int *gpuDist = nullptr, *gpuPath = nullptr, *gpuPivot = nullptr;
    const size_t localBytes = std::max(size_t(1), local.size()) * sizeof(unsigned int);
    cudaCheck(cudaMalloc(&gpuDist, localBytes), "cudaMalloc distance");
    cudaCheck(cudaMalloc(&gpuPath, localBytes), "cudaMalloc path");
    cudaCheck(cudaMalloc(&gpuPivot, static_cast<size_t>(TILE) * padded * sizeof(unsigned int)), "cudaMalloc pivot");
    if (!local.empty()) {
        cudaCheck(cudaMemcpy(gpuDist, local.data(), local.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), "copy input");
        initializePath<<<(localRows * padded + 255) / 256, 256>>>(gpuPath, padded, localRows);
        cudaCheck(cudaGetLastError(), "initialize path");
    }
    std::vector<unsigned int> pivot(static_cast<size_t>(TILE) * padded);
    const dim3 threads(TILE, TILE);
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();
    for (int round = 0; round < tiles; ++round) {
        int owner = 0;
        while (round >= (owner + 1) * tiles / ranks) ++owner;
        if (rank == owner) {
            const int localRow = (round - firstTile) * TILE;
            pivotTile<<<1, threads>>>(gpuDist, gpuPath, padded, localRow, round);
            pivotRow<<<tiles, threads>>>(gpuDist, gpuPath, padded, localRow, round);
            cudaCheck(cudaGetLastError(), "pivot row kernels");
            cudaCheck(cudaMemcpy(pivot.data(), gpuDist + static_cast<size_t>(localRow) * padded,
                                 pivot.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), "download pivot row");
        }
        MPI_Bcast(pivot.data(), static_cast<int>(pivot.size()), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        cudaCheck(cudaMemcpy(gpuPivot, pivot.data(), pivot.size() * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "upload pivot row");
        if (localTiles) {
            pivotColumn<<<localTiles, threads>>>(gpuDist, gpuPath, gpuPivot, padded, firstTile, round);
            remainingTiles<<<dim3(tiles, localTiles), threads>>>(gpuDist, gpuPath, gpuPivot, padded, firstTile, round);
            cudaCheck(cudaGetLastError(), "update kernels");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "compute");
    MPI_Barrier(MPI_COMM_WORLD);
    const auto end = std::chrono::high_resolution_clock::now();
    if (!local.empty()) cudaCheck(cudaMemcpy(local.data(), gpuDist, local.size() * sizeof(unsigned int),
                                              cudaMemcpyDeviceToHost), "download result");
    if (rank == 0) paddedDist.resize(static_cast<size_t>(padded) * padded);
    MPI_Gatherv(local.data(), counts[rank], MPI_UNSIGNED,
                rank == 0 ? paddedDist.data() : nullptr, counts.data(), offsets.data(), MPI_UNSIGNED,
                0, MPI_COMM_WORLD);
    cudaFree(gpuDist); cudaFree(gpuPath); cudaFree(gpuPivot);
    int result = 0;
    if (rank == 0) {
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", duration.count());
        const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", ops / std::max(0.001, duration.count() / 1000.0) / 1e9);
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < n; ++row)
            std::copy_n(paddedDist.data() + static_cast<size_t>(row) * padded, n,
                        dist.data() + static_cast<size_t>(row) * n);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            result = validateResult(dist, numNodes) ? 0 : 1;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
