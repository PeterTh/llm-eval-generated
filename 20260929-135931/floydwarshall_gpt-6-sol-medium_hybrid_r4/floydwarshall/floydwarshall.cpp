#include <algorithm>
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
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA %s failed: %s\n", operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void mpiCheck(int status, const char* operation) {
    if (status != MPI_SUCCESS) {
        fprintf(stderr, "MPI %s failed\n", operation);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t numNodes,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    // The random sequence is intentionally serial to preserve the input graph.
    for (size_t i = 0; i < numNodes * numNodes; ++i)
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numNodes; ++i)
        dist[idx2(i, i, numNodes)] = 0;
}

void initializePathMatrix(std::vector<unsigned int>& path, size_t numNodes) {
#pragma omp parallel for schedule(static)
    for (size_t row = 0; row < numNodes; ++row)
        for (size_t col = 0; col < numNodes; ++col)
            path[row * numNodes + col] = static_cast<unsigned int>(row);
}

// Close the diagonal block in k order. Positive edge weights and a zero
// diagonal keep row/column k unchanged while the other entries are updated.
__global__ void diagonalKernel(unsigned int* dist, unsigned int* path,
                               int n, int localPivotRow, int pivot, int width) {
    __shared__ unsigned int block[TILE][TILE];
    int col = threadIdx.x, row = threadIdx.y;
    int index = (localPivotRow + row) * n + pivot + col;
    if (row < width && col < width) block[row][col] = dist[index];
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        unsigned int candidate = 0;
        if (row < width && col < width)
            candidate = block[row][k] + block[k][col];
        __syncthreads();
        if (row < width && col < width && candidate < block[row][col]) {
            block[row][col] = candidate;
            path[index] = pivot + k;
        }
        __syncthreads();
    }
    if (row < width && col < width) dist[index] = block[row][col];
}

// Extend the closed diagonal block across all columns of the pivot rows.
// source is a snapshot of the pivot rows, so every output cell is independent.
__global__ void pivotRowKernel(unsigned int* dist, unsigned int* path,
                               const unsigned int* source, int n,
                               int localPivotRow, int pivot, int width) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= width * n) return;
    int row = t / n, col = t % n;
    if (col >= pivot && col < pivot + width) return;
    unsigned int best = source[t], via = 0;
    for (int k = 0; k < width; ++k) {
        unsigned int candidate = dist[(localPivotRow + row) * n + pivot + k]
                               + source[k * n + col];
        if (candidate < best) {
            best = candidate;
            via = pivot + k;
        }
    }
    int index = (localPivotRow + row) * n + col;
    dist[index] = best;
    if (best < source[t]) path[index] = via;
}

// Update the pivot columns in local rows from a shared snapshot of each row.
__global__ void pivotColumnKernel(unsigned int* dist, unsigned int* path,
                                  const unsigned int* pivotRows, int n,
                                  int localRows, int localPivotRow,
                                  int pivot, int width) {
    int row = blockIdx.x, col = threadIdx.x;
    if (row >= localRows || (row >= localPivotRow && row < localPivotRow + width)) return;
    __shared__ unsigned int original[TILE];
    if (col < width) original[col] = dist[row * n + pivot + col];
    __syncthreads();
    if (col >= width) return;
    unsigned int best = original[col], via = 0;
    for (int k = 0; k < width; ++k) {
        unsigned int candidate = original[k] + pivotRows[k * n + pivot + col];
        if (candidate < best) {
            best = candidate;
            via = pivot + k;
        }
    }
    int index = row * n + pivot + col;
    dist[index] = best;
    if (best < original[col]) path[index] = via;
}

__global__ void remainingKernel(unsigned int* dist, unsigned int* path,
                                const unsigned int* pivotRows, int n,
                                int localRows, int localPivotRow,
                                int pivot, int width) {
    size_t t = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= static_cast<size_t>(localRows) * n) return;
    int row = t / n, col = t % n;
    if ((row >= localPivotRow && row < localPivotRow + width) ||
        (col >= pivot && col < pivot + width)) return;
    int index = row * n + col;
    unsigned int original = dist[index], best = original, via = 0;
    for (int k = 0; k < width; ++k) {
        unsigned int candidate = dist[row * n + pivot + k] + pivotRows[k * n + col];
        if (candidate < best) {
            best = candidate;
            via = pivot + k;
        }
    }
    dist[index] = best;
    if (best < original) path[index] = via;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t numNodes) {
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i)
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j)
            for (size_t k = 0; k < numNodes; ++k) {
                unsigned int ik = dist[idx2(k, i, numNodes)];
                unsigned int kj = dist[idx2(j, k, numNodes)];
                if (ik < INF && kj < INF && ik + kj < dist[idx2(j, i, numNodes)]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
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
    mpiCheck(MPI_Init(&argc, &argv), "Init");
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "Comm_size");
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            unsigned long long value = strtoull(argv[++i], &end, 10);
            if (*end || value > static_cast<unsigned long long>(std::numeric_limits<int>::max())) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes\n");
                MPI_Finalize();
                return 1;
            }
            numNodes = value;
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (numNodes * numNodes > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) fprintf(stderr, "Matrix is too large for MPI counts\n");
        MPI_Finalize();
        return 1;
    }
    int n = static_cast<int>(numNodes);
    int tiles = (n + TILE - 1) / TILE;
    int firstTile = tiles * rank / ranks;
    int lastTile = tiles * (rank + 1) / ranks;
    int firstRow = firstTile * TILE;
    int localRows = std::min(n, lastTile * TILE) - firstRow;
    std::vector<int> counts(ranks), offsets(ranks);
    for (int r = 0; r < ranks; ++r) {
        int begin = (tiles * r / ranks) * TILE;
        int end = std::min(n, (tiles * (r + 1) / ranks) * TILE);
        counts[r] = (end - begin) * n;
        offsets[r] = begin * n;
    }
    std::vector<unsigned int> dist, path;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        dist.resize(numNodes * numNodes);
        path.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
        initializePathMatrix(path, numNodes);
        printf("Computing shortest paths...\n");
    }
    std::vector<unsigned int> localDist(std::max(1, counts[rank]));
    std::vector<unsigned int> localPath(std::max(1, counts[rank]));
    mpiCheck(MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), offsets.data(), MPI_UNSIGNED,
                          localDist.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD), "Scatterv dist");
    mpiCheck(MPI_Scatterv(rank == 0 ? path.data() : nullptr, counts.data(), offsets.data(), MPI_UNSIGNED,
                          localPath.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD), "Scatterv path");

    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm), "Comm_split_type");
    int localRank, deviceCount;
    mpiCheck(MPI_Comm_rank(localComm, &localRank), "local Comm_rank");
    cudaCheck(cudaGetDeviceCount(&deviceCount), "GetDeviceCount");
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device found\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "SetDevice");
    mpiCheck(MPI_Comm_free(&localComm), "Comm_free");

    unsigned int *deviceDist, *devicePath, *devicePivot;
    size_t localBytes = static_cast<size_t>(std::max(1, counts[rank])) * sizeof(unsigned int);
    size_t pivotBytes = static_cast<size_t>(std::max(1, TILE * n)) * sizeof(unsigned int);
    cudaCheck(cudaMalloc(&deviceDist, localBytes), "Malloc dist");
    cudaCheck(cudaMalloc(&devicePath, localBytes), "Malloc path");
    cudaCheck(cudaMalloc(&devicePivot, pivotBytes), "Malloc pivot");
    if (counts[rank]) {
        cudaCheck(cudaMemcpy(deviceDist, localDist.data(), counts[rank] * sizeof(unsigned int), cudaMemcpyHostToDevice), "Copy dist");
        cudaCheck(cudaMemcpy(devicePath, localPath.data(), counts[rank] * sizeof(unsigned int), cudaMemcpyHostToDevice), "Copy path");
    }
    std::vector<unsigned int> pivotRows(static_cast<size_t>(std::max(1, TILE * n)));
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "Barrier");
    double start = MPI_Wtime();
    for (int tile = 0; tile < tiles; ++tile) {
        int pivot = tile * TILE;
        int width = std::min(TILE, n - pivot);
        int owner = 0;
        while (tile >= tiles * (owner + 1) / ranks) ++owner;
        int ownerFirstRow = (tiles * owner / ranks) * TILE;
        if (rank == owner) {
            int localPivotRow = pivot - firstRow;
            diagonalKernel<<<1, dim3(TILE, TILE)>>>(deviceDist, devicePath, n, localPivotRow, pivot, width);
            cudaCheck(cudaGetLastError(), "diagonal kernel");
            cudaCheck(cudaMemcpy(devicePivot, deviceDist + static_cast<size_t>(localPivotRow) * n,
                                 static_cast<size_t>(width) * n * sizeof(unsigned int), cudaMemcpyDeviceToDevice), "Snapshot pivot rows");
            pivotRowKernel<<<(width * n + 255) / 256, 256>>>(deviceDist, devicePath, devicePivot, n, localPivotRow, pivot, width);
            cudaCheck(cudaGetLastError(), "pivot row kernel");
            cudaCheck(cudaMemcpy(pivotRows.data(), deviceDist + static_cast<size_t>(pivot - ownerFirstRow) * n,
                                 static_cast<size_t>(width) * n * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Read pivot rows");
        }
        mpiCheck(MPI_Bcast(pivotRows.data(), width * n, MPI_UNSIGNED, owner, MPI_COMM_WORLD), "Bcast pivot rows");
        cudaCheck(cudaMemcpy(devicePivot, pivotRows.data(), static_cast<size_t>(width) * n * sizeof(unsigned int),
                             cudaMemcpyHostToDevice), "Broadcast pivot to device");
        if (localRows) {
            int localPivotRow = rank == owner ? pivot - firstRow : -TILE;
            pivotColumnKernel<<<localRows, TILE>>>(deviceDist, devicePath, devicePivot, n, localRows,
                                                    localPivotRow, pivot, width);
            cudaCheck(cudaGetLastError(), "pivot column kernel");
            int cells = localRows * n;
            remainingKernel<<<(cells + 255) / 256, 256>>>(deviceDist, devicePath, devicePivot, n, localRows,
                                                           localPivotRow, pivot, width);
            cudaCheck(cudaGetLastError(), "remaining kernel");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "Synchronize");
    if (counts[rank]) {
        cudaCheck(cudaMemcpy(localDist.data(), deviceDist, counts[rank] * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Read dist");
        cudaCheck(cudaMemcpy(localPath.data(), devicePath, counts[rank] * sizeof(unsigned int), cudaMemcpyDeviceToHost), "Read path");
    }
    mpiCheck(MPI_Gatherv(localDist.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? dist.data() : nullptr,
                         counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD), "Gatherv dist");
    mpiCheck(MPI_Gatherv(localPath.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? path.data() : nullptr,
                         counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD), "Gatherv path");
    double elapsed = MPI_Wtime() - start, duration;
    mpiCheck(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD), "Reduce time");
    cudaCheck(cudaFree(deviceDist), "Free dist");
    cudaCheck(cudaFree(devicePath), "Free path");
    cudaCheck(cudaFree(devicePivot), "Free pivot");
    int result = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000));
        double ops = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", duration > 0 ? ops / duration / 1e9 : 0.0);
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            result = validateResult(dist, numNodes) ? 0 : 1;
            printf("Validation: %s\n", result ? "FAILED" : "PASSED");
        }
    }
    mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD), "Bcast result");
    mpiCheck(MPI_Finalize(), "Finalize");
    return result;
}
