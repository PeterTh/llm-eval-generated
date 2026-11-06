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
constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Each thread owns four elements. The diagonal and pivot row/column remain
// unchanged within each relaxation because all input weights are nonnegative.
__global__ void closePivot(unsigned int* rows, size_t n, size_t pivotLocal, size_t pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        size_t gr = pivot + r, gc = pivot + x;
        tile[r][x] = (gr < n && gc < n) ? rows[(pivotLocal + r) * n + gc] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE && pivot + k < n; ++k) {
        for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
            int r = y + q * THREAD_ROWS;
            unsigned int candidate = tile[r][k] + tile[k][x];
            if (r != k && x != k && candidate < tile[r][x]) tile[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        if (pivot + r < n && pivot + x < n)
            rows[(pivotLocal + r) * n + pivot + x] = tile[r][x];
    }
}

__global__ void updatePivotRow(unsigned int* rows, size_t n, size_t pivotLocal, size_t pivot) {
    size_t col = static_cast<size_t>(blockIdx.x) * TILE;
    if (col == pivot) return;
    __shared__ unsigned int diag[TILE][TILE], tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        diag[r][x] = (pivot + r < n && pivot + x < n) ? rows[(pivotLocal + r) * n + pivot + x] : INF;
        tile[r][x] = (pivot + r < n && col + x < n) ? rows[(pivotLocal + r) * n + col + x] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE && pivot + k < n; ++k) {
        for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
            int r = y + q * THREAD_ROWS;
            unsigned int candidate = diag[r][k] + tile[k][x];
            if (r != k && candidate < tile[r][x]) tile[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        if (pivot + r < n && col + x < n) rows[(pivotLocal + r) * n + col + x] = tile[r][x];
    }
}

__global__ void updatePivotColumn(unsigned int* rows, const unsigned int* panel, size_t n,
                                  size_t rowStart, size_t localRows, size_t pivot) {
    size_t local = static_cast<size_t>(blockIdx.x) * TILE;
    if (rowStart + local == pivot) return;
    __shared__ unsigned int diag[TILE][TILE], tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        diag[r][x] = (pivot + r < n && pivot + x < n) ? panel[static_cast<size_t>(r) * n + pivot + x] : INF;
        tile[r][x] = (local + r < localRows && pivot + x < n) ? rows[(local + r) * n + pivot + x] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE && pivot + k < n; ++k) {
        for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
            int r = y + q * THREAD_ROWS;
            unsigned int candidate = tile[r][k] + diag[k][x];
            if (x != k && candidate < tile[r][x]) tile[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        if (local + r < localRows && pivot + x < n) rows[(local + r) * n + pivot + x] = tile[r][x];
    }
}

__global__ void updateRemaining(unsigned int* rows, const unsigned int* panel, size_t n,
                                 size_t rowStart, size_t localRows, size_t pivot) {
    size_t col = static_cast<size_t>(blockIdx.x) * TILE;
    size_t local = static_cast<size_t>(blockIdx.y) * TILE;
    if (col == pivot || rowStart + local == pivot) return;
    __shared__ unsigned int left[TILE][TILE], top[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    unsigned int value[TILE / THREAD_ROWS];
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        left[r][x] = (local + r < localRows && pivot + x < n) ? rows[(local + r) * n + pivot + x] : INF;
        top[r][x] = (pivot + r < n && col + x < n) ? panel[static_cast<size_t>(r) * n + col + x] : INF;
        value[q] = (local + r < localRows && col + x < n) ? rows[(local + r) * n + col + x] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE && pivot + k < n; ++k) {
        unsigned int t = top[k][x];
        for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
            unsigned int candidate = left[y + q * THREAD_ROWS][k] + t;
            if (candidate < value[q]) value[q] = candidate;
        }
    }
    for (int q = 0; q < TILE / THREAD_ROWS; ++q) {
        int r = y + q * THREAD_ROWS;
        if (local + r < localRows && col + x < n) rows[(local + r) * n + col + x] = value[q];
    }
}

// MPI's count is an int. Chunk transfers so large matrices do not overflow it.
static void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}
static void broadcast(unsigned int* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_UNSIGNED, root, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}
static size_t firstTile(int rank, int ranks, size_t tiles) {
    return (tiles / ranks) * rank + std::min(static_cast<size_t>(rank), tiles % ranks);
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    const size_t sample = std::min(n, size_t(10));
    size_t firstFailure = SIZE_MAX;
    #pragma omp parallel for collapse(2) reduction(min:firstFailure) schedule(static)
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t k = 0; k < n; ++k) {
                unsigned int ij = dist[idx2(j, i, n)], ik = dist[idx2(k, i, n)], kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < ij) {
                    firstFailure = std::min(firstFailure, (i * sample + j) * n + k);
                    break;
                }
            }
        }
    }
    if (firstFailure != SIZE_MAX) {
        size_t k = firstFailure % n;
        size_t pair = firstFailure / n;
        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
               pair / sample, pair % sample, k);
        return false;
    }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n > 0 && (n > SIZE_MAX / n || n * n > SIZE_MAX / sizeof(unsigned int))) {
        if (rank == 0) fprintf(stderr, "Matrix size is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&localComm);

    const size_t tiles = (n + TILE - 1) / TILE;
    const size_t startTile = firstTile(rank, ranks, tiles);
    const size_t endTile = firstTile(rank + 1, ranks, tiles);
    const size_t rowStart = std::min(n, startTile * TILE);
    const size_t localRows = std::min(n, endTile * TILE) - rowStart;
    std::vector<unsigned int> matrix;
    std::vector<unsigned int> local(localRows * n);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        matrix.resize(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            matrix[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) matrix[idx2(i, i, n)] = 0;
        if (!local.empty()) std::copy_n(matrix.data(), local.size(), local.data());
        for (int r = 1; r < ranks; ++r) {
            size_t begin = std::min(n, firstTile(r, ranks, tiles) * TILE);
            size_t end = std::min(n, firstTile(r + 1, ranks, tiles) * TILE);
            if (end > begin) transfer(matrix.data() + begin * n, (end - begin) * n, r, 0, true);
        }
    } else transfer(local.data(), local.size(), 0, 0, false);
    if (rank == 0 && !validate && !printResults) std::vector<unsigned int>().swap(matrix);

    unsigned int *deviceRows = nullptr, *devicePanel = nullptr, *hostPanel = nullptr;
    if (localRows) {
        CUDA(cudaMalloc(&deviceRows, local.size() * sizeof(unsigned int)));
        CUDA(cudaMemcpy(deviceRows, local.data(), local.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if (n) {
        CUDA(cudaMalloc(&devicePanel, TILE * n * sizeof(unsigned int)));
        CUDA(cudaMallocHost(&hostPanel, TILE * n * sizeof(unsigned int)));
    }
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    const dim3 threads(TILE, THREAD_ROWS);
    int owner = 0;
    for (size_t t = 0; t < tiles; ++t) {
        while (t >= firstTile(owner + 1, ranks, tiles)) ++owner;
        size_t pivot = t * TILE;
        size_t panelRows = std::min(static_cast<size_t>(TILE), n - pivot);
        if (rank == owner) {
            size_t pivotLocal = pivot - rowStart;
            closePivot<<<1, threads>>>(deviceRows, n, pivotLocal, pivot);
            updatePivotRow<<<static_cast<unsigned int>(tiles), threads>>>(deviceRows, n, pivotLocal, pivot);
            CUDA(cudaGetLastError());
            if (ranks > 1) {
                CUDA(cudaMemcpy(hostPanel, deviceRows + pivotLocal * n, panelRows * n * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            } else {
                CUDA(cudaMemcpyAsync(devicePanel, deviceRows + pivotLocal * n, panelRows * n * sizeof(unsigned int), cudaMemcpyDeviceToDevice));
            }
        }
        // A single-rank broadcast has no data to exchange; keep the panel on the GPU.
        if (ranks > 1) broadcast(hostPanel, panelRows * n, owner);
        else MPI_Bcast(hostPanel, 0, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        if (localRows) {
            if (ranks > 1) CUDA(cudaMemcpy(devicePanel, hostPanel, panelRows * n * sizeof(unsigned int), cudaMemcpyHostToDevice));
            updatePivotColumn<<<static_cast<unsigned int>(endTile - startTile), threads>>>(deviceRows, devicePanel, n, rowStart, localRows, pivot);
            dim3 grid(static_cast<unsigned int>(tiles), static_cast<unsigned int>(endTile - startTile));
            updateRemaining<<<grid, threads>>>(deviceRows, devicePanel, n, rowStart, localRows, pivot);
            CUDA(cudaGetLastError());
        }
    }
    if (localRows) CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    if (validate || printResults) {
        if (localRows) CUDA(cudaMemcpy(local.data(), deviceRows, local.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (rank == 0) {
            if (!local.empty()) std::copy_n(local.data(), local.size(), matrix.data());
            for (int r = 1; r < ranks; ++r) {
                size_t begin = std::min(n, firstTile(r, ranks, tiles) * TILE);
                size_t endRow = std::min(n, firstTile(r + 1, ranks, tiles) * TILE);
                if (endRow > begin) transfer(matrix.data() + begin * n, (endRow - begin) * n, r, 1, false);
            }
        } else transfer(local.data(), local.size(), 0, 1, true);
    }
    if (deviceRows) CUDA(cudaFree(deviceRows));
    if (devicePanel) CUDA(cudaFree(devicePanel));
    if (hostPanel) CUDA(cudaFreeHost(hostPanel));
    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        double ops = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", ops / (duration.count() / 1000.0) / 1e9);
        if (printResults) print_results_int(matrix, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(matrix, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
