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

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept { return j * n + i; }

static void checkCuda(cudaError_t error, MPI_Comm comm) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        MPI_Abort(comm, 1);
    }
}

static void checkMpi(int error, MPI_Comm comm) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(comm, 1);
    }
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              unsigned int rangeMin, unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    for (size_t i = 0; i < n * n; ++i)
        dist[i] = rangeMin + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
    for (size_t i = 0; i < n; ++i) dist[idx2(i, i, n)] = 0;
}

__global__ void initPaths(unsigned int* path, int rows, int stride) {
    size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t count = static_cast<size_t>(rows) * stride;
    if (index < count) path[index] = static_cast<unsigned int>(index % stride);
}

__global__ void pivotKernel(unsigned int* dist, unsigned int* path, int stride, int pivotLocalRow, int pivot) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    int j = threadIdx.x;
    int rowBase = pivotLocalRow + threadIdx.y;
    int colBase = pivot * TILE;
    for (int q = 0; q < 4; ++q) {
        int i = threadIdx.y + q * 8;
        tile[i][j] = dist[static_cast<size_t>(rowBase + q * 8) * stride + colBase + j];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int q = 0; q < 4; ++q) {
            int i = threadIdx.y + q * 8;
            unsigned int candidate = tile[i][k] + tile[k][j];
            if (candidate < tile[i][j]) {
                tile[i][j] = candidate;
                path[static_cast<size_t>(rowBase + q * 8) * stride + colBase + j] = pivot * TILE + k;
            }
        }
        __syncthreads();
    }
    for (int q = 0; q < 4; ++q)
        dist[static_cast<size_t>(rowBase + q * 8) * stride + colBase + j] = tile[threadIdx.y + q * 8][j];
}

// Mode 0 updates the pivot tile row; mode 1 updates local pivot tile columns.
template <int MODE>
__global__ void borderKernel(unsigned int* dist, unsigned int* path,
                             const unsigned int* pivotRow, int stride,
                             int localTileStart, int pivot, int tiles) {
    int tile = blockIdx.x;
    int rowTile = MODE == 0 ? pivot - localTileStart : tile;
    int colTile = MODE == 0 ? tile : pivot;
    if (tile >= tiles || tile == pivot && MODE == 0 ||
        MODE == 1 && localTileStart + tile == pivot) return;
    __shared__ unsigned int center[TILE][TILE + 1];
    __shared__ unsigned int target[TILE][TILE + 1];
    int j = threadIdx.x;
    int row = rowTile * TILE + threadIdx.y;
    int col = colTile * TILE + j;
    for (int q = 0; q < 4; ++q) {
        int i = threadIdx.y + q * 8;
        center[i][j] = pivotRow[static_cast<size_t>(i) * stride + pivot * TILE + j];
        target[i][j] = dist[static_cast<size_t>(row + q * 8) * stride + col];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int q = 0; q < 4; ++q) {
            int i = threadIdx.y + q * 8;
            unsigned int candidate = MODE == 0 ? center[i][k] + target[k][j]
                                               : target[i][k] + center[k][j];
            if (candidate < target[i][j]) {
                target[i][j] = candidate;
                path[static_cast<size_t>(row + q * 8) * stride + col] = pivot * TILE + k;
            }
        }
        __syncthreads();
    }
    for (int q = 0; q < 4; ++q)
        dist[static_cast<size_t>(row + q * 8) * stride + col] = target[threadIdx.y + q * 8][j];
}

__global__ void remainingKernel(unsigned int* dist, unsigned int* path,
                                const unsigned int* pivotRow, int stride,
                                int localTileStart, int pivot, int tiles) {
    int rowTile = blockIdx.y;
    int colTile = blockIdx.x;
    if (localTileStart + rowTile == pivot || colTile == pivot || colTile >= tiles) return;
    __shared__ unsigned int viaColumn[TILE][TILE + 1];
    __shared__ unsigned int viaRow[TILE][TILE + 1];
    int j = threadIdx.x;
    for (int q = 0; q < 4; ++q) {
        int i = threadIdx.y + q * 8;
        viaColumn[i][j] = dist[static_cast<size_t>(rowTile * TILE + i) * stride + pivot * TILE + j];
        viaRow[i][j] = pivotRow[static_cast<size_t>(i) * stride + colTile * TILE + j];
    }
    __syncthreads();
    for (int q = 0; q < 4; ++q) {
        int i = threadIdx.y + q * 8;
        size_t index = static_cast<size_t>(rowTile * TILE + i) * stride + colTile * TILE + j;
        unsigned int best = dist[index];
        unsigned int bestPath = path[index];
        for (int k = 0; k < TILE; ++k) {
            unsigned int candidate = viaColumn[i][k] + viaRow[k][j];
            if (candidate < best) {
                best = candidate;
                bestPath = pivot * TILE + k;
            }
        }
        dist[index] = best;
        path[index] = bestPath;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                unsigned int ik = dist[idx2(k, i, n)];
                unsigned int kj = dist[idx2(j, k, n)];
                if (ik < INF && kj < INF && ik + kj < dist[idx2(j, i, n)]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank = 0, world = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            unsigned long long value = strtoull(argv[++i], &end, 10);
            if (*end || value > INT_MAX - (TILE - 1)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes\n");
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            n = value;
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
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
    const int tiles = static_cast<int>((n + TILE - 1) / TILE);
    const int stride = tiles * TILE;
    if (static_cast<size_t>(stride) * stride > INT_MAX) {
        if (rank == 0) fprintf(stderr, "Graph exceeds MPI message limit\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    checkMpi(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm), MPI_COMM_WORLD);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(localComm, &localRank);
    checkCuda(cudaGetDeviceCount(&devices), MPI_COMM_WORLD);
    if (devices == 0) {
        if (rank == 0) fprintf(stderr, "CUDA device required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(localRank % devices), MPI_COMM_WORLD);
    MPI_Comm_free(&localComm);

    int base = tiles / world, extra = tiles % world;
    int startTile = rank * base + std::min(rank, extra);
    int localTiles = base + (rank < extra);
    int localRows = localTiles * TILE;
    std::vector<int> counts(world), offsets(world);
    for (int r = 0; r < world; ++r) {
        int first = r * base + std::min(r, extra);
        int num = base + (r < extra);
        if (static_cast<size_t>(num) * TILE * stride > INT_MAX ||
            static_cast<size_t>(first) * TILE * stride > INT_MAX) {
            if (rank == 0) fprintf(stderr, "Graph exceeds MPI message limit\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        counts[r] = num * TILE * stride;
        offsets[r] = first * TILE * stride;
    }
    std::vector<unsigned int> original;
    std::vector<unsigned int> padded;
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        original.resize(n * n);
        initializeDistanceMatrix(original, n, 1, MAX_DISTANCE);
        padded.resize(static_cast<size_t>(stride) * stride);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < stride; ++i) {
            unsigned int* row = padded.data() + static_cast<size_t>(i) * stride;
            std::fill(row, row + stride, INF);
            if (static_cast<size_t>(i) < n)
                std::copy_n(original.data() + static_cast<size_t>(i) * n, n, row);
            else row[i] = 0;
        }
    }
    std::vector<unsigned int> local(static_cast<size_t>(localRows) * stride);
    MPI_Scatterv(rank == 0 ? padded.data() : nullptr, counts.data(), offsets.data(), MPI_UNSIGNED,
                 local.data(), counts[rank], MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    unsigned int *dDist = nullptr, *dPath = nullptr, *dPivot = nullptr;
    size_t localBytes = std::max<size_t>(1, local.size()) * sizeof(unsigned int);
    size_t pivotBytes = std::max<size_t>(1, static_cast<size_t>(TILE) * stride) * sizeof(unsigned int);
    checkCuda(cudaMalloc(&dDist, localBytes), MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&dPath, localBytes), MPI_COMM_WORLD);
    checkCuda(cudaMalloc(&dPivot, pivotBytes), MPI_COMM_WORLD);
    if (!local.empty()) checkCuda(cudaMemcpy(dDist, local.data(), local.size() * sizeof(unsigned int), cudaMemcpyHostToDevice), MPI_COMM_WORLD);
    if (localRows) initPaths<<<(static_cast<size_t>(localRows) * stride + 255) / 256, 256>>>(dPath, localRows, stride);
    checkCuda(cudaGetLastError(), MPI_COMM_WORLD);
    unsigned int* pivotHost = nullptr;
    checkCuda(cudaHostAlloc(&pivotHost, pivotBytes, cudaHostAllocDefault), MPI_COMM_WORLD);
    dim3 threads(TILE, 8);
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto begin = std::chrono::steady_clock::now();
    for (int p = 0; p < tiles; ++p) {
        int owner = p < (base + 1) * extra ? p / (base + 1) : extra + (p - (base + 1) * extra) / base;
        if (rank == owner) {
            int pivotLocalRow = (p - startTile) * TILE;
            pivotKernel<<<1, threads>>>(dDist, dPath, stride, pivotLocalRow, p);
            borderKernel<0><<<tiles, threads>>>(dDist, dPath, dDist + static_cast<size_t>(pivotLocalRow) * stride,
                                                stride, startTile, p, tiles);
            checkCuda(cudaMemcpy(pivotHost, dDist + static_cast<size_t>(pivotLocalRow) * stride,
                                 TILE * static_cast<size_t>(stride) * sizeof(unsigned int), cudaMemcpyDeviceToHost), MPI_COMM_WORLD);
        }
        checkMpi(MPI_Bcast(pivotHost, TILE * stride, MPI_UNSIGNED, owner, MPI_COMM_WORLD), MPI_COMM_WORLD);
        const unsigned int* pivotDevice = dPivot;
        if (rank != owner) {
            checkCuda(cudaMemcpy(dPivot, pivotHost, TILE * static_cast<size_t>(stride) * sizeof(unsigned int),
                                 cudaMemcpyHostToDevice), MPI_COMM_WORLD);
        } else pivotDevice = dDist + static_cast<size_t>(p - startTile) * TILE * stride;
        if (localTiles) {
            borderKernel<1><<<localTiles, threads>>>(dDist, dPath, pivotDevice, stride, startTile, p, localTiles);
            remainingKernel<<<dim3(tiles, localTiles), threads>>>(dDist, dPath, pivotDevice, stride, startTile, p, tiles);
        }
        checkCuda(cudaGetLastError(), MPI_COMM_WORLD);
    }
    checkCuda(cudaDeviceSynchronize(), MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::steady_clock::now();
    if (!local.empty()) checkCuda(cudaMemcpy(local.data(), dDist, local.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost), MPI_COMM_WORLD);
    MPI_Gatherv(local.data(), counts[rank], MPI_UNSIGNED, rank == 0 ? padded.data() : nullptr,
                counts.data(), offsets.data(), MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    cudaFree(dDist);
    cudaFree(dPath);
    cudaFree(dPivot);
    cudaFreeHost(pivotHost);
    int result = 0;
    if (rank == 0) {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            std::copy_n(padded.data() + i * stride, n, original.data() + i * n);
        double seconds = std::chrono::duration<double>(end - begin).count();
        long milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
        printf("Computation time: %ld ms\n", milliseconds);
        printf("Performance: %.3f GOPS\n", static_cast<double>(n) * n * n / seconds / 1e9);
        if (printResults) print_results_int(original, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            result = validateResult(original, n) ? 0 : 1;
            printf("Validation: %s\n", result == 0 ? "PASSED" : "FAILED");
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
