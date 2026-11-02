#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

static void cudaCheck(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank, file, line,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), __FILE__, __LINE__)

// Matrices are stored as contiguous source rows, just as in the original code.
__global__ void fillPadding(unsigned int* matrix, size_t count) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(blockDim.x) * gridDim.x)
        matrix[i] = INF;
}

// Complete the diagonal tile for this group of intermediate vertices.
__global__ void pivotTile(unsigned int* matrix, size_t pitch, int localTile, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    int y = threadIdx.y, x = threadIdx.x;
    size_t pos = (static_cast<size_t>(localTile) * TILE + y) * pitch + pivot * TILE + x;
    tile[y][x] = matrix[pos];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int v = tile[y][k] + tile[k][x];
        __syncthreads();
        if (v < tile[y][x]) tile[y][x] = v;
        __syncthreads();
    }
    matrix[pos] = tile[y][x];
}

// Complete every other tile in the pivot row, using the completed diagonal tile.
__global__ void pivotRow(unsigned int* matrix, size_t pitch, int localTile, int pivot) {
    int columnTile = blockIdx.x;
    if (columnTile == pivot) return;
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int row[TILE][TILE];
    int y = threadIdx.y, x = threadIdx.x;
    size_t base = (static_cast<size_t>(localTile) * TILE + y) * pitch;
    diagonal[y][x] = matrix[base + pivot * TILE + x];
    row[y][x] = matrix[base + columnTile * TILE + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int v = diagonal[y][k] + row[k][x];
        __syncthreads();
        if (v < row[y][x]) row[y][x] = v;
        __syncthreads();
    }
    matrix[base + columnTile * TILE + x] = row[y][x];
}

// Complete the pivot column for all locally owned tile rows.
__global__ void pivotColumn(unsigned int* matrix, const unsigned int* pivotRowData,
                            size_t pitch, int firstTile, int pivot) {
    int localTile = blockIdx.x;
    if (firstTile + localTile == pivot) return;
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int column[TILE][TILE];
    int y = threadIdx.y, x = threadIdx.x;
    size_t pos = (static_cast<size_t>(localTile) * TILE + y) * pitch + pivot * TILE + x;
    diagonal[y][x] = pivotRowData[static_cast<size_t>(y) * pitch + pivot * TILE + x];
    column[y][x] = matrix[pos];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int v = column[y][k] + diagonal[k][x];
        __syncthreads();
        if (v < column[y][x]) column[y][x] = v;
        __syncthreads();
    }
    matrix[pos] = column[y][x];
}

// Update the remaining tiles from the completed pivot row and column.
__global__ void remainingTiles(unsigned int* matrix, const unsigned int* pivotRowData,
                               size_t pitch, int firstTile, int pivot) {
    int columnTile = blockIdx.x, localTile = blockIdx.y;
    if (columnTile == pivot || firstTile + localTile == pivot) return;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE];
    int y = threadIdx.y, x = threadIdx.x;
    size_t base = (static_cast<size_t>(localTile) * TILE + y) * pitch;
    left[y][x] = matrix[base + pivot * TILE + x];
    top[y][x] = pivotRowData[static_cast<size_t>(y) * pitch + columnTile * TILE + x];
    unsigned int value = matrix[base + columnTile * TILE + x];
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned int v = left[y][k] + top[k][x];
        value = min(value, v);
    }
    matrix[base + columnTile * TILE + x] = value;
}

static void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count) {
        int part = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, part, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, part, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += part;
        count -= part;
    }
}

static void broadcast(unsigned int* data, size_t count, int root) {
    while (count) {
        int part = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, part, MPI_UNSIGNED, root, MPI_COMM_WORLD);
        data += part;
        count -= part;
    }
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (dist[i * n + i] != 0) valid = 0;
    }
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j) {
            for (size_t k = 0; k < n; ++k) {
                unsigned int ik = dist[i * n + k], kj = dist[k * n + j];
                if (ik < INF && kj < INF && ik + kj < dist[i * n + j]) valid = 0;
            }
        }
    }
    return valid != 0;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) fprintf(stderr, "MPI does not provide the required thread support\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (value[0] == '-' || *value == '\0' || *end != '\0' ||
                parsed > static_cast<unsigned long long>(INT_MAX - TILE)) {
                if (!rank) fprintf(stderr, "Invalid number of nodes: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n && n > std::numeric_limits<size_t>::max() / n / sizeof(unsigned int)) {
        if (!rank) fprintf(stderr, "Matrix size is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, deviceCount;
    MPI_Comm_rank(localComm, &localRank);
    CUDA(cudaGetDeviceCount(&deviceCount));
    if (!deviceCount) {
        fprintf(stderr, "Rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % deviceCount));
    MPI_Comm_free(&localComm);

    int tiles = static_cast<int>((n + TILE - 1) / TILE);
    size_t pitch = static_cast<size_t>(tiles) * TILE;
    auto firstTileFor = [=](int r) { return static_cast<int>(static_cast<long long>(tiles) * r / ranks); };
    int firstTile = firstTileFor(rank), localTiles = firstTileFor(rank + 1) - firstTile;
    size_t firstRow = static_cast<size_t>(firstTile) * TILE;
    size_t actualRows = firstRow < n ? std::min(static_cast<size_t>(localTiles) * TILE, n - firstRow) : 0;
    std::vector<unsigned int> full;
    std::vector<unsigned int> local(actualRows * n);
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        full.resize(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            full[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) full[i * n + i] = 0;
        for (int r = 0; r < ranks; ++r) {
            size_t start = static_cast<size_t>(firstTileFor(r)) * TILE;
            size_t end = std::min(n, static_cast<size_t>(firstTileFor(r + 1)) * TILE);
            if (start >= end) continue;
            if (r == 0) std::copy_n(full.data() + start * n, (end - start) * n, local.data());
            else transfer(full.data() + start * n, (end - start) * n, r, 0, true);
        }
        if (!validate && !printResults) std::vector<unsigned int>().swap(full);
    } else {
        transfer(local.data(), local.size(), 0, 0, false);
    }

    unsigned int *deviceMatrix = nullptr, *devicePivot = nullptr, *hostPivot = nullptr;
    size_t deviceCountElements = static_cast<size_t>(localTiles) * TILE * pitch;
    if (deviceCountElements) {
        CUDA(cudaMalloc(&deviceMatrix, deviceCountElements * sizeof(unsigned int)));
        int blocks = static_cast<int>(std::min(size_t(65535), (deviceCountElements + 255) / 256));
        fillPadding<<<blocks, 256>>>(deviceMatrix, deviceCountElements);
        CUDA(cudaGetLastError());
        CUDA(cudaMemcpy2D(deviceMatrix, pitch * sizeof(unsigned int), local.data(), n * sizeof(unsigned int),
                          n * sizeof(unsigned int), actualRows, cudaMemcpyHostToDevice));
    }
    std::vector<unsigned int>().swap(local);
    if (tiles && ranks > 1) {
        CUDA(cudaMalloc(&devicePivot, TILE * pitch * sizeof(unsigned int)));
        CUDA(cudaMallocHost(&hostPivot, TILE * pitch * sizeof(unsigned int)));
    }
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) printf("Computing shortest paths...\n");
    double startTime = MPI_Wtime();
    dim3 threads(TILE, TILE);
    int owner = 0;
    for (int k = 0; k < tiles; ++k) {
        while (firstTileFor(owner + 1) <= k) ++owner;
        unsigned int* row = devicePivot;
        if (rank == owner) {
            int localPivot = k - firstTile;
            pivotTile<<<1, threads>>>(deviceMatrix, pitch, localPivot, k);
            pivotRow<<<tiles, threads>>>(deviceMatrix, pitch, localPivot, k);
            CUDA(cudaGetLastError());
            row = deviceMatrix + static_cast<size_t>(localPivot) * TILE * pitch;
            if (ranks > 1)
                CUDA(cudaMemcpy(hostPivot, row, TILE * pitch * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) {
            broadcast(hostPivot, TILE * pitch, owner);
            if (rank != owner && localTiles)
                CUDA(cudaMemcpy(devicePivot, hostPivot, TILE * pitch * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        if (localTiles) {
            pivotColumn<<<localTiles, threads>>>(deviceMatrix, row, pitch, firstTile, k);
            remainingTiles<<<dim3(tiles, localTiles), threads>>>(deviceMatrix, row, pitch, firstTile, k);
            CUDA(cudaGetLastError());
        }
    }
    CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - startTime;
    if (!rank) {
        long long millis = static_cast<long long>(elapsed * 1000.0);
        printf("Computation time: %lld ms\n", millis);
        printf("Performance: %.3f GOPS\n", elapsed > 0 ? double(n) * n * n / elapsed / 1e9 : 0.0);
    }

    if (validate || printResults) {
        local.resize(actualRows * n);
        if (actualRows)
            CUDA(cudaMemcpy2D(local.data(), n * sizeof(unsigned int), deviceMatrix, pitch * sizeof(unsigned int),
                              n * sizeof(unsigned int), actualRows, cudaMemcpyDeviceToHost));
        if (!rank) {
            for (int r = 0; r < ranks; ++r) {
                size_t begin = static_cast<size_t>(firstTileFor(r)) * TILE;
                size_t end = std::min(n, static_cast<size_t>(firstTileFor(r + 1)) * TILE);
                if (begin >= end) continue;
                if (r == 0) std::copy_n(local.data(), (end - begin) * n, full.data() + begin * n);
                else transfer(full.data() + begin * n, (end - begin) * n, r, 1, false);
            }
        } else transfer(local.data(), local.size(), 0, 1, true);
    }
    if (deviceMatrix) CUDA(cudaFree(deviceMatrix));
    if (devicePivot) CUDA(cudaFree(devicePivot));
    if (hostPivot) CUDA(cudaFreeHost(hostPivot));
    int result = 0;
    if (!rank) {
        if (printResults) print_results_int(full, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(full, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
