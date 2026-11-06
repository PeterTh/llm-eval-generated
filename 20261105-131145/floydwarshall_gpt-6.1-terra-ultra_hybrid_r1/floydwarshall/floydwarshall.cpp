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

static void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Each block uses 256 threads to process a 32 by 32 tile. The shared-memory
// tiles let each global load contribute to 32 min-plus operations.
__global__ void closePivot(unsigned int* matrix, size_t stride, size_t localPivot,
                           size_t pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int* base = matrix + localPivot * TILE * stride + pivot * TILE;
    #pragma unroll
    for (int r = 0; r < 4; ++r) tile[y + r * 8][x] = base[(y + r * 8) * stride + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = y + r * 8;
            unsigned int value = tile[row][k] + tile[k][x];
            if (value < tile[row][x]) tile[row][x] = value;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) base[(y + r * 8) * stride + x] = tile[y + r * 8][x];
}

// The closed diagonal tile makes each of these min-plus products independent
// of the order of k. ROW updates the pivot row, otherwise the pivot column.
template<bool ROW>
__global__ void updatePanel(unsigned int* matrix, const unsigned int* pivotRow,
                            size_t stride, size_t pivot, size_t firstTile,
                            size_t localPivot, size_t count) {
    const size_t tileIndex = blockIdx.x;
    if (tileIndex >= count) return;
    if (ROW ? tileIndex == pivot : firstTile + tileIndex == pivot) return;
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int panel[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const unsigned int* diag = pivotRow + pivot * TILE;
    unsigned int* target = ROW ? matrix + localPivot * TILE * stride + tileIndex * TILE
                               : matrix + tileIndex * TILE * stride + pivot * TILE;
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int row = y + r * 8;
        diagonal[row][x] = diag[row * stride + x];
        panel[row][x] = target[row * stride + x];
    }
    __syncthreads();
    unsigned int value[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) value[r] = panel[y + r * 8][x];
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = y + r * 8;
            unsigned int candidate = ROW ? diagonal[row][k] + panel[k][x]
                                         : panel[row][k] + diagonal[k][x];
            value[r] = min(value[r], candidate);
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) target[(y + r * 8) * stride + x] = value[r];
}

__global__ void updateRemaining(unsigned int* matrix, const unsigned int* pivotRow,
                                size_t stride, size_t pivot, size_t firstTile) {
    const size_t colTile = blockIdx.x, rowTile = blockIdx.y;
    if (colTile == pivot || firstTile + rowTile == pivot) return;
    __shared__ unsigned int column[TILE][TILE];
    __shared__ unsigned int row[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int* target = matrix + rowTile * TILE * stride + colTile * TILE;
    const unsigned int* colBase = matrix + rowTile * TILE * stride + pivot * TILE;
    const unsigned int* rowBase = pivotRow + colTile * TILE;
    unsigned int value[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int rr = y + r * 8;
        column[rr][x] = colBase[rr * stride + x];
        row[rr][x] = rowBase[rr * stride + x];
        value[r] = target[rr * stride + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned int right = row[k][x];
        #pragma unroll
        for (int r = 0; r < 4; ++r)
            value[r] = min(value[r], column[y + r * 8][k] + right);
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) target[(y + r * 8) * stride + x] = value[r];
}

// MPI counts are int, while the matrices and their offsets are size_t.
static void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}
static void broadcast(unsigned int* data, size_t count, int owner) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}
[[maybe_unused]] static int hostThreads(size_t elements) {
    return std::min(std::min(16, omp_get_max_threads()),
                    static_cast<int>(std::min(size_t(16), std::max(size_t(1), elements / 131072))));
}
static size_t tileStart(int rank, size_t tiles, int ranks) {
    return (tiles / ranks) * rank + std::min(static_cast<size_t>(rank), tiles % ranks);
}
static int tileOwner(size_t tile, size_t tiles, int ranks) {
    size_t base = tiles / ranks, extra = tiles % ranks;
    size_t boundary = (base + 1) * extra;
    return tile < boundary ? static_cast<int>(tile / (base + 1))
                           : static_cast<int>(extra + (tile - boundary) / base);
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[i * n + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[i*n+k] < INF && dist[k*n+j] < INF &&
                    dist[i*n+k] + dist[k*n+j] < dist[i*n+j]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
    return true;
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
        if (!rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = static_cast<size_t>(atoi(argv[++i]));
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n > static_cast<size_t>(INT_MAX) || (n && n > SIZE_MAX / n / sizeof(unsigned int))) {
        if (!rank) fprintf(stderr, "Graph size is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t tiles = (n + TILE - 1) / TILE, stride = tiles * TILE;
    const size_t firstTile = tileStart(rank, tiles, ranks);
    const size_t localTiles = tileStart(rank + 1, tiles, ranks) - firstTile;
    const size_t localCount = localTiles * TILE * stride;
    if (tiles > 65535 || localTiles > 65535) {
        if (!rank) fprintf(stderr, "Graph exceeds CUDA grid limits\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank, devices;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_free(&localComm);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA(cudaSetDevice(localRank % devices));

    std::vector<unsigned int> dist;
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        dist.resize(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            dist[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
        for (size_t i = 0; i < n; ++i) dist[i*n+i] = 0;
    }
    unsigned int *host = nullptr, *device = nullptr, *hostPivot = nullptr, *devicePivot = nullptr;
    if (localCount) {
        CUDA(cudaMallocHost(&host, localCount * sizeof(unsigned int)));
        CUDA(cudaMalloc(&device, localCount * sizeof(unsigned int)));
        if (ranks > 1) CUDA(cudaMalloc(&devicePivot, TILE * stride * sizeof(unsigned int)));
    }
    if (tiles && ranks > 1) CUDA(cudaMallocHost(&hostPivot, TILE * stride * sizeof(unsigned int)));
    const size_t maxTiles = (tiles + ranks - 1) / ranks;
    std::vector<unsigned int> staging;
    if (!rank && ranks > 1) staging.resize(maxTiles * TILE * stride);
    if (!rank) {
        for (int p = 0; p < ranks; ++p) {
            size_t start = tileStart(p, tiles, ranks) * TILE;
            size_t rows = (tileStart(p+1, tiles, ranks) - tileStart(p, tiles, ranks)) * TILE;
            unsigned int* buffer = p ? staging.data() : host;
            #pragma omp parallel for schedule(static) num_threads(hostThreads(rows * stride)) if(rows * stride >= 262144)
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < stride; ++j)
                    buffer[i*stride+j] = (start+i < n && j < n) ? dist[(start+i)*n+j] : ((start+i == j) ? 0 : INF);
            }
            if (p) transfer(buffer, rows * stride, p, true);
        }
    } else if (localCount) transfer(host, localCount, 0, false);
    if (localCount) CUDA(cudaMemcpy(device, host, localCount * sizeof(unsigned int), cudaMemcpyHostToDevice));
    if (!rank && !validate && !printResults) { std::vector<unsigned int>().swap(dist); std::vector<unsigned int>().swap(staging); }
    if (!rank) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    const dim3 threads(TILE, 8);
    for (size_t pivot = 0; pivot < tiles; ++pivot) {
        int owner = tileOwner(pivot, tiles, ranks);
        unsigned int* pivotRow = devicePivot;
        if (rank == owner) {
            size_t localPivot = pivot - firstTile;
            pivotRow = device + localPivot * TILE * stride;
            closePivot<<<1, threads>>>(device, stride, localPivot, pivot);
            updatePanel<true><<<static_cast<unsigned int>(tiles), threads>>>(device, pivotRow, stride, pivot, firstTile, localPivot, tiles);
            CUDA(cudaGetLastError());
            if (ranks > 1) CUDA(cudaMemcpy(hostPivot, pivotRow, TILE * stride * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) broadcast(hostPivot, TILE * stride, owner);
        if (localCount) {
            if (rank != owner) CUDA(cudaMemcpy(devicePivot, hostPivot, TILE * stride * sizeof(unsigned int), cudaMemcpyHostToDevice));
            updatePanel<false><<<static_cast<unsigned int>(localTiles), threads>>>(device, pivotRow, stride, pivot, firstTile, 0, localTiles);
            updateRemaining<<<dim3(static_cast<unsigned int>(tiles), static_cast<unsigned int>(localTiles)), threads>>>(device, pivotRow, stride, pivot, firstTile);
            CUDA(cudaGetLastError());
        }
    }
    if (localCount) CUDA(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (!rank) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        double ops = static_cast<double>(n) * n * n;
        double seconds = std::chrono::duration<double>(end - start).count();
        printf("Performance: %.3f GOPS\n", ops / seconds / 1e9);
    }
    if (validate || printResults) {
        if (localCount) CUDA(cudaMemcpy(host, device, localCount * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (!rank) {
            for (int p = 0; p < ranks; ++p) {
                size_t rowStart = tileStart(p, tiles, ranks) * TILE;
                size_t rows = (tileStart(p+1, tiles, ranks) - tileStart(p, tiles, ranks)) * TILE;
                unsigned int* buffer = p ? staging.data() : host;
                if (p) transfer(buffer, rows * stride, p, false);
                size_t realRows = rowStart < n ? std::min(rows, n - rowStart) : 0;
                #pragma omp parallel for schedule(static) num_threads(hostThreads(realRows * n)) if(realRows * n >= 262144)
                for (size_t i = 0; i < realRows; ++i)
                    std::copy_n(buffer + i * stride, n, dist.data() + (rowStart+i) * n);
            }
        } else if (localCount) transfer(host, localCount, 0, true);
    }
    int result = 0;
    if (!rank) {
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    if (devicePivot) CUDA(cudaFree(devicePivot));
    if (device) CUDA(cudaFree(device));
    if (host) CUDA(cudaFreeHost(host));
    if (hostPivot) CUDA(cudaFreeHost(hostPivot));
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
