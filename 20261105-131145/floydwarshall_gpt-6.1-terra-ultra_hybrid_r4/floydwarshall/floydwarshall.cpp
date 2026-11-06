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

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)

// Each block handles a 32 by 32 tile, with four elements per thread.
// Padding is INF, so it cannot introduce a shorter path between real nodes.
__global__ void diagonal(unsigned int* d, size_t stride, int local_pivot, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const size_t row = static_cast<size_t>(local_pivot) * TILE;
    const size_t col = static_cast<size_t>(pivot) * TILE;
    #pragma unroll
    for (int r = y; r < TILE; r += 8) tile[r][x] = d[(row + r) * stride + col + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = y; r < TILE; r += 8) {
            if (r != k && x != k) tile[r][x] = min(tile[r][x], tile[r][k] + tile[k][x]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = y; r < TILE; r += 8) d[(row + r) * stride + col + x] = tile[r][x];
}

__global__ void pivot_row(unsigned int* d, size_t stride, int local_pivot, int pivot) {
    const int tile_col = blockIdx.x;
    if (tile_col == pivot) return;
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t row = static_cast<size_t>(local_pivot) * TILE;
    #pragma unroll
    for (int r = y; r < TILE; r += 8) {
        diag[r][x] = d[(row + r) * stride + static_cast<size_t>(pivot) * TILE + x];
        tile[r][x] = d[(row + r) * stride + static_cast<size_t>(tile_col) * TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = y; r < TILE; r += 8) {
            if (r != k) tile[r][x] = min(tile[r][x], diag[r][k] + tile[k][x]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = y; r < TILE; r += 8) d[(row + r) * stride + static_cast<size_t>(tile_col) * TILE + x] = tile[r][x];
}

__global__ void pivot_column(unsigned int* d, const unsigned int* pivot_rows,
                             size_t stride, int first_tile, int pivot) {
    const int local_tile = blockIdx.x;
    if (first_tile + local_tile == pivot) return;
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t row = static_cast<size_t>(local_tile) * TILE;
    const size_t col = static_cast<size_t>(pivot) * TILE;
    #pragma unroll
    for (int r = y; r < TILE; r += 8) {
        diag[r][x] = pivot_rows[static_cast<size_t>(r) * stride + col + x];
        tile[r][x] = d[(row + r) * stride + col + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = y; r < TILE; r += 8) {
            if (x != k) tile[r][x] = min(tile[r][x], tile[r][k] + diag[k][x]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = y; r < TILE; r += 8) d[(row + r) * stride + col + x] = tile[r][x];
}

__global__ void remaining(unsigned int* d, const unsigned int* pivot_rows,
                          size_t stride, int first_tile, int pivot) {
    const int tile_col = blockIdx.x, local_tile = blockIdx.y;
    if (tile_col == pivot || first_tile + local_tile == pivot) return;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t row = static_cast<size_t>(local_tile) * TILE;
    const size_t col = static_cast<size_t>(tile_col) * TILE;
    #pragma unroll
    for (int r = y; r < TILE; r += 8) {
        left[r][x] = d[(row + r) * stride + static_cast<size_t>(pivot) * TILE + x];
        top[r][x] = pivot_rows[static_cast<size_t>(r) * stride + col + x];
    }
    __syncthreads();
    unsigned int value[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) value[t] = d[(row + y + t * 8) * stride + col + x];
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) value[t] = min(value[t], left[y + t * 8][k] + top[k][x]);
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) d[(row + y + t * 8) * stride + col + x] = value[t];
}

// MPI counts are int, while matrices may contain more than INT_MAX elements.
void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

void broadcast(unsigned int* data, size_t count, int owner) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
        data += chunk;
        count -= chunk;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[i * n + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[i*n+k] < INF && dist[k*n+j] < INF && dist[i*n+k] + dist[k*n+j] < dist[i*n+j]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\nOptions:\n", progName);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
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
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    if (n > static_cast<size_t>(INT_MAX - TILE) || (n && n > SIZE_MAX / n / sizeof(unsigned int))) {
        if (!rank) fprintf(stderr, "Matrix size is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank, devices;
    MPI_Comm_rank(local_comm, &local_rank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_Comm_free(&local_comm);

    const int tiles = static_cast<int>((n + TILE - 1) / TILE);
    const size_t stride = static_cast<size_t>(tiles) * TILE;
    auto tile_start = [=](int r) { return r * (tiles / ranks) + std::min(r, tiles % ranks); };
    auto row_start = [&](int r) { return std::min(n, static_cast<size_t>(tile_start(r)) * TILE); };
    const int first_tile = tile_start(rank), local_tiles = tile_start(rank + 1) - first_tile;
    const size_t first_row = row_start(rank), rows = row_start(rank + 1) - first_row;
    std::vector<unsigned int> dist;
    std::vector<unsigned int> local(rows * n);
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        unsigned int seed = 42;
        const double range = static_cast<double>(MAX_DISTANCE);
        // Generate slabs in original row-major order to preserve the random sequence
        // without retaining a full copy of the graph on rank zero.
        auto initialize = [&](std::vector<unsigned int>& slab, size_t begin, size_t slab_rows) {
            for (auto& value : slab)
                value = 1 + static_cast<unsigned int>(range * rand_r(&seed) / static_cast<double>(RAND_MAX));
            for (size_t i = 0; i < slab_rows; ++i) slab[i * n + begin + i] = 0;
        };
        initialize(local, first_row, rows);
        for (int r = 1; r < ranks; ++r) {
            size_t begin = row_start(r), slab_rows = row_start(r + 1) - begin;
            std::vector<unsigned int> slab(slab_rows * n);
            initialize(slab, begin, slab_rows);
            if (!slab.empty()) transfer(slab.data(), slab.size(), r, 0, true);
        }
    } else if (rows) transfer(local.data(), rows * n, 0, 0, false);

    const size_t local_padded_rows = static_cast<size_t>(local_tiles) * TILE;
    std::vector<unsigned int> padded(local_padded_rows * stride, INF);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) std::copy_n(local.data() + i * n, n, padded.data() + i * stride);
    local.clear();
    local.shrink_to_fit();
    unsigned int *device = nullptr, *device_pivot = nullptr, *host_pivot = nullptr;
    if (local_padded_rows) {
        CUDA_CHECK(cudaMalloc(&device, padded.size() * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(device, padded.data(), padded.size() * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if (stride && ranks > 1) {
        if (local_tiles) CUDA_CHECK(cudaMalloc(&device_pivot, TILE * stride * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&host_pivot, TILE * stride * sizeof(unsigned int)));
    }
    if (!rank) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    const dim3 threads(TILE, 8);
    for (int p = 0; p < tiles; ++p) {
        // The balanced contiguous tile distribution defines the pivot owner.
        int owner = 0;
        while (tile_start(owner + 1) <= p) ++owner;
        const unsigned int* pivot_data = device_pivot;
        if (rank == owner) {
            const int local_pivot = p - first_tile;
            pivot_data = device + static_cast<size_t>(local_pivot) * TILE * stride;
            diagonal<<<1, threads>>>(device, stride, local_pivot, p);
            CUDA_CHECK(cudaGetLastError());
            pivot_row<<<tiles, threads>>>(device, stride, local_pivot, p);
            CUDA_CHECK(cudaGetLastError());
            if (ranks > 1)
                CUDA_CHECK(cudaMemcpy(host_pivot, pivot_data, TILE * stride * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) broadcast(host_pivot, TILE * stride, owner);
        if (local_tiles) {
            if (rank != owner)
                CUDA_CHECK(cudaMemcpy(device_pivot, host_pivot, TILE * stride * sizeof(unsigned int), cudaMemcpyHostToDevice));
            pivot_column<<<local_tiles, threads>>>(device, pivot_data, stride, first_tile, p);
            CUDA_CHECK(cudaGetLastError());
            remaining<<<dim3(tiles, local_tiles), threads>>>(device, pivot_data, stride, first_tile, p);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    if (!rank) {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
        double ops = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", ops / (duration.count() / 1000.0) / 1e9);
    }
    if (validate || printResults) {
        if (local_padded_rows) CUDA_CHECK(cudaMemcpy(padded.data(), device, padded.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        local.resize(rows * n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i) std::copy_n(padded.data() + i * stride, n, local.data() + i * n);
        if (!rank) {
            dist.resize(n * n);
            std::copy_n(local.data(), rows * n, dist.data());
            for (int r = 1; r < ranks; ++r) {
                size_t begin = row_start(r), count = (row_start(r + 1) - begin) * n;
                if (count) transfer(dist.data() + begin * n, count, r, 1, false);
            }
        } else if (rows) transfer(local.data(), rows * n, 0, 1, true);
    }
    int status = 0;
    if (!rank) {
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (device) CUDA_CHECK(cudaFree(device));
    if (device_pivot) CUDA_CHECK(cudaFree(device_pivot));
    if (host_pivot) CUDA_CHECK(cudaFreeHost(host_pivot));
    MPI_Finalize();
    return status;
}
