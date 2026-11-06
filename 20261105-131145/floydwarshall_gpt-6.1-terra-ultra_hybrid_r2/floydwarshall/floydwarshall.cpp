#include <algorithm>
#include <climits>
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
constexpr int THREAD_ROWS = 8;

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e_ = (call); if (e_ != MPI_SUCCESS) { \
    char msg_[MPI_MAX_ERROR_STRING]; int len_; MPI_Error_string(e_, msg_, &len_); \
    std::fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, len_, msg_); \
    MPI_Abort(MPI_COMM_WORLD, e_); } } while (0)

// A tile is handled by 256 threads, each owning four elements.  The two
// barriers per pivot separate reads from writes within the shared tile.
__global__ void pivot_tile(unsigned int* d, size_t stride, int local_tile, int pivot) {
    __shared__ unsigned int a[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t row0 = static_cast<size_t>(local_tile) * TILE;
    size_t col0 = static_cast<size_t>(pivot) * TILE;
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS)
        a[y+t][x] = d[(row0+y+t)*stride+col0+x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int v[TILE / THREAD_ROWS];
        #pragma unroll
        for (int t = 0; t < TILE; t += THREAD_ROWS)
            v[t/THREAD_ROWS] = min(a[y+t][x], a[y+t][k] + a[k][x]);
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; t += THREAD_ROWS) a[y+t][x] = v[t/THREAD_ROWS];
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS)
        d[(row0+y+t)*stride+col0+x] = a[y+t][x];
}

__global__ void pivot_row(unsigned int* d, size_t stride, int local_tile, int pivot) {
    int tile = blockIdx.x;
    if (tile == pivot) return;
    __shared__ unsigned int a[TILE][TILE], p[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t row0 = static_cast<size_t>(local_tile) * TILE;
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS) {
        a[y+t][x] = d[(row0+y+t)*stride+static_cast<size_t>(tile)*TILE+x];
        p[y+t][x] = d[(row0+y+t)*stride+static_cast<size_t>(pivot)*TILE+x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int v[TILE / THREAD_ROWS];
        #pragma unroll
        for (int t = 0; t < TILE; t += THREAD_ROWS)
            v[t/THREAD_ROWS] = min(a[y+t][x], p[y+t][k] + a[k][x]);
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; t += THREAD_ROWS) a[y+t][x] = v[t/THREAD_ROWS];
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS)
        d[(row0+y+t)*stride+static_cast<size_t>(tile)*TILE+x] = a[y+t][x];
}

__global__ void pivot_column(unsigned int* d, const unsigned int* row, size_t stride,
                             int first_tile, int pivot) {
    int local_tile = blockIdx.x;
    if (first_tile + local_tile == pivot) return;
    __shared__ unsigned int a[TILE][TILE], p[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t row0 = static_cast<size_t>(local_tile) * TILE;
    size_t col0 = static_cast<size_t>(pivot) * TILE;
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS) {
        a[y+t][x] = d[(row0+y+t)*stride+col0+x];
        p[y+t][x] = row[static_cast<size_t>(y+t)*stride+col0+x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int v[TILE / THREAD_ROWS];
        #pragma unroll
        for (int t = 0; t < TILE; t += THREAD_ROWS)
            v[t/THREAD_ROWS] = min(a[y+t][x], a[y+t][k] + p[k][x]);
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < TILE; t += THREAD_ROWS) a[y+t][x] = v[t/THREAD_ROWS];
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS)
        d[(row0+y+t)*stride+col0+x] = a[y+t][x];
}

__global__ void remaining_tiles(unsigned int* d, const unsigned int* row, size_t stride,
                                int first_tile, int pivot) {
    int col_tile = blockIdx.x, local_tile = blockIdx.y;
    if (col_tile == pivot || first_tile + local_tile == pivot) return;
    __shared__ unsigned int left[TILE][TILE], top[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t row0 = static_cast<size_t>(local_tile) * TILE;
    size_t col0 = static_cast<size_t>(col_tile) * TILE;
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS) {
        left[y+t][x] = d[(row0+y+t)*stride+static_cast<size_t>(pivot)*TILE+x];
        top[y+t][x] = row[static_cast<size_t>(y+t)*stride+col0+x];
    }
    __syncthreads();
    unsigned int v[TILE / THREAD_ROWS];
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS)
        v[t/THREAD_ROWS] = d[(row0+y+t)*stride+col0+x];
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned int b = top[k][x];
        #pragma unroll
        for (int t = 0; t < TILE; t += THREAD_ROWS)
            v[t/THREAD_ROWS] = min(v[t/THREAD_ROWS], left[y+t][k] + b);
    }
    #pragma unroll
    for (int t = 0; t < TILE; t += THREAD_ROWS)
        d[(row0+y+t)*stride+col0+x] = v[t/THREAD_ROWS];
}

// MPI counts are ints; split large transfers instead of truncating them.
void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count) {
        int part = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_CHECK(MPI_Send(data, part, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD));
        else MPI_CHECK(MPI_Recv(data, part, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += part;
        count -= part;
    }
}
void broadcast(unsigned int* data, size_t count, int root) {
    while (count) {
        int part = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, part, MPI_UNSIGNED, root, MPI_COMM_WORLD));
        data += part;
        count -= part;
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[i*n+i] != 0) {
            std::printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                unsigned int ik = dist[i*n+k], kj = dist[k*n+j];
                if (ik < INF && kj < INF && ik + kj < dist[i*n+j]) {
                    std::printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* name) {
    std::printf("Usage: %s [options]\nOptions:\n", name);
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank, ranks;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) std::fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i+1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            unsigned long long parsed = std::strtoull(value, &end, 10);
            if (*value == '-' || *value == '\0' || *end != '\0' || parsed > static_cast<unsigned long long>(INT_MAX - TILE)) {
                if (!rank) std::fprintf(stderr, "Invalid number of nodes: %s\n", value);
                MPI_Finalize(); return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    int tiles = static_cast<int>((n + TILE - 1) / TILE);
    size_t stride = static_cast<size_t>(tiles) * TILE;
    if (stride && stride > std::numeric_limits<size_t>::max() / stride / sizeof(unsigned int)) {
        if (!rank) std::fprintf(stderr, "Matrix is too large\n");
        MPI_Finalize(); return 1;
    }
    auto tile_count = [=](int r) { return tiles / ranks + (r < tiles % ranks); };
    auto tile_start = [=](int r) { return r * (tiles / ranks) + std::min(r, tiles % ranks); };
    int local_tiles = tile_count(rank), first_tile = tile_start(rank);
    size_t local_elems = static_cast<size_t>(local_tiles) * TILE * stride;
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank, devices;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { std::fprintf(stderr, "No CUDA device available on rank %d\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    if (!rank) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        std::printf("Initializing graph...\n");
    }
    std::vector<unsigned int> local(local_elems);
    unsigned int seed = 42;
    if (!rank) {
        for (int r = 0; r < ranks; ++r) {
            size_t count = static_cast<size_t>(tile_count(r)) * TILE * stride;
            std::vector<unsigned int> buffer;
            unsigned int* dst;
            if (r == 0) dst = local.data();
            else { buffer.resize(count); dst = buffer.data(); }
            #pragma omp parallel for if(count > 1048576)
            for (size_t p = 0; p < count; ++p) dst[p] = INF;
            size_t start_row = static_cast<size_t>(tile_start(r)) * TILE;
            size_t rows = static_cast<size_t>(tile_count(r)) * TILE;
            for (size_t i = 0; i < rows && start_row+i < n; ++i) {
                for (size_t j = 0; j < n; ++j)
                    dst[i*stride+j] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
                dst[i*stride+start_row+i] = 0;
            }
            if (r) transfer(dst, count, r, 10, true);
        }
    } else transfer(local.data(), local_elems, 0, 10, false);

    unsigned int *device = nullptr, *device_row = nullptr, *host_row = nullptr;
    size_t row_elems = static_cast<size_t>(TILE) * stride;
    if (local_elems) {
        CUDA_CHECK(cudaMalloc(&device, local_elems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(device, local.data(), local_elems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if (ranks > 1 && row_elems) {
        if (local_elems) CUDA_CHECK(cudaMalloc(&device_row, row_elems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&host_row, row_elems * sizeof(unsigned int)));
    }
    if (!rank) std::printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    const dim3 threads(TILE, THREAD_ROWS);
    int owner = 0;
    for (int k = 0; k < tiles; ++k) {
        while (k >= tile_start(owner) + tile_count(owner)) ++owner;
        if (rank == owner) {
            int local_pivot = k - first_tile;
            unsigned int* pivot_data = device + static_cast<size_t>(local_pivot)*TILE*stride;
            pivot_tile<<<1, threads>>>(device, stride, local_pivot, k);
            pivot_row<<<tiles, threads>>>(device, stride, local_pivot, k);
            CUDA_CHECK(cudaGetLastError());
            if (ranks > 1)
                CUDA_CHECK(cudaMemcpy(host_row, pivot_data, row_elems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
            // The owner's GPU can update its remaining tiles while MPI distributes
            // the copied pivot row. With one rank no host transfer is needed.
            pivot_column<<<local_tiles, threads>>>(device, pivot_data, stride, first_tile, k);
            remaining_tiles<<<dim3(tiles, local_tiles), threads>>>(device, pivot_data, stride, first_tile, k);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ranks > 1) broadcast(host_row, row_elems, owner);
        if (rank != owner && local_tiles) {
            CUDA_CHECK(cudaMemcpy(device_row, host_row, row_elems*sizeof(unsigned int), cudaMemcpyHostToDevice));
            pivot_column<<<local_tiles, threads>>>(device, device_row, stride, first_tile, k);
            remaining_tiles<<<dim3(tiles, local_tiles), threads>>>(device, device_row, stride, first_tile, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!rank) {
        long milliseconds = static_cast<long>(max_elapsed * 1000.0);
        std::printf("Computation time: %ld ms\n", milliseconds);
        double ops = static_cast<double>(n) * n * n;
        std::printf("Performance: %.3f GOPS\n", ops / max_elapsed / 1e9);
    }
    int result = 0;
    if (validate || printResults) {
        if (local_elems) CUDA_CHECK(cudaMemcpy(local.data(), device, local_elems*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        std::vector<unsigned int> dist;
        if (!rank) {
            dist.resize(n*n);
            for (int r = 0; r < ranks; ++r) {
                size_t rows = static_cast<size_t>(tile_count(r))*TILE;
                size_t start_row = static_cast<size_t>(tile_start(r))*TILE;
                std::vector<unsigned int> buffer;
                const unsigned int* src;
                if (!r) src = local.data();
                else { buffer.resize(rows*stride); transfer(buffer.data(), buffer.size(), r, 11, false); src = buffer.data(); }
                size_t valid_rows = start_row < n ? std::min(rows, n-start_row) : 0;
                #pragma omp parallel for if(valid_rows * n > 1048576)
                for (size_t i = 0; i < valid_rows; ++i)
                    std::copy_n(src+i*stride, n, dist.data()+(start_row+i)*n);
            }
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                std::printf("Validating result...\n");
                bool valid = validateResult(dist, n);
                std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                result = valid ? 0 : 1;
            }
        } else transfer(local.data(), local_elems, 0, 11, true);
        MPI_CHECK(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    }
    if (device) CUDA_CHECK(cudaFree(device));
    if (device_row) CUDA_CHECK(cudaFree(device_row));
    if (host_row) CUDA_CHECK(cudaFreeHost(host_row));
    MPI_CHECK(MPI_Finalize());
    return result;
}
