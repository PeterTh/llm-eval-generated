#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

static int rank_id = 0;

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "Rank %d: CUDA error at %s:%d: %s\n", rank_id, __FILE__, __LINE__, cudaGetErrorString(e)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int e = (call); if (e != MPI_SUCCESS) { \
    char msg[MPI_MAX_ERROR_STRING]; int len; MPI_Error_string(e, msg, &len); \
    fprintf(stderr, "Rank %d: MPI error at %s:%d: %.*s\n", rank_id, __FILE__, __LINE__, len, msg); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Each block uses 256 threads, with four rows per thread. The pivot tile is
// closed first; the following two kernels update the pivot row and column.
__global__ void pivot_kernel(unsigned int* matrix, size_t pitch, int local_pivot, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        tile[y + 8*t][x] = matrix[(size_t)(local_pivot*TILE + y + 8*t)*pitch + pivot*TILE + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int row = y + 8*t;
            unsigned int v = tile[row][k] + tile[k][x];
            if (v < tile[row][x]) tile[row][x] = v;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        matrix[(size_t)(local_pivot*TILE + y + 8*t)*pitch + pivot*TILE + x] = tile[y + 8*t][x];
}

__global__ void pivot_row_kernel(unsigned int* matrix, size_t pitch, int local_pivot, int pivot) {
    int block = blockIdx.x;
    if (block == pivot) return;
    __shared__ unsigned int p[TILE][TILE], a[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int row = y + 8*t;
        p[row][x] = matrix[(size_t)(local_pivot*TILE + row)*pitch + pivot*TILE + x];
        a[row][x] = matrix[(size_t)(local_pivot*TILE + row)*pitch + block*TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int row = y + 8*t;
            unsigned int v = p[row][k] + a[k][x];
            if (v < a[row][x]) a[row][x] = v;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        matrix[(size_t)(local_pivot*TILE + y + 8*t)*pitch + block*TILE + x] = a[y + 8*t][x];
}

__global__ void pivot_column_kernel(unsigned int* matrix, const unsigned int* pivot_row,
                                    size_t pitch, int first_block, int pivot) {
    int block = blockIdx.x;
    if (first_block + block == pivot) return;
    __shared__ unsigned int p[TILE][TILE], a[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int row = y + 8*t;
        p[row][x] = pivot_row[(size_t)row*pitch + pivot*TILE + x];
        a[row][x] = matrix[(size_t)(block*TILE + row)*pitch + pivot*TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int row = y + 8*t;
            unsigned int v = a[row][k] + p[k][x];
            if (v < a[row][x]) a[row][x] = v;
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        matrix[(size_t)(block*TILE + y + 8*t)*pitch + pivot*TILE + x] = a[y + 8*t][x];
}

__global__ void remaining_kernel(unsigned int* matrix, const unsigned int* pivot_row,
                                 size_t pitch, int first_block, int pivot) {
    int col = blockIdx.x, row_block = blockIdx.y;
    if (col == pivot || first_block + row_block == pivot) return;
    __shared__ unsigned int left[TILE][TILE], top[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    unsigned int value[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int row = y + 8*t;
        left[row][x] = matrix[(size_t)(row_block*TILE + row)*pitch + pivot*TILE + x];
        top[row][x] = pivot_row[(size_t)row*pitch + col*TILE + x];
        value[t] = matrix[(size_t)(row_block*TILE + row)*pitch + col*TILE + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned int v = left[y + 8*t][k] + top[k][x];
            if (v < value[t]) value[t] = v;
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        matrix[(size_t)(row_block*TILE + y + 8*t)*pitch + col*TILE + x] = value[t];
}

// MPI's count is an int, so transfer large matrices in chunks.
static void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_CHECK(MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD));
        else MPI_CHECK(MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk; count -= chunk;
    }
}
static void broadcast(unsigned int* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_UNSIGNED, root, MPI_COMM_WORLD));
        data += chunk; count -= chunk;
    }
}
static int block_count(int rank, int ranks, int blocks) {
    return blocks / ranks + (rank < blocks % ranks);
}
static int block_start(int rank, int ranks, int blocks) {
    return rank * (blocks / ranks) + std::min(rank, blocks % ranks);
}
static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[i*n+i] != 0) {
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
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank_id));
    int ranks; MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank_id) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t n = 512;
    bool validate = false, print = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end; long long value = strtoll(argv[++i], &end, 10);
            if (*end || end == argv[i] || value < 0 || value > INT_MAX - TILE) {
                if (!rank_id) fprintf(stderr, "Invalid number of nodes: %s\n", argv[i]);
                MPI_Finalize(); return 1;
            }
            n = static_cast<size_t>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank_id) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!rank_id) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    int blocks = static_cast<int>((n + TILE - 1) / TILE);
    size_t pitch = static_cast<size_t>(blocks) * TILE;
    if (pitch && pitch > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / pitch) {
        if (!rank_id) fprintf(stderr, "Matrix is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm local;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank_id, MPI_INFO_NULL, &local));
    int local_rank; MPI_CHECK(MPI_Comm_rank(local, &local_rank));
    int devices; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank_id); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local));

    int first = block_start(rank_id, ranks, blocks), owned = block_count(rank_id, ranks, blocks);
    size_t local_size = static_cast<size_t>(owned) * TILE * pitch;
    std::vector<unsigned int> matrix;
    if (!rank_id) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        matrix.resize(pitch*pitch);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < pitch; ++i)
            for (size_t j = 0; j < pitch; ++j)
                matrix[i*pitch+j] = (i == j) ? 0 : INF;
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j) {
                unsigned int v = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
                matrix[i*pitch+j] = i == j ? 0 : v;
            }
    }
    std::vector<unsigned int> local_matrix(local_size);
    if (!rank_id) {
        if (local_size) std::copy_n(matrix.data(), local_size, local_matrix.data());
        for (int r = 1; r < ranks; ++r) {
            size_t count = static_cast<size_t>(block_count(r,ranks,blocks))*TILE*pitch;
            if (count) transfer(matrix.data() + static_cast<size_t>(block_start(r,ranks,blocks))*TILE*pitch, count, r, true);
        }
    } else transfer(local_matrix.data(), local_size, 0, false);
    unsigned int *device = nullptr, *device_pivot = nullptr, *host_pivot = nullptr;
    if (local_size) {
        CUDA_CHECK(cudaMalloc(&device, local_size*sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(device, local_matrix.data(), local_size*sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    size_t pivot_size = static_cast<size_t>(TILE)*pitch;
    if (ranks > 1 && pivot_size) {
        CUDA_CHECK(cudaMallocHost(&host_pivot, pivot_size*sizeof(unsigned int)));
        if (local_size) CUDA_CHECK(cudaMalloc(&device_pivot, pivot_size*sizeof(unsigned int)));
    }
    // Load kernels before timing so lazy CUDA module loading is not measured.
    if (owned) {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, pivot_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, pivot_row_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, pivot_column_kernel));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, remaining_kernel));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!rank_id) printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();
    dim3 threads(TILE, 8);
    for (int k = 0; k < blocks; ++k) {
        int owner = 0;
        // The number of ranks with blocks may exceed the number of blocks.
        int large = blocks % ranks, per = blocks / ranks;
        if (k < (per + 1)*large) owner = k / (per + 1);
        else owner = large + (k - (per + 1)*large) / per;
        unsigned int* pivot_ptr = device_pivot;
        if (rank_id == owner) {
            int local_k = k - first;
            pivot_ptr = device + static_cast<size_t>(local_k)*TILE*pitch;
            pivot_kernel<<<1, threads>>>(device, pitch, local_k, k);
            pivot_row_kernel<<<blocks, threads>>>(device, pitch, local_k, k);
            CUDA_CHECK(cudaGetLastError());
            if (ranks > 1) CUDA_CHECK(cudaMemcpy(host_pivot, pivot_ptr, pivot_size*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) {
            broadcast(host_pivot, pivot_size, owner);
            if (rank_id != owner && local_size)
                CUDA_CHECK(cudaMemcpy(device_pivot, host_pivot, pivot_size*sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        if (owned) {
            pivot_column_kernel<<<owned, threads>>>(device, pivot_ptr, pitch, first, k);
            remaining_kernel<<<dim3(blocks, owned), threads>>>(device, pivot_ptr, pitch, first, k);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    if (!rank_id) {
        double seconds = std::chrono::duration<double>(end - start).count();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        printf("Computation time: %lld ms\n", static_cast<long long>(ms));
        printf("Performance: %.3f GOPS\n", (double)n*n*n / seconds / 1e9);
    }
    if (print || validate) {
        if (local_size) CUDA_CHECK(cudaMemcpy(local_matrix.data(), device, local_size*sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (!rank_id) {
            if (local_size) std::copy_n(local_matrix.data(), local_size, matrix.data());
            for (int r = 1; r < ranks; ++r) {
                size_t count = static_cast<size_t>(block_count(r,ranks,blocks))*TILE*pitch;
                if (count) transfer(matrix.data() + static_cast<size_t>(block_start(r,ranks,blocks))*TILE*pitch, count, r, false);
            }
        } else transfer(local_matrix.data(), local_size, 0, true);
    }
    int valid = 1;
    if (!rank_id && (print || validate)) {
        std::vector<unsigned int> result(n*n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            std::copy_n(matrix.data()+i*pitch, n, result.data()+i*n);
        if (print) print_results_int(result, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            valid = validateResult(result, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    MPI_CHECK(MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (device) CUDA_CHECK(cudaFree(device));
    if (device_pivot) CUDA_CHECK(cudaFree(device_pivot));
    if (host_pivot) CUDA_CHECK(cudaFreeHost(host_pivot));
    MPI_CHECK(MPI_Finalize());
    return valid ? 0 : 1;
}
