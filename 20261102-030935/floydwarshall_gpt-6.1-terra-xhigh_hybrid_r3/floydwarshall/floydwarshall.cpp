#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <mpi.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

#define CUDA_CHECK(call) do { cudaError_t error_ = (call); if (error_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)
#define MPI_CHECK(call) do { int error_ = (call); if (error_ != MPI_SUCCESS) { \
    char message_[MPI_MAX_ERROR_STRING]; int length_ = 0; \
    MPI_Error_string(error_, message_, &length_); \
    fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length_, message_); \
    MPI_Abort(MPI_COMM_WORLD, 1); } } while (0)

// Each block has 32x8 threads, with four elements per thread. The matrix is
// padded to complete tiles, so the kernels need no edge branches.
__global__ void close_pivot(unsigned int* matrix, size_t stride, int local_block, int pivot_block) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t base = size_t(local_block) * TILE * stride + size_t(pivot_block) * TILE;
    #pragma unroll
    for (int q = 0; q < 4; ++q) tile[y + q * 8][x] = matrix[base + size_t(y + q * 8) * stride + x];
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int r = y + q * 8;
            if (r != k && x != k)
                tile[r][x] = min(tile[r][x], tile[r][k] + tile[k][x]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) matrix[base + size_t(y + q * 8) * stride + x] = tile[y + q * 8][x];
}

// Update either the pivot row or the pivot column using the closed diagonal tile.
template<bool ROW>
__global__ void update_cross(unsigned int* matrix, const unsigned int* pivot,
                             size_t stride, int pivot_block, int first_block,
                             int local_pivot_block) {
    int block = blockIdx.x;
    if ((ROW && block == pivot_block) || (!ROW && block + first_block == pivot_block)) return;
    __shared__ unsigned int diag[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t base = ROW ? size_t(local_pivot_block) * TILE * stride + size_t(block) * TILE
                      : size_t(block) * TILE * stride + size_t(pivot_block) * TILE;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int r = y + q * 8;
        diag[r][x] = pivot[size_t(r) * stride + size_t(pivot_block) * TILE + x];
        tile[r][x] = matrix[base + size_t(r) * stride + x];
    }
    __syncthreads();
    #pragma unroll 1
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int r = y + q * 8;
            unsigned int d = ROW ? diag[r][k] + tile[k][x] : tile[r][k] + diag[k][x];
            if ((ROW && r != k) || (!ROW && x != k)) tile[r][x] = min(tile[r][x], d);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q) matrix[base + size_t(y + q * 8) * stride + x] = tile[y + q * 8][x];
}

__global__ void update_remaining(unsigned int* matrix, const unsigned int* pivot,
                                 size_t stride, int pivot_block, int first_block) {
    int bc = blockIdx.x, br = blockIdx.y;
    if (bc == pivot_block || br + first_block == pivot_block) return;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int top[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t rowbase = size_t(br) * TILE * stride;
    unsigned int value[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int r = y + q * 8;
        left[r][x] = matrix[rowbase + size_t(r) * stride + size_t(pivot_block) * TILE + x];
        top[r][x] = pivot[size_t(r) * stride + size_t(bc) * TILE + x];
        value[q] = matrix[rowbase + size_t(r) * stride + size_t(bc) * TILE + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned int b = top[k][x];
        #pragma unroll
        for (int q = 0; q < 4; ++q) value[q] = min(value[q], left[y + q * 8][k] + b);
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q)
        matrix[rowbase + size_t(y + q * 8) * stride + size_t(bc) * TILE + x] = value[q];
}

// MPI's traditional count argument is an int; split transfers for large matrices.
void transfer(unsigned int* data, size_t count, int peer, int tag, bool send) {
    while (count) {
        int part = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_CHECK(MPI_Send(data, part, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD));
        else MPI_CHECK(MPI_Recv(data, part, MPI_UNSIGNED, peer, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += part; count -= part;
    }
}
void broadcast(unsigned int* data, size_t count, int root) {
    while (count) {
        int part = static_cast<int>(std::min(count, size_t(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, part, MPI_UNSIGNED, root, MPI_COMM_WORLD));
        data += part; count -= part;
    }
}
int block_start(int rank, int blocks, int ranks) {
    return rank * (blocks / ranks) + std::min(rank, blocks % ranks);
}
int block_owner(int block, int blocks, int ranks) {
    int larger = blocks % ranks, width = blocks / ranks;
    int split = (width + 1) * larger;
    return block < split ? block / (width + 1) : larger + (block - split) / width;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    size_t diagonal = n;
    #pragma omp parallel for reduction(min:diagonal)
    for (size_t i = 0; i < n; ++i)
        if (dist[i * n + i] != 0) diagonal = std::min(diagonal, i);
    if (diagonal != n) {
        printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", diagonal, diagonal);
        return false;
    }
    const size_t limit = std::min(n, size_t(10));
    size_t failure = limit * limit * n;
    #pragma omp parallel for reduction(min:failure) schedule(static)
    for (size_t ij = 0; ij < limit * limit; ++ij) {
        size_t i = ij / limit, j = ij % limit;
        unsigned int dij = dist[i * n + j];
        for (size_t k = 0; k < n; ++k) {
            unsigned int dik = dist[i * n + k], dkj = dist[k * n + j];
            if (dik < INF && dkj < INF && dik + dkj < dij) {
                failure = std::min(failure, ij * n + k);
                break;
            }
        }
    }
    if (failure != limit * limit * n) {
        size_t ij = failure / n;
        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", ij / limit, ij % limit, failure % n);
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t n = 512;
    bool validate = false, print_results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (arg[0] == '-' || !*arg || *end || value > INT_MAX - TILE) {
                if (!rank) fprintf(stderr, "Invalid number of nodes: %s\n", arg);
                MPI_Finalize(); return 1;
            }
            n = static_cast<size_t>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) print_results = true;
        else if (!strcmp(argv[i], "-h")) { if (!rank) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    size_t stride = ((n + TILE - 1) / TILE) * TILE;
    if (stride && stride > std::numeric_limits<size_t>::max() / stride / sizeof(unsigned int)) {
        if (!rank) fprintf(stderr, "Matrix size is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int blocks = static_cast<int>(stride / TILE);
    int first = block_start(rank, blocks, ranks);
    int local_blocks = block_start(rank + 1, blocks, ranks) - first;
    size_t local_count = size_t(local_blocks) * TILE * stride;
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, device_count = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    MPI_CHECK(MPI_Comm_free(&local_comm));
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (!device_count) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % device_count));

    std::vector<unsigned int> matrix;
    std::vector<unsigned int> local(local_count);
    if (!rank) {
        matrix.resize(stride * stride);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < stride; ++i)
            std::fill(matrix.begin() + i * stride, matrix.begin() + (i + 1) * stride, INF);
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i)
            for (size_t j = 0; j < n; ++j)
                matrix[i * stride + j] = 1 + static_cast<unsigned int>(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < stride; ++i) matrix[i * stride + i] = 0;
        if (local_count) std::copy_n(matrix.data(), local_count, local.data());
        for (int r = 1; r < ranks; ++r) {
            size_t offset = size_t(block_start(r, blocks, ranks)) * TILE * stride;
            size_t count = size_t(block_start(r + 1, blocks, ranks) - block_start(r, blocks, ranks)) * TILE * stride;
            if (count) transfer(matrix.data() + offset, count, r, 1, true);
        }
        if (!validate && !print_results) std::vector<unsigned int>().swap(matrix);
    } else if (local_count) transfer(local.data(), local_count, 0, 1, false);

    unsigned int *device = nullptr, *pivot_device = nullptr, *pivot_host = nullptr;
    size_t pivot_count = size_t(TILE) * stride;
    if (local_count) {
        CUDA_CHECK(cudaMalloc(&device, local_count * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(device, local.data(), local_count * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if (pivot_count) {
        CUDA_CHECK(cudaMalloc(&pivot_device, pivot_count * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&pivot_host, pivot_count * sizeof(unsigned int)));
    }
    if (!rank) printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    const dim3 threads(TILE, 8);
    for (int k = 0; k < blocks; ++k) {
        int owner = block_owner(k, blocks, ranks);
        unsigned int* pivot = pivot_device;
        if (rank == owner) {
            int local_k = k - first;
            pivot = device + size_t(local_k) * TILE * stride;
            close_pivot<<<1, threads>>>(device, stride, local_k, k);
            CUDA_CHECK(cudaGetLastError());
            update_cross<true><<<blocks, threads>>>(device, pivot, stride, k, first, local_k);
            CUDA_CHECK(cudaGetLastError());
            if (ranks > 1) CUDA_CHECK(cudaMemcpy(pivot_host, pivot, pivot_count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) {
            broadcast(pivot_host, pivot_count, owner);
            if (rank != owner) CUDA_CHECK(cudaMemcpy(pivot_device, pivot_host, pivot_count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        if (local_blocks) {
            update_cross<false><<<local_blocks, threads>>>(device, pivot, stride, k, first, 0);
            CUDA_CHECK(cudaGetLastError());
            update_remaining<<<dim3(blocks, local_blocks), threads>>>(device, pivot, stride, k, first);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, maximum = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!rank) {
        printf("Computation time: %ld ms\n", static_cast<long>(maximum * 1000.0));
        printf("Performance: %.3f GOPS\n", maximum > 0 ? double(n) * n * n / maximum / 1e9 : 0.0);
    }
    if (validate || print_results) {
        if (local_count) CUDA_CHECK(cudaMemcpy(local.data(), device, local_count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (!rank) {
            if (local_count) std::copy_n(local.data(), local_count, matrix.data());
            for (int r = 1; r < ranks; ++r) {
                size_t offset = size_t(block_start(r, blocks, ranks)) * TILE * stride;
                size_t count = size_t(block_start(r + 1, blocks, ranks) - block_start(r, blocks, ranks)) * TILE * stride;
                if (count) transfer(matrix.data() + offset, count, r, 2, false);
            }
        } else if (local_count) transfer(local.data(), local_count, 0, 2, true);
    }
    if (device) CUDA_CHECK(cudaFree(device));
    if (pivot_device) CUDA_CHECK(cudaFree(pivot_device));
    if (pivot_host) CUDA_CHECK(cudaFreeHost(pivot_host));
    int status = 0;
    if (!rank && (validate || print_results)) {
        std::vector<unsigned int> result(n * n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) std::copy_n(matrix.data() + i * stride, n, result.data() + i * n);
        if (print_results) print_results_int(result, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(result, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            if (!valid) status = 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return status;
}
