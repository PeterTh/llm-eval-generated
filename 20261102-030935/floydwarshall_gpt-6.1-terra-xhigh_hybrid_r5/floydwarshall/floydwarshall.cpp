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

static int world_rank = 0;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", world_rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

static void mpiCheck(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "Rank %d: %s: %.*s\n", world_rank, operation, length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}
#define MPI_CHECK(call) mpiCheck((call), #call)

// Each block handles a 32 x 32 tile, with four elements per thread.
// The diagonal is zero, so the pivot row and column do not change during
// each inner iteration. The barriers protect the remaining shared entries.
__global__ void pivotKernel(unsigned int* matrix, size_t n, size_t local_row,
                            size_t pivot_row) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t row = pivot_row + r, col = pivot_row + x;
        tile[r][x] = row < n && col < n ? matrix[(local_row + r) * n + col] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int r = y; r < TILE; r += THREAD_ROWS) {
            unsigned int candidate = tile[r][k] + tile[k][x];
            if (candidate < tile[r][x]) tile[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t row = pivot_row + r, col = pivot_row + x;
        if (row < n && col < n) matrix[(local_row + r) * n + col] = tile[r][x];
    }
}

// Update the pivot tile row on the rank that owns it.
__global__ void pivotRowKernel(unsigned int* matrix, size_t n, size_t local_row,
                               size_t pivot_row, size_t pivot_tile) {
    const size_t tile_col = blockIdx.x;
    if (tile_col == pivot_tile) return;
    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t col = tile_col * TILE + x;
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t row = pivot_row + r, pcol = pivot_row + x;
        pivot[r][x] = row < n && pcol < n ? matrix[(local_row + r) * n + pcol] : INF;
        tile[r][x] = row < n && col < n ? matrix[(local_row + r) * n + col] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int r = y; r < TILE; r += THREAD_ROWS) {
            unsigned int candidate = pivot[r][k] + tile[k][x];
            if (candidate < tile[r][x]) tile[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int r = y; r < TILE; r += THREAD_ROWS)
        if (pivot_row + r < n && col < n) matrix[(local_row + r) * n + col] = tile[r][x];
}

// Update the pivot tile column for every locally owned tile row.
__global__ void pivotColumnKernel(unsigned int* matrix, const unsigned int* pivot_rows,
                                  size_t n, size_t first_row, size_t local_rows,
                                  size_t pivot_row) {
    const size_t local_base = static_cast<size_t>(blockIdx.x) * TILE;
    if (first_row + local_base == pivot_row) return;
    __shared__ unsigned int pivot[TILE][TILE];
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t col = pivot_row + x;
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        const size_t lr = local_base + r;
        pivot[r][x] = pivot_row + r < n && col < n ? pivot_rows[static_cast<size_t>(r) * n + col] : INF;
        tile[r][x] = lr < local_rows && col < n ? matrix[lr * n + col] : INF;
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int r = y; r < TILE; r += THREAD_ROWS) {
            unsigned int candidate = tile[r][k] + pivot[k][x];
            if (candidate < tile[r][x]) tile[r][x] = candidate;
        }
        __syncthreads();
    }
    for (int r = y; r < TILE; r += THREAD_ROWS) {
        size_t lr = local_base + r;
        if (lr < local_rows && col < n) matrix[lr * n + col] = tile[r][x];
    }
}

// The two input tiles are read-only here; all 32 intermediate vertices can
// therefore be processed from shared memory without per-iteration barriers.
__global__ void remainingKernel(unsigned int* matrix, const unsigned int* pivot_rows,
                                size_t n, size_t first_row, size_t local_rows,
                                size_t pivot_row, size_t pivot_tile) {
    const size_t local_base = static_cast<size_t>(blockIdx.y) * TILE;
    const size_t tile_col = blockIdx.x;
    if (tile_col == pivot_tile || first_row + local_base == pivot_row) return;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t col = tile_col * TILE + x;
    unsigned int value[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * THREAD_ROWS;
        size_t lr = local_base + r;
        size_t pcol = pivot_row + x;
        left[r][x] = lr < local_rows && pcol < n ? matrix[lr * n + pcol] : INF;
        right[r][x] = pivot_row + r < n && col < n ? pivot_rows[static_cast<size_t>(r) * n + col] : INF;
        value[t] = lr < local_rows && col < n ? matrix[lr * n + col] : INF;
    }
    __syncthreads();
    #pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        unsigned int b = right[k][x];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned int candidate = left[y + t * THREAD_ROWS][k] + b;
            if (candidate < value[t]) value[t] = candidate;
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        size_t lr = local_base + y + t * THREAD_ROWS;
        if (lr < local_rows && col < n) matrix[lr * n + col] = value[t];
    }
}

// MPI counts are int, while matrices can be larger than INT_MAX elements.
static void transfer(unsigned int* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        if (send) MPI_CHECK(MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD));
        else MPI_CHECK(MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk;
        count -= chunk;
    }
}

static void broadcast(unsigned int* data, size_t count, int root) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_UNSIGNED, root, MPI_COMM_WORLD));
        data += chunk;
        count -= chunk;
    }
}

// glibc's rand_r advances its unsigned state by a fixed affine transform on
// every call. Jump to each thread's starting state to reproduce the serial
// sequence exactly, without serializing graph generation. Other C libraries
// retain the original rand_r loop because their state transition may differ.
static void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n) {
    const size_t total = dist.size();
#if defined(__GLIBC__) && UINT_MAX == 0xffffffffU
    unsigned int zero = 0, one = 1;
    (void)rand_r(&zero);
    (void)rand_r(&one);
    const unsigned int multiplier = one - zero, increment = zero;
    int threads = static_cast<int>(std::min(static_cast<size_t>(omp_get_max_threads()),
                                            std::max(size_t(1), total / 65536)));
    #pragma omp parallel num_threads(threads)
    {
        size_t id = static_cast<size_t>(omp_get_thread_num());
        size_t count = static_cast<size_t>(omp_get_num_threads());
        size_t start = (total / count) * id + std::min(id, total % count);
        size_t end = start + total / count + (id < total % count);
        size_t steps = start;
        unsigned int a = multiplier, c = increment, accumulated_a = 1, accumulated_c = 0;
        while (steps) {
            if (steps & 1) {
                accumulated_c = a * accumulated_c + c;
                accumulated_a = a * accumulated_a;
            }
            c = (a + 1U) * c;
            a *= a;
            steps >>= 1;
        }
        unsigned int seed = accumulated_a * 42U + accumulated_c;
        for (size_t i = start; i < end; ++i)
            dist[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
    }
#else
    unsigned int seed = 42;
    for (size_t i = 0; i < total; ++i)
        dist[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
#endif
    #pragma omp parallel for schedule(static) if(n >= 4096)
    for (size_t i = 0; i < n; ++i) dist[i * n + i] = 0;
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int bad_diagonal = 0;
    #pragma omp parallel for reduction(|:bad_diagonal) schedule(static) if(n >= 4096)
    for (size_t i = 0; i < n; ++i) bad_diagonal |= dist[i * n + i] != 0;
    if (bad_diagonal) {
        for (size_t i = 0; i < n; ++i) {
            if (dist[i * n + i] != 0) {
                printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
                break;
            }
        }
        return false;
    }
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k) {
                unsigned int ij = dist[i * n + j], ik = dist[i * n + k], kj = dist[k * n + j];
                if (ik < INF && kj < INF && ik + kj < ij) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

static void printUsage(const char* name) {
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
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));
    int ranks = 0;
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!world_rank) fprintf(stderr, "MPI does not support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long parsed = strtoull(arg, &end, 10);
            if (*arg == '-' || !*arg || *end || parsed > std::numeric_limits<size_t>::max()) {
                if (!world_rank) fprintf(stderr, "Invalid number of nodes: %s\n", arg);
                MPI_Finalize(); return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!world_rank) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (!world_rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (n > static_cast<size_t>(INT_MAX) || (n && n > std::numeric_limits<size_t>::max() / n / sizeof(unsigned int))) {
        if (!world_rank) fprintf(stderr, "Matrix size is too large\n");
        MPI_Finalize(); return 1;
    }
    MPI_Comm local_comm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &local_comm));
    int local_rank = 0, devices = 0;
    MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "Rank %d: no CUDA devices available\n", world_rank); MPI_Abort(MPI_COMM_WORLD, 1); }
    CUDA_CHECK(cudaSetDevice(local_rank % devices));
    MPI_CHECK(MPI_Comm_free(&local_comm));

    const size_t tiles = (n + TILE - 1) / TILE;
    std::vector<size_t> first(ranks), rows(ranks);
    size_t tile_start = 0;
    for (int r = 0; r < ranks; ++r) {
        size_t count = tiles / ranks + (static_cast<size_t>(r) < tiles % ranks);
        first[r] = std::min(n, tile_start * TILE);
        tile_start += count;
        rows[r] = std::min(n, tile_start * TILE) - first[r];
    }
    std::vector<unsigned int> dist, local;
    if (!world_rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        dist.resize(n * n);
        initializeDistanceMatrix(dist, n);
        for (int r = 1; r < ranks; ++r)
            if (rows[r]) transfer(dist.data() + first[r] * n, rows[r] * n, r, true);
    } else {
        local.resize(rows[world_rank] * n);
        transfer(local.data(), local.size(), 0, false);
    }
    unsigned int* device_matrix = nullptr;
    unsigned int* device_pivot = nullptr;
    unsigned int* host_pivot = nullptr;
    const size_t local_elements = rows[world_rank] * n;
    if (local_elements) {
        CUDA_CHECK(cudaMalloc(&device_matrix, local_elements * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(device_matrix, world_rank ? local.data() : dist.data(), local_elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    if (n) {
        CUDA_CHECK(cudaMalloc(&device_pivot, std::min(n, size_t(TILE)) * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMallocHost(&host_pivot, std::min(n, size_t(TILE)) * n * sizeof(unsigned int)));
    }
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (!world_rank) printf("Computing shortest paths...\n");
    double start = MPI_Wtime();
    dim3 threads(TILE, THREAD_ROWS);
    int owner = 0;
    for (size_t p = 0; p < tiles; ++p) {
        size_t pivot_row = p * TILE;
        while (owner + 1 < ranks && pivot_row >= first[owner] + rows[owner]) ++owner;
        size_t pivot_elements = std::min(size_t(TILE), n - pivot_row) * n;
        const unsigned int* pivot_device = device_pivot;
        if (world_rank == owner) {
            size_t offset = pivot_row - first[owner];
            pivotKernel<<<1, threads>>>(device_matrix, n, offset, pivot_row);
            pivotRowKernel<<<static_cast<unsigned int>(tiles), threads>>>(device_matrix, n, offset, pivot_row, p);
            CUDA_CHECK(cudaGetLastError());
            pivot_device = device_matrix + offset * n;
            if (ranks > 1) CUDA_CHECK(cudaMemcpy(host_pivot, pivot_device, pivot_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        }
        if (ranks > 1) {
            broadcast(host_pivot, pivot_elements, owner);
            if (world_rank != owner && local_elements)
                CUDA_CHECK(cudaMemcpy(device_pivot, host_pivot, pivot_elements * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
        if (local_elements) {
            size_t local_tiles = (rows[world_rank] + TILE - 1) / TILE;
            pivotColumnKernel<<<static_cast<unsigned int>(local_tiles), threads>>>(device_matrix, pivot_device, n, first[world_rank], rows[world_rank], pivot_row);
            dim3 grid(static_cast<unsigned int>(tiles), static_cast<unsigned int>(local_tiles));
            remainingKernel<<<grid, threads>>>(device_matrix, pivot_device, n, first[world_rank], rows[world_rank], pivot_row, p);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, max_elapsed = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!world_rank) {
        long milliseconds = static_cast<long>(max_elapsed * 1000.0);
        printf("Computation time: %ld ms\n", milliseconds);
        printf("Performance: %.3f GOPS\n", static_cast<double>(n) * n * n / max_elapsed / 1e9);
    }
    if (validate || printResults) {
        if (local_elements)
            CUDA_CHECK(cudaMemcpy(world_rank ? local.data() : dist.data(), device_matrix, local_elements * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        if (!world_rank) {
            for (int r = 1; r < ranks; ++r)
                if (rows[r]) transfer(dist.data() + first[r] * n, rows[r] * n, r, false);
        } else transfer(local.data(), local.size(), 0, true);
    }
    int status = 0;
    if (!world_rank) {
        if (printResults) print_results_int(dist, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    if (device_matrix) CUDA_CHECK(cudaFree(device_matrix));
    if (device_pivot) CUDA_CHECK(cudaFree(device_pivot));
    if (host_pivot) CUDA_CHECK(cudaFreeHost(host_pivot));
    MPI_Finalize();
    return status;
}
