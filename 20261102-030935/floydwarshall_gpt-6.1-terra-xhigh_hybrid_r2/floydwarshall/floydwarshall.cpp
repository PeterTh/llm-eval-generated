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

#define CUDA_CHECK(call) do { \
    cudaError_t error_ = (call); \
    if (error_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error_)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while (0)
#define MPI_CHECK(call) do { \
    int error_ = (call); \
    if (error_ != MPI_SUCCESS) { \
        char message_[MPI_MAX_ERROR_STRING]; int length_; \
        MPI_Error_string(error_, message_, &length_); \
        fprintf(stderr, "MPI error at %s:%d: %.*s\n", __FILE__, __LINE__, length_, message_); \
        MPI_Abort(MPI_COMM_WORLD, error_); \
    } \
} while (0)

// Each block has 32 x 8 threads; each thread owns four rows of a tile.
// Matrix rows are sources and columns are destinations, as in the original.
__global__ void pivot_tile(unsigned int* matrix, size_t stride, int local_pivot, int pivot) {
    __shared__ unsigned int tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < TILE; r += blockDim.y)
        tile[r][x] = matrix[(size_t)(local_pivot * TILE + r) * stride + pivot * TILE + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int values[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int r = y + q * 8;
            values[q] = min(tile[r][x], tile[r][k] + tile[k][x]);
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < 4; ++q) tile[y + q * 8][x] = values[q];
        __syncthreads();
    }
    for (int r = y; r < TILE; r += blockDim.y)
        matrix[(size_t)(local_pivot * TILE + r) * stride + pivot * TILE + x] = tile[r][x];
}

// Update either the pivot row or the pivot column using the closed pivot tile.
__global__ void pivot_edges(unsigned int* matrix, size_t stride, int local_pivot,
                            int pivot, int local_tiles, bool row_phase) {
    int tile_index = blockIdx.x;
    if (row_phase ? tile_index == pivot : tile_index == local_pivot) return;
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int edge[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    int row_tile = row_phase ? local_pivot : tile_index;
    int col_tile = row_phase ? tile_index : pivot;
    // The diagonal resides on the pivot owner. Column updates on other ranks
    // use the broadcast pivot row, and are performed by a separate kernel.
    if (row_phase || local_pivot >= 0) {
        for (int r = y; r < TILE; r += blockDim.y) {
            diagonal[r][x] = matrix[(size_t)(local_pivot * TILE + r) * stride + pivot * TILE + x];
            edge[r][x] = matrix[(size_t)(row_tile * TILE + r) * stride + col_tile * TILE + x];
        }
        __syncthreads();
        for (int k = 0; k < TILE; ++k) {
            unsigned int values[4];
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                int r = y + q * 8;
                unsigned int candidate = row_phase ? diagonal[r][k] + edge[k][x]
                                                   : edge[r][k] + diagonal[k][x];
                values[q] = min(edge[r][x], candidate);
            }
            __syncthreads();
            #pragma unroll
            for (int q = 0; q < 4; ++q) edge[y + q * 8][x] = values[q];
            __syncthreads();
        }
        for (int r = y; r < TILE; r += blockDim.y)
            matrix[(size_t)(row_tile * TILE + r) * stride + col_tile * TILE + x] = edge[r][x];
    }
    (void)local_tiles;
}

// Non-owner ranks obtain the closed diagonal tile in the broadcast pivot row.
__global__ void remote_column(unsigned int* matrix, const unsigned int* pivot_row,
                              size_t stride, int pivot) {
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int edge[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y, row_tile = blockIdx.x;
    for (int r = y; r < TILE; r += blockDim.y) {
        diagonal[r][x] = pivot_row[(size_t)r * stride + pivot * TILE + x];
        edge[r][x] = matrix[(size_t)(row_tile * TILE + r) * stride + pivot * TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int values[4];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            int r = y + q * 8;
            values[q] = min(edge[r][x], edge[r][k] + diagonal[k][x]);
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < 4; ++q) edge[y + q * 8][x] = values[q];
        __syncthreads();
    }
    for (int r = y; r < TILE; r += blockDim.y)
        matrix[(size_t)(row_tile * TILE + r) * stride + pivot * TILE + x] = edge[r][x];
}

__global__ void remaining_tiles(unsigned int* matrix, const unsigned int* pivot_row,
                                size_t stride, int pivot, int local_pivot) {
    int col_tile = blockIdx.x, row_tile = blockIdx.y;
    if (col_tile == pivot || row_tile == local_pivot) return;
    __shared__ unsigned int column[TILE][TILE];
    __shared__ unsigned int row[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    unsigned int values[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int r = y + q * 8;
        column[r][x] = matrix[(size_t)(row_tile * TILE + r) * stride + pivot * TILE + x];
        row[r][x] = pivot_row[(size_t)r * stride + col_tile * TILE + x];
        values[q] = matrix[(size_t)(row_tile * TILE + r) * stride + col_tile * TILE + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned int rhs = row[k][x];
        #pragma unroll
        for (int q = 0; q < 4; ++q)
            values[q] = min(values[q], column[y + q * 8][k] + rhs);
    }
    #pragma unroll
    for (int q = 0; q < 4; ++q)
        matrix[(size_t)(row_tile * TILE + y + q * 8) * stride + col_tile * TILE + x] = values[q];
}

// MPI counts are int, so split large transfers into legal-sized messages.
void send_elements(const unsigned int* data, size_t count, int peer, int tag, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Send(data, chunk, MPI_UNSIGNED, peer, tag, comm));
        data += chunk; count -= chunk;
    }
}
void receive_elements(unsigned int* data, size_t count, int peer, int tag, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Recv(data, chunk, MPI_UNSIGNED, peer, tag, comm, MPI_STATUS_IGNORE));
        data += chunk; count -= chunk;
    }
}
void broadcast_elements(unsigned int* data, size_t count, int root, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, static_cast<size_t>(INT_MAX)));
        MPI_CHECK(MPI_Bcast(data, chunk, MPI_UNSIGNED, root, comm));
        data += chunk; count -= chunk;
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
            for (size_t k = 0; k < n; ++k) {
                unsigned int ij = dist[i * n + j], ik = dist[i * n + k], kj = dist[k * n + j];
                if (ik < INF && kj < INF && ik + kj < ij) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
    return true;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, world_size;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
    size_t n = 512;
    bool validate = false, print_results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long value = strtoull(arg, &end, 10);
            if (*arg == '-' || *arg == '\0' || *end != '\0' || value > INT_MAX - TILE) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", arg);
                MPI_Finalize(); return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) print_results = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize(); return 0;
        } else {
            if (rank == 0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize(); return 1;
        }
    }
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    const int tiles = static_cast<int>((n + TILE - 1) / TILE);
    const size_t stride = static_cast<size_t>(tiles) * TILE;
    if (stride && stride > std::numeric_limits<size_t>::max() / stride / sizeof(unsigned int)) {
        if (rank == 0) fprintf(stderr, "Matrix is too large\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    std::vector<unsigned int> original;
    if (rank == 0) {
        original.resize(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i)
            original[i] = 1 + static_cast<unsigned int>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
        for (size_t i = 0; i < n; ++i) original[i * n + i] = 0;
    }
    // Ranks beyond the number of tile rows do not participate in computation.
    int active_count = std::min(world_size, tiles);
    MPI_Comm comm = MPI_COMM_NULL;
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, rank < active_count ? 0 : MPI_UNDEFINED, rank, &comm));
    if (rank < active_count) {
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        int local_rank; MPI_CHECK(MPI_Comm_rank(local_comm, &local_rank));
        MPI_CHECK(MPI_Comm_free(&local_comm));
        int devices = 0; CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (!devices) { fprintf(stderr, "Rank %d: no CUDA device available\n", rank); MPI_Abort(MPI_COMM_WORLD, 1); }
        CUDA_CHECK(cudaSetDevice(local_rank % devices));
        auto first_tile = [=](int r) { return (tiles / active_count) * r + std::min(r, tiles % active_count); };
        int first = first_tile(rank), local_tiles = first_tile(rank + 1) - first;
        size_t local_count = static_cast<size_t>(local_tiles) * TILE * stride;
        std::vector<unsigned int> host(local_count, INF);
        if (rank == 0) {
            // Generate the original random stream once, then distribute padded rows.
            for (int r = 0; r < active_count; ++r) {
                int begin = first_tile(r), count = first_tile(r + 1) - begin;
                std::vector<unsigned int> packed;
                unsigned int* dst;
                if (r == 0) dst = host.data();
                else { packed.assign(static_cast<size_t>(count) * TILE * stride, INF); dst = packed.data(); }
                #pragma omp parallel for schedule(static)
                for (size_t row = 0; row < static_cast<size_t>(count) * TILE; ++row) {
                    size_t global_row = static_cast<size_t>(begin) * TILE + row;
                    if (global_row < n) std::copy_n(original.data() + global_row * n, n, dst + row * stride);
                }
                if (r != 0) send_elements(dst, packed.size(), r, 0, comm);
            }
        } else receive_elements(host.data(), host.size(), 0, 0, comm);
        unsigned int *device = nullptr, *device_pivot = nullptr, *host_pivot = nullptr;
        CUDA_CHECK(cudaMalloc(&device, local_count * sizeof(unsigned int)));
        if (active_count > 1) {
            CUDA_CHECK(cudaMalloc(&device_pivot, TILE * stride * sizeof(unsigned int)));
            CUDA_CHECK(cudaMallocHost(&host_pivot, TILE * stride * sizeof(unsigned int)));
        }
        CUDA_CHECK(cudaMemcpy(device, host.data(), local_count * sizeof(unsigned int), cudaMemcpyHostToDevice));
        MPI_CHECK(MPI_Barrier(comm));
        if (rank == 0) printf("Computing shortest paths...\n");
        auto start = std::chrono::high_resolution_clock::now();
        dim3 threads(TILE, 8);
        for (int p = 0; p < tiles; ++p) {
            int owner = 0;
            while (first_tile(owner + 1) <= p) ++owner;
            int local_pivot = rank == owner ? p - first : -1;
            if (rank == owner) {
                pivot_tile<<<1, threads>>>(device, stride, local_pivot, p);
                pivot_edges<<<local_tiles, threads>>>(device, stride, local_pivot, p, local_tiles, false);
                pivot_edges<<<tiles, threads>>>(device, stride, local_pivot, p, local_tiles, true);
                CUDA_CHECK(cudaGetLastError());
                if (active_count > 1)
                    CUDA_CHECK(cudaMemcpy(host_pivot, device + static_cast<size_t>(local_pivot) * TILE * stride,
                                          TILE * stride * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            }
            if (active_count > 1) {
                broadcast_elements(host_pivot, TILE * stride, owner, comm);
                if (rank != owner)
                    CUDA_CHECK(cudaMemcpy(device_pivot, host_pivot, TILE * stride * sizeof(unsigned int), cudaMemcpyHostToDevice));
            }
            const unsigned int* pivot_data = rank == owner
                ? device + static_cast<size_t>(local_pivot) * TILE * stride : device_pivot;
            if (rank != owner) remote_column<<<local_tiles, threads>>>(device, pivot_data, stride, p);
            dim3 grid(tiles, local_tiles);
            remaining_tiles<<<grid, threads>>>(device, pivot_data, stride, p, local_pivot);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        MPI_CHECK(MPI_Barrier(comm));
        auto end = std::chrono::high_resolution_clock::now();
        if (rank == 0) {
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            printf("Computation time: %ld ms\n", static_cast<long>(duration.count()));
            double ops = static_cast<double>(n) * n * n;
            printf("Performance: %.3f GOPS\n", ops / (duration.count() / 1000.0) / 1e9);
        }
        if (print_results || validate) {
            CUDA_CHECK(cudaMemcpy(host.data(), device, local_count * sizeof(unsigned int), cudaMemcpyDeviceToHost));
            if (rank == 0) {
                for (int r = 0; r < active_count; ++r) {
                    int begin = first_tile(r), count = first_tile(r + 1) - begin;
                    std::vector<unsigned int> packed;
                    const unsigned int* src;
                    if (r == 0) src = host.data();
                    else {
                        packed.resize(static_cast<size_t>(count) * TILE * stride);
                        receive_elements(packed.data(), packed.size(), r, 1, comm);
                        src = packed.data();
                    }
                    #pragma omp parallel for schedule(static)
                    for (size_t row = 0; row < static_cast<size_t>(count) * TILE; ++row) {
                        size_t global_row = static_cast<size_t>(begin) * TILE + row;
                        if (global_row < n) std::copy_n(src + row * stride, n, original.data() + global_row * n);
                    }
                }
            } else send_elements(host.data(), host.size(), 0, 1, comm);
        }
        if (active_count > 1) {
            CUDA_CHECK(cudaFreeHost(host_pivot));
            CUDA_CHECK(cudaFree(device_pivot));
        }
        CUDA_CHECK(cudaFree(device));
        MPI_CHECK(MPI_Comm_free(&comm));
    } else {
        // MPI_Comm_split_type is collective on the world communicator.
        MPI_Comm local_comm;
        MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm));
        MPI_CHECK(MPI_Comm_free(&local_comm));
        if (rank == 0) printf("Computing shortest paths...\nComputation time: 0 ms\nPerformance: -nan GOPS\n");
    }
    int status = 0;
    if (rank == 0) {
        if (print_results) print_results_int(original, "DistanceMatrix");
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(original, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Finalize());
    return status;
}
