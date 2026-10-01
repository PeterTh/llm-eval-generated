#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

constexpr unsigned INF = 1000000000;
constexpr int TILE = 32;

static void cudaCheck(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// MPI counts are int even when the distributed matrix is larger than INT_MAX.
static void transfer(unsigned* data, size_t count, int peer, int kind, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (kind == 0) MPI_Bcast(data, chunk, MPI_UNSIGNED, peer, comm);
        else if (kind == 1) MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, comm);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

// Snapshot pivot k before moving to k+1. The pivot row and column are
// invariant at this step (nonnegative weights and a zero diagonal).
__global__ void preparePanel(unsigned* work, unsigned* panel, size_t n,
                             size_t base, int width, int k) {
    size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (j >= n || i >= width) return;
    unsigned pivot = work[size_t(k) * n + j];
    if (i == k) panel[size_t(k) * n + j] = pivot;
    else if (j != base + k) {
        size_t at = size_t(i) * n + j;
        work[at] = min(work[at], work[size_t(i) * n + base + k] + pivot);
    }
}

// Each warp evolves one row's pivot columns in registers. Save the exact
// d[i,k] at step k, allowing all output tiles to be updated independently.
__global__ void prepareFactors(const unsigned* dist, const unsigned* panel,
                               unsigned* factors, size_t n, size_t rows,
                               size_t base, int width) {
    size_t row = size_t(blockIdx.x) * 8 + threadIdx.y;
    int lane = threadIdx.x;
    if (row >= rows) return;
    unsigned value = lane < width ? dist[row * n + base + lane] : INF;
    for (int k = 0; k < width; ++k) {
        unsigned factor = __shfl_sync(0xffffffffu, value, k);
        if (lane == k) factors[row * TILE + k] = factor;
        if (lane < width) value = min(value, factor + panel[size_t(k) * n + base + lane]);
    }
}

__global__ void updateRows(unsigned* dist, unsigned* path, const unsigned* panel,
                           const unsigned* factors, size_t n, size_t rows,
                           size_t base, int width) {
    __shared__ unsigned right[TILE][TILE];
    __shared__ unsigned left[8][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t j = size_t(blockIdx.x) * TILE + x;
    size_t i = size_t(blockIdx.y) * 8 + y;
    left[y][x] = i < rows && x < width ? factors[i * TILE + x] : INF;
    for (int k = y; k < width; k += 8)
        right[k][x] = j < n ? panel[size_t(k) * n + j] : INF;
    __syncthreads();
    if (i >= rows || j >= n) return;
    size_t at = i * n + j;
    unsigned value = dist[at], intermediate = path[at];
    for (int k = 0; k < width; ++k) {
        unsigned candidate = left[y][k] + right[k][x];
        if (candidate < value) {
            value = candidate;
            intermediate = static_cast<unsigned>(base + k);
        }
    }
    dist[at] = value;
    path[at] = intermediate;
}

static bool validateResult(const std::vector<unsigned>& dist, size_t n) {
    int failed = 0;
    #pragma omp parallel for reduction(|:failed) schedule(static)
    for (size_t i = 0; i < n; ++i) failed |= dist[i * n + i] != 0;
    #pragma omp parallel for collapse(2) reduction(|:failed) schedule(static)
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[i*n+k] < INF && dist[k*n+j] < INF)
                    failed |= dist[i*n+k] + dist[k*n+j] < dist[i*n+j];
    return failed == 0;
}

static void printUsage(const char* program) {
    printf("Usage: %s [options]\n  -n <num>     Number of nodes (default: 512)\n"
           "  -v           Enable validation\n  -r           Print results for external validation\n"
           "  -h           Show this help message\n", program);
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t n = 512;
    bool validate = false, results = false, help = false, invalid = false;
    for (int a = 1; a < argc; ++a) {
        if (!strcmp(argv[a], "-n") && a + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++a];
            errno = 0;
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || *value == '-' || end == value || *end || parsed > UINT_MAX)
                invalid = true;
            else n = static_cast<size_t>(parsed);
        } else if (!strcmp(argv[a], "-v")) validate = true;
        else if (!strcmp(argv[a], "-r")) results = true;
        else if (!strcmp(argv[a], "-h")) help = true;
        else invalid = true;
    }
    if (n && n > std::numeric_limits<size_t>::max() / sizeof(unsigned) / n) invalid = true;
    if (help || invalid) {
        if (!rank) printUsage(argv[0]);
        MPI_Finalize();
        return invalid ? 1 : 0;
    }

    // One rank per GPU is recommended. Local rank also works with launcher
    // supplied CUDA_VISIBLE_DEVICES masks exposing one device per process.
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d requires a CUDA device\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);
    // Empty partitions do not participate in the pivot broadcasts.
    size_t tiles = (n + TILE - 1) / TILE;
    int active = static_cast<int>(std::min(size_t(world), std::max(size_t(1), tiles)));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = 0;
    if (rank < active) try {
        auto first = [&](int r) { return std::min(n, (tiles * size_t(r) / active) * TILE); };
        size_t begin = first(rank), rows = first(rank + 1) - begin;
        size_t count = rows * n, bytes = count * sizeof(unsigned);
        std::vector<unsigned> dist(count), path(count);
        if (!rank) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n"
                   "Validation: %s\nInitializing graph...\n", n, validate ? "enabled" : "disabled");
            // Keep rand_r's original stream, without replicating the full graph
            // on every host. Only one destination partition is staged at a time.
            unsigned seed = 42;
            for (int r = 0; r < active; ++r) {
                size_t start = first(r), nr = first(r + 1) - start;
                std::vector<unsigned> staging(r ? nr * n : 0);
                unsigned* out = r ? staging.data() : dist.data();
                for (size_t p = 0; p < nr * n; ++p)
                    out[p] = 1 + static_cast<unsigned>(200.0 * rand_r(&seed) / double(RAND_MAX));
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < nr; ++i) out[i * n + start + i] = 0;
                if (r) transfer(out, nr * n, r, 1, comm);
            }
        } else transfer(dist.data(), count, 0, 2, comm);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < n; ++j) path[i * n + j] = static_cast<unsigned>(begin + i);

        unsigned *deviceDist = nullptr, *devicePath = nullptr, *panel = nullptr;
        unsigned *work = nullptr, *factors = nullptr, *hostPanel = nullptr;
        size_t panelBytes = size_t(TILE) * n * sizeof(unsigned);
        if (n) {
            CUDA(cudaMalloc(&deviceDist, bytes));
            CUDA(cudaMalloc(&devicePath, bytes));
            CUDA(cudaMalloc(&panel, panelBytes));
            CUDA(cudaMalloc(&work, panelBytes));
            CUDA(cudaMalloc(&factors, rows * TILE * sizeof(unsigned)));
            CUDA(cudaMallocHost(&hostPanel, panelBytes));
            CUDA(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
        }
        MPI_Barrier(comm);
        if (!rank) printf("Computing shortest paths...\n");
        double start = MPI_Wtime();
        int owner = 0;
        for (size_t base = 0; base < n; base += TILE) {
            while (base >= first(owner + 1)) ++owner;
            int width = static_cast<int>(std::min(size_t(TILE), n - base));
            size_t panelCount = size_t(width) * n;
            if (rank == owner) {
                CUDA(cudaMemcpy(work, deviceDist + (base - begin) * n,
                                panelCount * sizeof(unsigned), cudaMemcpyDeviceToDevice));
                for (int k = 0; k < width; ++k)
                    preparePanel<<<dim3((n + 255) / 256, width), 256>>>(work, panel, n, base, width, k);
                CUDA(cudaGetLastError());
                if (active > 1)
                    CUDA(cudaMemcpy(hostPanel, panel, panelCount * sizeof(unsigned), cudaMemcpyDeviceToHost));
            }
            if (active > 1) {
                transfer(hostPanel, panelCount, owner, 0, comm);
                if (rank != owner)
                    CUDA(cudaMemcpy(panel, hostPanel, panelCount * sizeof(unsigned), cudaMemcpyHostToDevice));
            }
            prepareFactors<<<(rows + 7) / 8, dim3(TILE, 8)>>>(deviceDist, panel, factors, n, rows, base, width);
            updateRows<<<dim3((n + TILE - 1) / TILE, (rows + 7) / 8), dim3(TILE, 8)>>>(
                deviceDist, devicePath, panel, factors, n, rows, base, width);
            CUDA(cudaGetLastError());
        }
        CUDA(cudaDeviceSynchronize());
        double elapsed = MPI_Wtime() - start, maximum;
        MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Computation time: %lld ms\n", static_cast<long long>(maximum * 1000));
            printf("Performance: %.3f GOPS\n", maximum > 0 ? double(n)*n*n / maximum / 1e9 : 0.0);
        }
        if (validate || results) {
            if (n) CUDA(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
            if (!rank) {
                dist.resize(n * n);
                for (int r = 1; r < active; ++r)
                    transfer(dist.data() + first(r)*n, (first(r+1)-first(r))*n, r, 2, comm);
                if (results) print_results_int(dist, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    status = validateResult(dist, n) ? 0 : 1;
                    printf("Validation: %s\n", status ? "FAILED" : "PASSED");
                }
            } else transfer(dist.data(), count, 0, 1, comm);
        }
        if (n) {
            CUDA(cudaFree(deviceDist)); CUDA(cudaFree(devicePath));
            CUDA(cudaFree(panel)); CUDA(cudaFree(work)); CUDA(cudaFree(factors));
            CUDA(cudaFreeHost(hostPanel));
        }
        MPI_Comm_free(&comm);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
