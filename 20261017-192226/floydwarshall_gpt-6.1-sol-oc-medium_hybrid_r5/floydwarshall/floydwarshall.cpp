#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char text[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, text, &length);
        fprintf(stderr, "MPI: %.*s\n", length, text);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Chunk transfers so that neither matrices nor panels are limited by MPI's
// int count API. Only pinned host panels are passed to MPI: CUDA-aware MPI
// is not required.
static void broadcast(unsigned int* data, size_t count, int owner, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(data, chunk, MPI_UNSIGNED, owner, comm));
        data += chunk;
        count -= chunk;
    }
}

static void transfer(unsigned int* data, size_t count, int peer, bool send,
                     MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send)
            mpiCheck(MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, comm));
        else
            mpiCheck(MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE));
        data += chunk;
        count -= chunk;
    }
}

// Partition whole pivot tiles, keeping every broadcast panel on one owner.
static size_t firstRow(int rank, int ranks, size_t n) {
    size_t tiles = (n + TILE - 1) / TILE;
    size_t firstTile = (tiles / ranks) * rank + std::min(size_t(rank), tiles % ranks);
    return std::min(n, firstTile * TILE);
}

// Construct exact, time-ordered pivot-row snapshots. A block independently
// evolves a 32-column strip of the pivot panel and a copy of its diagonal
// tile. Unlike an ordinary blocked closure, these snapshots retain the value
// of row k BEFORE step k, preserving the original path/tie semantics too.
__global__ void pivotSnapshots(const unsigned int* __restrict__ input,
                               unsigned int* __restrict__ snapshots,
                               size_t n, size_t base, int width) {
    __shared__ unsigned int diagonal[TILE][TILE + 1];
    __shared__ unsigned int strip[TILE][TILE + 1];
    int x = threadIdx.x;
    size_t column = size_t(blockIdx.x) * TILE + x;
    for (int row = threadIdx.y; row < TILE; row += blockDim.y) {
        diagonal[row][x] = row < width && x < width ? input[size_t(row) * n + base + x] : INF;
        strip[row][x] = row < width && column < n ? input[size_t(row) * n + column] : INF;
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        unsigned int pivot = strip[k][x];
        unsigned int diagonalPivot = diagonal[k][x];
        if (threadIdx.y == 0 && column < n)
            snapshots[size_t(k) * n + column] = pivot;
        unsigned int nextStrip[4], nextDiagonal[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = threadIdx.y + r * 8;
            unsigned int left = diagonal[row][k];
            nextStrip[r] = min(strip[row][x], left + pivot);
            nextDiagonal[r] = min(diagonal[row][x], left + diagonalPivot);
        }
        // All reads of step k precede any writes of step k+1.
        __syncthreads();
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            int row = threadIdx.y + r * 8;
            strip[row][x] = nextStrip[r];
            diagonal[row][x] = nextDiagonal[r];
        }
        __syncthreads();
    }
}

// One warp evolves the pivot columns of each local row. Lane k records d(i,k)
// at step k; shuffle communication avoids block-wide synchronization.
__global__ void columnSnapshots(const unsigned int* __restrict__ dist,
                                const unsigned int* __restrict__ pivots,
                                unsigned int* __restrict__ columns,
                                size_t n, size_t rows, size_t base, int width) {
    int lane = threadIdx.x;
    size_t row = size_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= rows) return;
    unsigned int value = lane < width ? dist[row * n + base + lane] : INF;
    for (int k = 0; k < width; ++k) {
        unsigned int left = __shfl_sync(0xffffffffu, value, k);
        if (lane == k) columns[row * TILE + k] = left;
        if (lane < width)
            value = min(value, left + pivots[size_t(k) * n + base + lane]);
    }
}

// All 32 ordered relaxations are fused into a single read/write of each
// distance and path entry. Shared panels are reused across eight local rows.
__global__ void updateRows(unsigned int* __restrict__ dist,
                           unsigned int* __restrict__ path,
                           const unsigned int* __restrict__ pivots,
                           const unsigned int* __restrict__ columns,
                           size_t n, size_t rows, size_t base, int width) {
    __shared__ unsigned int panel[TILE][TILE + 1];
    __shared__ unsigned int left[8][TILE + 1];
    int x = threadIdx.x, y = threadIdx.y;
    size_t column = size_t(blockIdx.x) * TILE + x;
    size_t row = size_t(blockIdx.y) * 8 + y;
    for (int k = y; k < width; k += 8)
        panel[k][x] = column < n ? pivots[size_t(k) * n + column] : INF;
    left[y][x] = row < rows && x < width ? columns[row * TILE + x] : INF;
    __syncthreads();
    if (row >= rows || column >= n) return;
    size_t index = row * n + column;
    unsigned int original = dist[index], value = original, intermediate = 0;
    for (int k = 0; k < width; ++k) {
        unsigned int candidate = left[y][k] + panel[k][x];
        if (candidate < value) {
            value = candidate;
            intermediate = static_cast<unsigned int>(base + k);
        }
    }
    // Most late panels make few improvements. Avoid reading path entries
    // altogether, and write either matrix only when a relaxation succeeded.
    if (value < original) {
        dist[index] = value;
        path[index] = intermediate;
    }
}

static bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i)
        if (dist[idx2(i, i, n)] != 0) valid = 0;
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j) {
            for (size_t k = 0; k < n; ++k) {
                unsigned int a = dist[idx2(k, i, n)], b = dist[idx2(j, k, n)];
                if (a < INF && b < INF && a + b < dist[idx2(j, i, n)]) valid = 0;
            }
        }
    }
    return valid != 0;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
    printf("Launch with mpirun, normally one MPI rank per GPU; OMP_NUM_THREADS controls host threads.\n");
}

static int run(size_t n, bool validate, bool printResults, MPI_Comm comm) {
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(comm, &rank));
    mpiCheck(MPI_Comm_size(comm, &ranks));
    MPI_Comm local;
    mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
    int localRank, devices;
    mpiCheck(MPI_Comm_rank(local, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&local));

    size_t startRow = firstRow(rank, ranks, n);
    size_t rows = firstRow(rank + 1, ranks, n) - startRow;
    size_t count = rows * n;
    std::vector<unsigned int> host(count), hostPath(count);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads per rank: %d\n", ranks, omp_get_max_threads());
        printf("Initializing graph...\n");
        // Retain the exact rand_r stream; only one rank generates it, in the
        // original flat-array order. Stream rank-sized pieces to avoid an
        // O(n^2) allocation on rank zero during normal benchmark runs.
        unsigned int seed = 42;
        for (int p = 0; p < ranks; ++p) {
            size_t begin = firstRow(p, ranks, n);
            size_t length = (firstRow(p + 1, ranks, n) - begin) * n;
            std::vector<unsigned int> temporary(p == 0 ? 0 : length);
            unsigned int* data = p == 0 ? host.data() : temporary.data();
            for (size_t i = 0; i < length; ++i)
                data[i] = 1 + static_cast<unsigned int>(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
            #pragma omp parallel for schedule(static)
            for (size_t i = begin; i < firstRow(p + 1, ranks, n); ++i)
                data[(i - begin) * n + i] = 0;
            if (p != 0) transfer(data, length, p, true, comm);
        }
    } else {
        transfer(host.data(), count, 0, false, comm);
    }
    // The original path initializer assigns the source row to every entry.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i)
        std::fill_n(hostPath.data() + i * n, n, static_cast<unsigned int>(startRow + i));

    unsigned int *dist = nullptr, *path = nullptr, *panel = nullptr;
    unsigned int *pivots = nullptr, *columns = nullptr, *staging = nullptr;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    if (n) {
        CUDA_CHECK(cudaMalloc(&dist, count * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&path, count * sizeof(unsigned int)));
        if (ranks > 1) {
            CUDA_CHECK(cudaMalloc(&panel, TILE * n * sizeof(unsigned int)));
            CUDA_CHECK(cudaMallocHost(&staging, TILE * n * sizeof(unsigned int)));
        }
        CUDA_CHECK(cudaMalloc(&pivots, TILE * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&columns, rows * TILE * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpyAsync(dist, host.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(path, hostPath.data(), count * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned int>().swap(hostPath);
    if (rank == 0) printf("Computing shortest paths...\n");
    mpiCheck(MPI_Barrier(comm));
    double start = MPI_Wtime();
    int owner = 0;
    for (size_t base = 0; base < n; base += TILE) {
        while (base >= firstRow(owner + 1, ranks, n)) ++owner;
        int width = static_cast<int>(std::min(size_t(TILE), n - base));
        size_t bytes = size_t(width) * n * sizeof(unsigned int);
        if (ranks > 1) {
            // Synchronization also ensures the preceding local update has
            // completed before its owner exports the next pivot panel.
            if (rank == owner)
                CUDA_CHECK(cudaMemcpyAsync(staging, dist + (base - startRow) * n, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            broadcast(staging, size_t(width) * n, owner, comm);
            CUDA_CHECK(cudaMemcpyAsync(panel, staging, bytes, cudaMemcpyHostToDevice, stream));
        }
        const unsigned int* pivotInput = ranks == 1 ? dist + base * n : panel;
        pivotSnapshots<<<static_cast<unsigned int>((n + TILE - 1) / TILE), dim3(32, 8), 0, stream>>>(pivotInput, pivots, n, base, width);
        CUDA_CHECK(cudaGetLastError());
        columnSnapshots<<<static_cast<unsigned int>((rows + 3) / 4), dim3(32, 4), 0, stream>>>(dist, pivots, columns, n, rows, base, width);
        CUDA_CHECK(cudaGetLastError());
        updateRows<<<dim3(static_cast<unsigned int>((n + TILE - 1) / TILE), static_cast<unsigned int>((rows + 7) / 8)), dim3(32, 8), 0, stream>>>(dist, path, pivots, columns, n, rows, base, width);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double elapsed = MPI_Wtime() - start, seconds;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GOPS\n", seconds > 0 ? double(n) * double(n) * double(n) / seconds / 1e9 : 0.0);
    }

    int status = 0;
    if (validate || printResults) {
        if (count)
            CUDA_CHECK(cudaMemcpyAsync(host.data(), dist, count * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (rank == 0) {
            host.resize(n * n);
            for (int p = 1; p < ranks; ++p) {
                size_t begin = firstRow(p, ranks, n);
                transfer(host.data() + begin * n,
                         (firstRow(p + 1, ranks, n) - begin) * n, p, false, comm);
            }
            if (printResults) print_results_int(host, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(host, n) ? 0 : 1;
                printf("Validation: %s\n", status == 0 ? "PASSED" : "FAILED");
            }
        } else {
            transfer(host.data(), count, 0, true, comm);
        }
    }
    CUDA_CHECK(cudaFree(dist));
    CUDA_CHECK(cudaFree(path));
    CUDA_CHECK(cudaFree(panel));
    CUDA_CHECK(cudaFree(pivots));
    CUDA_CHECK(cudaFree(columns));
    if (staging) CUDA_CHECK(cudaFreeHost(staging));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status;
}

int main(int argc, char** argv) {
    int provided;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank, ranks;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t n = 512;
    bool validate = false, printResults = false, help = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* text = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long value = strtoull(text, &end, 10);
            if (errno || text[0] == '-' || end == text || *end || value > UINT_MAX) {
                if (rank == 0) fprintf(stderr, "Invalid node count: %s\n", text);
                status = 1;
                break;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) help = true;
        else {
            if (rank == 0) fprintf(stderr, "Unknown or incomplete option: %s\n", argv[i]);
            status = 1;
            break;
        }
    }
    // Check matrix arithmetic and CUDA's two-dimensional grid limit.
    if (n && (n > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / n ||
              (n + 7) / 8 > 65535)) {
        if (rank == 0) fprintf(stderr, "Node count exceeds supported matrix/grid size\n");
        status = 1;
    }
    if (help || status) {
        if (rank == 0) printUsage(argv[0]);
        mpiCheck(MPI_Finalize());
        return status;
    }
    // Do not involve ranks with no tiles in panel broadcasts or GPU work.
    int active = static_cast<int>(std::min(size_t(ranks), std::max(size_t(1), (n + TILE - 1) / TILE)));
    MPI_Comm comm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm));
    if (rank < active) {
        try {
            status = run(n, validate, printResults, comm);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        mpiCheck(MPI_Comm_free(&comm));
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return status;
}
