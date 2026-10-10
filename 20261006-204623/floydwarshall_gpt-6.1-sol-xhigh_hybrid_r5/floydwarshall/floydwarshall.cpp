#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 64;
constexpr int THREAD_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE / THREAD_ROWS;
static_assert(sizeof(unsigned int) == 4, "The benchmark uses 32-bit distances");

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

void checkMPI(int error, const char* operation) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        throw std::runtime_error(std::string(operation) + ": " + std::string(message, length));
    }
}

void checkCUDA(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(error));
}

#define CUDA_CHECK(call) checkCUDA((call), #call)
#define MPI_CHECK(call) checkMPI((call), #call)

struct ProcessGrid {
    MPI_Comm cart = MPI_COMM_NULL, row = MPI_COMM_NULL, column = MPI_COMM_NULL;
    int rank = 0, size = 1, dims[2] = {0, 0}, coords[2] = {0, 0};
    int localRank = 0, localSize = 1;

    ProcessGrid() {
        MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        MPI_CHECK(MPI_Dims_create(size, 2, dims));
        int periods[2] = {0, 0};
        MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &cart));
        MPI_CHECK(MPI_Cart_coords(cart, rank, 2, coords));
        MPI_CHECK(MPI_Comm_split(cart, coords[0], coords[1], &row));
        MPI_CHECK(MPI_Comm_split(cart, coords[1], coords[0], &column));
        MPI_Comm shared;
        MPI_CHECK(MPI_Comm_split_type(cart, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shared));
        MPI_CHECK(MPI_Comm_rank(shared, &localRank));
        MPI_CHECK(MPI_Comm_size(shared, &localSize));
        MPI_CHECK(MPI_Comm_free(&shared));
        // Respect explicit OpenMP settings; otherwise divide available CPUs among ranks.
        if (std::getenv("OMP_NUM_THREADS") == nullptr)
            omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    }

    ~ProcessGrid() {
        if (row != MPI_COMM_NULL) MPI_Comm_free(&row);
        if (column != MPI_COMM_NULL) MPI_Comm_free(&column);
        if (cart != MPI_COMM_NULL) MPI_Comm_free(&cart);
    }
};

size_t ownedTiles(size_t tiles, int coordinate, int processes) {
    return tiles > static_cast<size_t>(coordinate)
        ? (tiles - 1 - coordinate) / processes + 1 : 0;
}

// glibc rand_r advances its 32-bit LCG three times per output. Jumping the
// state, then calling the original rand_r, preserves every generated value.
struct SeedJump {
    unsigned int multiply = 1, add = 0;
    explicit SeedJump(uint64_t outputs) {
        uint64_t steps = outputs * 3;
        unsigned int m = 1103515245u, a = 12345u;
        while (steps != 0) {
            if (steps & 1) {
                add = add * m + a;
                multiply *= m;
            }
            a *= m + 1;
            m *= m;
            steps >>= 1;
        }
    }
    unsigned int operator()(unsigned int seed) const { return multiply * seed + add; }
};

bool canJumpRandomState() {
    const SeedJump one(1);
    for (unsigned int seed : {0u, 1u, 42u, 123456789u, UINT_MAX}) {
        unsigned int actual = seed;
        (void)rand_r(&actual);
        if (actual != one(seed)) return false;
    }
    return true;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, size_t n,
                              size_t rows, size_t columns, const ProcessGrid& grid) {
    if (rows == 0 || columns == 0) return;
    const int threads = std::min(omp_get_max_threads(),
        static_cast<int>(std::max<size_t>(1, std::min<size_t>(rows / 32, INT_MAX))));
    const bool jump = canJumpRandomState();
    const SeedJump gap(static_cast<uint64_t>(grid.dims[1] - 1) * TILE);
    const double range = static_cast<double>(MAX_DISTANCE);
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t r = 0; r < rows; ++r) {
        unsigned int* row = dist.data() + r * columns;
        std::fill(row, row + columns, INF);
        const size_t globalRow = (r / TILE * grid.dims[0] + grid.coords[0]) * TILE + r % TILE;
        if (!jump || globalRow >= n) continue;
        unsigned int seed = SeedJump(globalRow * n + static_cast<size_t>(grid.coords[1]) * TILE)(42);
        for (size_t c = 0; c < columns; c += TILE) {
            const size_t globalColumn = (c / TILE * grid.dims[1] + grid.coords[1]) * TILE;
            const size_t width = std::min<size_t>(TILE, n - globalColumn);
            for (size_t j = 0; j < width; ++j) {
                const unsigned int value = 1 + static_cast<unsigned int>(
                    range * rand_r(&seed) / static_cast<double>(RAND_MAX));
                row[c + j] = globalRow == globalColumn + j ? 0 : value;
            }
            seed = gap(seed);
        }
    }
    // Other libc implementations still get precisely the original random stream.
    if (!jump) {
        unsigned int seed = 42;
        for (size_t r = 0; r < n; ++r) {
            for (size_t c = 0; c < n; ++c) {
                const unsigned int value = 1 + static_cast<unsigned int>(
                    range * rand_r(&seed) / static_cast<double>(RAND_MAX));
                if (r / TILE % grid.dims[0] == static_cast<size_t>(grid.coords[0]) &&
                    c / TILE % grid.dims[1] == static_cast<size_t>(grid.coords[1])) {
                    const size_t lr = r / TILE / grid.dims[0] * TILE + r % TILE;
                    const size_t lc = c / TILE / grid.dims[1] * TILE + c % TILE;
                    dist[lr * columns + lc] = r == c ? 0 : value;
                }
            }
        }
    }
}

struct GPUStorage {
    unsigned int *dist = nullptr, *path = nullptr, *diagonal = nullptr;
    unsigned int *rowHistory = nullptr, *columnHistory = nullptr;
    unsigned int *hostDiagonal = nullptr, *hostRow = nullptr, *hostColumn = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t uploadsDone = nullptr;

    GPUStorage(size_t elements, size_t rows, size_t columns) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&uploadsDone, cudaEventDisableTiming));
        allocate(dist, elements);
        allocate(path, elements);
        allocate(diagonal, 2 * TILE * TILE);
        allocate(rowHistory, TILE * columns);
        allocate(columnHistory, TILE * rows);
        pin(hostDiagonal, 2 * TILE * TILE);
        pin(hostRow, TILE * columns);
        pin(hostColumn, TILE * rows);
    }
    static void allocate(unsigned int*& pointer, size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer),
                              std::max<size_t>(1, count) * sizeof(unsigned int)));
    }
    static void pin(unsigned int*& pointer, size_t count) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pointer),
                                  std::max<size_t>(1, count) * sizeof(unsigned int)));
    }
    ~GPUStorage() {
        if (stream) cudaStreamSynchronize(stream);
        cudaFree(dist); cudaFree(path); cudaFree(diagonal);
        cudaFree(rowHistory); cudaFree(columnHistory);
        cudaFreeHost(hostDiagonal); cudaFreeHost(hostRow); cudaFreeHost(hostColumn);
        if (uploadsDone) cudaEventDestroy(uploadsDone);
        if (stream) cudaStreamDestroy(stream);
    }
};

__global__ void initializePathMatrix(unsigned int* path, size_t elements, size_t columns,
                                     int processRows, int rowCoordinate) {
    for (size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
         index < elements; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t r = index / columns;
        path[index] = static_cast<unsigned int>((r / TILE * processRows + rowCoordinate) * TILE + r % TILE);
    }
}

// Save the pivot row and column BEFORE each k update. These histories let the
// panel and trailing kernels retain the scalar algorithm's exact k order,
// including strict-improvement path entries and ties, while communicating once
// per tile rather than once per vertex.
__global__ void updateDiagonal(unsigned int* dist, unsigned int* path,
                               unsigned int* history, size_t stride,
                               size_t tileRow, size_t tileColumn, unsigned int first, int width) {
    __shared__ unsigned int tile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = y + q * THREAD_ROWS;
        const size_t index = (tileRow * TILE + r) * stride + tileColumn * TILE + x;
        tile[r][x] = values[q] = dist[index];
        paths[q] = UINT_MAX;
        history[r * TILE + x] = INF;
        history[TILE * TILE + r * TILE + x] = INF;
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (y == 0) {
            history[k * TILE + x] = tile[k][x];
            history[TILE * TILE + x * TILE + k] = tile[x][k];
        }
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int r = y + q * THREAD_ROWS;
            // The pivot row/column cannot improve (zero diagonal, nonnegative weights).
            if (r != k && x != k) {
                const unsigned int candidate = tile[r][k] + tile[k][x];
                if (candidate < values[q]) {
                    tile[r][x] = values[q] = candidate;
                    paths[q] = first + k;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const size_t index = (tileRow * TILE + y + q * THREAD_ROWS) * stride + tileColumn * TILE + x;
        if (paths[q] != UINT_MAX) {
            dist[index] = values[q];
            path[index] = paths[q];
        }
    }
}

template<bool RowPanel>
__global__ void updatePanels(unsigned int* dist, unsigned int* path,
                             const unsigned int* diagonal, unsigned int* history,
                             size_t stride, size_t pivotLocal, size_t pivot,
                             int processes, int coordinate, unsigned int first, int width) {
    __shared__ unsigned int tile[TILE][TILE + 1], pivots[TILE][TILE];
    const size_t localTile = blockIdx.x;
    const size_t globalTile = localTile * processes + coordinate;
    const int x = threadIdx.x, y = threadIdx.y;
    if (globalTile == pivot) {
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int r = y + q * THREAD_ROWS;
            if (RowPanel)
                history[r * stride + localTile * TILE + x] = diagonal[r * TILE + x];
            else
                history[(localTile * TILE + r) * TILE + x] = diagonal[TILE * TILE + r * TILE + x];
        }
        return;
    }
    const size_t tileRow = RowPanel ? pivotLocal : localTile;
    const size_t tileColumn = RowPanel ? localTile : pivotLocal;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = y + q * THREAD_ROWS;
        const size_t index = (tileRow * TILE + r) * stride + tileColumn * TILE + x;
        tile[r][x] = values[q] = dist[index];
        paths[q] = UINT_MAX;
        pivots[r][x] = diagonal[(RowPanel ? TILE * TILE : 0) + r * TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (y == 0) {
            if (RowPanel) history[k * stride + localTile * TILE + x] = tile[k][x];
            else history[(localTile * TILE + x) * TILE + k] = tile[x][k];
        }
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int r = y + q * THREAD_ROWS;
            if ((RowPanel && r != k) || (!RowPanel && x != k)) {
                const unsigned int candidate = RowPanel
                    ? pivots[r][k] + tile[k][x] : tile[r][k] + pivots[k][x];
                if (candidate < values[q]) {
                    tile[r][x] = values[q] = candidate;
                    paths[q] = first + k;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const size_t index = (tileRow * TILE + y + q * THREAD_ROWS) * stride + tileColumn * TILE + x;
        if (paths[q] != UINT_MAX) {
            dist[index] = values[q];
            path[index] = paths[q];
        }
    }
}

template<bool FullTile>
__global__ void updateTrailing(unsigned int* dist, unsigned int* path,
                               const unsigned int* rowHistory, const unsigned int* columnHistory,
                               size_t stride, size_t tileColumns, size_t pivot,
                               int processRows, int processColumns, int rowCoordinate, int columnCoordinate,
                               unsigned int first, int width) {
    const size_t tileRow = blockIdx.x / tileColumns, tileColumn = blockIdx.x % tileColumns;
    if (tileRow * processRows + rowCoordinate == pivot ||
        tileColumn * processColumns + columnCoordinate == pivot) return;
    __shared__ unsigned int left[TILE][TILE], right[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int values[ROWS_PER_THREAD], paths[ROWS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int r = y + q * THREAD_ROWS;
        left[r][x] = columnHistory[(tileRow * TILE + r) * TILE + x];
        right[r][x] = rowHistory[r * stride + tileColumn * TILE + x];
        const size_t index = (tileRow * TILE + r) * stride + tileColumn * TILE + x;
        values[q] = dist[index];
        paths[q] = UINT_MAX;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        if (FullTile || k < width) {
            #pragma unroll
            for (int q = 0; q < ROWS_PER_THREAD; ++q) {
                const unsigned int candidate = left[y + q * THREAD_ROWS][k] + right[k][x];
                if (candidate < values[q]) {
                    values[q] = candidate;
                    paths[q] = first + k;
                }
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const size_t index = (tileRow * TILE + y + q * THREAD_ROWS) * stride + tileColumn * TILE + x;
        if (paths[q] != UINT_MAX) {
            dist[index] = values[q];
            path[index] = paths[q];
        }
    }
}

void broadcastAsync(unsigned int* data, size_t count, int root, MPI_Comm communicator,
                    std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count; ) {
        const int chunk = static_cast<int>(std::min<size_t>(count - offset, INT_MAX));
        requests.push_back(MPI_REQUEST_NULL);
        MPI_CHECK(MPI_Ibcast(data + offset, chunk, MPI_UNSIGNED, root, communicator, &requests.back()));
        offset += chunk;
    }
}

void floydWarshall(GPUStorage& gpu, size_t n, size_t rows, size_t columns, const ProcessGrid& grid) {
    const size_t tiles = (n + TILE - 1) / TILE;
    const dim3 threads(TILE, THREAD_ROWS);
    std::vector<MPI_Request> requests;
    requests.reserve(2);
    for (size_t pivot = 0; pivot < tiles; ++pivot) {
        // Only wait for DMA before reusing pinned buffers. Previous trailing work
        // may still run while the host enters the next MPI collective.
        if (pivot != 0) CUDA_CHECK(cudaEventSynchronize(gpu.uploadsDone));
        const int ownerRow = static_cast<int>(pivot % grid.dims[0]);
        const int ownerColumn = static_cast<int>(pivot % grid.dims[1]);
        const int owner = ownerRow * grid.dims[1] + ownerColumn;
        const unsigned int first = static_cast<unsigned int>(pivot * TILE);
        const int width = static_cast<int>(std::min<size_t>(TILE, n - pivot * TILE));
        if (grid.rank == owner) {
            updateDiagonal<<<1, threads, 0, gpu.stream>>>(gpu.dist, gpu.path, gpu.diagonal, columns,
                pivot / grid.dims[0], pivot / grid.dims[1], first, width);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(gpu.hostDiagonal, gpu.diagonal, 2 * TILE * TILE * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, gpu.stream));
            CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        }
        // Only the pivot's process row and column need the diagonal histories.
        // Independent nonblocking broadcasts avoid a global barrier per tile.
        requests.clear();
        if (grid.coords[0] == ownerRow)
            broadcastAsync(gpu.hostDiagonal, 2 * TILE * TILE, ownerColumn, grid.row, requests);
        if (grid.coords[1] == ownerColumn)
            broadcastAsync(gpu.hostDiagonal, 2 * TILE * TILE, ownerRow, grid.column, requests);
        MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        if (grid.rank != owner && (grid.coords[0] == ownerRow || grid.coords[1] == ownerColumn))
            CUDA_CHECK(cudaMemcpyAsync(gpu.diagonal, gpu.hostDiagonal, 2 * TILE * TILE * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, gpu.stream));
        if (grid.coords[0] == ownerRow && columns != 0) {
            updatePanels<true><<<static_cast<unsigned int>(columns / TILE), threads, 0, gpu.stream>>>(
                gpu.dist, gpu.path, gpu.diagonal, gpu.rowHistory, columns, pivot / grid.dims[0], pivot,
                grid.dims[1], grid.coords[1], first, width);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(gpu.hostRow, gpu.rowHistory, TILE * columns * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, gpu.stream));
        }
        if (grid.coords[1] == ownerColumn && rows != 0) {
            updatePanels<false><<<static_cast<unsigned int>(rows / TILE), threads, 0, gpu.stream>>>(
                gpu.dist, gpu.path, gpu.diagonal, gpu.columnHistory, columns, pivot / grid.dims[1], pivot,
                grid.dims[0], grid.coords[0], first, width);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(gpu.hostColumn, gpu.columnHistory, TILE * rows * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, gpu.stream));
        }
        if (grid.coords[0] == ownerRow || grid.coords[1] == ownerColumn)
            CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
        requests.clear();
        broadcastAsync(gpu.hostRow, TILE * columns, ownerRow, grid.column, requests);
        broadcastAsync(gpu.hostColumn, TILE * rows, ownerColumn, grid.row, requests);
        MPI_CHECK(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        if (grid.coords[0] != ownerRow && columns != 0)
            CUDA_CHECK(cudaMemcpyAsync(gpu.rowHistory, gpu.hostRow, TILE * columns * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, gpu.stream));
        if (grid.coords[1] != ownerColumn && rows != 0)
            CUDA_CHECK(cudaMemcpyAsync(gpu.columnHistory, gpu.hostColumn, TILE * rows * sizeof(unsigned int),
                                       cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaEventRecord(gpu.uploadsDone, gpu.stream));
        const size_t localTiles = rows / TILE * (columns / TILE);
        if (localTiles != 0) {
            // Full tiles need no per-k bounds tests in the cubic-work kernel.
            if (width == TILE) {
                updateTrailing<true><<<static_cast<unsigned int>(localTiles), threads, 0, gpu.stream>>>(
                    gpu.dist, gpu.path, gpu.rowHistory, gpu.columnHistory, columns, columns / TILE, pivot,
                    grid.dims[0], grid.dims[1], grid.coords[0], grid.coords[1], first, width);
            } else {
                updateTrailing<false><<<static_cast<unsigned int>(localTiles), threads, 0, gpu.stream>>>(
                    gpu.dist, gpu.path, gpu.rowHistory, gpu.columnHistory, columns, columns / TILE, pivot,
                    grid.dims[0], grid.dims[1], grid.coords[0], grid.coords[1], first, width);
            }
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
}

// Only materialize the complete matrix when results or validation are requested.
// Chunked transfers also avoid MPI's int-count limit on large problems.
std::vector<unsigned int> gatherResult(const std::vector<unsigned int>& local, size_t n,
                                       const ProcessGrid& grid) {
    const size_t tiles = (n + TILE - 1) / TILE;
    std::vector<unsigned int> result;
    if (grid.rank == 0) result.resize(n * n);
    if (grid.rank != 0) {
        for (size_t offset = 0; offset < local.size(); ) {
            const int count = static_cast<int>(std::min<size_t>(local.size() - offset, INT_MAX));
            MPI_CHECK(MPI_Send(local.data() + offset, count, MPI_UNSIGNED, 0, 0, grid.cart));
            offset += count;
        }
        return result;
    }
    std::vector<unsigned int> received;
    for (int source = 0; source < grid.size; ++source) {
        const int rowCoordinate = source / grid.dims[1], columnCoordinate = source % grid.dims[1];
        const size_t rows = ownedTiles(tiles, rowCoordinate, grid.dims[0]) * TILE;
        const size_t columns = ownedTiles(tiles, columnCoordinate, grid.dims[1]) * TILE;
        const unsigned int* data = local.data();
        if (source != 0) {
            received.resize(rows * columns);
            for (size_t offset = 0; offset < received.size(); ) {
                const int count = static_cast<int>(std::min<size_t>(received.size() - offset, INT_MAX));
                MPI_CHECK(MPI_Recv(received.data() + offset, count, MPI_UNSIGNED, source, 0, grid.cart, MPI_STATUS_IGNORE));
                offset += count;
            }
            data = received.data();
        }
        const int threads = std::min(omp_get_max_threads(),
            static_cast<int>(std::max<size_t>(1, std::min<size_t>(rows / 32, INT_MAX))));
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t r = 0; r < rows; ++r) {
            const size_t globalRow = (r / TILE * grid.dims[0] + rowCoordinate) * TILE + r % TILE;
            if (globalRow >= n) continue;
            for (size_t c = 0; c < columns; c += TILE) {
                const size_t globalColumn = (c / TILE * grid.dims[1] + columnCoordinate) * TILE;
                std::copy_n(data + r * columns + c, std::min<size_t>(TILE, n - globalColumn),
                            result.data() + globalRow * n + globalColumn);
            }
        }
    }
    return result;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    for (size_t i = 0; i < std::min<size_t>(n, 10); ++i) {
        for (size_t j = 0; j < std::min<size_t>(n, 10); ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int a = dist[idx2(k, i, n)], b = dist[idx2(j, k, n)];
                if (a < INF && b < INF && a + b < dist[idx2(j, i, n)]) {
                    printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                    return false;
                }
            }
        }
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

int runBenchmark(int argc, char** argv) {
    ProcessGrid grid;
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = strtoull(argument, &end, 10);
            if (errno != 0 || argument[0] == '-' || end == argument || *end != '\0' ||
                value > std::numeric_limits<size_t>::max()) {
                if (grid.rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                return 1;
            }
            n = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (grid.rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (grid.rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    if (n > UINT_MAX || (n != 0 && n > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / n))
        throw std::runtime_error("The matrix dimensions exceed the supported address or vertex range");
    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) throw std::runtime_error("A CUDA device is required on every MPI rank");
    CUDA_CHECK(cudaSetDevice(grid.localRank % devices));
    const size_t tiles = (n + TILE - 1) / TILE;
    const size_t rows = ownedTiles(tiles, grid.coords[0], grid.dims[0]) * TILE;
    const size_t columns = ownedTiles(tiles, grid.coords[1], grid.dims[1]) * TILE;
    if (columns != 0 && rows > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / columns)
        throw std::runtime_error("The padded local matrix is too large");
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, grid.localRank % devices));
    if (rows / TILE * (columns / TILE) > static_cast<size_t>(properties.maxGridSize[0]) ||
        rows / TILE > static_cast<size_t>(properties.maxGridSize[0]) ||
        columns / TILE > static_cast<size_t>(properties.maxGridSize[0]))
        throw std::runtime_error("The local tile grid exceeds CUDA launch limits");
    if (grid.rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI %d x %d, OpenMP, CUDA\n", grid.dims[0], grid.dims[1]);
        printf("Initializing graph...\n");
    }
    std::vector<unsigned int> local(rows * columns);
    initializeDistanceMatrix(local, n, rows, columns, grid);
    GPUStorage gpu(local.size(), rows, columns);
    if (!local.empty()) {
        CUDA_CHECK(cudaMemcpyAsync(gpu.dist, local.data(), local.size() * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, gpu.stream));
        const unsigned int blocks = static_cast<unsigned int>(std::min<size_t>((local.size() + 255) / 256, 65535));
        initializePathMatrix<<<blocks, 256, 0, gpu.stream>>>(gpu.path, local.size(), columns, grid.dims[0], grid.coords[0]);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(gpu.stream));
    if (grid.rank == 0) printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(grid.cart));
    const double start = MPI_Wtime();
    floydWarshall(gpu, n, rows, columns, grid);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid.cart));
    if (grid.rank == 0) {
        printf("Computation time: %.3f ms\n", duration * 1000.0);
        const double operations = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", duration > 0 ? operations / duration / 1e9 : 0.0);
    }
    int status = 0;
    if (printResults || validate) {
        if (!local.empty())
            CUDA_CHECK(cudaMemcpy(local.data(), gpu.dist, local.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        const auto result = gatherResult(local, n, grid);
        if (grid.rank == 0) {
            if (printResults) print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(result, n) ? 0 : 1;
                printf("Validation: %s\n", status == 0 ? "PASSED" : "FAILED");
            }
        }
        MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, grid.cart));
    }
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    const int initialized = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initialized != MPI_SUCCESS) return 1;
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int status = 1;
    try {
        if (provided < MPI_THREAD_FUNNELED)
            throw std::runtime_error("MPI_THREAD_FUNNELED support is required");
        status = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return status;
}
