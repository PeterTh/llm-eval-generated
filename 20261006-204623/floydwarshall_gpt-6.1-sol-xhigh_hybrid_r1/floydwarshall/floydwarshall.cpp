#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 64;
constexpr int TILE_ELEMENTS = TILE * TILE;
constexpr int THREAD_ROWS = 8;
constexpr int THREAD_COLS = 32;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

[[noreturn]] void fail(const char* message) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s\n", rank, message);
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    std::exit(EXIT_FAILURE);
}

void checkMPI(int status) {
    if (status != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(status, message, &length);
        fail(message);
    }
}

void checkCUDA(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        char message[512];
        snprintf(message, sizeof(message), "%s: %s", operation, cudaGetErrorString(status));
        fail(message);
    }
}
#define CUDA_CHECK(operation) checkCUDA((operation), #operation)

struct DeviceBuffer {
    unsigned int* data = nullptr;
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),
                              std::max(count, size_t{1}) * sizeof(unsigned int)));
    }
    ~DeviceBuffer() { cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct HostBuffer {
    unsigned int* data = nullptr;
    explicit HostBuffer(size_t count) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&data),
                                  std::max(count, size_t{1}) * sizeof(unsigned int)));
    }
    ~HostBuffer() { cudaFreeHost(data); }
    HostBuffer(const HostBuffer&) = delete;
    HostBuffer& operator=(const HostBuffer&) = delete;
};

// Tile-major local storage; tile rows and columns are block-cyclic on a 2-D grid.
struct Distribution {
    int rank, size, rows, cols, row, col, tileCount, localRows, localCols;
    MPI_Comm rowComm = MPI_COMM_NULL, colComm = MPI_COMM_NULL;

    explicit Distribution(size_t n) {
        checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
        checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));
        int dimensions[2] = {0, 0};
        checkMPI(MPI_Dims_create(size, 2, dimensions));
        rows = dimensions[0];
        cols = dimensions[1];
        row = rank / cols;
        col = rank % cols;
        tileCount = static_cast<int>((n + TILE - 1) / TILE);
        localRows = tileCount > row ? 1 + (tileCount - 1 - row) / rows : 0;
        localCols = tileCount > col ? 1 + (tileCount - 1 - col) / cols : 0;
        checkMPI(MPI_Comm_split(MPI_COMM_WORLD, row, col, &rowComm));
        checkMPI(MPI_Comm_split(MPI_COMM_WORLD, col, row, &colComm));
    }
    ~Distribution() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&colComm);
    }
    size_t elements() const {
        return static_cast<size_t>(localRows) * localCols * TILE_ELEMENTS;
    }
};

#if defined(__GLIBC__)
// glibc rand_r advances its 32-bit LCG three times per value. Exponentiate
// that affine transition to reproduce the serial stream at any matrix offset.
unsigned int seedAt(size_t offset) {
    uint32_t multiplier = 1, increment = 0;
    for (int step = 0; step < 3; ++step) {
        multiplier *= uint32_t{1103515245};
        increment = increment * uint32_t{1103515245} + uint32_t{12345};
    }
    uint32_t seed = 42;
    while (offset) {
        if (offset & 1) seed = multiplier * seed + increment;
        increment *= multiplier + uint32_t{1};
        multiplier *= multiplier;
        offset >>= 1;
    }
    return seed;
}
#endif

void initializeDistanceMatrix(unsigned int* dist, size_t n, const Distribution& grid) {
#if !defined(__GLIBC__)
    // Other libc implementations may use a different rand_r transition.
    // Generate exact row seeds once, rather than assuming their RNG algorithm.
    std::vector<unsigned int> rowSeeds(n);
    if (grid.rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            rowSeeds[i] = seed;
            for (size_t j = 0; j < n; ++j) (void)rand_r(&seed);
        }
    }
    for (size_t offset = 0; offset < n;) {
        int count = static_cast<int>(std::min(n - offset, size_t{INT_MAX}));
        checkMPI(MPI_Bcast(rowSeeds.data() + offset, count, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
        offset += count;
    }
#endif
    #pragma omp parallel for collapse(2) schedule(static)
    for (int ti = 0; ti < grid.localRows; ++ti) {
        for (int tj = 0; tj < grid.localCols; ++tj) {
            const size_t firstRow = static_cast<size_t>(grid.row + ti * grid.rows) * TILE;
            const size_t firstCol = static_cast<size_t>(grid.col + tj * grid.cols) * TILE;
            unsigned int* tile = dist + (static_cast<size_t>(ti) * grid.localCols + tj) * TILE_ELEMENTS;
            for (int i = 0; i < TILE; ++i) {
                size_t row = firstRow + i;
                unsigned int seed = 42;
                if (row < n && firstCol < n) {
#if defined(__GLIBC__)
                    seed = seedAt(row * n + firstCol);
#else
                    seed = rowSeeds[row];
                    for (size_t j = 0; j < firstCol; ++j) (void)rand_r(&seed);
#endif
                }
                for (int j = 0; j < TILE; ++j) {
                    size_t col = firstCol + j;
                    unsigned int value = INF;
                    if (row < n && col < n)
                        value = 1 + static_cast<unsigned int>(
                            static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
                    tile[i * TILE + j] = row == col ? 0 : value;
                }
            }
        }
    }
}

__global__ void initializePathMatrix(unsigned int* path, size_t count,
                                     int localCols, int rankRow, int gridRows) {
    for (size_t element = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         element < count; element += static_cast<size_t>(gridDim.x) * blockDim.x) {
        size_t tileRow = element / (static_cast<size_t>(localCols) * TILE_ELEMENTS);
        unsigned int row = static_cast<unsigned int>((rankRow + tileRow * gridRows) * TILE
                           + (element % TILE_ELEMENTS) / TILE);
        // The original initializer assigns the source vertex to every entry.
        path[element] = row;
    }
}

// Path entries only change on strict improvement. A UINT_MAX register marker
// avoids loading old paths or writing either matrix when no relaxation occurs.
// Record the pivot row/column as they stand at each intermediate vertex k.
// Using fully closed panels instead would preserve distances but change the
// original strict-relaxation path matrix. These snapshots preserve both.
__global__ void closePivot(unsigned int* dist, unsigned int* path,
                           unsigned int* pivot, size_t offset, unsigned int firstK) {
    __shared__ unsigned int tile[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int via[TILE / THREAD_ROWS][TILE / THREAD_COLS];
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE / THREAD_COLS; ++c) {
            int i = y + r * THREAD_ROWS, j = x + c * THREAD_COLS;
            tile[i][j] = dist[offset + i * TILE + j];
            via[r][c] = UINT_MAX;
        }
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int j = y * THREAD_COLS + x; j < TILE; j += THREAD_ROWS * THREAD_COLS)
            pivot[k * TILE + j] = tile[k][j];
        for (int i = y * THREAD_COLS + x; i < TILE; i += THREAD_ROWS * THREAD_COLS)
            pivot[TILE_ELEMENTS + i * TILE + k] = tile[i][k];
        #pragma unroll
        for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
            int i = y + r * THREAD_ROWS;
            const unsigned int left = tile[i][k];
            #pragma unroll
            for (int c = 0; c < TILE / THREAD_COLS; ++c) {
                int j = x + c * THREAD_COLS;
                unsigned int candidate = left + tile[k][j];
                // With a zero diagonal, pivot row/column never change at k.
                // Avoid writing them, so their concurrent shared reads are safe.
                if (i != k && j != k && candidate < tile[i][j]) {
                    tile[i][j] = candidate;
                    via[r][c] = firstK + k;
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE / THREAD_COLS; ++c) {
            int i = y + r * THREAD_ROWS, j = x + c * THREAD_COLS;
            if (via[r][c] != UINT_MAX) {
                dist[offset + i * TILE + j] = tile[i][j];
                path[offset + i * TILE + j] = via[r][c];
            }
        }
    }
}

// Row panels use the recorded pivot columns; column panels use pivot rows.
template<bool RowPanel>
__global__ void closePanel(unsigned int* dist, unsigned int* path,
                           const unsigned int* pivot, unsigned int* snapshots,
                           int localCols, int pivotLocal, int rankCoordinate,
                           int gridDimension, int globalPivot) {
    int panel = blockIdx.x;
    if (rankCoordinate + panel * gridDimension == globalPivot) return;
    size_t tileIndex = RowPanel ? static_cast<size_t>(pivotLocal) * localCols + panel
                               : static_cast<size_t>(panel) * localCols + pivotLocal;
    size_t offset = tileIndex * TILE_ELEMENTS;
    __shared__ unsigned int tile[TILE][TILE];
    __shared__ unsigned int dependency[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    unsigned int via[TILE / THREAD_ROWS][TILE / THREAD_COLS];
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE / THREAD_COLS; ++c) {
            int i = y + r * THREAD_ROWS, j = x + c * THREAD_COLS;
            tile[i][j] = dist[offset + i * TILE + j];
            dependency[i][j] = pivot[(RowPanel ? TILE_ELEMENTS : 0) + i * TILE + j];
            via[r][c] = UINT_MAX;
        }
    }
    __syncthreads();
    unsigned int* tape = snapshots + static_cast<size_t>(panel) * TILE_ELEMENTS;
    for (int k = 0; k < TILE; ++k) {
        for (int t = y * THREAD_COLS + x; t < TILE; t += THREAD_ROWS * THREAD_COLS) {
            if (RowPanel) tape[k * TILE + t] = tile[k][t];
            else tape[t * TILE + k] = tile[t][k];
        }
        #pragma unroll
        for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
            int i = y + r * THREAD_ROWS;
            unsigned int left = RowPanel ? dependency[i][k] : tile[i][k];
            #pragma unroll
            for (int c = 0; c < TILE / THREAD_COLS; ++c) {
                int j = x + c * THREAD_COLS;
                unsigned int right = RowPanel ? tile[k][j] : dependency[k][j];
                unsigned int candidate = left + right;
                if ((RowPanel ? i != k : j != k) && candidate < tile[i][j]) {
                    tile[i][j] = candidate;
                    via[r][c] = static_cast<unsigned int>(globalPivot * TILE + k);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE / THREAD_COLS; ++c) {
            int i = y + r * THREAD_ROWS, j = x + c * THREAD_COLS;
            if (via[r][c] != UINT_MAX) {
                dist[offset + i * TILE + j] = tile[i][j];
                path[offset + i * TILE + j] = via[r][c];
            }
        }
    }
}

__global__ void updateTiles(unsigned int* __restrict__ dist,
                            unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ rowSnapshots,
                            const unsigned int* __restrict__ colSnapshots,
                            int localCols, int rankRow, int rankCol,
                            int gridRows, int gridCols, int globalPivot) {
    int ti = blockIdx.y, tj = blockIdx.x;
    if (rankRow + ti * gridRows == globalPivot || rankCol + tj * gridCols == globalPivot) return;
    size_t offset = (static_cast<size_t>(ti) * localCols + tj) * TILE_ELEMENTS;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    unsigned int value[TILE / THREAD_ROWS][TILE / THREAD_COLS];
    unsigned int via[TILE / THREAD_ROWS][TILE / THREAD_COLS];
    const int x = threadIdx.x, y = threadIdx.y;
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE / THREAD_COLS; ++c) {
            int i = y + r * THREAD_ROWS, j = x + c * THREAD_COLS;
            left[i][j] = colSnapshots[static_cast<size_t>(ti) * TILE_ELEMENTS + i * TILE + j];
            right[i][j] = rowSnapshots[static_cast<size_t>(tj) * TILE_ELEMENTS + i * TILE + j];
            value[r][c] = dist[offset + i * TILE + j];
            via[r][c] = UINT_MAX;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
            unsigned int a = left[y + r * THREAD_ROWS][k];
            #pragma unroll
            for (int c = 0; c < TILE / THREAD_COLS; ++c) {
                unsigned int candidate = a + right[k][x + c * THREAD_COLS];
                if (candidate < value[r][c]) {
                    value[r][c] = candidate;
                    via[r][c] = static_cast<unsigned int>(globalPivot * TILE + k);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        #pragma unroll
        for (int c = 0; c < TILE / THREAD_COLS; ++c) {
            int i = y + r * THREAD_ROWS, j = x + c * THREAD_COLS;
            if (via[r][c] != UINT_MAX) {
                dist[offset + i * TILE + j] = value[r][c];
                path[offset + i * TILE + j] = via[r][c];
            }
        }
    }
}

// Split messages to respect MPI's int count even for very large graphs.
void broadcastAsync(unsigned int* buffer, size_t count, int root, MPI_Comm comm,
                     std::vector<MPI_Request>& requests) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t{INT_MAX}));
        MPI_Request request;
        checkMPI(MPI_Ibcast(buffer + offset, chunk, MPI_UNSIGNED, root, comm, &request));
        requests.push_back(request);
        offset += chunk;
    }
}

void waitBroadcasts(std::vector<MPI_Request>& requests) {
    if (!requests.empty())
        checkMPI(MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    requests.clear();
}

double floydWarshall(DeviceBuffer& dist, DeviceBuffer& path, const Distribution& grid,
                     cudaStream_t stream, bool cudaAware) {
    DeviceBuffer pivot(2 * TILE_ELEMENTS);
    size_t rowCount = static_cast<size_t>(grid.localCols) * TILE_ELEMENTS;
    size_t colCount = static_cast<size_t>(grid.localRows) * TILE_ELEMENTS;
    DeviceBuffer rowSnapshots(rowCount), colSnapshots(colCount);
    HostBuffer hostPivot(cudaAware ? 0 : 2 * TILE_ELEMENTS);
    HostBuffer hostRows(cudaAware ? 0 : rowCount), hostCols(cudaAware ? 0 : colCount);
    std::vector<MPI_Request> requests;
    requests.reserve(8);
    const dim3 threads(THREAD_COLS, THREAD_ROWS);
    cudaEvent_t pivotUploaded, panelsUploaded;
    CUDA_CHECK(cudaEventCreateWithFlags(&pivotUploaded, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&panelsUploaded, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventRecord(pivotUploaded, stream));
    CUDA_CHECK(cudaEventRecord(panelsUploaded, stream));
    checkMPI(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();

    for (int round = 0; round < grid.tileCount; ++round) {
        int pivotRow = round % grid.rows, pivotCol = round % grid.cols;
        bool ownsRow = grid.row == pivotRow, ownsCol = grid.col == pivotCol;
        bool owner = ownsRow && ownsCol;
        if (owner) {
            size_t offset = (static_cast<size_t>(round / grid.rows) * grid.localCols
                            + round / grid.cols) * TILE_ELEMENTS;
            closePivot<<<1, threads, 0, stream>>>(dist.data, path.data, pivot.data, offset,
                                                 static_cast<unsigned int>(round * TILE));
            CUDA_CHECK(cudaGetLastError());
        }
        // Only the pivot process row/column needs the small dependency tile.
        bool exchangePivot = (ownsRow && grid.cols > 1) || (ownsCol && grid.rows > 1);
        if (exchangePivot) {
            if (owner || cudaAware) {
                if (owner && !cudaAware)
                    CUDA_CHECK(cudaMemcpyAsync(hostPivot.data, pivot.data, 2 * TILE_ELEMENTS * sizeof(unsigned int),
                                               cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
            if (!cudaAware && !owner)
                CUDA_CHECK(cudaEventSynchronize(pivotUploaded));
            unsigned int* buffer = cudaAware ? pivot.data : hostPivot.data;
            if (ownsRow && grid.cols > 1)
                broadcastAsync(buffer, 2 * TILE_ELEMENTS, pivotCol, grid.rowComm, requests);
            if (ownsCol && grid.rows > 1)
                broadcastAsync(buffer, 2 * TILE_ELEMENTS, pivotRow, grid.colComm, requests);
            waitBroadcasts(requests);
            if (!owner && !cudaAware) {
                CUDA_CHECK(cudaMemcpyAsync(pivot.data, hostPivot.data, 2 * TILE_ELEMENTS * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaEventRecord(pivotUploaded, stream));
            }
        }
        if (owner) {
            CUDA_CHECK(cudaMemcpyAsync(rowSnapshots.data + static_cast<size_t>(round / grid.cols) * TILE_ELEMENTS,
                                       pivot.data, TILE_ELEMENTS * sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(colSnapshots.data + static_cast<size_t>(round / grid.rows) * TILE_ELEMENTS,
                                       pivot.data + TILE_ELEMENTS, TILE_ELEMENTS * sizeof(unsigned int),
                                       cudaMemcpyDeviceToDevice, stream));
        }
        if (ownsRow && grid.localCols > 0) {
            closePanel<true><<<grid.localCols, threads, 0, stream>>>(dist.data, path.data, pivot.data,
                rowSnapshots.data, grid.localCols, round / grid.rows, grid.col, grid.cols, round);
            CUDA_CHECK(cudaGetLastError());
        }
        if (ownsCol && grid.localRows > 0) {
            closePanel<false><<<grid.localRows, threads, 0, stream>>>(dist.data, path.data, pivot.data,
                colSnapshots.data, grid.localCols, round / grid.cols, grid.row, grid.rows, round);
            CUDA_CHECK(cudaGetLastError());
        }
        // Nonblocking orthogonal broadcasts let MPI progress both panels together.
        // Host staging is portable; CUDA-aware MPI uses device buffers directly.
        if (!cudaAware) {
            if (ownsRow && grid.rows > 1 && rowCount)
                CUDA_CHECK(cudaMemcpyAsync(hostRows.data, rowSnapshots.data, rowCount * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost, stream));
            if (ownsCol && grid.cols > 1 && colCount)
                CUDA_CHECK(cudaMemcpyAsync(hostCols.data, colSnapshots.data, colCount * sizeof(unsigned int),
                                           cudaMemcpyDeviceToHost, stream));
        }
        bool sendsPanel = (ownsRow && grid.rows > 1 && rowCount) || (ownsCol && grid.cols > 1 && colCount);
        if (sendsPanel || (cudaAware && (grid.rows > 1 || grid.cols > 1)))
            CUDA_CHECK(cudaStreamSynchronize(stream));
        // MPI may overwrite a pinned receive buffer only after its previous
        // asynchronous upload has finished. Waiting on the upload event still
        // allows the preceding GPU tile update to overlap communication.
        if (!cudaAware && (grid.rows > 1 || grid.cols > 1))
            CUDA_CHECK(cudaEventSynchronize(panelsUploaded));
        if (grid.rows > 1)
            broadcastAsync(cudaAware ? rowSnapshots.data : hostRows.data, rowCount,
                           pivotRow, grid.colComm, requests);
        if (grid.cols > 1)
            broadcastAsync(cudaAware ? colSnapshots.data : hostCols.data, colCount,
                           pivotCol, grid.rowComm, requests);
        waitBroadcasts(requests);
        if (!cudaAware) {
            if (!ownsRow && grid.rows > 1 && rowCount)
                CUDA_CHECK(cudaMemcpyAsync(rowSnapshots.data, hostRows.data, rowCount * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, stream));
            if (!ownsCol && grid.cols > 1 && colCount)
                CUDA_CHECK(cudaMemcpyAsync(colSnapshots.data, hostCols.data, colCount * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(panelsUploaded, stream));
        }
        if (grid.localRows > 0 && grid.localCols > 0) {
            updateTiles<<<dim3(grid.localCols, grid.localRows), threads, 0, stream>>>(dist.data, path.data,
                rowSnapshots.data, colSnapshots.data, grid.localCols, grid.row, grid.col,
                grid.rows, grid.cols, round);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    double elapsed = MPI_Wtime() - start;
    CUDA_CHECK(cudaEventDestroy(pivotUploaded));
    CUDA_CHECK(cudaEventDestroy(panelsUploaded));
    return elapsed;
}

std::vector<unsigned int> gatherDistances(const DeviceBuffer& dist, size_t n,
                                          const Distribution& grid, cudaStream_t stream) {
    HostBuffer local(grid.elements());
    CUDA_CHECK(cudaMemcpyAsync(local.data, dist.data, grid.elements() * sizeof(unsigned int),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned int> result;
    if (grid.rank == 0) result.resize(n * n);
    if (grid.rank != 0) {
        for (size_t offset = 0; offset < grid.elements();) {
            int count = static_cast<int>(std::min(grid.elements() - offset, size_t{INT_MAX}));
            checkMPI(MPI_Send(local.data + offset, count, MPI_UNSIGNED, 0, 1, MPI_COMM_WORLD));
            offset += count;
        }
        return result;
    }
    std::vector<unsigned int> received;
    for (int rank = 0; rank < grid.size; ++rank) {
        int rankRow = rank / grid.cols, rankCol = rank % grid.cols;
        int tileRows = grid.tileCount > rankRow ? 1 + (grid.tileCount - 1 - rankRow) / grid.rows : 0;
        int tileCols = grid.tileCount > rankCol ? 1 + (grid.tileCount - 1 - rankCol) / grid.cols : 0;
        size_t count = static_cast<size_t>(tileRows) * tileCols * TILE_ELEMENTS;
        const unsigned int* buffer = local.data;
        if (rank != 0) {
            received.resize(count);
            for (size_t offset = 0; offset < count;) {
                int chunk = static_cast<int>(std::min(count - offset, size_t{INT_MAX}));
                checkMPI(MPI_Recv(received.data() + offset, chunk, MPI_UNSIGNED, rank, 1,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                offset += chunk;
            }
            buffer = received.data();
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int ti = 0; ti < tileRows; ++ti) {
            for (int tj = 0; tj < tileCols; ++tj) {
                size_t firstRow = static_cast<size_t>(rankRow + ti * grid.rows) * TILE;
                size_t firstCol = static_cast<size_t>(rankCol + tj * grid.cols) * TILE;
                const unsigned int* tile = buffer + (static_cast<size_t>(ti) * tileCols + tj) * TILE_ELEMENTS;
                size_t columns = std::min(size_t{TILE}, n - firstCol);
                for (size_t i = 0; i < TILE && firstRow + i < n; ++i)
                    std::memcpy(result.data() + (firstRow + i) * n + firstCol,
                                tile + i * TILE, columns * sizeof(unsigned int));
            }
        }
    }
    return result;
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        if (dist[idx2(i, i, n)] != 0) valid = 0;
    }
    size_t sample = std::min(n, size_t{10});
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t k = 0; k < n; ++k) {
                unsigned int a = dist[idx2(k, i, n)], b = dist[idx2(j, k, n)];
                if (a < INF && b < INF && a + b < dist[idx2(j, i, n)]) valid = 0;
            }
        }
    }
    if (!valid) printf("Validation failed: diagonal or triangle inequality violated\n");
    return valid != 0;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int rank = 0;
    checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* argument = argv[++i];
            errno = 0;
            unsigned long long parsed = strtoull(argument, &end, 10);
            if (errno || argument[0] == '-' || end == argument || *end != '\0' ||
                parsed > static_cast<unsigned long long>(INT_MAX - TILE)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    size_t padded = (numNodes + TILE - 1) / TILE * TILE;
    if (padded && padded > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / padded)
        fail("Matrix size exceeds the addressable memory range");

    MPI_Comm nodeComm;
    checkMPI(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm));
    int localRank = 0, localSize = 0, deviceCount = 0;
    checkMPI(MPI_Comm_rank(nodeComm, &localRank));
    checkMPI(MPI_Comm_size(nodeComm, &localSize));
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) fail("The hybrid benchmark requires a CUDA GPU on every MPI rank");
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    checkMPI(MPI_Comm_free(&nodeComm));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::min(8, std::max(1, omp_get_num_procs() / localSize)));

    bool cudaAware = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    cudaAware = MPIX_Query_cuda_support() != 0;
#endif
    int aware = cudaAware ? 1 : 0;
    checkMPI(MPI_Allreduce(MPI_IN_PLACE, &aware, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    cudaAware = aware != 0;

    Distribution grid(numNodes);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    DeviceBuffer dist(grid.elements()), path(grid.elements());
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Parallelization: MPI %dx%d, OpenMP %d threads/rank, CUDA (%s MPI)\n",
               grid.rows, grid.cols, omp_get_max_threads(), cudaAware ? "CUDA-aware" : "host-staged");
        printf("Initializing graph...\n");
    }
    {
        HostBuffer initial(grid.elements());
        initializeDistanceMatrix(initial.data, numNodes, grid);
        CUDA_CHECK(cudaMemcpyAsync(dist.data, initial.data, grid.elements() * sizeof(unsigned int),
                                   cudaMemcpyHostToDevice, stream));
        if (grid.elements()) {
            unsigned int blocks = static_cast<unsigned int>(std::min(size_t{65535}, (grid.elements() + 255) / 256));
            initializePathMatrix<<<blocks, 256, 0, stream>>>(path.data, grid.elements(), grid.localCols, grid.row, grid.rows);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    if (rank == 0) printf("Computing shortest paths...\n");
    double elapsed = floydWarshall(dist, path, grid, stream, cudaAware), seconds = 0;
    checkMPI(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        double ops = static_cast<double>(numNodes) * numNodes * numNodes;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }
    int status = 0;
    if (printResults || validate) {
        std::vector<unsigned int> result = gatherDistances(dist, numNodes, grid, stream);
        if (rank == 0) {
            if (printResults) print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(result, numNodes) ? 0 : 1;
                printf("Validation: %s\n", status == 0 ? "PASSED" : "FAILED");
            }
        }
    }
    checkMPI(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    checkMPI(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support is required");
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::bad_alloc&) {
        fail("Host matrix allocation failed");
    }
    checkMPI(MPI_Finalize());
    return status;
}
