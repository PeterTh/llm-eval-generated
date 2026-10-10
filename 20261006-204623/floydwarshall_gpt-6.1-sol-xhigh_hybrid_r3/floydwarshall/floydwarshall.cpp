#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OPEN_MPI)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int BLOCK_ROWS = 8;

// During computation, via is one plus the smallest possible largest internal
// vertex on a shortest path (zero denotes a direct edge). Minimizing this on
// equal distances preserves the original ascending-k, strict-update path
// matrix even though blocked Floyd-Warshall changes the evaluation order.
struct alignas(8) Cell {
    unsigned int distance;
    unsigned int via;
};
static_assert(sizeof(Cell) == 2 * sizeof(unsigned int));
int worldRank = -1;

void cudaCheck(cudaError_t error, const char* expression) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Rank %d: %s: %s\n", worldRank, expression, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
void mpiCheck(int error, const char* expression) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, expression, length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)
#define MPI_CHECK(call) mpiCheck((call), #call)

struct Grid {
    MPI_Comm cart = MPI_COMM_NULL, row = MPI_COMM_NULL, column = MPI_COMM_NULL;
    int size, rank, dims[2] = {0, 0}, coords[2], rows, columns;
    size_t elements;
    explicit Grid(int tiles, int tileWidth = 32) {
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        MPI_CHECK(MPI_Dims_create(size, 2, dims));
        int periodic[2] = {0, 0};
        MPI_CHECK(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, 0, &cart));
        MPI_CHECK(MPI_Comm_rank(cart, &rank));
        MPI_CHECK(MPI_Cart_coords(cart, rank, 2, coords));
        MPI_CHECK(MPI_Comm_split(cart, coords[0], coords[1], &row));
        MPI_CHECK(MPI_Comm_split(cart, coords[1], coords[0], &column));
        rows = tiles > coords[0] ? (tiles - 1 - coords[0]) / dims[0] + 1 : 0;
        columns = tiles > coords[1] ? (tiles - 1 - coords[1]) / dims[1] + 1 : 0;
        elements = size_t(rows) * columns * tileWidth * tileWidth;
    }
    ~Grid() {
        MPI_Comm_free(&row);
        MPI_Comm_free(&column);
        MPI_Comm_free(&cart);
    }
    Grid(const Grid&) = delete;
    Grid& operator=(const Grid&) = delete;
};

struct DeviceCells {
    Cell* data = nullptr;
    explicit DeviceCells(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data),
                              std::max(count, size_t(1)) * sizeof(Cell)));
    }
    ~DeviceCells() { cudaFree(data); }
    DeviceCells(const DeviceCells&) = delete;
    DeviceCells& operator=(const DeviceCells&) = delete;
};
struct PinnedCells {
    Cell* data = nullptr;
    explicit PinnedCells(size_t count) {
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&data),
                                 std::max(count, size_t(1)) * sizeof(Cell), cudaHostAllocPortable));
    }
    ~PinnedCells() { cudaFreeHost(data); }
    PinnedCells(const PinnedCells&) = delete;
    PinnedCells& operator=(const PinnedCells&) = delete;
};
int hostThreads(size_t work) {
    return std::min(omp_get_max_threads(), int(std::min(size_t(16), work / 32768 + 1)));
}

// glibc rand_r advances a 32-bit LCG three times per output. Exponentiating
// this affine map gives each tile row its exact position in the original
// random stream without generating or replicating the global matrix.
unsigned int randomSeedAt(size_t offset) {
    uint32_t multiplier = 1, increment = 0;
    for (int i = 0; i < 3; ++i) {
        multiplier *= 1103515245u;
        increment = increment * 1103515245u + 12345u;
    }
    uint32_t seed = 42;
    while (offset) {
        if (offset & 1) seed = multiplier * seed + increment;
        increment *= multiplier + 1;
        multiplier *= multiplier;
        offset >>= 1;
    }
    return seed;
}

template<int B>
void initializeMatrix(std::vector<Cell>& local, size_t n, const Grid& grid) {
    constexpr int TILE = B;
    constexpr int TILE_CELLS = B * B;
#if !defined(__GLIBC__)
    // Preserve the platform's rand_r sequence on other C libraries as well.
    std::vector<unsigned int> rowSeeds(n);
    if (grid.rank == 0) {
        unsigned int seed = 42;
        for (size_t i = 0; i < n; ++i) {
            rowSeeds[i] = seed;
            for (size_t j = 0; j < n; ++j) (void)rand_r(&seed);
        }
    }
    for (size_t start = 0; start < n;) {
        const int count = int(std::min(n - start, size_t(INT_MAX)));
        MPI_CHECK(MPI_Bcast(rowSeeds.data() + start, count, MPI_UNSIGNED, 0, grid.cart));
        start += count;
    }
#endif
    const size_t tileRows = size_t(grid.rows) * grid.columns * TILE;
    const int threads = hostThreads(local.size());
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t line = 0; line < tileRows; ++line) {
        const size_t tile = line / TILE;
        const size_t row = size_t(grid.coords[0] + int(tile / grid.columns) * grid.dims[0])
                           * TILE + line % TILE;
        const size_t firstColumn = size_t(grid.coords[1] + int(tile % grid.columns)
                                         * grid.dims[1]) * TILE;
        unsigned int seed = 42;
        if (row < n && firstColumn < n) {
#if defined(__GLIBC__)
            seed = randomSeedAt(row * n + firstColumn);
#else
            seed = rowSeeds[row];
            for (size_t j = 0; j < firstColumn; ++j) (void)rand_r(&seed);
#endif
        }
        for (int j = 0; j < TILE; ++j) {
            const size_t column = firstColumn + j;
            unsigned int distance = INF;
            if (row < n && column < n)
                distance = 1 + static_cast<unsigned int>(
                    double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
            if (row == column) distance = 0;
            local[tile * TILE_CELLS + (line % TILE) * TILE + j] = {distance, 0};
        }
    }
}

__device__ __forceinline__ bool relax(Cell& current, Cell left, Cell right,
                                     unsigned int vertex) {
    const unsigned int distance = left.distance + right.distance;
    const unsigned int via = max(vertex, max(left.via, right.via));
    if (distance < current.distance || (distance == current.distance && via < current.via)) {
        current = {distance, via};
        return true;
    }
    return false;
}

template<int B>
__global__ void updateDiagonal(Cell* diagonal, unsigned int firstVertex) {
    constexpr int TILE = B;
    constexpr int ROWS_PER_THREAD = B / BLOCK_ROWS;
    __shared__ Cell tile[TILE][TILE];
    const int column = threadIdx.x, firstRow = threadIdx.y;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = firstRow + r * BLOCK_ROWS;
        tile[row][column] = diagonal[row * TILE + column];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = firstRow + r * BLOCK_ROWS;
            // Zero diagonals and positive edges make pivot rows/columns
            // invariant. Leaving them untouched avoids shared-memory races.
            if (row != k && column != k) {
                Cell value = tile[row][column];
                if (relax(value, tile[row][k], tile[k][column], firstVertex + k + 1))
                    tile[row][column] = value;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = firstRow + r * BLOCK_ROWS;
        diagonal[row * TILE + column] = tile[row][column];
    }
}

template<int B>
__global__ void updatePanels(Cell* matrix, const Cell* diagonal, int rows, int columns,
                             int pivotRow, int pivotColumn, bool ownsRow, bool ownsColumn,
                             unsigned int firstVertex) {
    constexpr int TILE = B;
    constexpr int TILE_CELLS = B * B;
    constexpr int ROWS_PER_THREAD = B / BLOCK_ROWS;
    const bool rowPanel = blockIdx.y == 0;
    const int index = blockIdx.x;
    if (rowPanel ? (!ownsRow || index >= columns || (ownsColumn && index == pivotColumn))
                 : (!ownsColumn || index >= rows || (ownsRow && index == pivotRow))) return;
    const size_t tileIndex = rowPanel ? size_t(pivotRow) * columns + index
                                     : size_t(index) * columns + pivotColumn;
    Cell* output = matrix + tileIndex * TILE_CELLS;
    extern __shared__ Cell shared[];
    Cell (*pivot)[TILE] = reinterpret_cast<Cell (*)[TILE]>(shared);
    Cell (*tile)[TILE] = reinterpret_cast<Cell (*)[TILE]>(shared + TILE_CELLS);
    const int column = threadIdx.x, firstRow = threadIdx.y;
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = firstRow + r * BLOCK_ROWS;
        pivot[row][column] = diagonal[row * TILE + column];
        tile[row][column] = output[row * TILE + column];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = firstRow + r * BLOCK_ROWS;
            if (rowPanel ? row != k : column != k) {
                Cell value = tile[row][column];
                const Cell left = rowPanel ? pivot[row][k] : tile[row][k];
                const Cell right = rowPanel ? tile[k][column] : pivot[k][column];
                if (relax(value, left, right, firstVertex + k + 1)) tile[row][column] = value;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = firstRow + r * BLOCK_ROWS;
        output[row * TILE + column] = tile[row][column];
    }
}

template<int B>
__global__ void packColumn(const Cell* matrix, Cell* panel, size_t count,
                           int columns, int pivotColumn) {
    constexpr int TILE_CELLS = B * B;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x)
        panel[i] = matrix[(i / TILE_CELLS * columns + pivotColumn) * TILE_CELLS + i % TILE_CELLS];
}

template<int B>
__global__ void updateRemaining(Cell* matrix, const Cell* columnPanel, const Cell* rowPanel,
                                size_t tiles, int columns, int pivotRow, int pivotColumn,
                                bool ownsRow, bool ownsColumn, unsigned int firstVertex) {
    constexpr int TILE = B;
    constexpr int TILE_CELLS = B * B;
    constexpr int ROWS_PER_THREAD = B / BLOCK_ROWS;
    extern __shared__ Cell shared[];
    Cell (*left)[TILE] = reinterpret_cast<Cell (*)[TILE]>(shared);
    Cell (*right)[TILE] = reinterpret_cast<Cell (*)[TILE]>(shared + TILE_CELLS);
    const int column = threadIdx.x, firstRow = threadIdx.y;
    for (size_t tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int tileRow = int(tile / columns), tileColumn = int(tile % columns);
        if ((ownsRow && tileRow == pivotRow) || (ownsColumn && tileColumn == pivotColumn)) continue;
        Cell values[ROWS_PER_THREAD];
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = firstRow + r * BLOCK_ROWS;
            const int element = row * TILE + column;
            left[row][column] = columnPanel[size_t(tileRow) * TILE_CELLS + element];
            right[row][column] = rowPanel[size_t(tileColumn) * TILE_CELLS + element];
            values[r] = matrix[tile * TILE_CELLS + element];
        }
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            const Cell b = right[k][column];
            #pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r)
                relax(values[r], left[firstRow + r * BLOCK_ROWS][k], b, firstVertex + k + 1);
        }
        #pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const int row = firstRow + r * BLOCK_ROWS;
            matrix[tile * TILE_CELLS + row * TILE + column] = values[r];
        }
        __syncthreads();
    }
}

template<int B>
__global__ void restorePathIndices(Cell* matrix, size_t count, int columns,
                                   int columnCoordinate, int columnRanks) {
    constexpr int TILE = B;
    constexpr int TILE_CELLS = B * B;
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += size_t(gridDim.x) * blockDim.x) {
        const unsigned int column = unsigned(columnCoordinate
            + int(i / TILE_CELLS % columns) * columnRanks) * TILE + i % TILE;
        matrix[i].via = matrix[i].via ? matrix[i].via - 1 : column;
    }
}
unsigned int blocksFor(size_t elements) {
    return unsigned(std::min(size_t(65535), (elements + 255) / 256));
}

// Chunk collectives to support matrices larger than MPI's int count.
void broadcastAsync(Cell* cells, size_t count, int root, MPI_Comm communicator,
                    std::vector<MPI_Request>& requests, bool deviceBuffer = false) {
    auto* bytes = reinterpret_cast<unsigned char*>(cells);
    size_t remaining = count * sizeof(Cell);
    do {
        const int chunk = int(std::min(remaining, size_t(INT_MAX)));
        if (deviceBuffer) {
            // CUDA-aware Open MPI supports blocking collectives even on
            // versions that cannot accept GPU buffers in nonblocking ones.
            MPI_CHECK(MPI_Bcast(bytes, chunk, MPI_BYTE, root, communicator));
        } else {
            MPI_Request request;
            MPI_CHECK(MPI_Ibcast(bytes, chunk, MPI_BYTE, root, communicator, &request));
            requests.push_back(request);
        }
        bytes += chunk;
        remaining -= chunk;
    } while (remaining);
}
void waitBroadcasts(std::vector<MPI_Request>& requests) {
    if (!requests.empty()) {
        MPI_CHECK(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
        requests.clear();
    }
}

bool deviceMPIAvailable(const Grid& grid) {
    int available = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    available = MPIX_Query_cuda_support();
#endif
    int supported = 0;
    MPI_CHECK(MPI_Allreduce(&available, &supported, 1, MPI_INT, MPI_MIN, grid.cart));
    return supported != 0;
}

template<int B>
double floydWarshall(Cell* matrix, int tiles, const Grid& grid) {
    constexpr int TILE = B;
    constexpr int TILE_CELLS = B * B;
    const bool deviceMPI = deviceMPIAvailable(grid);
    const size_t rowCount = size_t(grid.columns) * TILE_CELLS;
    const size_t columnCount = size_t(grid.rows) * TILE_CELLS;
    DeviceCells diagonal(TILE_CELLS), rowPanel(rowCount), columnPanel(columnCount);
    // Double-buffer GPU receives: MPI can fill the next panel while the
    // previous interior update is still reading the other panel on the GPU.
    DeviceCells nextRow(deviceMPI ? rowCount : 0), nextColumn(deviceMPI ? columnCount : 0);
    PinnedCells hostDiagonal(deviceMPI ? 0 : TILE_CELLS),
                hostRow(deviceMPI ? 0 : rowCount), hostColumn(deviceMPI ? 0 : columnCount);
    cudaStream_t stream;
    cudaEvent_t copied;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
    const dim3 threads(TILE, BLOCK_ROWS);
    std::vector<MPI_Request> requests;
    requests.reserve(4);
    // Resolve CUDA's lazy kernel loading before timing the algorithm, just as
    // matrix allocation and graph initialization are outside the original timer.
    if constexpr (B > 32) {
        CUDA_CHECK(cudaFuncSetAttribute(updatePanels<B>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        2 * TILE_CELLS * sizeof(Cell)));
        CUDA_CHECK(cudaFuncSetAttribute(updateRemaining<B>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        2 * TILE_CELLS * sizeof(Cell)));
    }
    cudaFuncAttributes attributes;
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, updateDiagonal<B>));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, updatePanels<B>));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, packColumn<B>));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, updateRemaining<B>));
    CUDA_CHECK(cudaFuncGetAttributes(&attributes, restorePathIndices<B>));
    MPI_CHECK(MPI_Barrier(grid.cart));
    const double start = MPI_Wtime();
    for (int k = 0; k < tiles; ++k) {
        // Wait only for previous host-to-device copies; the preceding interior
        // update can continue while the CPU progresses communication.
        if (k && grid.size > 1) CUDA_CHECK(cudaEventSynchronize(copied));
        const int ownerRow = k % grid.dims[0], ownerColumn = k % grid.dims[1];
        const bool ownsRow = grid.coords[0] == ownerRow;
        const bool ownsColumn = grid.coords[1] == ownerColumn;
        const bool ownsDiagonal = ownsRow && ownsColumn;
        const int pivotRow = k / grid.dims[0], pivotColumn = k / grid.dims[1];
        Cell* rowData = deviceMPI && (k & 1) ? nextRow.data : rowPanel.data;
        Cell* columnData = deviceMPI && (k & 1) ? nextColumn.data : columnPanel.data;
        Cell* localDiagonal = matrix;
        if (ownsDiagonal) {
            localDiagonal += (size_t(pivotRow) * grid.columns + pivotColumn) * TILE_CELLS;
            updateDiagonal<B><<<1, threads, 0, stream>>>(localDiagonal, k * TILE);
            CUDA_CHECK(cudaGetLastError());
            if (grid.size > 1) {
                if (!deviceMPI)
                    CUDA_CHECK(cudaMemcpyAsync(hostDiagonal.data, localDiagonal,
                        TILE_CELLS * sizeof(Cell), cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
        Cell* pivotData = ownsDiagonal ? localDiagonal : diagonal.data;
        Cell* pivotMessage = deviceMPI ? pivotData : hostDiagonal.data;
        if (ownsRow && grid.dims[1] > 1)
            broadcastAsync(pivotMessage, TILE_CELLS, ownerColumn, grid.row, requests, deviceMPI);
        if (ownsColumn && grid.dims[0] > 1)
            broadcastAsync(pivotMessage, TILE_CELLS, ownerRow, grid.column, requests, deviceMPI);
        waitBroadcasts(requests);
        if ((ownsRow || ownsColumn) && !ownsDiagonal && !deviceMPI)
            CUDA_CHECK(cudaMemcpyAsync(diagonal.data, hostDiagonal.data,
                TILE_CELLS * sizeof(Cell), cudaMemcpyHostToDevice, stream));
        if ((ownsRow || ownsColumn) && std::max(grid.rows, grid.columns)) {
            updatePanels<B><<<dim3(std::max(grid.rows, grid.columns), 2), threads, 2 * TILE_CELLS * sizeof(Cell), stream>>>(
                matrix, pivotData, grid.rows, grid.columns,
                pivotRow, pivotColumn, ownsRow, ownsColumn, k * TILE);
            CUDA_CHECK(cudaGetLastError());
        }
        const Cell* localRow = rowData;
        if (ownsRow) localRow = matrix + size_t(pivotRow) * grid.columns * TILE_CELLS;
        if (ownsColumn && columnCount) {
            packColumn<B><<<blocksFor(columnCount), 256, 0, stream>>>(
                matrix, columnData, columnCount, grid.columns, pivotColumn);
            CUDA_CHECK(cudaGetLastError());
        }
        bool staged = false;
        if (!deviceMPI && ownsRow && grid.dims[0] > 1 && rowCount) {
            CUDA_CHECK(cudaMemcpyAsync(hostRow.data, localRow, rowCount * sizeof(Cell),
                cudaMemcpyDeviceToHost, stream));
            staged = true;
        }
        if (!deviceMPI && ownsColumn && grid.dims[1] > 1 && columnCount) {
            CUDA_CHECK(cudaMemcpyAsync(hostColumn.data, columnData,
                columnCount * sizeof(Cell), cudaMemcpyDeviceToHost, stream));
            staged = true;
        }
        if (staged || (deviceMPI && grid.size > 1 && (ownsRow || ownsColumn)))
            CUDA_CHECK(cudaStreamSynchronize(stream));
        if (grid.dims[0] > 1)
            broadcastAsync(deviceMPI ? const_cast<Cell*>(localRow) : hostRow.data,
                           rowCount, ownerRow, grid.column, requests, deviceMPI);
        if (grid.dims[1] > 1)
            broadcastAsync(deviceMPI ? columnData : hostColumn.data,
                           columnCount, ownerColumn, grid.row, requests, deviceMPI);
        waitBroadcasts(requests);
        if (!deviceMPI && !ownsRow && rowCount)
            CUDA_CHECK(cudaMemcpyAsync(rowData, hostRow.data, rowCount * sizeof(Cell),
                cudaMemcpyHostToDevice, stream));
        if (!deviceMPI && !ownsColumn && columnCount)
            CUDA_CHECK(cudaMemcpyAsync(columnData, hostColumn.data,
                columnCount * sizeof(Cell), cudaMemcpyHostToDevice, stream));
        if (grid.size > 1) CUDA_CHECK(cudaEventRecord(copied, stream));
        const size_t localTiles = size_t(grid.rows) * grid.columns;
        if (grid.rows > int(ownsRow) && grid.columns > int(ownsColumn)) {
            updateRemaining<B><<<unsigned(std::min(localTiles, size_t(65535))), threads, 2 * TILE_CELLS * sizeof(Cell), stream>>>(
                matrix, columnData, localRow, localTiles, grid.columns,
                pivotRow, pivotColumn, ownsRow, ownsColumn, k * TILE);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    if (grid.elements) {
        restorePathIndices<B><<<blocksFor(grid.elements), 256, 0, stream>>>(
            matrix, grid.elements, grid.columns, grid.coords[1], grid.dims[1]);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    CUDA_CHECK(cudaEventDestroy(copied));
    CUDA_CHECK(cudaStreamDestroy(stream));
    return elapsed;
}

template<int B>
void gatherResults(const std::vector<Cell>& local, size_t n, int tiles, const Grid& grid,
                   std::vector<unsigned int>& dist, std::vector<unsigned int>* path = nullptr) {
    constexpr int TILE = B;
    constexpr int TILE_CELLS = B * B;
    if (grid.rank == 0) {
        dist.resize(n * n);
        if (path) path->resize(n * n);
    }
    std::vector<Cell> received;
    for (int rank = 0; rank < grid.size; ++rank) {
        int coordinates[2];
        MPI_CHECK(MPI_Cart_coords(grid.cart, rank, 2, coordinates));
        const int rows = tiles > coordinates[0] ? (tiles - 1 - coordinates[0]) / grid.dims[0] + 1 : 0;
        const int columns = tiles > coordinates[1] ? (tiles - 1 - coordinates[1]) / grid.dims[1] + 1 : 0;
        const size_t count = size_t(rows) * columns * TILE_CELLS;
        if (grid.rank == 0 && rank) received.resize(count);
        if (rank && (grid.rank == 0 || grid.rank == rank)) {
            auto* bytes = reinterpret_cast<unsigned char*>(grid.rank == 0 ? received.data()
                           : const_cast<Cell*>(local.data()));
            for (size_t remaining = count * sizeof(Cell); remaining;) {
                const int chunk = int(std::min(remaining, size_t(INT_MAX)));
                if (grid.rank == 0)
                    MPI_CHECK(MPI_Recv(bytes, chunk, MPI_BYTE, rank, 0, grid.cart, MPI_STATUS_IGNORE));
                else
                    MPI_CHECK(MPI_Send(bytes, chunk, MPI_BYTE, 0, 0, grid.cart));
                bytes += chunk;
                remaining -= chunk;
            }
        }
        if (grid.rank != 0) continue;
        const Cell* data = rank ? received.data() : local.data();
        const int threads = hostThreads(count);
        #pragma omp parallel for schedule(static) num_threads(threads)
        for (size_t line = 0; line < size_t(rows) * columns * TILE; ++line) {
            const size_t tile = line / TILE;
            const size_t row = size_t(coordinates[0] + int(tile / columns) * grid.dims[0])
                               * TILE + line % TILE;
            const size_t firstColumn = size_t(coordinates[1] + int(tile % columns) * grid.dims[1]) * TILE;
            if (row >= n) continue;
            for (int j = 0; j < TILE && firstColumn + j < n; ++j) {
                const Cell value = data[tile * TILE_CELLS + (line % TILE) * TILE + j];
                dist[row * n + firstColumn + j] = value.distance;
                if (path) (*path)[row * n + firstColumn + j] = value.via;
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& dist, size_t n) {
    int invalid = 0;
    const int threads = hostThreads(n * n);
    #pragma omp parallel for reduction(|:invalid) schedule(static) num_threads(threads)
    for (size_t i = 0; i < n; ++i) invalid |= dist[i * n + i] != 0;
    const size_t sample = std::min(n, size_t(10));
    #pragma omp parallel for collapse(2) reduction(|:invalid) schedule(static) num_threads(threads)
    for (size_t i = 0; i < sample; ++i) {
        for (size_t j = 0; j < sample; ++j) {
            for (size_t k = 0; k < n; ++k) {
                const unsigned int a = dist[i * n + k], b = dist[k * n + j];
                if (a < INF && b < INF) invalid |= a + b < dist[i * n + j];
            }
        }
    }
    return !invalid;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\n", name);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
    printf("Launch one MPI rank per GPU; OMP_NUM_THREADS controls host parallelism.\n");
}

template<int B>
int runDistributedBenchmark(size_t n, bool validate, bool printResults) {
    constexpr int TILE = B;
    const int tiles = int((n + TILE - 1) / TILE);
    Grid grid(tiles, B);
    if (grid.rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI grid: %d x %d; CUDA tile: %d x %d\n", grid.dims[0], grid.dims[1], TILE, TILE);
        printf("Initializing graph...\n");
    }
    std::vector<Cell> local(grid.elements);
    initializeMatrix<B>(local, n, grid);
    DeviceCells matrix(grid.elements);
    if (grid.elements)
        CUDA_CHECK(cudaMemcpy(matrix.data, local.data(), grid.elements * sizeof(Cell), cudaMemcpyHostToDevice));
    if (grid.rank == 0) printf("Computing shortest paths...\n");
    const double elapsed = floydWarshall<B>(matrix.data, tiles, grid);
    double seconds = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid.cart));
    if (grid.rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        const double operations = double(n) * double(n) * double(n);
        printf("Performance: %.3f GOPS\n", seconds > 0 ? operations / seconds / 1e9 : 0.0);
    }
    int status = 0;
    if (printResults || validate) {
        if (grid.elements)
            CUDA_CHECK(cudaMemcpy(local.data(), matrix.data, grid.elements * sizeof(Cell), cudaMemcpyDeviceToHost));
        std::vector<unsigned int> dist;
        gatherResults<B>(local, n, tiles, grid, dist);
        if (grid.rank == 0) {
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(dist, n) ? 0 : 1;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, grid.cart));
    return status;
}

int runBenchmark(size_t n, bool validate, bool printResults) {
    // Node-local ranks respect CUDA_VISIBLE_DEVICES. A launcher exposing one
    // GPU per rank naturally selects device zero on each rank.
    MPI_Comm node;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &node));
    int localRank, devices = 0;
    MPI_CHECK(MPI_Comm_rank(node, &localRank));
    MPI_CHECK(MPI_Comm_free(&node));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required.\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    int sharedBytes = 0, minimumSharedBytes = 0, ranks = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&sharedBytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, localRank % devices));
    MPI_CHECK(MPI_Allreduce(&sharedBytes, &minimumSharedBytes, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    // Wider tiles amortize panel exchange at scale. Small matrices retain more
    // GPU blocks with 32-wide tiles; GPUs without 64 KiB opt-in use these too.
    if (minimumSharedBytes >= 2 * 64 * 64 * int(sizeof(Cell)) &&
        ((ranks > 1 && n >= 1024) || n >= 8192))
        return runDistributedBenchmark<64>(n, validate, printResults);
    return runDistributedBenchmark<32>(n, validate, printResults);
}

int main(int argc, char** argv) {
    int provided = 0;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        if (worldRank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    size_t n = 512;
    bool validate = false, printResults = false;
    int status = 0;
    bool run = true;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* argument = argv[++i];
            const unsigned long long value = strtoull(argument, &end, 10);
            if (errno || argument[0] == '-' || end == argument || *end ||
                value > unsigned(INT_MAX - 64) ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(Cell) / value)) {
                if (worldRank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", argument);
                status = 1;
                run = false;
                break;
            }
            n = size_t(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (worldRank == 0) printUsage(argv[0]);
            run = false;
            break;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            run = false;
            break;
        }
    }
    if (run) {
        try {
            status = runBenchmark(n, validate, printResults);
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
    }
    MPI_CHECK(MPI_Finalize());
    return status;
}
