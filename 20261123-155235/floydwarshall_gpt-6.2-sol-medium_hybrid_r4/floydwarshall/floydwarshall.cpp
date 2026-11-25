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
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int Tile = 32;
constexpr int TileCells = Tile * Tile;
constexpr int ThreadRows = 8;
constexpr int BatchTiles = 2048;
constexpr unsigned int MaxDistance = 200;
constexpr unsigned int PaddingDistance = 255;

void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

template <typename T, bool Pinned = false>
class Buffer {
public:
    explicit Buffer(size_t count) {
        if (count == 0) {
            return;
        }
        if (count > SIZE_MAX / sizeof(T)) {
            CUDA_CHECK(cudaErrorInvalidValue);
        }
        if constexpr (Pinned) {
            CUDA_CHECK(cudaMallocHost(&data_, count * sizeof(T)));
        } else {
            CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
        }
    }

    ~Buffer() {
        if (data_) {
            if constexpr (Pinned) {
                CUDA_CHECK(cudaFreeHost(data_));
            } else {
                CUDA_CHECK(cudaFree(data_));
            }
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    T* get() const { return data_; }

private:
    T* data_ = nullptr;
};

// Pack the distance above the provenance of that distance. Zero provenance
// denotes an unchanged direct edge (whose original path entry is its source).
// Otherwise it is one plus the smallest possible largest intermediate vertex
// of a shortest path. This is the iteration of the last strict improvement in
// scalar Floyd-Warshall. Carrying it through tile products also resolves ties
// in the original iteration order.
// Nine upper bits suffice even for a sum of two padded distances.
template <typename Word>
constexpr int PathBits = sizeof(Word) * CHAR_BIT - 9;

template <typename Word>
__device__ __forceinline__ Word through(Word right, Word pivot) {
    constexpr Word pathMask = (Word{1} << PathBits<Word>) - 1;
    const Word path = right & pathMask;
    return (right & ~pathMask) | (path > pivot ? path : pivot);
}

template <typename Word>
__device__ __forceinline__ Word relax(Word current, Word left, Word right) {
    constexpr Word pathMask = (Word{1} << PathBits<Word>) - 1;
    const Word distance = (left & ~pathMask) + (right & ~pathMask);
    const Word leftPath = left & pathMask;
    const Word rightPath = right & pathMask;
    const Word candidate = distance | (leftPath > rightPath ? leftPath : rightPath);
    return current < candidate ? current : candidate;
}

template <typename Word>
__global__ void closePivot(Word* matrixTile, Word* pivotTile, int firstVertex) {
    __shared__ Word distances[Tile][Tile];
    const int column = threadIdx.x;
    const int firstRow = threadIdx.y;

    #pragma unroll
    for (int row = firstRow; row < Tile; row += ThreadRows) {
        distances[row][column] = matrixTile[row * Tile + column];
    }
    __syncthreads();

    for (int k = 0; k < Tile; ++k) {
        // With a zero diagonal, neither input to this iteration can improve.
        if (column != k) {
            const Word via = through(distances[k][column], Word(firstVertex) + k + 1);
            #pragma unroll
            for (int row = firstRow; row < Tile; row += ThreadRows) {
                if (row != k) {
                    distances[row][column] = relax(distances[row][column], distances[row][k], via);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row = firstRow; row < Tile; row += ThreadRows) {
        const Word value = distances[row][column];
        matrixTile[row * Tile + column] = value;
        pivotTile[row * Tile + column] = value;
    }
}

template <typename Word>
__device__ __forceinline__ void updateTile(Word* target, const Word* left,
                                            const Word* right, Word* panel, int firstVertex) {
    __shared__ Word leftValues[Tile][Tile];
    __shared__ Word rightValues[Tile][Tile];
    Word values[Tile / ThreadRows];
    const int column = threadIdx.x;

    #pragma unroll
    for (int i = 0; i < Tile / ThreadRows; ++i) {
        const int row = threadIdx.y + i * ThreadRows;
        const int cell = row * Tile + column;
        leftValues[row][column] = left[cell];
        rightValues[row][column] = through(right[cell], Word(firstVertex) + row + 1);
        values[i] = target[cell];
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < Tile; ++k) {
        const Word via = rightValues[k][column];
        #pragma unroll
        for (int i = 0; i < Tile / ThreadRows; ++i) {
            const int row = threadIdx.y + i * ThreadRows;
            values[i] = relax(values[i], leftValues[row][k], via);
        }
    }

    #pragma unroll
    for (int i = 0; i < Tile / ThreadRows; ++i) {
        const int cell = (threadIdx.y + i * ThreadRows) * Tile + column;
        target[cell] = values[i];
        if (panel) {
            panel[cell] = values[i];
        }
    }
}

template <typename Word, bool Horizontal>
__global__ void updatePanel(Word* matrix, const Word* pivot, Word* panel, int localColumns,
                            int pivotIndex, int pivotCrossIndex, bool ownsPivot, int firstVertex) {
    const int index = blockIdx.x;
    Word* destination = panel + size_t(index) * TileCells;
    if (ownsPivot && index == pivotCrossIndex) {
        #pragma unroll
        for (int row = threadIdx.y; row < Tile; row += ThreadRows) {
            const int cell = row * Tile + threadIdx.x;
            destination[cell] = pivot[cell];
        }
        return;
    }

    const size_t tile = Horizontal ? size_t(pivotIndex) * localColumns + index
                                   : size_t(index) * localColumns + pivotIndex;
    Word* target = matrix + tile * TileCells;
    updateTile(target, Horizontal ? pivot : target, Horizontal ? target : pivot,
               destination, firstVertex);
}

template <typename Word>
__global__ void updateRemaining(Word* matrix, const Word* column, const Word* row,
                                int localColumns, int otherColumns, int excludedRow,
                                int excludedColumn, int firstVertex) {
    int localRow = blockIdx.x / otherColumns;
    int localColumn = blockIdx.x % otherColumns;
    if (localRow >= excludedRow) {
        ++localRow;
    }
    if (localColumn >= excludedColumn) {
        ++localColumn;
    }
    Word* target = matrix + (size_t(localRow) * localColumns + localColumn) * TileCells;
    updateTile(target, column + size_t(localRow) * TileCells,
               row + size_t(localColumn) * TileCells, static_cast<Word*>(nullptr), firstVertex);
}

struct Grid {
    // Global tile (i, j) belongs to process (i % rows, j % columns).
    MPI_Comm comm;
    MPI_Comm rowComm;
    MPI_Comm columnComm;
    int rank;
    int rows;
    int columns;
    int row;
    int column;
    int tileRows;
    int tileColumns;

    static int tileCount(int total, int coordinate, int dimension) {
        return (total - coordinate + dimension - 1) / dimension;
    }
};

[[maybe_unused]] int hostThreads(size_t cells) {
    const int available = omp_get_max_threads();
    const int limit = std::getenv("OMP_NUM_THREADS") ? available : std::min(available, 16);
    return static_cast<int>(std::min<size_t>(limit, std::max<size_t>(1, cells / 8192)));
}

// glibc rand_r advances its 32-bit LCG three times per generated value.
// Exponentiate the affine transition to start each tile at its serial offset.
#ifdef __GLIBC__
struct RandomStep {
    uint32_t factor = 1;
    uint32_t increment = 0;

    unsigned int operator()(unsigned int seed) const { return factor * seed + increment; }
};

RandomStep randomStep(uint64_t count) {
    RandomStep step;
    uint32_t factor = 1103515245u;
    uint32_t increment = 12345u;
    for (count *= 3; count; count >>= 1) {
        if (count & 1) {
            step.factor *= factor;
            step.increment = factor * step.increment + increment;
        }
        increment *= factor + 1;
        factor *= factor;
    }
    return step;
}
#else
unsigned int advanceRandom(unsigned int seed, uint64_t count) {
    while (count--) {
        rand_r(&seed);
    }
    return seed;
}
#endif

template <typename Word>
void initialize(Word* matrix, int n, const Grid& grid, cudaStream_t stream) {
    const size_t localTiles = size_t(grid.tileRows) * grid.tileColumns;
    if (!localTiles) {
        return;
    }

#ifdef __GLIBC__
    const auto nextRow = randomStep(n);
#else
    // Other C libraries need only provide rand_r, so obtain the row offsets
    // directly from that implementation.
    std::vector<unsigned int> rowSeeds(n);
    if (grid.rank == 0) {
        unsigned int seed = 42;
        for (auto& rowSeed : rowSeeds) {
            rowSeed = seed;
            seed = advanceRandom(seed, n);
        }
    }
    MPI_Bcast(rowSeeds.data(), n, MPI_UNSIGNED, 0, grid.comm);
#endif

    const size_t batchTiles = std::min<size_t>(BatchTiles, localTiles);
    Buffer<Word, true> first(batchTiles * TileCells);
    Buffer<Word, true> second(std::min(batchTiles, localTiles - batchTiles) * TileCells);
    cudaEvent_t ready[2];
    for (auto& event : ready) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    }

    int slot = 0;
    for (size_t offset = 0; offset < localTiles; offset += batchTiles, slot ^= 1) {
        CUDA_CHECK(cudaEventSynchronize(ready[slot]));
        Word* staging = slot ? second.get() : first.get();
        const size_t count = std::min(batchTiles, localTiles - offset);
        #pragma omp parallel for schedule(static) num_threads(hostThreads(count * TileCells))
        for (size_t index = 0; index < count; ++index) {
            const size_t tile = offset + index;
            const int firstRow = (tile / grid.tileColumns * grid.rows + grid.row) * Tile;
            const int firstColumn = (tile % grid.tileColumns * grid.columns + grid.column) * Tile;
            Word* output = staging + index * TileCells;
#ifdef __GLIBC__
            unsigned int rowSeed = randomStep(uint64_t(firstRow) * n + firstColumn)(42);
#endif
            for (int i = 0; i < Tile; ++i) {
                const int row = firstRow + i;
#ifdef __GLIBC__
                unsigned int seed = rowSeed;
                rowSeed = nextRow(rowSeed);
#else
                unsigned int seed = row < n ? advanceRandom(rowSeeds[row], firstColumn) : 0;
#endif
                for (int j = 0; j < Tile; ++j) {
                    const int column = firstColumn + j;
                    unsigned int distance = PaddingDistance;
                    if (row < n && column < n) {
                        distance = 1 + static_cast<unsigned int>(
                            double(MaxDistance) * rand_r(&seed) / double(RAND_MAX));
                    }
                    output[i * Tile + j] = Word(row == column ? 0 : distance) << PathBits<Word>;
                }
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(matrix + offset * TileCells, staging,
                                   count * TileCells * sizeof(Word), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(ready[slot], stream));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (auto& event : ready) {
        CUDA_CHECK(cudaEventDestroy(event));
    }
}

template <typename Word>
MPI_Datatype wordType() {
    return sizeof(Word) == sizeof(uint32_t) ? MPI_UINT32_T : MPI_UINT64_T;
}

template <typename Word>
void broadcast(Word* data, size_t count, int root, MPI_Comm comm,
               std::vector<MPI_Request>& requests) {
    while (count) {
        const int chunk = static_cast<int>(std::min<size_t>(count, INT_MAX));
        requests.emplace_back();
        MPI_Ibcast(data, chunk, wordType<Word>(), root, comm, &requests.back());
        data += chunk;
        count -= chunk;
    }
}

void finishBroadcasts(std::vector<MPI_Request>& requests) {
    if (!requests.empty()) {
        MPI_Waitall(static_cast<int>(requests.size()), requests.data(), MPI_STATUSES_IGNORE);
        requests.clear();
    }
}

template <typename Word>
void stage(Word* destination, const Word* source, size_t cells,
           cudaMemcpyKind direction, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(destination, source, cells * sizeof(Word), direction, stream));
}

template <typename Word>
void floydWarshall(Word* matrix, int tiles, const Grid& grid, cudaStream_t stream,
                   Word* pivot, Word* row, Word* column, Word* hostPivot,
                   Word* hostRow, Word* hostColumn, cudaEvent_t stagingReady) {
    const size_t rowCells = size_t(grid.tileColumns) * TileCells;
    const size_t columnCells = size_t(grid.tileRows) * TileCells;
    const dim3 threads(Tile, ThreadRows);
    std::vector<MPI_Request> requests;
    requests.reserve(4);

    for (int p = 0; p < tiles; ++p) {
        const int pivotRow = p % grid.rows;
        const int pivotColumn = p % grid.columns;
        const int localPivotRow = p / grid.rows;
        const int localPivotColumn = p / grid.columns;
        const bool onPivotRow = grid.row == pivotRow;
        const bool onPivotColumn = grid.column == pivotColumn;
        const bool owner = onPivotRow && onPivotColumn;
        const bool exchangePivot = (onPivotRow && hostColumn) || (onPivotColumn && hostRow);

        if (stagingReady) {
            CUDA_CHECK(cudaEventSynchronize(stagingReady));
        }

        // Close the diagonal tile, then share it with its process row and column.
        if (owner) {
            const size_t tile = size_t(localPivotRow) * grid.tileColumns + localPivotColumn;
            closePivot<<<1, threads, 0, stream>>>(matrix + tile * TileCells, pivot, p * Tile);
            if (exchangePivot) {
                stage(hostPivot, pivot, TileCells, cudaMemcpyDeviceToHost, stream);
                CUDA_CHECK(cudaStreamSynchronize(stream));
            }
        }
        if (onPivotRow && hostColumn) {
            broadcast(hostPivot, TileCells, pivotColumn, grid.rowComm, requests);
        }
        if (onPivotColumn && hostRow) {
            broadcast(hostPivot, TileCells, pivotRow, grid.columnComm, requests);
        }
        if (!owner) {
            finishBroadcasts(requests);
            if (exchangePivot) {
                stage(pivot, hostPivot, TileCells, cudaMemcpyHostToDevice, stream);
            }
        }

        // Close each tile of the two pivot panels using the diagonal tile.
        if (onPivotRow) {
            updatePanel<Word, true><<<grid.tileColumns, threads, 0, stream>>>(
                matrix, pivot, row, grid.tileColumns, localPivotRow, localPivotColumn, owner, p * Tile);
            if (hostRow) {
                stage(hostRow, row, rowCells, cudaMemcpyDeviceToHost, stream);
            }
        }
        if (onPivotColumn) {
            updatePanel<Word, false><<<grid.tileRows, threads, 0, stream>>>(
                matrix, pivot, column, grid.tileColumns, localPivotColumn, localPivotRow, owner, p * Tile);
            if (hostColumn) {
                stage(hostColumn, column, columnCells, cudaMemcpyDeviceToHost, stream);
            }
        }
        // The pivot owner can update its own panels while the pivot travels.
        finishBroadcasts(requests);
        if ((onPivotRow && hostRow) || (onPivotColumn && hostColumn)) {
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        // Broadcast each panel across the other grid dimension. Both transfers
        // use pinned staging so the MPI implementation needs only host buffers.
        if (hostRow) {
            broadcast(hostRow, rowCells, pivotRow, grid.columnComm, requests);
        }
        if (hostColumn) {
            broadcast(hostColumn, columnCells, pivotColumn, grid.rowComm, requests);
        }
        finishBroadcasts(requests);

        if (hostRow && !onPivotRow) {
            stage(row, hostRow, rowCells, cudaMemcpyHostToDevice, stream);
        }
        if (hostColumn && !onPivotColumn) {
            stage(column, hostColumn, columnCells, cudaMemcpyHostToDevice, stream);
        }
        if (stagingReady) {
            // Mark when MPI's host buffers can be reused, ahead of the GPU tile updates.
            CUDA_CHECK(cudaEventRecord(stagingReady, stream));
        }

        // Every other tile can now use the two panels resident on its GPU.
        const int otherRows = grid.tileRows - onPivotRow;
        const int otherColumns = grid.tileColumns - onPivotColumn;
        if (otherRows && otherColumns) {
            updateRemaining<<<size_t(otherRows) * otherColumns, threads, 0, stream>>>(
                matrix, column, row, grid.tileColumns, otherColumns,
                onPivotRow ? localPivotRow : grid.tileRows,
                onPivotColumn ? localPivotColumn : grid.tileColumns, p * Tile);
        }
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename Word>
std::vector<unsigned int> gatherDistances(const Word* matrix, int n, int tiles, const Grid& grid) {
    std::vector<unsigned int> distances;
    if (grid.rank == 0) {
        distances.resize(size_t(n) * n);
    }
    if (!n) {
        return distances;
    }

    const size_t localTiles = size_t(grid.tileRows) * grid.tileColumns;
    if (grid.rank != 0) {
        Buffer<Word, true> staging(std::min<size_t>(BatchTiles, localTiles) * TileCells);
        for (size_t offset = 0; offset < localTiles; offset += BatchTiles) {
            const size_t count = std::min<size_t>(BatchTiles, localTiles - offset) * TileCells;
            CUDA_CHECK(cudaMemcpy(staging.get(), matrix + offset * TileCells,
                                  count * sizeof(Word), cudaMemcpyDeviceToHost));
            MPI_Send(staging.get(), count, wordType<Word>(), 0, 0, grid.comm);
        }
        return distances;
    }

    // Rank zero has the largest domain. Receive and unpack one batch at a time.
    Buffer<Word, true> staging(std::min<size_t>(BatchTiles, localTiles) * TileCells);
    for (int source = 0; source < grid.rows * grid.columns; ++source) {
        const int processRow = source / grid.columns;
        const int processColumn = source % grid.columns;
        const int tileRows = Grid::tileCount(tiles, processRow, grid.rows);
        const int tileColumns = Grid::tileCount(tiles, processColumn, grid.columns);
        const size_t sourceTiles = size_t(tileRows) * tileColumns;
        for (size_t offset = 0; offset < sourceTiles; offset += BatchTiles) {
            const size_t count = std::min<size_t>(BatchTiles, sourceTiles - offset);
            if (source == 0) {
                CUDA_CHECK(cudaMemcpy(staging.get(), matrix + offset * TileCells,
                                      count * TileCells * sizeof(Word), cudaMemcpyDeviceToHost));
            } else {
                MPI_Recv(staging.get(), count * TileCells, wordType<Word>(), source, 0,
                         grid.comm, MPI_STATUS_IGNORE);
            }

            #pragma omp parallel for schedule(static) num_threads(hostThreads(count * TileCells))
            for (size_t index = 0; index < count; ++index) {
                const size_t tile = offset + index;
                const int row = (tile / tileColumns * grid.rows + processRow) * Tile;
                const int column = (tile % tileColumns * grid.columns + processColumn) * Tile;
                const Word* input = staging.get() + index * TileCells;
                const int width = std::min(Tile, n - column);
                for (int i = 0; i < std::min(Tile, n - row); ++i) {
                    unsigned int* output = distances.data() + size_t(row + i) * n + column;
                    for (int j = 0; j < width; ++j) {
                        output[j] = input[i * Tile + j] >> PathBits<Word>;
                    }
                }
            }
        }
    }
    return distances;
}

template <typename Word>
__global__ void extractSamples(const Word* matrix, unsigned int* output, size_t count,
                               int n, int sample, int rows, int columns,
                               int processRow, int processColumn, int tileColumns) {
    for (size_t index = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += size_t(gridDim.x) * blockDim.x) {
        const int line = index / n;
        const int position = index % n;
        const int row = line == 0 || line > sample ? position : line - 1;
        const int column = line == 0 ? position : line > sample ? line - sample - 1 : position;
        const int tileRow = row / Tile;
        const int tileColumn = column / Tile;
        unsigned int distance = 0;
        if (tileRow % rows == processRow && tileColumn % columns == processColumn) {
            const size_t localTile = size_t(tileRow / rows) * tileColumns + tileColumn / columns;
            distance = matrix[localTile * TileCells + row % Tile * Tile + column % Tile] >> PathBits<Word>;
        }
        output[index] = distance;
    }
}

template <typename Word>
std::vector<unsigned int> gatherSamples(const Word* matrix, int n, int sample, const Grid& grid) {
    const size_t count = size_t(n) * (2 * sample + 1);
    std::vector<unsigned int> samples(count);
    if (count) {
        Buffer<unsigned int> device(count);
        const int blocks = std::min<size_t>((count + 255) / 256, 65535);
        extractSamples<<<blocks, 256>>>(matrix, device.get(), count, n, sample,
                                       grid.rows, grid.columns, grid.row, grid.column, grid.tileColumns);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(samples.data(), device.get(), count * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }
    for (size_t offset = 0; offset < count;) {
        const int chunk = std::min<size_t>(count - offset, INT_MAX);
        auto* data = samples.data() + offset;
        MPI_Reduce(grid.rank == 0 ? MPI_IN_PLACE : data, grid.rank == 0 ? data : nullptr,
                   chunk, MPI_UNSIGNED,
                   MPI_BOR, 0, grid.comm);
        offset += chunk;
    }
    return samples;
}

template <typename Distance>
bool validateResult(Distance distance, int n, int sample) {
    for (int i = 0; i < n; ++i) {
        if (distance(i, i) != 0) {
            printf("Validation failed: diagonal element [%d,%d] is not zero\n", i, i);
            return false;
        }
    }

    for (int i = 0; i < sample; ++i) {
        for (int j = 0; j < sample; ++j) {
            for (int k = 0; k < n; ++k) {
                if (distance(i, k) + distance(k, j) < distance(i, j)) {
                    printf("Validation failed: triangle inequality violated at [%d,%d,%d]\n", i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

template <typename Word>
int run(int n, int tiles, bool validate, bool printResults, const Grid& grid) {
    const size_t rowCells = size_t(grid.tileColumns) * TileCells;
    const size_t columnCells = size_t(grid.tileRows) * TileCells;
    Buffer<Word> matrix(size_t(grid.tileRows) * rowCells);
    Buffer<Word> pivot(tiles ? TileCells : 0), row(rowCells), column(columnCells);
    Buffer<Word, true> hostPivot(grid.rows > 1 || grid.columns > 1 ? TileCells : 0);
    Buffer<Word, true> hostRow(grid.rows > 1 ? rowCells : 0);
    Buffer<Word, true> hostColumn(grid.columns > 1 ? columnCells : 0);
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t stagingReady = nullptr;
    if (hostPivot.get()) {
        CUDA_CHECK(cudaEventCreateWithFlags(&stagingReady, cudaEventDisableTiming));
    }

    initialize(matrix.get(), n, grid, stream);
    if (tiles) {
        // Resolve CUDA's lazy kernel loading and any PTX JIT before timing.
        cudaFuncAttributes attributes;
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, closePivot<Word>));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, (updatePanel<Word, true>)));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, (updatePanel<Word, false>)));
        CUDA_CHECK(cudaFuncGetAttributes(&attributes, updateRemaining<Word>));
    }
    if (grid.rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(grid.comm);
    const double start = MPI_Wtime();
    floydWarshall(matrix.get(), tiles, grid, stream, pivot.get(), row.get(), column.get(),
                  hostPivot.get(), hostRow.get(), hostColumn.get(), stagingReady);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, grid.comm);
    if (stagingReady) {
        CUDA_CHECK(cudaEventDestroy(stagingReady));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (grid.rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GOPS\n", double(n) * n * n / seconds / 1e9);
    }

    std::vector<unsigned int> distances;
    if (printResults) {
        distances = gatherDistances(matrix.get(), n, tiles, grid);
        if (grid.rank == 0) {
            print_results_int(distances, "DistanceMatrix");
        }
    }

    int result = 0;
    if (validate) {
        if (grid.rank == 0) {
            printf("Validating result...\n");
        }
        const int sample = std::min(n, 10);
        std::vector<unsigned int> samples;
        if (!printResults) {
            samples = gatherSamples(matrix.get(), n, sample, grid);
        }
        if (grid.rank == 0) {
            const bool valid = validateResult([&](int i, int j) {
                if (printResults) {
                    return distances[size_t(i) * n + j];
                }
                if (i == j) {
                    return samples[i];
                }
                if (i < sample) {
                    return samples[size_t(1 + i) * n + j];
                }
                return samples[size_t(1 + sample + j) * n + i];
            }, n, sample);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            result = valid ? 0 : 1;
        }
    }
    return result;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n = 512;
    bool validate = false;
    bool printResults = false;
    int result = 0;
    bool help = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            const long value = strtol(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || value < 0 || value > INT_MAX) {
                if (rank == 0) {
                    printf("Invalid node count: %s\n", argv[i]);
                }
                result = 1;
                break;
            }
            n = static_cast<int>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            help = true;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            result = 1;
            break;
        }
    }

    if (help || result) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return result;
    }

    const int tiles = n / Tile + (n % Tile != 0);
    const int limit = std::max(1, tiles);
    int rows = 1;
    int columns = 1;
    // Maximize the number of ranks with work, then choose the most square grid.
    for (int height = 1; height <= limit && height <= worldSize / height; ++height) {
        const int width = std::min(limit, worldSize / height);
        if (height * width >= rows * columns) {
            rows = height;
            columns = width;
        }
    }

    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < rows * columns ? 0 : MPI_UNDEFINED, rank, &active);
    if (active != MPI_COMM_NULL) {
        MPI_Comm node;
        MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_free(&node);
        int deviceCount;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            fprintf(stderr, "Rank %d: no CUDA device is available\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

        Grid grid{active, MPI_COMM_NULL, MPI_COMM_NULL, rank, rows, columns,
                  rank / columns, rank % columns,
                  Grid::tileCount(tiles, rank / columns, rows),
                  Grid::tileCount(tiles, rank % columns, columns)};
        MPI_Comm_split(active, grid.row, grid.column, &grid.rowComm);
        MPI_Comm_split(active, grid.column, grid.row, &grid.columnComm);

        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %d\n", n);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
        }
        // Include the padded vertices in the bit-width check.
        if (size_t(tiles) * Tile < (size_t{1} << PathBits<uint32_t>)) {
            result = run<uint32_t>(n, tiles, validate, printResults, grid);
        } else {
            result = run<uint64_t>(n, tiles, validate, printResults, grid);
        }
        MPI_Comm_free(&grid.columnComm);
        MPI_Comm_free(&grid.rowComm);
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
