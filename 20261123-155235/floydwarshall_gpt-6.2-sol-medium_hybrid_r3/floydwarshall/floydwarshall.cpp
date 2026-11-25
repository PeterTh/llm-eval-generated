#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <vector>

#include "../common/results_output.hpp"

namespace {

constexpr int TILE = 32;
constexpr int TILE_AREA = TILE * TILE;
constexpr int THREAD_ROWS = 8;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr unsigned int PADDING_DISTANCE = 255;

// Low bits contain one plus the largest internal vertex, or zero for a direct edge.
// Among shortest paths, the serial loop first reaches the final distance at the
// smallest possible largest internal vertex. Lexicographic min with (sum, max)
// preserves this choice when pivots are blocked. A zero witness denotes the
// original path entry, which is the source vertex.
template <typename Word>
constexpr int WITNESS_BITS = sizeof(Word) == 4 ? 23 : 32;

template <typename Word>
constexpr Word WITNESS_MASK = (Word{1} << WITNESS_BITS<Word>) - 1;

template <typename Word>
__host__ __device__ Word encode(unsigned int distance) {
    return static_cast<Word>(distance) << WITNESS_BITS<Word>;
}

template <typename Word>
__device__ __forceinline__ Word relax(Word current, Word left, Word right, Word pivot) {
    const Word leftWitness = left & WITNESS_MASK<Word>;
    const Word rightWitness = right & WITNESS_MASK<Word>;
    const Word endWitness = leftWitness > rightWitness ? leftWitness : rightWitness;
    const Word witness = endWitness > pivot ? endWitness : pivot;
    const Word distance = (left & ~WITNESS_MASK<Word>) + (right & ~WITNESS_MASK<Word>);
    const Word candidate = distance | witness;
    return candidate < current ? candidate : current;
}

void checkCuda(cudaError_t result) {
    if (result != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d CUDA error: %s\n", rank, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template <typename Word>
struct DeviceBuffer {
    Word* data = nullptr;

    explicit DeviceBuffer(size_t count) {
        if (count > SIZE_MAX / sizeof(Word)) {
            throw std::length_error("CUDA allocation size exceeds addressable memory");
        }
        if (count != 0) {
            checkCuda(cudaMalloc(&data, count * sizeof(Word)));
        }
    }

    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename Operation>
void mpiChunks(size_t count, Operation operation) {
    for (size_t offset = 0; offset < count;) {
        const int chunk = static_cast<int>(std::min(count - offset, size_t{INT_MAX}));
        operation(offset, chunk);
        offset += chunk;
    }
}

template <typename Word>
struct BroadcastBuffer {
    DeviceBuffer<Word> device;
    Word* host = nullptr;

    BroadcastBuffer(size_t count, bool stageOnHost) : device(count) {
        if (stageOnHost) {
            checkCuda(cudaMallocHost(&host, count * sizeof(Word)));
        }
    }

    ~BroadcastBuffer() { if (host) cudaFreeHost(host); }

    void stage(size_t count, cudaStream_t stream) const {
        if (host) {
            checkCuda(cudaMemcpyAsync(host, device.data, count * sizeof(Word),
                                      cudaMemcpyDeviceToHost, stream));
        }
    }

    void broadcast(size_t count, int root, MPI_Comm comm, MPI_Request& request) const {
        if (host && count <= INT_MAX) {
            MPI_Ibcast(host, static_cast<int>(count), mpiType(), root, comm, &request);
        } else {
            Word* buffer = host ? host : device.data;
            mpiChunks(count, [&](size_t offset, int chunk) {
                MPI_Bcast(buffer + offset, chunk, mpiType(), root, comm);
            });
        }
    }

    void upload(size_t count, cudaStream_t stream) const {
        if (host) {
            checkCuda(cudaMemcpyAsync(device.data, host, count * sizeof(Word),
                                      cudaMemcpyHostToDevice, stream));
        }
    }

    static MPI_Datatype mpiType() {
        return sizeof(Word) == 4 ? MPI_UINT32_T : MPI_UINT64_T;
    }
};

__host__ __device__ size_t tileOffset(int row, int col, int localCols) {
    return (static_cast<size_t>(row) * localCols + col) * TILE_AREA;
}

template <typename Word>
__global__ void closePivot(const Word* input, Word* output, int pivot) {
    __shared__ Word tile[TILE][TILE + 1];
    const int x = threadIdx.x;
    const int y = threadIdx.y;

    #pragma unroll
    for (int row = y; row < TILE; row += THREAD_ROWS) {
        tile[row][x] = input[row * TILE + x];
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        // The zero diagonal leaves the pivot row and column invariant at step k.
        // Keep those locations read-only so every thread can relax in place.
        if (x != k) {
            #pragma unroll
            for (int row = y; row < TILE; row += THREAD_ROWS) {
                if (row != k) {
                    tile[row][x] = relax(tile[row][x], tile[row][k], tile[k][x],
                                         static_cast<Word>(pivot) + k + 1);
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int row = y; row < TILE; row += THREAD_ROWS) {
        output[row * TILE + x] = tile[row][x];
    }
}

template <typename Word, bool WRITE_PANEL>
__device__ __forceinline__ void updateTile(Word* output, const Word* left, const Word* right,
                                           Word* panel, int pivot) {
    __shared__ Word a[TILE][TILE + 1];
    __shared__ Word b[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    Word best[TILE / THREAD_ROWS];

    #pragma unroll
    for (int i = 0; i < TILE / THREAD_ROWS; ++i) {
        const int row = y + i * THREAD_ROWS;
        const int index = row * TILE + x;
        a[row][x] = left[index];
        b[row][x] = right[index];
        best[i] = output[index];
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const Word rightValue = b[k][x];
        #pragma unroll
        for (int i = 0; i < TILE / THREAD_ROWS; ++i) {
            best[i] = relax(best[i], a[y + i * THREAD_ROWS][k], rightValue,
                            static_cast<Word>(pivot) + k + 1);
        }
    }

    #pragma unroll
    for (int i = 0; i < TILE / THREAD_ROWS; ++i) {
        const int index = (y + i * THREAD_ROWS) * TILE + x;
        output[index] = best[i];
        if constexpr (WRITE_PANEL) {
            panel[index] = best[i];
        }
    }
}

template <typename Word>
__global__ void updatePanels(Word* matrix, const Word* diagonal, Word* rowPanel, Word* colPanel,
                             int localCols, int pivotRow, int pivotCol, int rowCount, int pivot) {
    const int tile = blockIdx.x;
    if (tile < rowCount) {
        Word* output = matrix + tileOffset(pivotRow, tile, localCols);
        Word* panel = rowPanel + static_cast<size_t>(tile) * TILE_AREA;
        if (tile == pivotCol) {
            for (int row = threadIdx.y; row < TILE; row += THREAD_ROWS) {
                const int index = row * TILE + threadIdx.x;
                output[index] = diagonal[index];
                panel[index] = diagonal[index];
                colPanel[static_cast<size_t>(pivotRow) * TILE_AREA + index] = diagonal[index];
            }
        } else {
            updateTile<Word, true>(output, diagonal, output, panel, pivot);
        }
    } else {
        const int row = tile - rowCount;
        if (row != pivotRow) {
            Word* output = matrix + tileOffset(row, pivotCol, localCols);
            updateTile<Word, true>(output, output, diagonal,
                                   colPanel + static_cast<size_t>(row) * TILE_AREA, pivot);
        }
    }
}

template <typename Word>
__global__ void updateRemaining(Word* matrix, const Word* rowPanel, const Word* colPanel,
                                int localCols, int colsToUpdate, int skipRow, int skipCol,
                                int pivot) {
    int row = blockIdx.x / colsToUpdate;
    int col = blockIdx.x % colsToUpdate;
    row += row >= skipRow;
    col += col >= skipCol;
    updateTile<Word, false>(matrix + tileOffset(row, col, localCols),
                            colPanel + static_cast<size_t>(row) * TILE_AREA,
                            rowPanel + static_cast<size_t>(col) * TILE_AREA, nullptr, pivot);
}

struct Grid {
    int rows = 1;
    int cols = 1;
    int row;
    int col;
    int localRows;
    int localCols;
    int tiles;
    MPI_Comm active = MPI_COMM_NULL;
    MPI_Comm rowComm = MPI_COMM_NULL;
    MPI_Comm colComm = MPI_COMM_NULL;

    Grid(int rank, int size, int tileCount) : tiles(tileCount) {
        const int limit = std::max(1, tiles);
        // Use as many ranks as there are useful tiles, favoring a square grid.
        for (int r = 1; r <= std::min(limit, size); ++r) {
            const int c = std::min(limit, size / r);
            if (r * c > rows * cols ||
                (r * c == rows * cols && std::abs(r - c) < std::abs(rows - cols))) {
                rows = r;
                cols = c;
            }
        }

        MPI_Comm_split(MPI_COMM_WORLD, rank < rows * cols ? 0 : MPI_UNDEFINED, rank, &active);
        if (active != MPI_COMM_NULL) {
            row = rank / cols;
            col = rank % cols;
            localRows = count(row, rows);
            localCols = count(col, cols);
            MPI_Comm_split(active, row, col, &rowComm);
            MPI_Comm_split(active, col, row, &colComm);
        }
    }

    ~Grid() {
        if (active != MPI_COMM_NULL) {
            MPI_Comm_free(&rowComm);
            MPI_Comm_free(&colComm);
            MPI_Comm_free(&active);
        }
    }

    int count(int coordinate, int dimension) const {
        return tiles / dimension + (coordinate < tiles % dimension);
    }
};

int hostThreads(size_t tiles) {
    // Keep enough work per thread to amortize creating the OpenMP team.
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + tiles / 32));
}

#if defined(__GLIBC__)
struct SeedStep {
    unsigned int multiplier;
    unsigned int increment;

    unsigned int apply(unsigned int seed) const {
        return multiplier * seed + increment;
    }
};

SeedStep compose(SeedStep outer, SeedStep inner) {
    return {outer.multiplier * inner.multiplier, outer.apply(inner.increment)};
}

SeedStep skipRandom(size_t calls) {
    // glibc rand_r advances this 32-bit LCG three times per call.
    constexpr SeedStep lcg{1103515245u, 12345u};
    SeedStep power = compose(lcg, compose(lcg, lcg));
    SeedStep result{1, 0};
    for (; calls; calls >>= 1, power = compose(power, power)) {
        if (calls & 1) {
            result = compose(power, result);
        }
    }
    return result;
}
#endif

template <typename Word>
void initializeGraph(Word* matrix, int n, const Grid& grid) {
#if defined(__GLIBC__)
    const SeedStep nextRow = skipRandom(n);

    const int threads = hostThreads(static_cast<size_t>(grid.localRows) * grid.localCols);
    #pragma omp parallel for collapse(2) schedule(static) if(threads > 1) num_threads(threads)
    for (int row = 0; row < grid.localRows; ++row) {
        for (int col = 0; col < grid.localCols; ++col) {
            const int firstRow = (row * grid.rows + grid.row) * TILE;
            const int firstCol = (col * grid.cols + grid.col) * TILE;
            Word* tile = matrix + tileOffset(row, col, grid.localCols);
            unsigned int rowSeed = skipRandom(static_cast<size_t>(firstRow) * n + firstCol).apply(42);

            for (int y = 0; y < TILE; ++y, rowSeed = nextRow.apply(rowSeed)) {
                unsigned int seed = rowSeed;
                for (int x = 0; x < TILE; ++x) {
                    const int i = firstRow + y;
                    const int j = firstCol + x;
                    unsigned int distance = PADDING_DISTANCE;
                    if (i < n && j < n) {
                        distance = 1 + static_cast<unsigned int>(
                            static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                    }
                    tile[y * TILE + x] = encode<Word>(i == j ? 0 : distance);
                }
            }
        }
    }
#else
    // rand_r is implementation-defined. Capture row seeds in original sequence
    // on other libcs, then generate the owned tile rows in parallel.
    std::vector<unsigned int> seeds(n);
    unsigned int seed = 42;
    for (int i = 0; i < n; ++i) {
        seeds[i] = seed;
        for (int j = 0; j < n; ++j) {
            rand_r(&seed);
        }
    }
    const int threads = hostThreads(static_cast<size_t>(grid.localRows) * grid.localCols);
    #pragma omp parallel for schedule(static) if(threads > 1) num_threads(threads)
    for (int row = 0; row < grid.localRows; ++row) {
        for (int y = 0; y < TILE; ++y) {
            const int i = (row * grid.rows + grid.row) * TILE + y;
            unsigned int rowSeed = i < n ? seeds[i] : 0;
            int nextCol = 0;
            for (int col = 0; col < grid.localCols; ++col) {
                const int firstCol = (col * grid.cols + grid.col) * TILE;
                Word* tile = matrix + tileOffset(row, col, grid.localCols);
                for (; i < n && nextCol < firstCol; ++nextCol) {
                    rand_r(&rowSeed);
                }
                for (int x = 0; x < TILE; ++x) {
                    const int j = firstCol + x;
                    unsigned int distance = PADDING_DISTANCE;
                    if (i < n && j < n) {
                        distance = 1 + static_cast<unsigned int>(
                            static_cast<double>(MAX_DISTANCE) * rand_r(&rowSeed) / RAND_MAX);
                        ++nextCol;
                    }
                    tile[y * TILE + x] = encode<Word>(i == j ? 0 : distance);
                }
            }
        }
    }
#endif
}

template <typename Word>
double floydWarshall(Word* matrix, const Grid& grid, cudaStream_t stream, bool cudaAware) {
    const size_t rowWords = static_cast<size_t>(grid.localCols) * TILE_AREA;
    const size_t colWords = static_cast<size_t>(grid.localRows) * TILE_AREA;
    const bool distributed = grid.rows > 1 || grid.cols > 1;
    BroadcastBuffer<Word> diagonal(TILE_AREA, !cudaAware && distributed);
    BroadcastBuffer<Word> rowPanel(rowWords, !cudaAware && grid.rows > 1);
    BroadcastBuffer<Word> colPanel(colWords, !cudaAware && grid.cols > 1);
    const dim3 threads(TILE, THREAD_ROWS);
    cudaEvent_t diagonalUploaded = nullptr;
    cudaEvent_t panelsUploaded = nullptr;
    if (distributed && !cudaAware) {
        checkCuda(cudaEventCreateWithFlags(&diagonalUploaded, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&panelsUploaded, cudaEventDisableTiming));
    }

    MPI_Barrier(grid.active);
    const double start = MPI_Wtime();
    for (int k = 0; k < grid.tiles; ++k) {
        const int ownerRow = k % grid.rows;
        const int ownerCol = k % grid.cols;
        const bool ownsRow = grid.row == ownerRow;
        const bool ownsCol = grid.col == ownerCol;
        const int pivotRow = ownsRow ? k / grid.rows : grid.localRows;
        const int pivotCol = ownsCol ? k / grid.cols : grid.localCols;
        const int pivot = k * TILE;

        // Close the pivot tile and share it with its process row and column.
        MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (ownsRow && ownsCol) {
            closePivot<<<1, threads, 0, stream>>>(
                matrix + tileOffset(pivotRow, pivotCol, grid.localCols), diagonal.device.data, pivot);
            checkCuda(cudaGetLastError());
            if (distributed) {
                diagonal.stage(TILE_AREA, stream);
                checkCuda(cudaStreamSynchronize(stream));
            }
        }
        if (ownsRow != ownsCol) {
            if (cudaAware) {
                checkCuda(cudaStreamSynchronize(stream));
            } else {
                checkCuda(cudaEventSynchronize(diagonalUploaded));
            }
        }
        if (ownsRow && grid.cols > 1) {
            diagonal.broadcast(TILE_AREA, ownerCol, grid.rowComm, requests[0]);
        }
        if (ownsCol && grid.rows > 1) {
            diagonal.broadcast(TILE_AREA, ownerRow, grid.colComm, requests[1]);
        }
        if (distributed && (ownsRow || ownsCol)) {
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }

        if (ownsRow != ownsCol) {
            diagonal.upload(TILE_AREA, stream);
            if (diagonalUploaded) {
                checkCuda(cudaEventRecord(diagonalUploaded, stream));
            }
        }

        // Update the pivot panels and broadcast each in the orthogonal direction.
        const int rowCount = ownsRow ? grid.localCols : 0;
        const int colCount = ownsCol ? grid.localRows : 0;
        if (rowCount + colCount != 0) {
            updatePanels<<<rowCount + colCount, threads, 0, stream>>>(
                matrix, diagonal.device.data, rowPanel.device.data, colPanel.device.data,
                grid.localCols, pivotRow, pivotCol, rowCount, pivot);
            checkCuda(cudaGetLastError());
            if (grid.rows > 1 && ownsRow) {
                rowPanel.stage(rowWords, stream);
            }
            if (grid.cols > 1 && ownsCol) {
                colPanel.stage(colWords, stream);
            }
            if ((grid.rows > 1 && ownsRow) || (grid.cols > 1 && ownsCol)) {
                checkCuda(cudaStreamSynchronize(stream));
            }
        }

        const bool receivesPanels = !ownsCol || !ownsRow;
        if (receivesPanels) {
            // Host staging can be reused as soon as the upload completes. A
            // CUDA-aware receive must also wait for the last kernel consuming it.
            if (cudaAware) {
                checkCuda(cudaStreamSynchronize(stream));
            } else {
                checkCuda(cudaEventSynchronize(panelsUploaded));
            }
        }
        if (grid.cols > 1) {
            colPanel.broadcast(colWords, ownerCol, grid.rowComm, requests[0]);
        }
        if (grid.rows > 1) {
            rowPanel.broadcast(rowWords, ownerRow, grid.colComm, requests[1]);
        }
        if (distributed) {
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }

        if (!ownsRow) {
            rowPanel.upload(rowWords, stream);
        }
        if (!ownsCol) {
            colPanel.upload(colWords, stream);
        }
        if (receivesPanels && panelsUploaded) {
            checkCuda(cudaEventRecord(panelsUploaded, stream));
        }

        // Each rank now has both operands for all its remaining tiles.
        const int rowsToUpdate = grid.localRows - ownsRow;
        const int colsToUpdate = grid.localCols - ownsCol;
        const size_t blocks = static_cast<size_t>(rowsToUpdate) * colsToUpdate;
        if (blocks != 0) {
            updateRemaining<<<blocks, threads, 0, stream>>>(
                matrix, rowPanel.device.data, colPanel.device.data, grid.localCols,
                colsToUpdate, pivotRow, pivotCol, pivot);
            checkCuda(cudaGetLastError());
        }
    }
    checkCuda(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    if (panelsUploaded) {
        checkCuda(cudaEventDestroy(diagonalUploaded));
        checkCuda(cudaEventDestroy(panelsUploaded));
    }
    double duration = 0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, grid.active);
    return duration;
}

template <typename Word>
std::vector<unsigned int> gatherDistances(const Word* matrix, int n, const Grid& grid, int rank) {
    const size_t localWords = tileOffset(grid.localRows, 0, grid.localCols);
    std::vector<Word> packed(localWords);
    if (localWords != 0) {
        checkCuda(cudaMemcpy(packed.data(), matrix, localWords * sizeof(Word), cudaMemcpyDeviceToHost));
    }

    if (rank != 0) {
        mpiChunks(localWords, [&](size_t offset, int count) {
            MPI_Send(packed.data() + offset, count, BroadcastBuffer<Word>::mpiType(),
                     0, 0, grid.active);
        });
        return {};
    }

    std::vector<unsigned int> distances(static_cast<size_t>(n) * n);
    for (int source = 0; source < grid.rows * grid.cols; ++source) {
        const int processRow = source / grid.cols;
        const int processCol = source % grid.cols;
        const int rows = grid.count(processRow, grid.rows);
        const int cols = grid.count(processCol, grid.cols);
        if (source != 0) {
            // Coordinate zero owns at least as many tile rows and columns as any peer.
            mpiChunks(tileOffset(rows, 0, cols), [&](size_t offset, int count) {
                MPI_Recv(packed.data() + offset, count, BroadcastBuffer<Word>::mpiType(),
                         source, 0, grid.active, MPI_STATUS_IGNORE);
            });
        }

        const int threads = hostThreads(static_cast<size_t>(rows) * cols);
        #pragma omp parallel for collapse(2) schedule(static) if(threads > 1) num_threads(threads)
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const int firstRow = (row * grid.rows + processRow) * TILE;
                const int firstCol = (col * grid.cols + processCol) * TILE;
                const Word* tile = packed.data() + tileOffset(row, col, cols);
                for (int y = 0; y < std::min(TILE, n - firstRow); ++y) {
                    const size_t offset = static_cast<size_t>(firstRow + y) * n + firstCol;
                    for (int x = 0; x < std::min(TILE, n - firstCol); ++x) {
                        distances[offset + x] = tile[y * TILE + x] >> WITNESS_BITS<Word>;
                    }
                }
            }
        }
    }
    return distances;
}

template <typename Word>
__global__ void extractValidationEntries(const Word* matrix, unsigned int* entries, size_t count,
                                         int n, int sample, int rows, int cols, int row, int col,
                                         int localCols) {
    for (size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < count; index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const int strip = index / n;
        const int vertex = index % n;
        const int i = strip < sample ? strip : vertex;
        const int j = strip < sample ? vertex : (strip < 2 * sample ? strip - sample : vertex);
        const int tileRow = i / TILE;
        const int tileCol = j / TILE;
        unsigned int value = 0;
        if (tileRow % rows == row && tileCol % cols == col) {
            const size_t offset = tileOffset(tileRow / rows, tileCol / cols, localCols);
            value = matrix[offset + (i % TILE) * TILE + j % TILE] >> WITNESS_BITS<Word>;
        }
        entries[index] = value;
    }
}

template <typename Word>
std::vector<unsigned int> gatherValidationEntries(const Word* matrix, int n, int sample,
                                                   const Grid& grid, int rank) {
    const size_t count = static_cast<size_t>(2 * sample + 1) * n;
    DeviceBuffer<unsigned int> device(count);
    std::vector<unsigned int> local(count);
    if (count != 0) {
        const unsigned int blocks = std::min((count + 255) / 256, size_t{65535});
        extractValidationEntries<<<blocks, 256>>>(matrix, device.data, count, n, sample,
            grid.rows, grid.cols, grid.row, grid.col, grid.localCols);
        checkCuda(cudaGetLastError());
        checkCuda(cudaMemcpy(local.data(), device.data, count * sizeof(unsigned int),
                             cudaMemcpyDeviceToHost));
        // Each entry has one owner. Combining the strips avoids gathering the full matrix.
        mpiChunks(count, [&](size_t offset, int size) {
            unsigned int* entries = local.data() + offset;
            MPI_Reduce(rank == 0 ? MPI_IN_PLACE : entries, rank == 0 ? entries : nullptr,
                       size, MPI_UNSIGNED, MPI_SUM, 0, grid.active);
        });
    }
    return local;
}

template <typename Distance>
bool validateResult(int n, int sample, Distance distance) {
    for (int i = 0; i < n; ++i) {
        if (distance(i, i) != 0) {
            printf("Validation failed: diagonal element [%d,%d] is not zero\n", i, i);
            return false;
        }
    }

    for (int i = 0; i < sample; ++i) {
        for (int j = 0; j < sample; ++j) {
            const unsigned int direct = distance(i, j);
            for (int k = 0; k < n; ++k) {
                if (distance(i, k) + distance(k, j) < direct) {
                    printf("Validation failed: triangle inequality violated at [%d,%d,%d]\n", i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void printUsage(const char* program) {
    printf("Usage: %s [options]\n", program);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

template <typename Word>
int run(int n, bool validate, bool printResults, int rank, const Grid& grid) {
    MPI_Comm nodeComm;
    MPI_Comm_split_type(grid.active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int gpuCount = 0;
    checkCuda(cudaGetDeviceCount(&gpuCount));
    if (gpuCount == 0) {
        checkCuda(cudaErrorNoDevice);
    }
    checkCuda(cudaSetDevice(localRank % gpuCount));
    int cudaAware = 0;
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
    cudaAware = MPIX_Query_cuda_support();
#endif
    MPI_Allreduce(MPI_IN_PLACE, &cudaAware, 1, MPI_INT, MPI_MIN, grid.active);

    const size_t localWords = tileOffset(grid.localRows, 0, grid.localCols);
    DeviceBuffer<Word> matrix(localWords);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %d\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    if (localWords != 0) {
        std::unique_ptr<Word[]> initial(new Word[localWords]);
        initializeGraph(initial.get(), n, grid);
        checkCuda(cudaMemcpy(matrix.data, initial.get(), localWords * sizeof(Word),
                             cudaMemcpyHostToDevice));
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    double duration = 0;
    if (n != 0) {
        cudaStream_t stream;
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        duration = floydWarshall(matrix.data, grid, stream, cudaAware);
        checkCuda(cudaStreamDestroy(stream));
    }
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(duration * 1000));
        const double operations = static_cast<double>(n) * n * n;
        printf("Performance: %.3f GOPS\n", duration > 0 ? operations / duration / 1e9 : 0);
    }

    std::vector<unsigned int> distances;
    if (printResults) {
        distances = gatherDistances(matrix.data, n, grid, rank);
        if (rank == 0) {
            print_results_int(distances, "DistanceMatrix");
        }
    }

    int status = 0;
    if (validate) {
        const int sample = std::min(n, 10);
        if (rank == 0) {
            printf("Validating result...\n");
        }
        bool valid = true;
        if (printResults) {
            if (rank == 0) {
                valid = validateResult(n, sample, [&](int i, int j) {
                    return distances[static_cast<size_t>(i) * n + j];
                });
            }
        } else {
            auto entries = gatherValidationEntries(matrix.data, n, sample, grid, rank);
            if (rank == 0) {
                valid = validateResult(n, sample, [&](int i, int j) {
                    if (i == j) {
                        return entries[static_cast<size_t>(2 * sample) * n + i];
                    }
                    if (i < sample) {
                        return entries[static_cast<size_t>(i) * n + j];
                    }
                    return entries[static_cast<size_t>(sample + j) * n + i];
                });
            }
        }
        if (rank == 0) {
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    return status;
}

int benchmark(int argc, char** argv, int rank, int size) {
    int n = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* number = argv[++i];
            char* end = nullptr;
            errno = 0;
            const long parsed = std::strtol(number, &end, 10);
            if (errno == ERANGE || end == number || *end != '\0' ||
                parsed < 0 || parsed > INT_MAX - TILE) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid number of nodes: %s\n", number);
                }
                return 1;
            }
            n = static_cast<int>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (std::strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (std::strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    const Grid grid(rank, size, (n + TILE - 1) / TILE);
    int status = 0;
    if (grid.active != MPI_COMM_NULL) {
        if (static_cast<uint32_t>(grid.tiles) * TILE <= WITNESS_MASK<uint32_t>) {
            status = run<uint32_t>(n, validate, printResults, rank, grid);
        } else {
            status = run<uint64_t>(n, validate, printResults, rank, grid);
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}
} // namespace

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return 1;
    }
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int status = 1;
    try {
        status = benchmark(argc, argv, rank, size);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
