#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mpi.h>
#if __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include "../common/results_output.hpp"

namespace {
int worldRank = 0;

[[noreturn]] void fail(const char* operation, int code) {
    std::fprintf(stderr, "Rank %d: %s failed (error %d)\n", worldRank, operation, code);
    MPI_Abort(MPI_COMM_WORLD, code ? code : 1);
    std::abort();
}

#define MPI_CHECK(call) do { const int checkStatus = (call); \
    if (checkStatus != MPI_SUCCESS) fail(#call, checkStatus); } while (0)
#define CUDA_CHECK(call) do { const cudaError_t checkStatus = (call); \
    if (checkStatus != cudaSuccess) { \
        std::fprintf(stderr, "%s\n", cudaGetErrorString(checkStatus)); \
        fail(#call, static_cast<int>(checkStatus)); } } while (0)
#define BLAS_CHECK(call) do { const cublasStatus_t checkStatus = (call); \
    if (checkStatus != CUBLAS_STATUS_SUCCESS) fail(#call, static_cast<int>(checkStatus)); } while (0)
#define SOLVER_CHECK(call) do { const cusolverStatus_t checkStatus = (call); \
    if (checkStatus != CUSOLVER_STATUS_SUCCESS) fail(#call, static_cast<int>(checkStatus)); } while (0)

int mpiCount(size_t count) {
    if (count > static_cast<size_t>(INT_MAX))
        throw std::runtime_error("MPI panel exceeds the supported count range");
    return static_cast<int>(count);
}

template<class T> class DeviceBuffer {
    T* pointer = nullptr;
public:
    explicit DeviceBuffer(size_t count) {
        if (count > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("Device allocation size overflow");
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), std::max(size_t(1), count) * sizeof(T)));
    }
    ~DeviceBuffer() { cudaFree(pointer); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* data() { return pointer; }
    const T* data() const { return pointer; }
};

class PinnedBuffer {
    double* pointer = nullptr;
public:
    explicit PinnedBuffer(size_t count) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pointer), std::max(size_t(1), count) * sizeof(double)));
        std::fill_n(pointer, count, 0.0);
    }
    ~PinnedBuffer() { cudaFreeHost(pointer); }
    double* data() { return pointer; }
};

struct Tile { int row, column; };

struct Grid {
    int size, rows, columns, row, column;
    MPI_Comm rowComm = MPI_COMM_NULL, columnComm = MPI_COMM_NULL;
    Grid() {
        MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
        int dimensions[2] = {0, 0};
        MPI_CHECK(MPI_Dims_create(size, 2, dimensions));
        rows = dimensions[0]; columns = dimensions[1];
        row = worldRank / columns; column = worldRank % columns;
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, row, column, &rowComm));
        MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, column, row, &columnComm));
    }
    ~Grid() {
        MPI_Comm_free(&rowComm);
        MPI_Comm_free(&columnComm);
    }
    int owner(int i, int j) const { return (i % rows) * columns + j % columns; }
    std::vector<Tile> tiles(int blocks, int rank) const {
        std::vector<Tile> result;
        for (int i = rank / columns; i < blocks; i += rows)
            for (int j = rank % columns; j <= i; j += columns)
                result.push_back({i, j});
        return result;
    }
};

struct Accelerator {
    cudaStream_t stream, bulkStream;
    cudaEvent_t bulkDone, panelsReady;
    cublasHandle_t blas, bulkBlas;
    cusolverDnHandle_t solver;
    bool deviceMPI = false;
    Accelerator() {
        int capability = 0, allCapable = 0;
        #if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        capability = MPIX_Query_cuda_support() != 0;
        #endif
        MPI_CHECK(MPI_Allreduce(&capability, &allCapable, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
        deviceMPI = allCapable != 0;
        int lowPriority, highPriority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, highPriority));
        CUDA_CHECK(cudaStreamCreateWithPriority(&bulkStream, cudaStreamNonBlocking, lowPriority));
        CUDA_CHECK(cudaEventCreateWithFlags(&bulkDone, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&panelsReady, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(bulkDone, bulkStream));
        BLAS_CHECK(cublasCreate(&blas));
        BLAS_CHECK(cublasCreate(&bulkBlas));
        SOLVER_CHECK(cusolverDnCreate(&solver));
        BLAS_CHECK(cublasSetStream(blas, stream));
        BLAS_CHECK(cublasSetStream(bulkBlas, bulkStream));
        SOLVER_CHECK(cusolverDnSetStream(solver, stream));
        // All arithmetic is FP64, including generation and validation.
        BLAS_CHECK(cublasSetMathMode(blas, CUBLAS_DEFAULT_MATH));
        BLAS_CHECK(cublasSetMathMode(bulkBlas, CUBLAS_DEFAULT_MATH));
    }
    ~Accelerator() {
        cusolverDnDestroy(solver);
        cublasDestroy(blas);
        cublasDestroy(bulkBlas);
        cudaEventDestroy(bulkDone);
        cudaEventDestroy(panelsReady);
        cudaStreamDestroy(stream);
        cudaStreamDestroy(bulkStream);
    }
    void synchronize() { CUDA_CHECK(cudaStreamSynchronize(stream)); }
    void synchronizeAll() {
        synchronize();
        CUDA_CHECK(cudaStreamSynchronize(bulkStream));
    }
};

int solverWorkspace(Accelerator& gpu, int dimension, int leading, double* diagonal) {
    int count = 0;
    SOLVER_CHECK(cusolverDnDpotrf_bufferSize(gpu.solver, CUBLAS_FILL_MODE_LOWER, dimension, diagonal, leading, &count));
    return count;
}

// glibc rand_r advances its 32-bit LCG three times per output. Jumping to
// each row's initial state preserves the original seed-42 stream exactly,
// independent of MPI ownership and the OpenMP schedule.
unsigned int rowSeed(uint64_t outputs) {
    uint32_t multiplier = 1103515245u, increment = 12345u;
    uint32_t resultMultiplier = 1, resultIncrement = 0;
    for (uint64_t steps = outputs * 3; steps; steps >>= 1) {
        if (steps & 1) {
            resultIncrement = multiplier * resultIncrement + increment;
            resultMultiplier = multiplier * resultMultiplier;
        }
        increment = (multiplier + 1u) * increment;
        multiplier *= multiplier;
    }
    return resultMultiplier * 42u + resultIncrement;
}

bool compatibleRandomGenerator() {
    unsigned int seed = 42;
    for (uint64_t i = 1; i <= 16; ++i) {
        rand_r(&seed);
        if (seed != rowSeed(i)) return false;
    }
    return sizeof(unsigned int) == 4;
}

std::vector<unsigned int> prepareRowSeeds(int n) {
    int localCompatible = compatibleRandomGenerator() ? 1 : 0, allCompatible = 0;
    MPI_CHECK(MPI_Allreduce(&localCompatible, &allCompatible, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (allCompatible) return {};
    // Preserve rand_r semantics on other C libraries too. Only this less
    // common initialization path needs a serial scan of the RNG states.
    std::vector<unsigned int> seeds(n);
    if (worldRank == 0) {
        unsigned int seed = 42;
        for (int i = 0; i < n; ++i) {
            seeds[i] = seed;
            for (int j = 0; j < n; ++j) rand_r(&seed);
        }
    }
    MPI_CHECK(MPI_Bcast(seeds.data(), n, MPI_UNSIGNED, 0, MPI_COMM_WORLD));
    return seeds;
}

__global__ void addDiagonal(double* matrix, const int* rows, const int* columns,
                            size_t tileElements, int blockSize, int tileCount, int n) {
    const int tile = blockIdx.x;
    if (tile < tileCount && rows[tile] == columns[tile]) {
        for (int i = threadIdx.x; i < blockSize; i += blockDim.x)
            matrix[static_cast<size_t>(tile) * tileElements + i * (blockSize + 1)] +=
                rows[tile] * blockSize + i < n ? static_cast<double>(n) : 1.0;
    }
}

__global__ void packTiles(const double* const* sources, double* destination,
                          size_t tileElements, int count) {
    const int tile = blockIdx.x;
    if (tile < count)
        for (size_t i = threadIdx.x; i < tileElements; i += blockDim.x)
            destination[static_cast<size_t>(tile) * tileElements + i] = sources[tile][i];
}

__global__ void identityDiagonal(double* matrix, int b) {
    for (int i = threadIdx.x; i < b; i += blockDim.x) matrix[i * (b + 1)] = 1.0;
}

__global__ void clearDiagonalUpper(double* matrix, const int* rows, const int* columns,
                                   size_t tileElements, int blockSize, int tileCount) {
    const int tile = blockIdx.x;
    if (tile < tileCount && rows[tile] == columns[tile])
        for (size_t i = threadIdx.x; i < tileElements; i += blockDim.x)
            if (i % blockSize < i / blockSize)
                matrix[static_cast<size_t>(tile) * tileElements + i] = 0.0;
}

__global__ void tileErrors(const double* reconstructed, const double* original,
                           const int* rows, const int* columns, size_t tileElements,
                           int blockSize, int n, double* errors) {
    __shared__ double absolute[256], relative[256];
    const int tile = blockIdx.x;
    double maxAbsolute = 0.0, maxRelative = 0.0;
    for (size_t i = threadIdx.x; i < tileElements; i += blockDim.x) {
        const int row = rows[tile] * blockSize + i % blockSize;
        const int column = columns[tile] * blockSize + i / blockSize;
        if (row < n && column < n && column <= row) {
            const size_t offset = static_cast<size_t>(tile) * tileElements + i;
            const double expected = original[offset], actual = reconstructed[offset];
            const bool finite = isfinite(actual) && isfinite(expected);
            const double error = finite ? fabs(actual - expected) : CUDART_INF;
            maxAbsolute = fmax(maxAbsolute, error);
            maxRelative = fmax(maxRelative, finite ? error / (fabs(expected) + 1e-10) : CUDART_INF);
        }
    }
    absolute[threadIdx.x] = maxAbsolute;
    relative[threadIdx.x] = maxRelative;
    __syncthreads();
    for (int stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            absolute[threadIdx.x] = fmax(absolute[threadIdx.x], absolute[threadIdx.x + stride]);
            relative[threadIdx.x] = fmax(relative[threadIdx.x], relative[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        errors[2 * tile] = absolute[0];
        errors[2 * tile + 1] = relative[0];
    }
}

class DistributedMatrix {
    const Grid& grid;
    Accelerator& gpu;
    const std::vector<unsigned int>& seeds;
    int n, b, blocks;
    size_t elements;
    std::vector<Tile> tiles;
    std::vector<int> rowBlocks, rowSlot, columnSlot, tileMap;
    std::vector<int> counts, displacements;
    DeviceBuffer<double> matrix, rowPanel, columnPanel, sendPanel, diagonal;
    DeviceBuffer<int> tileRows, tileColumns, info;
    int workspaceSize;
    DeviceBuffer<double> workspace;
    DeviceBuffer<const double*> leftPointers, rightPointers;
    DeviceBuffer<double*> outputPointers;
    DeviceBuffer<const double*> bulkLeftPointers, bulkRightPointers;
    DeviceBuffer<double*> bulkOutputPointers;
    PinnedBuffer hostRow, hostColumn, hostSend, hostDiagonal;
    std::vector<const double*> left, right, bulkLeft, bulkRight;
    std::vector<double*> output, bulkOutput;
    std::vector<int> active, bulkActive;

    int localRows() const { return (blocks + grid.rows - 1 - grid.row) / grid.rows; }
    int localColumns() const { return (blocks + grid.columns - 1 - grid.column) / grid.columns; }
    size_t capacity() const { return std::max(tiles.size(), static_cast<size_t>(localRows())); }
    int width(int k) const { return std::min(b, n - k * b); }
    double* rowData(int k) { return rowPanel.data() + (k & 1) * static_cast<size_t>(localRows()) * elements; }
    double* columnData(int k) { return columnPanel.data() + (k & 1) * static_cast<size_t>(localColumns()) * elements; }
    double* tile(int i, int j) {
        const int index = tileMap[static_cast<size_t>(i) * blocks + j];
        if (index < 0) throw std::runtime_error("Invalid local tile ownership");
        return matrix.data() + static_cast<size_t>(index) * elements;
    }

    // Row broadcasts supply L_ik; column all-gathers supply only the L_jk
    // needed by this process column. CUDA-aware MPI moves device panels
    // directly; ordinary MPI uses pinned staging with the same distribution.
    void exchangePanel(int k, int firstBlock) {
        // Only the critical stream is synchronized. Alternating device panel
        // buffers let the previous bulk update overlap this communication.
        gpu.synchronize();
        const int firstRow = static_cast<int>(std::lower_bound(rowBlocks.begin(), rowBlocks.end(), firstBlock) - rowBlocks.begin());
        const int count = static_cast<int>(rowBlocks.size()) - firstRow;
        if (grid.column == k % grid.columns && count) {
            #pragma omp parallel for schedule(static) if(count > 8)
            for (int i = 0; i < count; ++i) left[i] = tile(rowBlocks[firstRow + i], k);
            CUDA_CHECK(cudaMemcpyAsync(leftPointers.data(), left.data(), count * sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
            packTiles<<<count, 256, 0, gpu.stream>>>(leftPointers.data(), rowData(k) + firstRow * elements, elements, count);
            CUDA_CHECK(cudaGetLastError());
            if (!gpu.deviceMPI)
                CUDA_CHECK(cudaMemcpyAsync(hostRow.data() + firstRow * elements, rowData(k) + firstRow * elements,
                    count * elements * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
            gpu.synchronize();
        }
        double* broadcast = gpu.deviceMPI ? rowData(k) : hostRow.data();
        MPI_CHECK(MPI_Bcast(broadcast + firstRow * elements, mpiCount(count * elements), MPI_DOUBLE,
                            k % grid.columns, grid.rowComm));

        int sendCount = 0;
        for (int r = 0; r < grid.rows; ++r) {
            int number = 0, first = 0;
            for (int j = grid.column; j < blocks; j += grid.columns)
                if (j % grid.rows == r && j >= firstBlock) {
                    if (number == 0) first = columnSlot[j];
                    ++number;
                }
            counts[r] = mpiCount(static_cast<size_t>(number) * elements);
            displacements[r] = mpiCount(static_cast<size_t>(first) * elements);
            if (r == grid.row) sendCount = number;
        }
        active.clear();
        for (int j : rowBlocks)
            if (j >= firstBlock && j % grid.columns == grid.column) active.push_back(j);
        if (gpu.deviceMPI) {
            #pragma omp parallel for schedule(static) if(sendCount > 8)
            for (int i = 0; i < sendCount; ++i) left[i] = rowData(k) + rowSlot[active[i]] * elements;
            if (sendCount) {
                CUDA_CHECK(cudaMemcpyAsync(leftPointers.data(), left.data(), sendCount * sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
                packTiles<<<sendCount, 256, 0, gpu.stream>>>(leftPointers.data(), sendPanel.data(), elements, sendCount);
                CUDA_CHECK(cudaGetLastError());
                gpu.synchronize();
            }
            MPI_CHECK(MPI_Allgatherv(sendPanel.data(), mpiCount(sendCount * elements), MPI_DOUBLE,
                columnData(k), counts.data(), displacements.data(), MPI_DOUBLE, grid.columnComm));
            return;
        }
        #pragma omp parallel for schedule(static) if(sendCount > 8)
        for (int i = 0; i < sendCount; ++i)
            std::memcpy(hostSend.data() + i * elements, hostRow.data() + rowSlot[active[i]] * elements, elements * sizeof(double));

        MPI_Request request;
        MPI_CHECK(MPI_Iallgatherv(hostSend.data(), mpiCount(sendCount * elements), MPI_DOUBLE,
            hostColumn.data(), counts.data(), displacements.data(), MPI_DOUBLE, grid.columnComm, &request));
        if (grid.column != k % grid.columns)
            CUDA_CHECK(cudaMemcpyAsync(rowData(k) + firstRow * elements, hostRow.data() + firstRow * elements,
                count * elements * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
        MPI_CHECK(MPI_Wait(&request, MPI_STATUS_IGNORE));
        // Each process-row group has a contiguous live suffix in columnPanel.
        for (int r = 0; r < grid.rows; ++r)
            if (counts[r])
                CUDA_CHECK(cudaMemcpyAsync(columnData(k) + displacements[r], hostColumn.data() + displacements[r],
                    static_cast<size_t>(counts[r]) * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
    }

    // In row-major tile order, partial last-row tiles form a suffix, with
    // the partial diagonal last. Three uniform batches avoid padded FLOPs.
    void gemmBatches(cublasHandle_t handle, int inner, int leading, bool generating,
                     const double* const* a, const double* const* bPointers,
                     double* const* c, int count, const std::vector<int>* indices,
                     double alpha, double beta) {
        auto tileAt = [&](int i) -> const Tile& { return tiles[indices ? (*indices)[i] : i]; };
        int full = count, partialDiagonal = 0;
        const int edge = blocks ? width(blocks - 1) : b;
        if (edge < b && count) {
            full = 0;
            while (full < count && tileAt(full).row != blocks - 1) ++full;
            partialDiagonal = tileAt(count - 1).column == blocks - 1 ? 1 : 0;
        }
        const cublasOperation_t opA = generating ? CUBLAS_OP_T : CUBLAS_OP_N;
        const cublasOperation_t opB = generating ? CUBLAS_OP_N : CUBLAS_OP_T;
        auto launch = [&](int rows, int columns, int offset, int number) {
            if (number)
                BLAS_CHECK(cublasDgemmBatched(handle, opA, opB, rows, columns, inner, &alpha,
                    a + offset, leading, bPointers + offset, leading, &beta, c + offset, b, number));
        };
        launch(b, b, 0, full);
        launch(edge, b, full, count - full - partialDiagonal);
        launch(edge, edge, count - partialDiagonal, partialDiagonal);
    }

    void launchBatch(const std::vector<int>& indices, int k, double* destination, double alpha, bool bulk) {
        const int count = mpiCount(indices.size());
        if (!count) return;
        auto& a = bulk ? bulkLeft : left;
        auto& bPointers = bulk ? bulkRight : right;
        auto& c = bulk ? bulkOutput : output;
        auto* deviceA = bulk ? bulkLeftPointers.data() : leftPointers.data();
        auto* deviceB = bulk ? bulkRightPointers.data() : rightPointers.data();
        auto* deviceC = bulk ? bulkOutputPointers.data() : outputPointers.data();
        const cudaStream_t stream = bulk ? gpu.bulkStream : gpu.stream;
        #pragma omp parallel for schedule(static) if(count > 8)
        for (int i = 0; i < count; ++i) {
            const int index = indices[i];
            a[i] = grid.size == 1 ? tile(tiles[index].row, k)
                : rowData(k) + rowSlot[tiles[index].row] * elements;
            bPointers[i] = grid.size == 1 ? tile(tiles[index].column, k)
                : columnData(k) + columnSlot[tiles[index].column] * elements;
            c[i] = destination + static_cast<size_t>(index) * elements;
        }
        CUDA_CHECK(cudaMemcpyAsync(deviceA, a.data(), count * sizeof(double*), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceB, bPointers.data(), count * sizeof(double*), cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(deviceC, c.data(), count * sizeof(double*), cudaMemcpyHostToDevice, stream));
        const double beta = 1.0;
        gemmBatches(bulk ? gpu.bulkBlas : gpu.blas, width(k), b, false,
            deviceA, deviceB, deviceC, count, &indices, alpha, beta);
    }

    void multiplyPanel(int k, int firstBlock, double* destination, double alpha, bool lookahead = false) {
        if (grid.size == 1) gpu.synchronize();
        if (lookahead) {
            // Trailing tiles must have received the previous update before
            // applying this one. The next panel and MPI exchange have already
            // run concurrently with that previous update.
            CUDA_CHECK(cudaEventSynchronize(gpu.bulkDone));
            CUDA_CHECK(cudaEventRecord(gpu.panelsReady, gpu.stream));
        }
        active.clear(); bulkActive.clear();
        for (size_t i = 0; i < tiles.size(); ++i)
            if (tiles[i].column >= firstBlock) {
                auto& indices = lookahead && tiles[i].column > firstBlock ? bulkActive : active;
                indices.push_back(static_cast<int>(i));
            }
        launchBatch(active, k, destination, alpha, false);
        if (lookahead) {
            CUDA_CHECK(cudaStreamWaitEvent(gpu.bulkStream, gpu.panelsReady, 0));
            launchBatch(bulkActive, k, destination, alpha, true);
            CUDA_CHECK(cudaEventRecord(gpu.bulkDone, gpu.bulkStream));
        }
    }

public:
    DistributedMatrix(const Grid& processGrid, Accelerator& accelerator, int size, const std::vector<unsigned int>& seedTable)
        : grid(processGrid), gpu(accelerator), seeds(seedTable), n(size),
          // Retain at least eight tile rows per process row when increasing
          // the tile size, so larger BLAS calls do not sacrifice concurrency.
          b(std::max(1, std::min(size, size / grid.rows >= 4096 ? 512 : 256))),
          blocks(size / b + (size % b != 0)), elements(static_cast<size_t>(b) * b),
          tiles(grid.tiles(blocks, worldRank)), rowSlot(blocks, -1), columnSlot(blocks, -1),
          tileMap(static_cast<size_t>(blocks) * blocks, -1), counts(grid.rows), displacements(grid.rows),
          matrix(tiles.size() * elements), rowPanel(2 * static_cast<size_t>(localRows()) * elements),
          columnPanel(2 * static_cast<size_t>(localColumns()) * elements),
          sendPanel(static_cast<size_t>(localRows()) * elements), diagonal(elements),
          tileRows(tiles.size()), tileColumns(tiles.size()), info(1),
          workspaceSize(std::max(solverWorkspace(gpu, b, b, diagonal.data()),
              solverWorkspace(gpu, blocks ? width(blocks - 1) : b, b, diagonal.data()))), workspace(workspaceSize),
          leftPointers(capacity()), rightPointers(capacity()), outputPointers(capacity()),
          bulkLeftPointers(capacity()), bulkRightPointers(capacity()), bulkOutputPointers(capacity()),
          hostRow(static_cast<size_t>(localRows()) * elements), hostColumn(static_cast<size_t>(localColumns()) * elements),
          hostSend(static_cast<size_t>(localRows()) * elements), hostDiagonal(elements),
          left(capacity()), right(capacity()), bulkLeft(capacity()), bulkRight(capacity()),
          output(capacity()), bulkOutput(capacity()) {
        for (int i = grid.row; i < blocks; i += grid.rows) {
            rowSlot[i] = static_cast<int>(rowBlocks.size()); rowBlocks.push_back(i);
        }
        int slot = 0;
        for (int r = 0; r < grid.rows; ++r)
            for (int j = grid.column; j < blocks; j += grid.columns)
                if (j % grid.rows == r) columnSlot[j] = slot++;
        std::vector<int> rows(tiles.size()), columns(tiles.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < tiles.size(); ++i) {
            rows[i] = tiles[i].row; columns[i] = tiles[i].column;
            tileMap[static_cast<size_t>(rows[i]) * blocks + columns[i]] = static_cast<int>(i);
        }
        CUDA_CHECK(cudaMemcpyAsync(tileRows.data(), rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(tileColumns.data(), columns.data(), columns.size() * sizeof(int), cudaMemcpyHostToDevice, gpu.stream));
        gpu.synchronize();
    }

    void generatePositiveDefiniteMatrix() {
        if (tiles.empty()) return;
        std::vector<int> randomBlocks, randomSlot(blocks, -1);
        for (const Tile& t : tiles) {
            if (randomSlot[t.row] < 0) { randomSlot[t.row] = static_cast<int>(randomBlocks.size()); randomBlocks.push_back(t.row); }
            if (randomSlot[t.column] < 0) { randomSlot[t.column] = static_cast<int>(randomBlocks.size()); randomBlocks.push_back(t.column); }
        }
        // Stream balanced column chunks of B instead of replicating complete
        // row slabs. Temporary storage is O(n * chunk / sqrt(ranks)), and
        // chunks stay aligned for native FP64 tensor-core GEMMs where present.
        const size_t chunks = (static_cast<size_t>(n) + 4095) / 4096;
        const int leading = static_cast<int>(((static_cast<size_t>(n) + chunks - 1) / chunks + 7) / 8 * 8);
        const size_t randomTileElements = static_cast<size_t>(b) * leading;
        std::vector<double> randomMatrix(randomBlocks.size() * randomTileElements, 0.0);
        std::vector<unsigned int> rowStates(randomBlocks.size() * b);
        DeviceBuffer<double> randomDevice(randomMatrix.size());
        const int count = mpiCount(tiles.size());
        #pragma omp parallel for schedule(static) if(count > 8)
        for (int i = 0; i < count; ++i) {
            left[i] = randomDevice.data() + randomSlot[tiles[i].row] * randomTileElements;
            right[i] = randomDevice.data() + randomSlot[tiles[i].column] * randomTileElements;
            output[i] = matrix.data() + static_cast<size_t>(i) * elements;
        }
        CUDA_CHECK(cudaMemcpyAsync(leftPointers.data(), left.data(), count * sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(rightPointers.data(), right.data(), count * sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(outputPointers.data(), output.data(), count * sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemsetAsync(matrix.data(), 0, localElements() * sizeof(double), gpu.stream));
        for (int firstColumn = 0; firstColumn < n; firstColumn += leading) {
            const int columns = std::min(leading, n - firstColumn);
            const int inner = (columns + 7) / 8 * 8;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rowStates.size(); ++i) {
                const int block = static_cast<int>(i / b), localRow = static_cast<int>(i % b);
                const int globalRow = randomBlocks[block] * b + localRow;
                if (globalRow < n) {
                    unsigned int seed = firstColumn ? rowStates[i]
                        : seeds.empty() ? rowSeed(static_cast<uint64_t>(globalRow) * n) : seeds[globalRow];
                    double* row = randomMatrix.data() + block * randomTileElements + static_cast<size_t>(localRow) * leading;
                    for (int j = 0; j < columns; ++j) row[j] = rand_r(&seed) / static_cast<double>(RAND_MAX) - 0.5;
                    for (int j = columns; j < inner; ++j) row[j] = 0.0;
                    rowStates[i] = seed;
                }
            }
            CUDA_CHECK(cudaMemcpyAsync(randomDevice.data(), randomMatrix.data(), randomMatrix.size() * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
            gemmBatches(gpu.blas, inner, leading, true, leftPointers.data(), rightPointers.data(),
                outputPointers.data(), count, nullptr, 1.0, firstColumn ? 1.0 : 0.0);
            gpu.synchronize();
        }
        addDiagonal<<<count, 256, 0, gpu.stream>>>(matrix.data(), tileRows.data(), tileColumns.data(), elements, b, count, n);
        CUDA_CHECK(cudaGetLastError());
        gpu.synchronize();
    }

    void saveOriginal(DeviceBuffer<double>& original) {
        CUDA_CHECK(cudaMemcpyAsync(original.data(), matrix.data(), tiles.size() * elements * sizeof(double), cudaMemcpyDeviceToDevice, gpu.stream));
        gpu.synchronize();
    }
    size_t localElements() const { return tiles.size() * elements; }

    // Allocate solver scratch once and load its CUDA kernels before timing.
    // This factors a separate identity tile and leaves the input untouched.
    void prepareCholesky() {
        if (!blocks) return;
        DeviceBuffer<double> scratch(2 * elements);
        CUDA_CHECK(cudaMemsetAsync(diagonal.data(), 0, elements * sizeof(double), gpu.stream));
        CUDA_CHECK(cudaMemsetAsync(scratch.data(), 0, 2 * elements * sizeof(double), gpu.stream));
        identityDiagonal<<<1, 256, 0, gpu.stream>>>(diagonal.data(), b);
        CUDA_CHECK(cudaGetLastError());
        SOLVER_CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, b, diagonal.data(), b,
            workspace.data(), workspaceSize, info.data()));
        const double* diagonalPointer = diagonal.data();
        const double* scratchPointer = scratch.data();
        double* scratchOutput = scratch.data();
        CUDA_CHECK(cudaMemcpyAsync(leftPointers.data(), &diagonalPointer, sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(outputPointers.data(), &scratchOutput, sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        const double one = 1.0, zero = 0.0;
        BLAS_CHECK(cublasDtrsmBatched(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, b, b, &one, leftPointers.data(), b, outputPointers.data(), b, 1));
        BLAS_CHECK(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, width(blocks - 1), b, &one, diagonal.data(), b, scratch.data(), b));
        CUDA_CHECK(cudaMemcpyAsync(leftPointers.data(), &scratchPointer, sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        CUDA_CHECK(cudaMemcpyAsync(rightPointers.data(), &scratchPointer, sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        double* gemmOutput = scratch.data() + elements;
        CUDA_CHECK(cudaMemcpyAsync(outputPointers.data(), &gemmOutput, sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
        BLAS_CHECK(cublasDgemmBatched(gpu.blas, CUBLAS_OP_N, CUBLAS_OP_T, b, b, b, &one,
            leftPointers.data(), b, rightPointers.data(), b, &zero, outputPointers.data(), b, 1));
        gpu.synchronize();
        BLAS_CHECK(cublasDgemmBatched(gpu.bulkBlas, CUBLAS_OP_N, CUBLAS_OP_T, b, b, b, &one,
            leftPointers.data(), b, rightPointers.data(), b, &zero, outputPointers.data(), b, 1));
        gpu.synchronizeAll();
    }

    bool choleskyDecomposition() {
        for (int k = 0; k < blocks; ++k) {
            gpu.synchronize();
            const int owner = grid.owner(k, k);
            int status = 0;
            if (worldRank == owner) {
                double* diagonalTile = tile(k, k);
                SOLVER_CHECK(cusolverDnDpotrf(gpu.solver, CUBLAS_FILL_MODE_LOWER, width(k), diagonalTile, b,
                    workspace.data(), workspaceSize, info.data()));
                CUDA_CHECK(cudaMemcpyAsync(&status, info.data(), sizeof(int), cudaMemcpyDeviceToHost, gpu.stream));
                if (gpu.deviceMPI)
                    CUDA_CHECK(cudaMemcpyAsync(diagonal.data(), diagonalTile, elements * sizeof(double), cudaMemcpyDeviceToDevice, gpu.stream));
                else
                    CUDA_CHECK(cudaMemcpyAsync(hostDiagonal.data(), diagonalTile, elements * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
                gpu.synchronize();
            }
            MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, owner, MPI_COMM_WORLD));
            if (status) {
                gpu.synchronizeAll();
                if (worldRank == 0) std::printf("Error: Matrix is not positive definite at diagonal element %d\n", k * b + status - 1);
                return false;
            }
            if (k + 1 == blocks) continue;
            if (grid.column == k % grid.columns) {
                MPI_CHECK(MPI_Bcast(gpu.deviceMPI ? diagonal.data() : hostDiagonal.data(), mpiCount(elements), MPI_DOUBLE, k % grid.rows, grid.columnComm));
                if (!gpu.deviceMPI)
                    CUDA_CHECK(cudaMemcpyAsync(diagonal.data(), hostDiagonal.data(), elements * sizeof(double), cudaMemcpyHostToDevice, gpu.stream));
                active.clear();
                for (int i : rowBlocks) if (i > k) active.push_back(i);
                const int count = static_cast<int>(active.size());
                #pragma omp parallel for schedule(static) if(count > 8)
                for (int i = 0; i < count; ++i) { left[i] = diagonal.data(); output[i] = tile(active[i], k); }
                if (count) {
                    CUDA_CHECK(cudaMemcpyAsync(leftPointers.data(), left.data(), count * sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
                    CUDA_CHECK(cudaMemcpyAsync(outputPointers.data(), output.data(), count * sizeof(double*), cudaMemcpyHostToDevice, gpu.stream));
                    const double alpha = 1.0;
                    const int full = count - (active.back() == blocks - 1 && width(blocks - 1) < b ? 1 : 0);
                    if (full)
                        BLAS_CHECK(cublasDtrsmBatched(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, b, b, &alpha, leftPointers.data(), b, outputPointers.data(), b, full));
                    if (full < count)
                        BLAS_CHECK(cublasDtrsm(gpu.blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER,
                            CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, width(blocks - 1), b, &alpha, diagonal.data(), b,
                            tile(blocks - 1, k), b));
                }
            }
            if (k + 1 < blocks) {
                if (grid.size > 1) exchangePanel(k, k + 1);
                multiplyPanel(k, k + 1, matrix.data(), -1.0, true);
            }
        }
        gpu.synchronizeAll();
        if (!tiles.empty()) {
            clearDiagonalUpper<<<mpiCount(tiles.size()), 256, 0, gpu.stream>>>(matrix.data(), tileRows.data(), tileColumns.data(), elements, b, mpiCount(tiles.size()));
            CUDA_CHECK(cudaGetLastError());
        }
        gpu.synchronize();
        return true;
    }

    bool validateCholesky(const DeviceBuffer<double>& original) {
        DeviceBuffer<double> reconstructed(localElements()), deviceErrors(tiles.size() * 2);
        CUDA_CHECK(cudaMemsetAsync(reconstructed.data(), 0, localElements() * sizeof(double), gpu.stream));
        for (int k = 0; k < blocks; ++k) {
            if (grid.size > 1) exchangePanel(k, k);
            else gpu.synchronize();
            multiplyPanel(k, k, reconstructed.data(), 1.0);
        }
        std::vector<double> errors(tiles.size() * 2);
        if (!tiles.empty()) {
            tileErrors<<<mpiCount(tiles.size()), 256, 0, gpu.stream>>>(reconstructed.data(), original.data(), tileRows.data(), tileColumns.data(), elements, b, n, deviceErrors.data());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(errors.data(), deviceErrors.data(), errors.size() * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
        }
        gpu.synchronize();
        double maxError = 0.0, relativeError = 0.0;
        #pragma omp parallel for reduction(max:maxError,relativeError) schedule(static)
        for (size_t i = 0; i < tiles.size(); ++i) {
            maxError = std::max(maxError, errors[2 * i]);
            relativeError = std::max(relativeError, errors[2 * i + 1]);
        }
        double local[2] = {maxError, relativeError}, global[2];
        MPI_CHECK(MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (worldRank == 0) {
            std::printf("Max absolute error: %.10e\nMax relative error: %.10e\n", global[0], global[1]);
            if (global[1] > 1e-6) std::printf("Validation failed: relative error too large\n");
        }
        return std::isfinite(global[1]) && global[1] <= 1e-6;
    }

    void printResults() {
        std::vector<double> packed(localElements());
        CUDA_CHECK(cudaMemcpyAsync(packed.data(), matrix.data(), packed.size() * sizeof(double), cudaMemcpyDeviceToHost, gpu.stream));
        gpu.synchronize();
        constexpr size_t chunk = 1 << 26;
        if (worldRank != 0) {
            for (size_t offset = 0; offset < packed.size(); offset += chunk)
                MPI_CHECK(MPI_Send(packed.data() + offset, static_cast<int>(std::min(chunk, packed.size() - offset)), MPI_DOUBLE, 0, 70, MPI_COMM_WORLD));
            return;
        }
        std::vector<double> result(static_cast<size_t>(n) * n, 0.0);
        for (int source = 0; source < grid.size; ++source) {
            const std::vector<Tile> sourceTiles = grid.tiles(blocks, source);
            if (source) {
                packed.resize(sourceTiles.size() * elements);
                for (size_t offset = 0; offset < packed.size(); offset += chunk)
                    MPI_CHECK(MPI_Recv(packed.data() + offset, static_cast<int>(std::min(chunk, packed.size() - offset)), MPI_DOUBLE, source, 70, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            }
            #pragma omp parallel for schedule(static)
            for (size_t t = 0; t < sourceTiles.size(); ++t)
                for (int i = 0; i < b && sourceTiles[t].row * b + i < n; ++i)
                    for (int j = 0; j < b && sourceTiles[t].column * b + j < n; ++j) {
                        const size_t row = sourceTiles[t].row * b + i, column = sourceTiles[t].column * b + j;
                        if (column <= row) result[row * n + column] = packed[t * elements + static_cast<size_t>(j) * b + i];
                    }
        }
        print_results(result, "CholeskyL");
    }
};

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\nOptions:\n", program);
    std::printf("  -n <num>     Matrix size (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
}

int run(int argc, char** argv) {
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const long long parsed = std::strtoll(argv[++i], &end, 10);
            if (errno || !end || *end || end == argv[i] || parsed < 0 || parsed > INT_MAX ||
                static_cast<unsigned long long>(parsed) * parsed > std::numeric_limits<size_t>::max() / sizeof(double)) {
                if (worldRank == 0) std::fprintf(stderr, "Invalid matrix size: %s\n", argv[i]);
                return 1;
            }
            n = static_cast<int>(parsed);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) results = true;
        else if (std::strcmp(argv[i], "-h") == 0) { if (worldRank == 0) printUsage(argv[0]); return 0; }
        else {
            if (worldRank == 0) { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm));
    int localRank, localSize, devices = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_size(localComm, &localSize));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("A CUDA device is required on every MPI rank");
    // Also supports launchers that expose only the assigned GPU to each rank.
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
    MPI_CHECK(MPI_Comm_free(&localComm));

    Grid grid;
    Accelerator accelerator;
    const std::vector<unsigned int> seeds = prepareRowSeeds(n);
    DistributedMatrix matrix(grid, accelerator, n, seeds);
    DeviceBuffer<double> original(validate ? matrix.localElements() : 0);
    if (worldRank == 0) {
        std::printf("Cholesky Decomposition Benchmark\nMatrix size: %d x %d\nValidation: %s\n", n, n, validate ? "enabled" : "disabled");
        std::printf("MPI grid: %d x %d; OpenMP threads per rank: %d; CUDA enabled\n", grid.rows, grid.columns, omp_get_max_threads());
        std::printf("Generating positive definite matrix...\n");
    }
    matrix.generatePositiveDefiniteMatrix();
    if (validate) matrix.saveOriginal(original);
    matrix.prepareCholesky();
    if (worldRank == 0) std::printf("Computing Cholesky decomposition...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    const bool success = matrix.choleskyDecomposition();
    const double localSeconds = MPI_Wtime() - start;
    double seconds;
    MPI_CHECK(MPI_Allreduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
    if (!success) { if (worldRank == 0) std::printf("Cholesky decomposition failed\n"); return 1; }
    if (worldRank == 0) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double operations = static_cast<double>(n) * n * n / 3.0;
        std::printf("Performance: %.3f GFLOPS\n", seconds > 0 ? operations / seconds / 1e9 : 0.0);
    }
    if (results) matrix.printResults();
    if (validate) {
        if (worldRank == 0) std::printf("Validating result...\n");
        const bool valid = matrix.validateCholesky(original);
        if (worldRank == 0) std::printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        return valid ? 0 : 1;
    }
    return 0;
}
} // namespace

int main(int argc, char** argv) {
    int provided = 0;
    const int initialized = MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (initialized != MPI_SUCCESS) return 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) fail("MPI_THREAD_FUNNELED support", 1);
    int result = 1;
    try { result = run(argc, argv); }
    catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return result;
}
