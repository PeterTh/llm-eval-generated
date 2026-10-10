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

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;
constexpr int TILE_ELEMENTS = TILE * TILE;

// via is one plus the smallest possible highest intermediate vertex on a
// shortest path; zero denotes the original edge. Lexicographic (distance, via)
// minimization preserves the original ascending-k algorithm's last strict
// improvement, including ties, despite the blocked traversal order.
struct alignas(8) Cell {
    unsigned int distance;
    unsigned int via;
};
static_assert(sizeof(Cell) == 2 * sizeof(unsigned int), "MPI cell layout");

int worldRank = 0;

void checkMPI(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING];
    int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "Rank %d: %s: %.*s\n", worldRank, operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
    std::abort();
}

void checkCUDA(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "Rank %d: %s: %s\n", worldRank, operation,
                 cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}
#define MPI_CHECK(call) checkMPI((call), #call)
#define CUDA_CHECK(call) checkCUDA((call), #call)

__device__ __forceinline__ void relax(unsigned int& distance, unsigned int& via,
                                    unsigned int leftDistance, unsigned int leftVia,
                                    unsigned int rightDistance, unsigned int rightVia,
                                    unsigned int pivot) {
    const unsigned int candidate = leftDistance + rightDistance;
    const unsigned int witness = max(max(leftVia, rightVia), pivot);
    if (candidate < distance || (candidate == distance && witness < via)) {
        distance = candidate;
        via = witness;
    }
}

__global__ void closePivot(Cell* matrix, Cell* pivot, size_t tileIndex,
                           unsigned int firstVertex) {
    __shared__ unsigned int distance[TILE][TILE];
    __shared__ unsigned int via[TILE][TILE];
    const int column = threadIdx.x;
    #pragma unroll
    for (int row = threadIdx.y; row < TILE; row += THREAD_ROWS) {
        const Cell value = matrix[tileIndex * TILE_ELEMENTS + row * TILE + column];
        distance[row][column] = value.distance;
        via[row][column] = value.via;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int row = threadIdx.y; row < TILE; row += THREAD_ROWS) {
            // The pivot row/column cannot improve (positive edges, zero
            // diagonal). Avoid writing them while other threads read them.
            if (row != k && column != k)
                relax(distance[row][column], via[row][column],
                      distance[row][k], via[row][k], distance[k][column],
                      via[k][column], firstVertex + k + 1);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int row = threadIdx.y; row < TILE; row += THREAD_ROWS) {
        const Cell value{distance[row][column], via[row][column]};
        matrix[tileIndex * TILE_ELEMENTS + row * TILE + column] = value;
        pivot[row * TILE + column] = value;
    }
}

// A closed diagonal tile allows each panel element to be computed independently
// as a min-plus product; only the initial shared-memory load needs a barrier.
template<bool RowPanel>
__global__ void updatePanel(Cell* matrix, const Cell* pivot, Cell* panel,
                            int localColumns, int localPivot, int coordinate,
                            int gridExtent, int pivotBlock) {
    __shared__ unsigned int pd[TILE][TILE], pv[TILE][TILE];
    __shared__ unsigned int sd[TILE][TILE], sv[TILE][TILE];
    const int block = blockIdx.x;
    const size_t tileIndex = RowPanel ? size_t(localPivot) * localColumns + block
                                      : size_t(block) * localColumns + localPivot;
    const int column = threadIdx.x;
    if (block * gridExtent + coordinate == pivotBlock) {
        #pragma unroll
        for (int row = threadIdx.y; row < TILE; row += THREAD_ROWS)
            panel[size_t(block) * TILE_ELEMENTS + row * TILE + column] =
                pivot[row * TILE + column];
        return;
    }
    #pragma unroll
    for (int row = threadIdx.y; row < TILE; row += THREAD_ROWS) {
        const Cell p = pivot[row * TILE + column];
        const Cell value = matrix[tileIndex * TILE_ELEMENTS + row * TILE + column];
        pd[row][column] = p.distance;
        pv[row][column] = p.via;
        sd[row][column] = value.distance;
        sv[row][column] = value.via;
    }
    __syncthreads();
    unsigned int distance[TILE / THREAD_ROWS], via[TILE / THREAD_ROWS];
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        distance[r] = sd[row][column];
        via[r] = sv[row][column];
    }
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
            const int row = threadIdx.y + r * THREAD_ROWS;
            if (RowPanel)
                relax(distance[r], via[r], pd[row][k], pv[row][k],
                      sd[k][column], sv[k][column], pivotBlock * TILE + k + 1);
            else
                relax(distance[r], via[r], sd[row][k], sv[row][k],
                      pd[k][column], pv[k][column], pivotBlock * TILE + k + 1);
        }
    }
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const int element = (threadIdx.y + r * THREAD_ROWS) * TILE + column;
        const Cell value{distance[r], via[r]};
        matrix[tileIndex * TILE_ELEMENTS + element] = value;
        panel[size_t(block) * TILE_ELEMENTS + element] = value;
    }
}

__global__ void updateRemaining(Cell* matrix, const Cell* rowPanel,
                                const Cell* columnPanel, int localColumns,
                                int rankRow, int rankColumn, int gridRows,
                                int gridColumns, int pivotBlock) {
    const size_t tileIndex = blockIdx.x;
    const int tileRow = tileIndex / localColumns;
    const int tileColumn = tileIndex % localColumns;
    if (tileRow * gridRows + rankRow == pivotBlock ||
        tileColumn * gridColumns + rankColumn == pivotBlock) return;
    __shared__ unsigned int ld[TILE][TILE], lv[TILE][TILE];
    __shared__ unsigned int rd[TILE][TILE], rv[TILE][TILE];
    const int column = threadIdx.x;
    unsigned int distance[TILE / THREAD_ROWS], via[TILE / THREAD_ROWS];
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
        const int row = threadIdx.y + r * THREAD_ROWS;
        const int element = row * TILE + column;
        const Cell left = columnPanel[size_t(tileRow) * TILE_ELEMENTS + element];
        const Cell right = rowPanel[size_t(tileColumn) * TILE_ELEMENTS + element];
        ld[row][column] = left.distance;
        lv[row][column] = left.via;
        rd[row][column] = right.distance;
        rv[row][column] = right.via;
        const Cell value = matrix[tileIndex * TILE_ELEMENTS + element];
        distance[r] = value.distance;
        via[r] = value.via;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int r = 0; r < TILE / THREAD_ROWS; ++r) {
            const int row = threadIdx.y + r * THREAD_ROWS;
            relax(distance[r], via[r], ld[row][k], lv[row][k],
                  rd[k][column], rv[k][column], pivotBlock * TILE + k + 1);
        }
    }
    #pragma unroll
    for (int r = 0; r < TILE / THREAD_ROWS; ++r)
        matrix[tileIndex * TILE_ELEMENTS +
               (threadIdx.y + r * THREAD_ROWS) * TILE + column] = Cell{distance[r], via[r]};
}

struct Grid {
    int rows, columns, row, column, blocks, localRows, localColumns;
    MPI_Comm rowComm, columnComm;
    size_t cells() const { return size_t(localRows) * localColumns * TILE_ELEMENTS; }
};

int blockCount(int total, int coordinate, int extent) {
    return total / extent + (coordinate < total % extent);
}

// glibc rand_r advances this 32-bit LCG three times per result. Jumping its
// state reproduces the original stream exactly without a root-only O(n^2)
// initialization or scattering a full matrix.
struct Affine {
    uint32_t multiplier = 1, increment = 0;
    uint32_t operator()(uint32_t seed) const { return multiplier * seed + increment; }
};

Affine randomJump(uint64_t steps) {
    Affine result, power{1103515245u, 12345u};
    for (; steps; steps >>= 1) {
        if (steps & 1) {
            result.increment = power.multiplier * result.increment + power.increment;
            result.multiplier *= power.multiplier;
        }
        power.increment *= power.multiplier + 1;
        power.multiplier *= power.multiplier;
    }
    return result;
}

void initializeLocal(std::vector<Cell>& matrix, size_t n, const Grid& grid) {
    #if defined(__GLIBC__)
    std::vector<unsigned int> rowSeeds(size_t(grid.localRows) * TILE);
    std::vector<Affine> columnJumps(grid.localColumns);
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rowSeeds.size(); ++row) {
        const size_t globalRow = (row / TILE * grid.rows + grid.row) * TILE + row % TILE;
        rowSeeds[row] = globalRow < n ? randomJump(3 * globalRow * n)(42) : 42;
    }
    for (int column = 0; column < grid.localColumns; ++column)
        columnJumps[column] = randomJump(3ULL * (column * grid.columns + grid.column) * TILE);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int tileRow = 0; tileRow < grid.localRows; ++tileRow) {
        for (int tileColumn = 0; tileColumn < grid.localColumns; ++tileColumn) {
            const size_t firstRow = size_t(tileRow * grid.rows + grid.row) * TILE;
            const size_t firstColumn = size_t(tileColumn * grid.columns + grid.column) * TILE;
            const size_t offset = (size_t(tileRow) * grid.localColumns + tileColumn) * TILE_ELEMENTS;
            for (int row = 0; row < TILE; ++row) {
                unsigned int seed = columnJumps[tileColumn](rowSeeds[size_t(tileRow) * TILE + row]);
                for (int column = 0; column < TILE; ++column) {
                    unsigned int distance = INF;
                    if (firstRow + row < n && firstColumn + column < n) {
                        distance = 1 + static_cast<unsigned int>(
                            double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
                        if (firstRow + row == firstColumn + column) distance = 0;
                    } else if (firstRow + row == firstColumn + column) distance = 0;
                    matrix[offset + row * TILE + column] = Cell{distance, 0};
                }
            }
        }
    }
    #else
    // Preserve the platform's rand_r stream on other libc implementations too.
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < matrix.size(); ++i) matrix[i] = Cell{INF, 0};
    unsigned int seed = 42;
    for (size_t row = 0; row < n; ++row) {
        for (size_t column = 0; column < n; ++column) {
            unsigned int distance = 1 + static_cast<unsigned int>(
                double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
            if (row == column) distance = 0;
            if (int(row / TILE % grid.rows) == grid.row &&
                int(column / TILE % grid.columns) == grid.column) {
                const size_t tile = row / TILE / grid.rows * grid.localColumns + column / TILE / grid.columns;
                matrix[tile * TILE_ELEMENTS + row % TILE * TILE + column % TILE] = Cell{distance, 0};
            }
        }
    }
    #endif
}

// Chunk counts to avoid MPI's int count limit. MPI operates on the two unsigned
// fields, so cell transfers also work between hosts of different endianness.
void broadcastCells(Cell* data, size_t count, int root, MPI_Comm comm,
                    std::vector<MPI_Request>& requests) {
    constexpr size_t chunkLimit = INT_MAX / 2;
    for (size_t offset = 0; offset < count; offset += chunkLimit) {
        requests.push_back(MPI_REQUEST_NULL);
        MPI_CHECK(MPI_Ibcast(data + offset, int(2 * std::min(chunkLimit, count - offset)),
                            MPI_UNSIGNED, root, comm, &requests.back()));
    }
}

void finishBroadcasts(std::vector<MPI_Request>& requests) {
    if (!requests.empty())
        MPI_CHECK(MPI_Waitall(int(requests.size()), requests.data(), MPI_STATUSES_IGNORE));
    requests.clear();
}

class Buffers {
public:
    Cell *matrix = nullptr, *pivot = nullptr, *row = nullptr, *column = nullptr;
    Cell *hostPivot = nullptr, *hostRow = nullptr, *hostColumn = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t stagingReady = nullptr;
    size_t rowCells, columnCells;

    explicit Buffers(const Grid& grid)
        : rowCells(size_t(grid.localColumns) * TILE_ELEMENTS),
          columnCells(size_t(grid.localRows) * TILE_ELEMENTS) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&stagingReady, cudaEventDisableTiming));
        allocateDevice(matrix, grid.cells());
        allocateDevice(pivot, TILE_ELEMENTS);
        allocateDevice(row, rowCells);
        allocateDevice(column, columnCells);
        allocateHost(hostPivot, TILE_ELEMENTS);
        allocateHost(hostRow, rowCells);
        allocateHost(hostColumn, columnCells);
    }
    ~Buffers() {
        cudaFree(matrix); cudaFree(pivot); cudaFree(row); cudaFree(column);
        cudaFreeHost(hostPivot); cudaFreeHost(hostRow); cudaFreeHost(hostColumn);
        cudaEventDestroy(stagingReady);
        cudaStreamDestroy(stream);
    }
private:
    static void allocateDevice(Cell*& pointer, size_t cells) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer), std::max(cells, size_t(1)) * sizeof(Cell)));
    }
    static void allocateHost(Cell*& pointer, size_t cells) {
        CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&pointer), std::max(cells, size_t(1)) * sizeof(Cell)));
    }
};

void floydWarshall(Buffers& buffers, const Grid& grid) {
    const dim3 threads(TILE, THREAD_ROWS);
    const bool distributed = grid.rows > 1 || grid.columns > 1;
    std::vector<MPI_Request> requests;
    requests.reserve(4);
    for (int block = 0; block < grid.blocks; ++block) {
        const int ownerRow = block % grid.rows, ownerColumn = block % grid.columns;
        const bool inRow = grid.row == ownerRow, inColumn = grid.column == ownerColumn;
        if (inRow && inColumn) {
            const size_t index = size_t(block / grid.rows) * grid.localColumns + block / grid.columns;
            closePivot<<<1, threads, 0, buffers.stream>>>(buffers.matrix, buffers.pivot, index, block * TILE);
            CUDA_CHECK(cudaGetLastError());
            if (distributed) {
                CUDA_CHECK(cudaMemcpyAsync(buffers.hostPivot, buffers.pivot,
                                          TILE_ELEMENTS * sizeof(Cell), cudaMemcpyDeviceToHost, buffers.stream));
                CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
            }
        }
        if (inRow && grid.columns > 1)
            broadcastCells(buffers.hostPivot, TILE_ELEMENTS, ownerColumn, grid.rowComm, requests);
        if (inColumn && grid.rows > 1)
            broadcastCells(buffers.hostPivot, TILE_ELEMENTS, ownerRow, grid.columnComm, requests);
        finishBroadcasts(requests);
        if ((inRow || inColumn) && !(inRow && inColumn))
            CUDA_CHECK(cudaMemcpyAsync(buffers.pivot, buffers.hostPivot,
                                      TILE_ELEMENTS * sizeof(Cell), cudaMemcpyHostToDevice, buffers.stream));
        if (inRow && grid.localColumns) {
            updatePanel<true><<<grid.localColumns, threads, 0, buffers.stream>>>(
                buffers.matrix, buffers.pivot, buffers.row, grid.localColumns,
                block / grid.rows, grid.column, grid.columns, block);
            CUDA_CHECK(cudaGetLastError());
            if (grid.rows > 1)
                CUDA_CHECK(cudaMemcpyAsync(buffers.hostRow, buffers.row,
                                          buffers.rowCells * sizeof(Cell), cudaMemcpyDeviceToHost, buffers.stream));
        }
        if (inColumn && grid.localRows) {
            updatePanel<false><<<grid.localRows, threads, 0, buffers.stream>>>(
                buffers.matrix, buffers.pivot, buffers.column, grid.localColumns,
                block / grid.columns, grid.row, grid.rows, block);
            CUDA_CHECK(cudaGetLastError());
            if (grid.columns > 1)
                CUDA_CHECK(cudaMemcpyAsync(buffers.hostColumn, buffers.column,
                                          buffers.columnCells * sizeof(Cell), cudaMemcpyDeviceToHost, buffers.stream));
        }
        if (distributed && (inRow || inColumn)) CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
        // A prior asynchronous upload must finish before MPI reuses its pinned
        // source buffer. This event precedes the trailing kernel, allowing that
        // kernel to keep running while the next panels arrive on the host.
        if (distributed) CUDA_CHECK(cudaEventSynchronize(buffers.stagingReady));
        // Two-dimensional distribution: each panel travels only along the
        // process rows/columns that need it. Portable pinned staging does not
        // require a CUDA-aware MPI installation. The two broadcasts overlap.
        if (grid.rows > 1)
            broadcastCells(buffers.hostRow, buffers.rowCells, ownerRow, grid.columnComm, requests);
        if (grid.columns > 1)
            broadcastCells(buffers.hostColumn, buffers.columnCells, ownerColumn, grid.rowComm, requests);
        finishBroadcasts(requests);
        if (!inRow && buffers.rowCells)
            CUDA_CHECK(cudaMemcpyAsync(buffers.row, buffers.hostRow, buffers.rowCells * sizeof(Cell),
                                      cudaMemcpyHostToDevice, buffers.stream));
        if (!inColumn && buffers.columnCells)
            CUDA_CHECK(cudaMemcpyAsync(buffers.column, buffers.hostColumn, buffers.columnCells * sizeof(Cell),
                                      cudaMemcpyHostToDevice, buffers.stream));
        if (distributed) CUDA_CHECK(cudaEventRecord(buffers.stagingReady, buffers.stream));
        if (grid.cells()) {
            updateRemaining<<<grid.cells() / TILE_ELEMENTS, threads, 0, buffers.stream>>>(
                buffers.matrix, buffers.row, buffers.column, grid.localColumns,
                grid.row, grid.column, grid.rows, grid.columns, block);
            CUDA_CHECK(cudaGetLastError());
        }
        // No global GPU barrier: this rank's trailing update can overlap the
        // next pivot's MPI communication; its stream enforces data dependencies.
    }
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
}

void gatherResult(const std::vector<Cell>& local, size_t n, const Grid& grid,
                  int ranks, std::vector<unsigned int>& distance,
                  std::vector<unsigned int>& path) {
    constexpr size_t chunkLimit = INT_MAX / 2;
    if (worldRank != 0) {
        for (size_t offset = 0; offset < local.size(); offset += chunkLimit)
            MPI_CHECK(MPI_Send(local.data() + offset, int(2 * std::min(chunkLimit, local.size() - offset)),
                               MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD));
        return;
    }
    distance.resize(n * n);
    if (n <= 256) path.resize(n * n);
    std::vector<Cell> received;
    for (int rank = 0; rank < ranks; ++rank) {
        const int rankRow = rank / grid.columns, rankColumn = rank % grid.columns;
        const int rows = blockCount(grid.blocks, rankRow, grid.rows);
        const int columns = blockCount(grid.blocks, rankColumn, grid.columns);
        const size_t cells = size_t(rows) * columns * TILE_ELEMENTS;
        const Cell* data = local.data();
        if (rank != 0) {
            received.resize(cells);
            for (size_t offset = 0; offset < cells; offset += chunkLimit)
                MPI_CHECK(MPI_Recv(received.data() + offset, int(2 * std::min(chunkLimit, cells - offset)),
                                   MPI_UNSIGNED, rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            data = received.data();
        }
        #pragma omp parallel for collapse(2) schedule(static)
        for (int tileRow = 0; tileRow < rows; ++tileRow) {
            for (int tileColumn = 0; tileColumn < columns; ++tileColumn) {
                const size_t firstRow = size_t(tileRow * grid.rows + rankRow) * TILE;
                const size_t firstColumn = size_t(tileColumn * grid.columns + rankColumn) * TILE;
                const size_t offset = (size_t(tileRow) * columns + tileColumn) * TILE_ELEMENTS;
                for (int row = 0; row < TILE && firstRow + row < n; ++row) {
                    for (int column = 0; column < TILE && firstColumn + column < n; ++column) {
                        const size_t index = (firstRow + row) * n + firstColumn + column;
                        const Cell value = data[offset + row * TILE + column];
                        distance[index] = value.distance;
                        if (!path.empty()) path[index] = value.via ? value.via - 1 : firstRow + row;
                    }
                }
            }
        }
    }
}

bool validateResult(const std::vector<unsigned int>& distance,
                    const std::vector<unsigned int>& path, size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i) valid &= distance[i * n + i] == 0;
    #pragma omp parallel for collapse(2) reduction(&:valid) schedule(static)
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i) {
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j) {
            for (size_t k = 0; k < n; ++k)
                if (distance[i * n + k] < INF && distance[k * n + j] < INF)
                    valid &= distance[i * n + k] + distance[k * n + j] >= distance[i * n + j];
        }
    }
    // Small validation runs additionally check every distance AND path entry
    // against the original loop order. This is never a computation fallback.
    if (n <= 256) {
        std::vector<unsigned int> reference(n * n), referencePath(n * n);
        unsigned int seed = 42;
        for (size_t i = 0; i < n * n; ++i) {
            reference[i] = 1 + static_cast<unsigned int>(
                double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
            referencePath[i] = i / n;
        }
        for (size_t i = 0; i < n; ++i) reference[i * n + i] = 0;
        for (size_t k = 0; k < n; ++k)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const unsigned int candidate = reference[i * n + k] + reference[k * n + j];
                    if (candidate < reference[i * n + j]) {
                        reference[i * n + j] = candidate;
                        referencePath[i * n + j] = k;
                    }
                }
        if (distance != reference || path != referencePath) {
            std::printf("Validation failed: exact distance/path comparison\n");
            valid = 0;
        }
    }
    return valid != 0;
}

void printUsage(const char* program) {
    std::printf("Usage: %s [options]\n", program);
    std::printf("Options:\n");
    std::printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    std::printf("  -v           Enable validation\n");
    std::printf("  -r           Print results for external validation\n");
    std::printf("  -h           Show this help message\n");
    std::printf("Run one MPI rank per GPU; OMP_NUM_THREADS controls host parallelism.\n");
}

int run(int argc, char** argv) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const unsigned long long value = std::strtoull(argv[++i], &end, 10);
            if (errno || end == argv[i] || *end || argv[i][0] == '-' || value >= UINT_MAX ||
                value > std::numeric_limits<size_t>::max() ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(Cell) / value)) {
                if (!worldRank) std::fprintf(stderr, "Invalid number of nodes: %s\n", argv[i]);
                return 1;
            }
            n = size_t(value);
        } else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) {
            if (!worldRank) printUsage(argv[0]);
            return 0;
        } else {
            if (!worldRank) {
                std::printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    int ranks = 1, dimensions[2] = {0, 0};
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    MPI_CHECK(MPI_Dims_create(ranks, 2, dimensions));
    Grid grid{};
    grid.rows = dimensions[0]; grid.columns = dimensions[1];
    grid.row = worldRank / grid.columns; grid.column = worldRank % grid.columns;
    grid.blocks = int((n + TILE - 1) / TILE);
    grid.localRows = blockCount(grid.blocks, grid.row, grid.rows);
    grid.localColumns = blockCount(grid.blocks, grid.column, grid.columns);
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, grid.row, grid.column, &grid.rowComm));
    MPI_CHECK(MPI_Comm_split(MPI_COMM_WORLD, grid.column, grid.row, &grid.columnComm));
    MPI_Comm localComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, worldRank, MPI_INFO_NULL, &localComm));
    int localRank = 0, devices = 0;
    MPI_CHECK(MPI_Comm_rank(localComm, &localRank));
    MPI_CHECK(MPI_Comm_free(&localComm));
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        std::fprintf(stderr, "Rank %d: the MPI/OpenMP/CUDA benchmark requires a CUDA GPU\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    // Respect an explicit OpenMP setting. Otherwise avoid every rank spawning
    // a full node's worth of threads, especially for the default small graph.
    if (std::getenv("OMP_NUM_THREADS") == nullptr) omp_set_num_threads(std::min(8, omp_get_max_threads()));
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, localRank % devices));
    // These streaming kernels reuse their operands in shared memory, rather
    // than L1. Configuring them here also loads the kernels before timing.
    CUDA_CHECK(cudaFuncSetCacheConfig(closePivot, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetCacheConfig(updatePanel<true>, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetCacheConfig(updatePanel<false>, cudaFuncCachePreferShared));
    CUDA_CHECK(cudaFuncSetCacheConfig(updateRemaining, cudaFuncCachePreferShared));
    if (grid.cells() / TILE_ELEMENTS > size_t(properties.maxGridSize[0])) {
        std::fprintf(stderr, "Rank %d: local matrix exceeds the CUDA grid limit\n", worldRank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (!worldRank) {
        std::printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        std::printf("Number of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        std::printf("MPI grid: %d x %d; CUDA tile: %d; OpenMP threads/rank: %d\n",
                    grid.rows, grid.columns, TILE, omp_get_max_threads());
        std::printf("Initializing graph...\n");
    }
    std::vector<Cell> local(grid.cells());
    initializeLocal(local, n, grid);
    Buffers buffers(grid);
    if (!local.empty())
        CUDA_CHECK(cudaMemcpyAsync(buffers.matrix, local.data(), local.size() * sizeof(Cell),
                                  cudaMemcpyHostToDevice, buffers.stream));
    CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
    if (!printResults && !validate) std::vector<Cell>().swap(local);
    if (!worldRank) std::printf("Computing shortest paths...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    floydWarshall(buffers, grid);
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_CHECK(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!worldRank) {
        std::printf("Computation time: %.3f ms\n", seconds * 1000);
        std::printf("Performance: %.3f GOPS\n", seconds > 0 ? double(n) * n * n / seconds / 1e9 : 0);
    }
    int status = 0;
    if (printResults || validate) {
        if (!local.empty())
            CUDA_CHECK(cudaMemcpyAsync(local.data(), buffers.matrix, local.size() * sizeof(Cell),
                                      cudaMemcpyDeviceToHost, buffers.stream));
        CUDA_CHECK(cudaStreamSynchronize(buffers.stream));
        std::vector<unsigned int> distance, path;
        gatherResult(local, n, grid, ranks, distance, path);
        if (!worldRank) {
            if (printResults) print_results_int(distance, "DistanceMatrix");
            if (validate) {
                std::printf("Validating result...\n");
                status = validateResult(distance, path, n) ? 0 : 1;
                std::printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_CHECK(MPI_Comm_free(&grid.rowComm));
    MPI_CHECK(MPI_Comm_free(&grid.columnComm));
    return status;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));
    MPI_CHECK(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    if (provided < MPI_THREAD_FUNNELED) {
        std::fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int status = 1;
    try {
        status = run(argc, argv);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "Rank %d: %s\n", worldRank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_CHECK(MPI_Finalize());
    return status;
}
