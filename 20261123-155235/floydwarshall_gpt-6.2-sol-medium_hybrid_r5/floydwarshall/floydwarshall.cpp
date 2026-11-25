#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
constexpr int tileSize = 32;
constexpr int threadRows = 8;
constexpr int rowsPerThread = tileSize / threadRows;
constexpr int tileElements = tileSize * tileSize;

// A real distance starts at at most MAX_DISTANCE + 1 and only decreases.
using Distance = unsigned char;
constexpr Distance paddedDistance = UCHAR_MAX;
static_assert(MAX_DISTANCE + 1 < paddedDistance);

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[idx2(i, i, numNodes)] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];
                
                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", 
                               i, j, k);
                        return false;
                    }
                }
            }
        }
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

#define CUDA_CHECK(operation) checkCuda((operation), #operation)

template <typename T>
struct DeviceBuffer {
    T* data = nullptr;

    explicit DeviceBuffer(size_t size) {
        if (size)
            CUDA_CHECK(cudaMalloc(&data, size * sizeof(*data)));
    }

    ~DeviceBuffer() {
        if (data)
            CUDA_CHECK(cudaFree(data));
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

template <typename T>
struct PinnedBuffer {
    T* data = nullptr;

    explicit PinnedBuffer(size_t size) {
        if (size)
            CUDA_CHECK(cudaMallocHost(&data, size * sizeof(*data)));
    }

    ~PinnedBuffer() {
        if (data)
            CUDA_CHECK(cudaFreeHost(data));
    }

    PinnedBuffer(const PinnedBuffer&) = delete;
    PinnedBuffer& operator=(const PinnedBuffer&) = delete;
};

struct ProcessGrid {
    // Global tile (i, j) belongs to rank (i % rows, j % cols).
    int rows;
    int cols;
    int row;
    int col;
    int localRows;
    int localCols;
    MPI_Comm rowComm;
    MPI_Comm colComm;
};

// rand_r on glibc advances its 32-bit LCG three times per output. Jump
// directly to the first draw needed for a local tile row.
#if defined(__GLIBC__)
unsigned int seedAt(size_t offset) {
    std::uint32_t seed = 42;
    std::uint32_t multiplier = 1103515245u;
    std::uint32_t increment = 12345u;
    for (size_t steps = 3 * offset; steps; steps >>= 1) {
        if (steps & 1)
            seed = multiplier * seed + increment;
        increment *= multiplier + 1u;
        multiplier *= multiplier;
    }
    return seed;
}
#endif

void initializeDistances(Distance* dist, int nodes, const ProcessGrid& grid,
                         size_t firstTile, size_t count) {
#if !defined(__GLIBC__)
    // Other C libraries can have a different rand_r state transition.
    std::vector<unsigned int> seeds(count * tileSize);
    unsigned int serialSeed = 42;
    for (int row = 0; row < nodes; ++row) {
        for (int col = 0; col < nodes; ++col) {
            const int tileRow = row / tileSize;
            const int tileCol = col / tileSize;
            if (col % tileSize == 0 && tileRow % grid.rows == grid.row &&
                tileCol % grid.cols == grid.col) {
                const size_t tile = static_cast<size_t>(tileRow / grid.rows) *
                                    grid.localCols + tileCol / grid.cols;
                if (tile >= firstTile && tile - firstTile < count)
                    seeds[(tile - firstTile) * tileSize + row % tileSize] = serialSeed;
            }
            rand_r(&serialSeed);
        }
    }
#endif

    const int threads = static_cast<int>(std::min<size_t>(omp_get_max_threads(), count));
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (size_t localTile = 0; localTile < count; ++localTile) {
        const size_t tile = firstTile + localTile;
        const int firstRow = (static_cast<int>(tile / grid.localCols) *
                              grid.rows + grid.row) * tileSize;
        const int firstCol = (static_cast<int>(tile % grid.localCols) *
                              grid.cols + grid.col) * tileSize;
        for (int i = 0; i < tileSize; ++i) {
            const int row = firstRow + i;
            unsigned int seed = 42;
            if (row < nodes && firstCol < nodes) {
#if defined(__GLIBC__)
                seed = seedAt(static_cast<size_t>(row) * nodes + firstCol);
#else
                seed = seeds[localTile * tileSize + i];
#endif
            }
            for (int j = 0; j < tileSize; ++j) {
                const int col = firstCol + j;
                unsigned int weight = paddedDistance;
                if (row < nodes && col < nodes)
                    weight = 1u + static_cast<unsigned int>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                dist[localTile * tileElements + i * tileSize + j] = row == col ? 0u : weight;
            }
        }
    }
}

__global__ void initializePaths(unsigned int* path, int localCols, int rankRow, int gridRows) {
    const int tileRow = (blockIdx.x / localCols) * gridRows + rankRow;
    path += static_cast<size_t>(blockIdx.x) * tileElements;
    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int element = i * (tileSize * threadRows) + threadIdx.x;
        path[element] = tileRow * tileSize + element / tileSize;
    }
}

__device__ __forceinline__ void relax(unsigned int candidate, unsigned int pivot,
                                      unsigned int& distance, unsigned int& path) {
    if (candidate < distance) {
        distance = candidate;
        path = pivot;
    }
}

// Record each pivot row and column at its own step. Later pivots can change
// these values, so a tile's final contents would lose the serial path choices.
__global__ void updateDiagonal(Distance* dist, unsigned int* path,
                               Distance* snapshots, Distance* rowPanel,
                               Distance* colPanel, int firstPivot) {
    __shared__ unsigned int tile[tileSize][tileSize];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    unsigned int previous[rowsPerThread];

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = y + i * threadRows;
        tile[row][x] = dist[row * tileSize + x];
        previous[i] = path[row * tileSize + x];
    }
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        const unsigned int right = tile[k][x];
        if (y == 0) {
            snapshots[k * tileSize + x] = right;
            rowPanel[k * tileSize + x] = right;
        }

        #pragma unroll
        for (int i = 0; i < rowsPerThread; ++i) {
            const int row = y + i * threadRows;
            const unsigned int left = tile[row][k];
            if (x == k) {
                snapshots[tileElements + row * tileSize + k] = left;
                colPanel[row * tileSize + k] = left;
            }

            // The zero diagonal keeps both pivot operands unchanged in this step.
            const unsigned int candidate = left + right;
            if (row != k && x != k && candidate < tile[row][x]) {
                tile[row][x] = candidate;
                previous[i] = firstPivot + k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = y + i * threadRows;
        dist[row * tileSize + x] = tile[row][x];
        path[row * tileSize + x] = previous[i];
    }
}

template <bool pivotRow>
__global__ void updatePanel(Distance* dist, unsigned int* path,
                            const Distance* diagonal, Distance* snapshots,
                            int localCols, int fixedTile, int skipTile, int firstPivot) {
    __shared__ unsigned int tile[tileSize][tileSize];
    __shared__ unsigned int fixed[tileSize][tileSize];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    const int block = blockIdx.x;
    const int index = block + (skipTile >= 0 && block >= skipTile);
    const size_t target = static_cast<size_t>(pivotRow ? fixedTile : index) * localCols +
                          (pivotRow ? index : fixedTile);
    dist += target * tileElements;
    path += target * tileElements;
    snapshots += static_cast<size_t>(index) * tileElements;
    unsigned int previous[rowsPerThread];

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = y + i * threadRows;
        const int element = row * tileSize + x;
        tile[row][x] = dist[element];
        fixed[row][x] = diagonal[element];
        previous[i] = path[element];
    }
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        const unsigned int right = pivotRow ? tile[k][x] : fixed[k][x];
        if (pivotRow && y == 0)
            snapshots[k * tileSize + x] = right;

        #pragma unroll
        for (int i = 0; i < rowsPerThread; ++i) {
            const int row = y + i * threadRows;
            const unsigned int left = pivotRow ? fixed[row][k] : tile[row][k];
            if (!pivotRow && x == k)
                snapshots[row * tileSize + k] = left;

            const unsigned int candidate = left + right;
            if ((pivotRow ? row != k : x != k) && candidate < tile[row][x]) {
                tile[row][x] = candidate;
                previous[i] = firstPivot + k;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = y + i * threadRows;
        dist[row * tileSize + x] = tile[row][x];
        path[row * tileSize + x] = previous[i];
    }
}

__global__ void updateRemaining(Distance* dist, unsigned int* path,
                                const Distance* rowPanel, const Distance* colPanel,
                                int localCols, int remainingCols, int skipRow, int skipCol,
                                int firstPivot) {
    __shared__ unsigned int left[tileSize][tileSize];
    __shared__ unsigned int right[tileSize][tileSize];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    int tileRow = blockIdx.x / remainingCols;
    int tileCol = blockIdx.x % remainingCols;
    tileRow += skipRow >= 0 && tileRow >= skipRow;
    tileCol += skipCol >= 0 && tileCol >= skipCol;
    const size_t target = static_cast<size_t>(tileRow) * localCols + tileCol;
    dist += target * tileElements;
    path += target * tileElements;
    rowPanel += static_cast<size_t>(tileCol) * tileElements;
    colPanel += static_cast<size_t>(tileRow) * tileElements;
    unsigned int distance[rowsPerThread];
    unsigned int previous[rowsPerThread];

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int row = y + i * threadRows;
        const int element = row * tileSize + x;
        left[row][x] = colPanel[element];
        right[row][x] = rowPanel[element];
        distance[i] = dist[element];
        previous[i] = UINT_MAX;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < tileSize; ++k) {
        const unsigned int operand = right[k][x];
        #pragma unroll
        for (int i = 0; i < rowsPerThread; ++i)
            relax(left[y + i * threadRows][k] + operand, firstPivot + k,
                  distance[i], previous[i]);
    }

    #pragma unroll
    for (int i = 0; i < rowsPerThread; ++i) {
        const int element = (y + i * threadRows) * tileSize + x;
        if (previous[i] != UINT_MAX) {
            dist[element] = distance[i];
            path[element] = previous[i];
        }
    }
}

struct Broadcast {
    Distance* device;
    Distance* host;
    int count;
    int position;
    int root;
    MPI_Comm comm;
};

void exchangePanels(const Broadcast (&panels)[2], bool deviceMpi, cudaStream_t stream,
                    cudaEvent_t hostReady) {
    bool sending = false;
    bool receiving = false;
    for (const auto& panel : panels) {
        if (panel.comm == MPI_COMM_NULL)
            continue;
        if (panel.position == panel.root) {
            if (!deviceMpi) {
                CUDA_CHECK(cudaMemcpyAsync(panel.host, panel.device, panel.count * sizeof(Distance),
                                           cudaMemcpyDeviceToHost, stream));
            }
            sending = true;
        } else {
            receiving = true;
        }
    }
    if (!sending && !receiving)
        return;
    // A device collective can write buffers still in use by an earlier kernel.
    if (sending || deviceMpi)
        CUDA_CHECK(cudaStreamSynchronize(stream));
    else
        CUDA_CHECK(cudaEventSynchronize(hostReady));

    // Some CUDA-aware MPI implementations support blocking device collectives only.
    if (deviceMpi) {
        for (const auto& panel : panels) {
            if (panel.comm != MPI_COMM_NULL)
                MPI_Bcast(panel.device, panel.count, MPI_UNSIGNED_CHAR, panel.root, panel.comm);
        }
        return;
    }

    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    for (int i = 0; i < 2; ++i) {
        const auto& panel = panels[i];
        if (panel.comm != MPI_COMM_NULL)
            MPI_Ibcast(panel.host, panel.count, MPI_UNSIGNED_CHAR, panel.root, panel.comm, &requests[i]);
    }
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    for (const auto& panel : panels) {
        if (panel.comm != MPI_COMM_NULL && panel.position != panel.root) {
            CUDA_CHECK(cudaMemcpyAsync(panel.device, panel.host, panel.count * sizeof(Distance),
                                       cudaMemcpyHostToDevice, stream));
        }
    }
    // Let CUDA consume the received panels in stream order. MPI can reuse the
    // pinned buffers as soon as the copies finish, even during the next kernel.
    if (receiving)
        CUDA_CHECK(cudaEventRecord(hostReady, stream));
}

bool cudaAwareMpi(MPI_Comm comm) {
    int supported = 0;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    supported = MPIX_Query_cuda_support();
#endif
    int allSupported = 0;
    MPI_Allreduce(&supported, &allSupported, 1, MPI_INT, MPI_MIN, comm);
    return allSupported != 0;
}

struct PanelWorkspace {
    const int rowCount;
    const int colCount;
    const bool staged;
    DeviceBuffer<Distance> diagonal;
    DeviceBuffer<Distance> rowPanel;
    DeviceBuffer<Distance> colPanel;
    PinnedBuffer<Distance> hostDiagonal;
    PinnedBuffer<Distance> hostRow;
    PinnedBuffer<Distance> hostCol;
    cudaEvent_t diagonalReady = nullptr;
    cudaEvent_t panelsReady = nullptr;

    PanelWorkspace(const ProcessGrid& grid, bool deviceMpi)
        : rowCount(grid.localCols * tileElements), colCount(grid.localRows * tileElements),
          staged(!deviceMpi && (grid.rows > 1 || grid.cols > 1)),
          diagonal(rowCount > 0 ? 2 * tileElements : 0),
          rowPanel(rowCount), colPanel(colCount),
          hostDiagonal(staged ? 2 * tileElements : 0),
          hostRow(staged && grid.rows > 1 ? rowCount : 0),
          hostCol(staged && grid.cols > 1 ? colCount : 0) {
        if (staged) {
            CUDA_CHECK(cudaEventCreateWithFlags(&diagonalReady, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&panelsReady, cudaEventDisableTiming));
        }
    }

    ~PanelWorkspace() {
        if (staged) {
            CUDA_CHECK(cudaEventDestroy(diagonalReady));
            CUDA_CHECK(cudaEventDestroy(panelsReady));
        }
    }
};

void floydWarshall(Distance* dist, unsigned int* path, int tiles,
                   const ProcessGrid& grid, bool deviceMpi, cudaStream_t stream,
                   const PanelWorkspace& work) {
    const dim3 threads(tileSize, threadRows);
    auto* diagonalRow = work.diagonal.data;
    auto* diagonalCol = work.diagonal.data + tileElements;
    auto* hostDiagonalRow = work.hostDiagonal.data;
    auto* hostDiagonalCol = hostDiagonalRow ? hostDiagonalRow + tileElements : nullptr;

    for (int pivot = 0; pivot < tiles; ++pivot) {
        const int ownerRow = pivot % grid.rows;
        const int ownerCol = pivot % grid.cols;
        const int localPivotRow = grid.row == ownerRow ? pivot / grid.rows : -1;
        const int localPivotCol = grid.col == ownerCol ? pivot / grid.cols : -1;
        const int firstPivot = pivot * tileSize;

        if (localPivotRow >= 0 && localPivotCol >= 0) {
            const size_t offset = (static_cast<size_t>(localPivotRow) * grid.localCols +
                                   localPivotCol) * tileElements;
            updateDiagonal<<<1, threads, 0, stream>>>(
                dist + offset, path + offset, work.diagonal.data,
                work.rowPanel.data + static_cast<size_t>(localPivotCol) * tileElements,
                work.colPanel.data + static_cast<size_t>(localPivotRow) * tileElements, firstPivot);
        }

        const Broadcast diagonalBroadcasts[2] = {
            {diagonalRow, hostDiagonalRow, tileElements, grid.row, ownerRow,
             localPivotCol >= 0 && grid.rows > 1 ? grid.colComm : MPI_COMM_NULL},
            {diagonalCol, hostDiagonalCol, tileElements, grid.col, ownerCol,
             localPivotRow >= 0 && grid.cols > 1 ? grid.rowComm : MPI_COMM_NULL}
        };
        if (localPivotRow >= 0 || localPivotCol >= 0)
            exchangePanels(diagonalBroadcasts, deviceMpi, stream, work.diagonalReady);

        const int remainingRows = grid.localRows - (localPivotRow >= 0);
        const int remainingCols = grid.localCols - (localPivotCol >= 0);
        if (localPivotRow >= 0 && remainingCols > 0) {
            updatePanel<true><<<remainingCols, threads, 0, stream>>>(
                dist, path, diagonalCol, work.rowPanel.data, grid.localCols,
                localPivotRow, localPivotCol, firstPivot);
        }
        if (localPivotCol >= 0 && remainingRows > 0) {
            updatePanel<false><<<remainingRows, threads, 0, stream>>>(
                dist, path, diagonalRow, work.colPanel.data, grid.localCols,
                localPivotCol, localPivotRow, firstPivot);
        }

        const Broadcast panelBroadcasts[2] = {
            {work.rowPanel.data, work.hostRow.data, work.rowCount, grid.row, ownerRow,
             grid.rows > 1 ? grid.colComm : MPI_COMM_NULL},
            {work.colPanel.data, work.hostCol.data, work.colCount, grid.col, ownerCol,
             grid.cols > 1 ? grid.rowComm : MPI_COMM_NULL}
        };
        exchangePanels(panelBroadcasts, deviceMpi, stream, work.panelsReady);

        const unsigned int remaining = static_cast<unsigned int>(
            static_cast<size_t>(remainingRows) * remainingCols);
        if (remaining > 0) {
            updateRemaining<<<remaining, threads, 0, stream>>>(
                dist, path, work.rowPanel.data, work.colPanel.data, grid.localCols, remainingCols,
                localPivotRow, localPivotCol, firstPivot);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void chooseGrid(int ranks, int tiles, int& rows, int& cols) {
    // Allow idle ranks for tiny graphs or to avoid a very elongated grid.
    rows = std::min(tiles, static_cast<int>(std::sqrt(ranks)));
    cols = std::min(tiles, ranks / rows);
    const int next = rows + 1;
    if (next <= tiles && static_cast<long long>(next) * (ranks / next) > rows * cols) {
        rows = next;
        cols = std::min(tiles, ranks / rows);
    }
}

std::vector<unsigned int> collectDistances(const Distance* deviceDist, int nodes, int tiles,
                                          const ProcessGrid& grid, MPI_Comm comm) {
    const int stripeElements = grid.localCols * tileElements;
    std::vector<Distance> stripe(stripeElements);
    std::vector<unsigned int> matrix;
    const bool root = grid.row == 0 && grid.col == 0;
    if (root) {
        matrix.resize(static_cast<size_t>(nodes) * nodes);
        stripe.resize(static_cast<size_t>((tiles + grid.cols - 1) / grid.cols) * tileElements);
    }
    for (int tileRow = 0; tileRow < tiles; ++tileRow) {
        const int ownerRow = tileRow % grid.rows;
        if (grid.row == ownerRow) {
            const size_t offset = static_cast<size_t>(tileRow / grid.rows) * stripeElements;
            CUDA_CHECK(cudaMemcpy(stripe.data(), deviceDist + offset,
                                  stripeElements * sizeof(Distance), cudaMemcpyDeviceToHost));
            if (!root)
                MPI_Send(stripe.data(), stripeElements, MPI_UNSIGNED_CHAR, 0, 0, comm);
        }
        if (!root)
            continue;

        for (int col = 0; col < grid.cols; ++col) {
            const int count = (tiles - 1 - col) / grid.cols + 1;
            const int owner = ownerRow * grid.cols + col;
            if (owner != 0)
                MPI_Recv(stripe.data(), count * tileElements, MPI_UNSIGNED_CHAR, owner, 0,
                         comm, MPI_STATUS_IGNORE);

            const int firstRow = tileRow * tileSize;
            const int rows = std::min(tileSize, nodes - firstRow);
            for (int tile = 0; tile < count; ++tile) {
                const int firstCol = (tile * grid.cols + col) * tileSize;
                const int cols = std::min(tileSize, nodes - firstCol);
                for (int i = 0; i < rows; ++i) {
                    std::copy_n(stripe.data() + static_cast<size_t>(tile) * tileElements + i * tileSize,
                                cols, matrix.data() + idx2(firstCol, firstRow + i, nodes));
                }
            }
        }
    }
    return matrix;
}

int runBenchmark(int nodes, bool validate, bool printResults, int rank, int size,
                 int localRank) {
    const int tiles = (nodes + tileSize - 1) / tileSize;
    ProcessGrid grid{};
    chooseGrid(size, std::max(1, tiles), grid.rows, grid.cols);

    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < grid.rows * grid.cols ? 0 : MPI_UNDEFINED, rank, &active);
    if (active == MPI_COMM_NULL)
        return 0;

    grid.row = rank / grid.cols;
    grid.col = rank % grid.cols;
    grid.localRows = (tiles + grid.rows - 1 - grid.row) / grid.rows;
    grid.localCols = (tiles + grid.cols - 1 - grid.col) / grid.cols;
    MPI_Comm_split(active, grid.row, grid.col, &grid.rowComm);
    MPI_Comm_split(active, grid.col, grid.row, &grid.colComm);

    int devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "Rank %d: no CUDA device is visible\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    const size_t localTiles = static_cast<size_t>(grid.localRows) * grid.localCols;
    if (grid.localRows > INT_MAX / tileElements || grid.localCols > INT_MAX / tileElements ||
        localTiles > INT_MAX) {
        fprintf(stderr, "Rank %d: local tile grid exceeds CUDA or MPI count limits\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const size_t localElements = localTiles * tileElements;
    DeviceBuffer<Distance> dist(localElements);
    DeviceBuffer<unsigned int> path(localElements);
    const bool deviceMpi = grid.rows * grid.cols > 1 && cudaAwareMpi(active);
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %d\n", nodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }
    if (localElements > 0) {
#if defined(__GLIBC__)
        constexpr size_t maxBatchTiles = 32768;
        const size_t batchTiles = std::min(localTiles, maxBatchTiles);
#else
        const size_t batchTiles = localTiles;
#endif
        PinnedBuffer<Distance> initial(batchTiles * tileElements);
        for (size_t first = 0; first < localTiles; first += batchTiles) {
            const size_t count = std::min(batchTiles, localTiles - first);
            initializeDistances(initial.data, nodes, grid, first, count);
            CUDA_CHECK(cudaMemcpyAsync(dist.data + first * tileElements, initial.data,
                                       count * tileElements * sizeof(Distance),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        initializePaths<<<static_cast<unsigned>(localTiles), tileSize * threadRows, 0, stream>>>(
            path.data, grid.localCols, grid.row, grid.rows);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    const PanelWorkspace work(grid, deviceMpi);
    if (rank == 0)
        printf("Computing shortest paths...\n");
    MPI_Barrier(active);
    const double start = MPI_Wtime();
    if (tiles > 0)
        floydWarshall(dist.data, path.data, tiles, grid, deviceMpi, stream, work);
    const double elapsed = MPI_Wtime() - start;
    double duration = 0.0;
    MPI_Reduce(&elapsed, &duration, 1, MPI_DOUBLE, MPI_MAX, 0, active);
    CUDA_CHECK(cudaStreamDestroy(stream));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", static_cast<long>(duration * 1000.0));
        const double ops = static_cast<double>(nodes) * nodes * nodes;
        printf("Performance: %.3f GOPS\n", ops / duration / 1e9);
    }

    int status = 0;
    if (validate || printResults) {
        const auto result = collectDistances(dist.data, nodes, tiles, grid, active);
        if (rank == 0) {
            if (printResults)
                print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(result, nodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                status = valid ? 0 : 1;
            }
        }
    }

    MPI_Comm_free(&grid.rowComm);
    MPI_Comm_free(&grid.colComm);
    MPI_Comm_free(&active);
    return status;
}

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0)
            fprintf(stderr, "MPI must support MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int nodes = 512;
    bool validate = false;
    bool printResults = false;
    int status = 0;
    bool run = true;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* argument = argv[++i];
            errno = 0;
            const long count = strtol(argument, nullptr, 10);
            if (errno || count < 0 || count > INT_MAX - tileSize) {
                if (rank == 0)
                    fprintf(stderr, "Invalid node count: %s\n", argument);
                status = 1;
                run = false;
                break;
            }
            nodes = static_cast<int>(count);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else {
            const bool help = strcmp(argv[i], "-h") == 0;
            if (rank == 0) {
                if (!help)
                    printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = help ? 0 : 1;
            run = false;
            break;
        }
    }

    if (run) {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int localRank = 0;
        MPI_Comm_rank(local, &localRank);
        MPI_Comm_free(&local);
        status = runBenchmark(nodes, validate, printResults, rank, size, localRank);
        MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return status;
}
