#include <algorithm>
#include <array>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#if defined(OMPI_MAJOR_VERSION)
#include <mpi-ext.h>
#endif

#include "../common/results_output.hpp"

namespace {

using Distance = unsigned int;
constexpr Distance INF = 1000000000;
constexpr Distance MAX_DISTANCE = 200;
constexpr int tileSize = 32;
constexpr int tileElements = tileSize * tileSize;
constexpr int threadRows = 8;

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

struct Buffer {
    Distance* device = nullptr;
    Distance* host = nullptr;

    Buffer(size_t count, bool pinned = false) {
        if (count) {
            checkCuda(cudaMalloc(&device, count * sizeof(Distance)));
            if (pinned) {
                checkCuda(cudaMallocHost(&host, count * sizeof(Distance)));
            }
        }
    }

    ~Buffer() {
        if (host) {
            checkCuda(cudaFreeHost(host));
        }
        if (device) {
            checkCuda(cudaFree(device));
        }
    }

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Grid {
    // Global tile (i,j) belongs to process (i % rows, j % cols).
    int rows;
    int cols;
    int row;
    int col;
    int localRows;
    int localCols;
    MPI_Comm rowComm;
    MPI_Comm colComm;

    size_t elements() const {
        return static_cast<size_t>(localRows) * localCols * tileElements;
    }
};

__host__ __device__ Distance* tileAt(Distance* matrix, int row, int col, int stride) {
    return matrix + (static_cast<size_t>(row) * stride + col) * tileElements;
}

// Only the rows and columns outside the current intermediate are written.
// The zero diagonal ensures that the inputs to each relaxation stay fixed.
__global__ void closePivot(Distance* tile) {
    __shared__ Distance values[tileSize][tileSize];
    const int x = threadIdx.x;
    const int y = threadIdx.y;

    for (int i = y; i < tileSize; i += threadRows) {
        values[i][x] = tile[i * tileSize + x];
    }
    __syncthreads();

    for (int k = 0; k < tileSize; ++k) {
        for (int i = y; i < tileSize; i += threadRows) {
            if (i != k && x != k) {
                values[i][x] = min(values[i][x], values[i][k] + values[k][x]);
            }
        }
        __syncthreads();
    }

    for (int i = y; i < tileSize; i += threadRows) {
        tile[i * tileSize + x] = values[i][x];
    }
}

__global__ void updatePanels(Distance* matrix, const Distance* pivot, Distance* packedColumn,
                             int localCols, int pivotRow, int pivotCol, int rowTiles, bool ownsPivot) {
    const int block = blockIdx.x;
    const bool onRow = block < rowTiles;
    const int tile = onRow ? block : block - rowTiles;
    const int x = threadIdx.x;
    const int y = threadIdx.y;

    if (ownsPivot && (tile == (onRow ? pivotCol : pivotRow))) {
        if (!onRow && packedColumn) {
            for (int i = y; i < tileSize; i += threadRows) {
                packedColumn[tile * static_cast<size_t>(tileElements) + i * tileSize + x] =
                    pivot[i * tileSize + x];
            }
        }
        return;
    }

    Distance* target = onRow ? tileAt(matrix, pivotRow, tile, localCols)
                             : tileAt(matrix, tile, pivotCol, localCols);
    const Distance* leftTile = onRow ? pivot : target;
    const Distance* rightTile = onRow ? target : pivot;
    __shared__ Distance left[tileSize][tileSize];
    __shared__ Distance right[tileSize][tileSize];
    Distance best[tileSize / threadRows];
    #pragma unroll
    for (int i = 0; i < tileSize / threadRows; ++i) {
        const int r = y + i * threadRows;
        const Distance lhs = leftTile[r * tileSize + x];
        const Distance rhs = rightTile[r * tileSize + x];
        left[r][x] = lhs;
        right[r][x] = rhs;
        best[i] = onRow ? rhs : lhs;
    }
    __syncthreads();

    // The closed diagonal contains every path within the pivot tile.
    // Multiplying by it closes the panel without loop-carried dependencies.

    #pragma unroll
    for (int k = 0; k < tileSize; ++k) {
        const Distance rhs = right[k][x];
        #pragma unroll
        for (int i = 0; i < tileSize / threadRows; ++i) {
            best[i] = min(best[i], left[y + i * threadRows][k] + rhs);
        }
    }

    #pragma unroll
    for (int i = y; i < tileSize; i += threadRows) {
        const Distance value = best[(i - y) / threadRows];
        target[i * tileSize + x] = value;
        if (!onRow && packedColumn) {
            packedColumn[tile * static_cast<size_t>(tileElements) + i * tileSize + x] = value;
        }
    }
}

__global__ void updateInterior(Distance* matrix, const Distance* rowPanel,
                               const Distance* colPanel, int colPanelStride, int localCols,
                               int interiorCols, int skipRow, int skipCol) {
    int row = blockIdx.x / interiorCols;
    int col = blockIdx.x % interiorCols;
    row += skipRow >= 0 && row >= skipRow;
    col += skipCol >= 0 && col >= skipCol;
    const int x = threadIdx.x;
    const int y = threadIdx.y;

    Distance* target = tileAt(matrix, row, col, localCols);
    const Distance* leftTile = colPanel + static_cast<size_t>(row) * colPanelStride * tileElements;
    const Distance* rightTile = rowPanel + static_cast<size_t>(col) * tileElements;
    __shared__ Distance left[tileSize][tileSize];
    __shared__ Distance right[tileSize][tileSize];
    Distance best[tileSize / threadRows];

    #pragma unroll
    for (int i = 0; i < tileSize / threadRows; ++i) {
        const int r = y + i * threadRows;
        left[r][x] = leftTile[r * tileSize + x];
        right[r][x] = rightTile[r * tileSize + x];
        best[i] = target[r * tileSize + x];
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < tileSize; ++k) {
        const Distance rhs = right[k][x];
        #pragma unroll
        for (int i = 0; i < tileSize / threadRows; ++i) {
            best[i] = min(best[i], left[y + i * threadRows][k] + rhs);
        }
    }

    #pragma unroll
    for (int i = 0; i < tileSize / threadRows; ++i) {
        target[(y + i * threadRows) * tileSize + x] = best[i];
    }
}

#if defined(__GLIBC__)
// glibc rand_r advances a 32-bit LCG three times per result.
// Composing its affine steps gives each thread the original seed at any index.
unsigned int distanceSeed(uint64_t index) {
    uint32_t value = 42;
    uint32_t multiplier = 1103515245u;
    uint32_t increment = 12345u;
    for (uint64_t steps = 3 * index; steps; steps >>= 1) {
        if (steps & 1) {
            value = multiplier * value + increment;
        }
        increment *= multiplier + 1;
        multiplier *= multiplier;
    }
    return value;
}
#endif

int hostThreads(size_t elements) {
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), 1 + elements / 16384));
}

void initializeMatrix(Distance* matrix, int nodes, const Grid& grid) {
#if !defined(__GLIBC__)
    // On other C libraries, record row seeds using the platform's rand_r.
    std::vector<unsigned int> rowSeeds(nodes);
    unsigned int seed = 42;
    for (int row = 0; row < nodes; ++row) {
        rowSeeds[row] = seed;
        for (int col = 0; col < nodes; ++col) {
            rand_r(&seed);
        }
    }
#endif

    const int threads = hostThreads(grid.elements());
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int localRow = 0; localRow < grid.localRows * tileSize; ++localRow) {
        const int row = (localRow / tileSize * grid.rows + grid.row) * tileSize
                        + localRow % tileSize;
#if !defined(__GLIBC__)
        unsigned int rowSeed = row < nodes ? rowSeeds[row] : 0;
        int seedCol = 0;
#endif
        for (int block = 0; block < grid.localCols; ++block) {
            const int firstCol = (block * grid.cols + grid.col) * tileSize;
            auto* dest = tileAt(matrix, localRow / tileSize, block, grid.localCols)
                         + localRow % tileSize * tileSize;

            if (row >= nodes || firstCol >= nodes) {
                for (int j = 0; j < tileSize; ++j) {
                    dest[j] = row == firstCol + j ? 0 : INF;
                }
                continue;
            }

#if defined(__GLIBC__)
            unsigned int seed = distanceSeed(static_cast<uint64_t>(row) * nodes + firstCol);
#else
            for (; seedCol < firstCol; ++seedCol) {
                rand_r(&rowSeed);
            }
            unsigned int& seed = rowSeed;
#endif
            const int count = std::min(tileSize, nodes - firstCol);
            for (int j = 0; j < count; ++j) {
                const Distance weight = 1 + static_cast<Distance>(
                    (static_cast<double>(MAX_DISTANCE) * rand_r(&seed)) / RAND_MAX);
                dest[j] = row == firstCol + j ? 0 : weight;
            }
            std::fill(dest + count, dest + tileSize, INF);
#if !defined(__GLIBC__)
            seedCol += count;
#endif
        }
    }
}

struct Broadcast {
    MPI_Comm comm = MPI_COMM_NULL;
    int root = 0;
    bool send = false;
    Distance* device = nullptr;
    Distance* host = nullptr;
    size_t count = 0;
};

void exchange(const std::array<Broadcast, 2>& transfers, cudaStream_t stream,
              cudaEvent_t staged) {
    if (transfers[0].comm == MPI_COMM_NULL && transfers[1].comm == MPI_COMM_NULL) {
        return;
    }
    Distance* copied = nullptr;
    for (const auto& transfer : transfers) {
        if (transfer.comm != MPI_COMM_NULL && transfer.send && transfer.host
            && transfer.host != copied) {
            checkCuda(cudaMemcpyAsync(transfer.host, transfer.device,
                                      transfer.count * sizeof(Distance),
                                      cudaMemcpyDeviceToHost, stream));
            copied = transfer.host;
        }
    }
    if (copied || !staged) {
        checkCuda(cudaStreamSynchronize(stream));
    } else {
        // Reuse pinned receive storage once the previous copy has consumed it.
        // The previous GPU update can continue while MPI fills the host buffer.
        checkCuda(cudaEventSynchronize(staged));
    }

    // Keep each collective's count within the range of the MPI int interface.
    const size_t largest = std::max(transfers[0].count, transfers[1].count);
    for (size_t offset = 0; offset < largest; offset += INT_MAX) {
        MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        for (size_t i = 0; i < transfers.size(); ++i) {
            const auto& transfer = transfers[i];
            if (transfer.comm != MPI_COMM_NULL && offset < transfer.count) {
                auto* buffer = transfer.host ? transfer.host : transfer.device;
                const int count = static_cast<int>(
                    std::min<size_t>(INT_MAX, transfer.count - offset));
                if (staged) {
                    MPI_Ibcast(buffer + offset, count, MPI_UNSIGNED, transfer.root,
                               transfer.comm, &requests[i]);
                } else {
                    MPI_Bcast(buffer + offset, count, MPI_UNSIGNED, transfer.root, transfer.comm);
                }
            }
        }
        if (staged) {
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
        }
    }

    bool received = false;
    for (const auto& transfer : transfers) {
        if (transfer.comm != MPI_COMM_NULL && !transfer.send && transfer.host) {
            checkCuda(cudaMemcpyAsync(transfer.device, transfer.host,
                                      transfer.count * sizeof(Distance),
                                      cudaMemcpyHostToDevice, stream));
            received = true;
        }
    }
    if (received) {
        checkCuda(cudaEventRecord(staged, stream));
    }
}

void floydWarshall(Distance* matrix, int tiles, const Grid& grid, cudaStream_t stream,
                   Buffer& diagonal, Buffer& row, Buffer& column,
                   cudaEvent_t diagonalStaged, cudaEvent_t panelsStaged) {
    const dim3 threads(tileSize, threadRows);
    // Each round closes the diagonal, updates its two panels, then updates
    // every other tile from the corresponding parts of those panels.
    for (int k = 0; k < tiles; ++k) {
        const int ownerRow = k % grid.rows;
        const int ownerCol = k % grid.cols;
        const int pivotRow = k / grid.rows;
        const int pivotCol = k / grid.cols;
        const bool ownsRow = grid.row == ownerRow;
        const bool ownsCol = grid.col == ownerCol;
        const bool ownsPivot = ownsRow && ownsCol;
        Distance* pivot = diagonal.device;
        if (ownsPivot) {
            pivot = tileAt(matrix, pivotRow, pivotCol, grid.localCols);
            closePivot<<<1, threads, 0, stream>>>(pivot);
        }

        // Only ranks updating a pivot panel need the diagonal tile.
        exchange({{
            {ownsRow && grid.cols > 1 ? grid.rowComm : MPI_COMM_NULL,
             ownerCol, ownsPivot, pivot, diagonal.host, tileElements},
            {ownsCol && grid.rows > 1 ? grid.colComm : MPI_COMM_NULL,
             ownerRow, ownsPivot, pivot, diagonal.host, tileElements}
        }}, stream, diagonalStaged);

        const int rowTiles = ownsRow ? grid.localCols : 0;
        const int colTiles = ownsCol ? grid.localRows : 0;
        if (rowTiles + colTiles) {
            updatePanels<<<rowTiles + colTiles, threads, 0, stream>>>(
                matrix, pivot, column.device, grid.localCols, pivotRow, pivotCol, rowTiles, ownsPivot);
        }

        Distance* rowPanel = ownsRow ? tileAt(matrix, pivotRow, 0, grid.localCols) : row.device;
        Distance* colPanel = grid.cols > 1 ? column.device
                                           : tileAt(matrix, 0, pivotCol, grid.localCols);
        exchange({{
            {grid.rows > 1 ? grid.colComm : MPI_COMM_NULL, ownerRow, ownsRow,
             rowPanel, row.host, static_cast<size_t>(grid.localCols) * tileElements},
            {grid.cols > 1 ? grid.rowComm : MPI_COMM_NULL, ownerCol, ownsCol,
             colPanel, column.host, static_cast<size_t>(grid.localRows) * tileElements}
        }}, stream, panelsStaged);

        const int interiorRows = grid.localRows - ownsRow;
        const int interiorCols = grid.localCols - ownsCol;
        if (interiorRows && interiorCols) {
            updateInterior<<<static_cast<unsigned int>(static_cast<size_t>(interiorRows)
                                                       * interiorCols), threads, 0, stream>>>(
                matrix, rowPanel, colPanel, grid.cols > 1 ? 1 : grid.localCols,
                grid.localCols, interiorCols, ownsRow ? pivotRow : -1, ownsCol ? pivotCol : -1);
        }
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaStreamSynchronize(stream));
}

int localTiles(int tiles, int processes, int coordinate) {
    return (tiles - 1 - coordinate) / processes + 1;
}

void unpackMatrix(const Distance* matrix, int nodes, const Grid& grid, Distance* result) {
    const int threads = hostThreads(grid.elements());
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int localRow = 0; localRow < grid.localRows * tileSize; ++localRow) {
        const int row = (localRow / tileSize * grid.rows + grid.row) * tileSize
                        + localRow % tileSize;
        if (row >= nodes) {
            continue;
        }
        for (int block = 0; block < grid.localCols; ++block) {
            const int col = (block * grid.cols + grid.col) * tileSize;
            const auto* src = matrix
                + (static_cast<size_t>(localRow / tileSize) * grid.localCols + block)
                    * tileElements + localRow % tileSize * tileSize;
            std::copy_n(src, std::min(tileSize, nodes - col),
                        result + static_cast<size_t>(row) * nodes + col);
        }
    }
}

std::vector<Distance> collectMatrix(const Buffer& matrix, int nodes, int tiles,
                                    const Grid& grid, int rank, MPI_Comm comm) {
    // Rank zero can reuse a single maximum-sized receive buffer for all ranks.
    std::unique_ptr<Distance[]> packed(new Distance[grid.elements()]);
    if (grid.elements()) {
        checkCuda(cudaMemcpy(packed.get(), matrix.device, grid.elements() * sizeof(Distance),
                             cudaMemcpyDeviceToHost));
    }

    if (rank != 0) {
        for (size_t offset = 0; offset < grid.elements(); offset += INT_MAX) {
            const int count = static_cast<int>(std::min<size_t>(INT_MAX, grid.elements() - offset));
            MPI_Send(packed.get() + offset, count, MPI_UNSIGNED, 0, 0, comm);
        }
        return {};
    }

    std::vector<Distance> result(static_cast<size_t>(nodes) * nodes);
    unpackMatrix(packed.get(), nodes, grid, result.data());
    for (int source = 1; source < grid.rows * grid.cols; ++source) {
        Grid sourceGrid = grid;
        sourceGrid.row = source / grid.cols;
        sourceGrid.col = source % grid.cols;
        sourceGrid.localRows = localTiles(tiles, grid.rows, sourceGrid.row);
        sourceGrid.localCols = localTiles(tiles, grid.cols, sourceGrid.col);
        for (size_t offset = 0; offset < sourceGrid.elements(); offset += INT_MAX) {
            const int count = static_cast<int>(
                std::min<size_t>(INT_MAX, sourceGrid.elements() - offset));
            MPI_Recv(packed.get() + offset, count, MPI_UNSIGNED, source, 0, comm, MPI_STATUS_IGNORE);
        }
        unpackMatrix(packed.get(), nodes, sourceGrid, result.data());
    }
    return result;
}

bool validateResult(const std::vector<Distance>& dist, int nodes) {
    for (int i = 0; i < nodes; ++i) {
        if (dist[static_cast<size_t>(i) * nodes + i] != 0) {
            printf("Validation failed: diagonal element [%d,%d] is not zero\n", i, i);
            return false;
        }
    }

    const int sample = std::min(nodes, 10);
    for (int i = 0; i < sample; ++i) {
        for (int j = 0; j < sample; ++j) {
            const auto* row = dist.data() + static_cast<size_t>(i) * nodes;
            for (int k = 0; k < nodes; ++k) {
                const Distance left = row[k];
                const Distance right = dist[static_cast<size_t>(k) * nodes + j];
                if (left < INF && right < INF && left + right < row[j]) {
                    printf("Validation failed: triangle inequality violated at [%d,%d,%d]\n",
                           i, j, k);
                    return false;
                }
            }
        }
    }
    return true;
}

void selectDevice(MPI_Comm comm) {
    MPI_Comm local;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);

    int devices;
    checkCuda(cudaGetDeviceCount(&devices));
    if (devices == 0) {
        fprintf(stderr, "No CUDA device is visible to an active MPI rank\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    checkCuda(cudaSetDevice(localRank % devices));
}

int runBenchmark(int nodes, int tiles, bool validate, bool printResults, int rank,
                 const Grid& grid, MPI_Comm comm) {
    selectDevice(comm);
    bool pinned = grid.rows > 1 || grid.cols > 1;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
    pinned = pinned && !MPIX_Query_cuda_support();
#endif

    cudaStream_t stream;
    checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t diagonalStaged = nullptr;
    cudaEvent_t panelsStaged = nullptr;
    if (pinned) {
        checkCuda(cudaEventCreateWithFlags(&diagonalStaged, cudaEventDisableTiming));
        checkCuda(cudaEventCreateWithFlags(&panelsStaged, cudaEventDisableTiming));
    }
    Buffer matrix(grid.elements());
    Buffer diagonal(grid.rows > 1 || grid.cols > 1 ? tileElements : 0, pinned);
    Buffer row(grid.rows > 1 ? static_cast<size_t>(grid.localCols) * tileElements : 0, pinned);
    Buffer column(grid.cols > 1 ? static_cast<size_t>(grid.localRows) * tileElements : 0, pinned);
    {
        std::unique_ptr<Distance[]> initial(new Distance[grid.elements()]);
        initializeMatrix(initial.get(), nodes, grid);
        if (grid.elements()) {
            checkCuda(cudaMemcpyAsync(matrix.device, initial.get(),
                                      grid.elements() * sizeof(Distance),
                                      cudaMemcpyHostToDevice, stream));
            checkCuda(cudaStreamSynchronize(stream));
        }
    }

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }
    MPI_Barrier(comm);
    const double start = MPI_Wtime();
    floydWarshall(matrix.device, tiles, grid, stream, diagonal, row, column,
                  diagonalStaged, panelsStaged);
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (pinned) {
        checkCuda(cudaEventDestroy(diagonalStaged));
        checkCuda(cudaEventDestroy(panelsStaged));
    }
    checkCuda(cudaStreamDestroy(stream));

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        const double ops = static_cast<double>(nodes) * nodes * nodes;
        printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
    }
    if (printResults || validate) {
        auto result = collectMatrix(matrix, nodes, tiles, grid, rank, comm);
        if (rank == 0) {
            if (printResults) {
                print_results_int(result, "DistanceMatrix");
            }
            if (validate) {
                printf("Validating result...\n");
                const bool valid = validateResult(result, nodes);
                printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                return valid ? EXIT_SUCCESS : EXIT_FAILURE;
            }
        }
    }
    return EXIT_SUCCESS;
}

int benchmarkMain(int argc, char** argv) {
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int nodes = 512;
    bool validate = false;
    bool printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const long count = strtol(value, &end, 10);
            if (errno || end == value || *end || count < 0 || count > INT_MAX - tileSize) {
                if (rank == 0) {
                    fprintf(stderr, "Invalid node count: %s\n", value);
                    printUsage(argv[0]);
                }
                return EXIT_FAILURE;
            }
            nodes = static_cast<int>(count);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return EXIT_SUCCESS;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return EXIT_FAILURE;
        }
    }

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %d\n", nodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    const int tiles = (nodes + tileSize - 1) / tileSize;
    const int extent = std::max(1, tiles);
    int activeRanks = static_cast<int>(std::min<int64_t>(ranks, int64_t{extent} * extent));
    int dims[2] = {0, 0};
    // Each active rank must own at least one tile for a nonempty graph.
    for (;; --activeRanks) {
        dims[0] = dims[1] = 0;
        MPI_Dims_create(activeRanks, 2, dims);
        if (dims[0] <= extent && dims[1] <= extent) {
            break;
        }
    }

    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &comm);
    int status = EXIT_SUCCESS;
    if (comm != MPI_COMM_NULL) {
        const int row = rank / dims[1];
        const int col = rank % dims[1];
        Grid grid{dims[0], dims[1], row, col,
                  tiles ? localTiles(tiles, dims[0], row) : 0,
                  tiles ? localTiles(tiles, dims[1], col) : 0, MPI_COMM_NULL, MPI_COMM_NULL};
        if (grid.cols > 1) {
            MPI_Comm_split(comm, row, col, &grid.rowComm);
        }
        if (grid.rows > 1) {
            MPI_Comm_split(comm, col, row, &grid.colComm);
        }
        if (static_cast<size_t>(grid.localRows) * grid.localCols > INT_MAX) {
            fprintf(stderr, "Rank %d: local tile count exceeds the CUDA grid limit\n", rank);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        status = runBenchmark(nodes, tiles, validate, printResults, rank, grid, comm);
        if (grid.rowComm != MPI_COMM_NULL) {
            MPI_Comm_free(&grid.rowComm);
        }
        if (grid.colComm != MPI_COMM_NULL) {
            MPI_Comm_free(&grid.colComm);
        }
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

} // namespace

int main(int argc, char** argv) {
    int provided;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        return EXIT_FAILURE;
    }
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI must provide MPI_THREAD_FUNNELED for OpenMP host work\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    int status = EXIT_FAILURE;
    try {
        status = benchmarkMain(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    MPI_Finalize();
    return status;
}
