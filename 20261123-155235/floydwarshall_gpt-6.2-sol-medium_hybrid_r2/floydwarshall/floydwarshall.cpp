#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
// The generated weights fit in a byte and distances only decrease. Widen
// before adding so a path involving a padded vertex cannot wrap around.
using Distance = unsigned char;
constexpr Distance PaddingDistance = UCHAR_MAX;
static_assert(MAX_DISTANCE + 1 < PaddingDistance);
constexpr int Tile = 32;
constexpr int ThreadRows = 8;
constexpr size_t TileElements = Tile * Tile;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
        std::abort();
    }
}

#define CUDA_CHECK(call) checkCuda((call), #call)

enum class Phase { Diagonal, Row, Column };

__global__ void initializePaths(unsigned int* path, size_t elements, int tileCols,
                                int gridRows, int rankRow) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
         index += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const size_t tileRow = index / TileElements / tileCols;
        path[index] = (tileRow * gridRows + rankRow) * Tile + index % TileElements / Tile;
    }
}

// Record the pivot row and column as they were at each scalar iteration.
// Finalized panels give the same distances but can change the recorded pivots.
template <Phase phase>
__global__ void updatePanel(Distance* dist, unsigned int* path, int tileCols,
                            int pivotRow, int pivotCol, unsigned int firstPivot,
                            const Distance* diagonal, Distance* rowHistory,
                            Distance* colHistory, Distance* diagonalRowSlot = nullptr,
                            Distance* diagonalColSlot = nullptr) {
    const int tileRow = phase == Phase::Column ? blockIdx.x : pivotRow;
    const int tileCol = phase == Phase::Row ? blockIdx.x : pivotCol;
    if ((phase == Phase::Row && tileCol == pivotCol) ||
        (phase == Phase::Column && tileRow == pivotRow)) {
        return;
    }
    const size_t offset = (static_cast<size_t>(tileRow) * tileCols + tileCol) * TileElements;
    __shared__ unsigned int tile[Tile][Tile + 1];
    __shared__ unsigned int pivotValues[Tile][Tile];
    const int x = threadIdx.x;
    const int firstRow = threadIdx.y;
    unsigned int via[Tile / ThreadRows];
    #pragma unroll
    for (int r = 0; r < Tile / ThreadRows; ++r) {
        const int y = firstRow + r * ThreadRows;
        tile[y][x] = dist[offset + y * Tile + x];
        via[r] = UINT_MAX;
        if constexpr (phase != Phase::Diagonal) {
            pivotValues[y][x] = diagonal[y * Tile + x];
        }
    }
    __syncthreads();

    #pragma unroll 1
    for (int k = 0; k < Tile; ++k) {
        if (firstRow == 0) {
            if constexpr (phase != Phase::Column) {
                const size_t base = phase == Phase::Diagonal ? 0 : tileCol * TileElements;
                const Distance value = tile[k][x];
                rowHistory[base + k * Tile + x] = value;
                if constexpr (phase == Phase::Diagonal) {
                    if (diagonalRowSlot) {
                        diagonalRowSlot[k * Tile + x] = value;
                    }
                }
            }
            if constexpr (phase != Phase::Row) {
                const size_t base = phase == Phase::Diagonal ? 0 : tileRow * TileElements;
                const Distance value = tile[x][k];
                colHistory[base + k * Tile + x] = value;
                if constexpr (phase == Phase::Diagonal) {
                    if (diagonalColSlot) {
                        diagonalColSlot[k * Tile + x] = value;
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < Tile / ThreadRows; ++r) {
            const int y = firstRow + r * ThreadRows;
            // The zero diagonal keeps the current pivot operands unchanged.
            if ((phase == Phase::Column || y != k) && (phase == Phase::Row || x != k)) {
                const unsigned int left = phase == Phase::Row ? pivotValues[k][y] : tile[y][k];
                const unsigned int right = phase == Phase::Column ? pivotValues[k][x] : tile[k][x];
                const unsigned int candidate = left + right;
                if (candidate < tile[y][x]) {
                    tile[y][x] = candidate;
                    via[r] = firstPivot + k;
                }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int r = 0; r < Tile / ThreadRows; ++r) {
        const int y = firstRow + r * ThreadRows;
        if (via[r] != UINT_MAX) {
            dist[offset + y * Tile + x] = tile[y][x];
            path[offset + y * Tile + x] = via[r];
        }
    }
}

__global__ void updateRemaining(Distance* dist, unsigned int* path, int tileCols,
                                int pivotRow, int pivotCol, unsigned int firstPivot,
                                const Distance* rowHistory, const Distance* colHistory) {
    const int tileRow = blockIdx.y + (pivotRow >= 0 && blockIdx.y >= static_cast<unsigned>(pivotRow));
    const int tileCol = blockIdx.x + (pivotCol >= 0 && blockIdx.x >= static_cast<unsigned>(pivotCol));
    const size_t offset = (static_cast<size_t>(tileRow) * tileCols + tileCol) * TileElements;
    const size_t rowOffset = tileCol * TileElements;
    const size_t colOffset = tileRow * TileElements;
    const int x = threadIdx.x;
    const int firstRow = threadIdx.y;
    __shared__ unsigned int rows[Tile][Tile];
    __shared__ unsigned int cols[Tile][Tile];
    unsigned int best[Tile / ThreadRows];
    unsigned int via[Tile / ThreadRows];
    #pragma unroll
    for (int r = 0; r < Tile / ThreadRows; ++r) {
        const int y = firstRow + r * ThreadRows;
        rows[y][x] = rowHistory[rowOffset + y * Tile + x];
        cols[y][x] = colHistory[colOffset + y * Tile + x];
        best[r] = dist[offset + y * Tile + x];
        via[r] = UINT_MAX;
    }
    __syncthreads();

    #pragma unroll
    for (int k = 0; k < Tile; ++k) {
        const unsigned int right = rows[k][x];
        #pragma unroll
        for (int r = 0; r < Tile / ThreadRows; ++r) {
            const int y = firstRow + r * ThreadRows;
            const unsigned int candidate = cols[k][y] + right;
            if (candidate < best[r]) {
                best[r] = candidate;
                via[r] = firstPivot + k;
            }
        }
    }

    #pragma unroll
    for (int r = 0; r < Tile / ThreadRows; ++r) {
        const int y = firstRow + r * ThreadRows;
        if (via[r] != UINT_MAX) {
            dist[offset + y * Tile + x] = best[r];
            path[offset + y * Tile + x] = via[r];
        }
    }
}

int localTileCount(int tiles, int gridSize, int coordinate) {
    return (tiles + gridSize - 1 - coordinate) / gridSize;
}

int hostThreads(size_t elements) {
    const size_t work = std::max<size_t>(1, (elements + 32767) / 32768);
    return static_cast<int>(std::min<size_t>(omp_get_max_threads(), work));
}

#ifdef __GLIBC__
struct SeedJump {
    unsigned int multiplier = 1;
    unsigned int increment = 0;

    unsigned int apply(unsigned int seed) const {
        return multiplier * seed + increment;
    }
};

SeedJump seedJump(size_t samples) {
    // glibc's rand_r takes three 32-bit linear congruential steps per sample.
    SeedJump result;
    SeedJump power{1103515245u, 12345u};
    for (size_t steps = samples * 3; steps != 0; steps >>= 1) {
        if (steps & 1) {
            result = {power.multiplier * result.multiplier, power.apply(result.increment)};
        }
        power = {power.multiplier * power.multiplier, power.apply(power.increment)};
    }
    return result;
}
#endif

void initializeDistances(Distance* dist, size_t n, int tileRows, int tileCols,
                         const int* grid, const int* coord) {
    const size_t localRows = static_cast<size_t>(tileRows) * Tile;
    const size_t firstCol = static_cast<size_t>(coord[1]) * Tile;
    const size_t columnGap = static_cast<size_t>(grid[1] - 1) * Tile;
#ifdef __GLIBC__
    const SeedJump skipColumns = seedJump(columnGap);
#else
    // Other C libraries can implement rand_r with a different recurrence.
    std::vector<unsigned int> rowSeeds(n);
    unsigned int rowSeed = 42;
    for (size_t row = 0; row < n; ++row) {
        rowSeeds[row] = rowSeed;
        for (size_t col = 0; col < n; ++col) {
            rand_r(&rowSeed);
        }
    }
#endif
    #pragma omp parallel for schedule(static) num_threads(hostThreads(localRows * tileCols * Tile))
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t row = (localRow / Tile * grid[0] + coord[0]) * Tile + localRow % Tile;
        unsigned int seed = 42;
        if (row < n && firstCol < n) {
#ifdef __GLIBC__
            seed = seedJump(row * n + firstCol).apply(seed);
#else
            seed = rowSeeds[row];
            for (size_t i = 0; i < firstCol; ++i) {
                rand_r(&seed);
            }
#endif
        }
        for (int tileCol = 0; tileCol < tileCols; ++tileCol) {
            const size_t colStart = firstCol + static_cast<size_t>(tileCol) * grid[1] * Tile;
            const size_t offset = (localRow / Tile * tileCols + tileCol) *
                                  TileElements + localRow % Tile * Tile;
            for (int x = 0; x < Tile; ++x) {
                const size_t col = colStart + x;
                Distance weight = PaddingDistance;
                if (row < n && col < n) {
                    weight = 1 + static_cast<Distance>(
                        static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
                }
                dist[offset + x] = row == col ? 0 : weight;
            }
            if (row < n && colStart + Tile + columnGap < n) {
#ifdef __GLIBC__
                seed = skipColumns.apply(seed);
#else
                for (size_t i = 0; i < columnGap; ++i) {
                    rand_r(&seed);
                }
#endif
            }
        }
    }
}

struct Panel {
    Distance* device = nullptr;
    Distance* host = nullptr;
    cudaEvent_t received = nullptr;
    int count;
    int peers;
    int coordinate;
    MPI_Comm communicator;

    Panel(int count, int peers, int coordinate, MPI_Comm communicator)
        : count(count), peers(peers), coordinate(coordinate), communicator(communicator) {
        const size_t bytes = static_cast<size_t>(count) * sizeof(Distance);
        CUDA_CHECK(cudaMalloc(&device, bytes));
        if (peers > 1) {
            CUDA_CHECK(cudaMallocHost(&host, bytes));
            CUDA_CHECK(cudaEventCreateWithFlags(&received, cudaEventDisableTiming | cudaEventBlockingSync));
        }
    }

    ~Panel() {
        if (host) {
            CUDA_CHECK(cudaEventDestroy(received));
            CUDA_CHECK(cudaFreeHost(host));
        }
        CUDA_CHECK(cudaFree(device));
    }

    Panel(const Panel&) = delete;
    Panel& operator=(const Panel&) = delete;
};

void exchangePanels(Panel& a, int rootA, Panel& b, int rootB, cudaStream_t stream) {
    Panel* panels[] = {&a, &b};
    const int roots[] = {rootA, rootB};
    MPI_Request requests[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
    bool copying = false;
    for (int i = 0; i < 2; ++i) {
        const Panel& panel = *panels[i];
        if (roots[i] < 0 || panel.peers == 1) {
            continue;
        }
        if (panel.coordinate == roots[i]) {
            CUDA_CHECK(cudaMemcpyAsync(panel.host, panel.device, panel.count * sizeof(Distance),
                                       cudaMemcpyDeviceToHost, stream));
            copying = true;
        } else {
            // MPI may reuse this buffer once the earlier device upload has finished.
            CUDA_CHECK(cudaEventSynchronize(panel.received));
            MPI_Ibcast(panel.host, panel.count, MPI_UNSIGNED_CHAR, roots[i], panel.communicator, &requests[i]);
        }
    }
    if (copying) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (int i = 0; i < 2; ++i) {
            const Panel& panel = *panels[i];
            if (roots[i] >= 0 && panel.peers > 1 && panel.coordinate == roots[i]) {
                MPI_Ibcast(panel.host, panel.count, MPI_UNSIGNED_CHAR, roots[i], panel.communicator, &requests[i]);
            }
        }
    }
    if (requests[0] == MPI_REQUEST_NULL && requests[1] == MPI_REQUEST_NULL) {
        return;
    }
    MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);
    for (int i = 0; i < 2; ++i) {
        const Panel& panel = *panels[i];
        if (roots[i] >= 0 && panel.peers > 1 && panel.coordinate != roots[i]) {
            CUDA_CHECK(cudaMemcpyAsync(panel.device, panel.host, panel.count * sizeof(Distance),
                                       cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaEventRecord(panel.received, stream));
        }
    }
}

double floydWarshall(Distance* dist, unsigned int* path, int tiles, int tileRows, int tileCols,
                     const int* grid, const int* coord, MPI_Comm active, MPI_Comm row,
                     MPI_Comm col, cudaStream_t stream) {
    // Row histories feed the ranks below and above their producer in the
    // process column. Column histories travel across the process row.
    Panel diagRows(TileElements, grid[0], coord[0], col);
    Panel diagCols(TileElements, grid[1], coord[1], row);
    Panel rowHistory(tileCols * TileElements, grid[0], coord[0], col);
    Panel colHistory(tileRows * TileElements, grid[1], coord[1], row);
    const dim3 threads(Tile, ThreadRows);

    MPI_Barrier(active);
    const auto start = std::chrono::steady_clock::now();
    for (int pivot = 0; pivot < tiles; ++pivot) {
        const int ownerRow = pivot % grid[0];
        const int ownerCol = pivot % grid[1];
        const int pivotRow = coord[0] == ownerRow ? pivot / grid[0] : -1;
        const int pivotCol = coord[1] == ownerCol ? pivot / grid[1] : -1;
        const unsigned int firstPivot = static_cast<unsigned int>(pivot) * Tile;

        if (pivotRow >= 0 && pivotCol >= 0) {
            // A broadcast includes the diagonal slot even though those ranks
            // already computed the diagonal tile or one of its panels.
            auto* rowSlot = grid[0] > 1 ? rowHistory.device + pivotCol * TileElements : nullptr;
            auto* colSlot = grid[1] > 1 ? colHistory.device + pivotRow * TileElements : nullptr;
            updatePanel<Phase::Diagonal><<<1, threads, 0, stream>>>(
                dist, path, tileCols, pivotRow, pivotCol, firstPivot,
                nullptr, diagRows.device, diagCols.device, rowSlot, colSlot);
        }
        exchangePanels(diagRows, pivotCol >= 0 ? ownerRow : -1,
                       diagCols, pivotRow >= 0 ? ownerCol : -1, stream);

        if (pivotRow >= 0 && tileCols > (pivotCol >= 0)) {
            updatePanel<Phase::Row><<<tileCols, threads, 0, stream>>>(
                dist, path, tileCols, pivotRow, pivotCol, firstPivot,
                diagCols.device, rowHistory.device, nullptr);
        }
        if (pivotCol >= 0 && tileRows > (pivotRow >= 0)) {
            updatePanel<Phase::Column><<<tileRows, threads, 0, stream>>>(
                dist, path, tileCols, pivotRow, pivotCol, firstPivot,
                diagRows.device, nullptr, colHistory.device);
        }
        exchangePanels(rowHistory, ownerRow, colHistory, ownerCol, stream);

        const int remainingRows = tileRows - (pivotRow >= 0);
        const int remainingCols = tileCols - (pivotCol >= 0);
        if (remainingRows > 0 && remainingCols > 0) {
            updateRemaining<<<dim3(remainingCols, remainingRows), threads, 0, stream>>>(
                dist, path, tileCols, pivotRow, pivotCol, firstPivot,
                rowHistory.device, colHistory.device);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

void unpackDistances(const Distance* local, std::vector<unsigned int>& dist, size_t n,
                     int tileRows, int tileCols, const int* grid, const int* coord) {
    const size_t localRows = static_cast<size_t>(tileRows) * Tile;
    #pragma omp parallel for schedule(static) num_threads(hostThreads(localRows * tileCols * Tile))
    for (size_t localRow = 0; localRow < localRows; ++localRow) {
        const size_t row = (localRow / Tile * grid[0] + coord[0]) * Tile + localRow % Tile;
        if (row >= n) {
            continue;
        }
        for (int tileCol = 0; tileCol < tileCols; ++tileCol) {
            const size_t col = (static_cast<size_t>(tileCol) * grid[1] + coord[1]) * Tile;
            if (col < n) {
                const size_t offset = (localRow / Tile * tileCols + tileCol) *
                                      TileElements + localRow % Tile * Tile;
                std::copy_n(local + offset, std::min<size_t>(Tile, n - col), dist.data() + row * n + col);
            }
        }
    }
}

std::vector<unsigned int> gatherDistances(const Distance* deviceDist, size_t n, int tiles,
                                          int tileRows, int tileCols, const int* grid,
                                          const int* coord, MPI_Comm active, int rank) {
    const size_t elements = static_cast<size_t>(tileRows) * tileCols * TileElements;
    std::vector<Distance> local(elements);
    CUDA_CHECK(cudaMemcpy(local.data(), deviceDist, elements * sizeof(Distance), cudaMemcpyDeviceToHost));
    if (rank != 0) {
        for (size_t offset = 0; offset < elements;) {
            const int count = std::min<size_t>(elements - offset, INT_MAX);
            MPI_Send(local.data() + offset, count, MPI_UNSIGNED_CHAR, 0, 0, active);
            offset += count;
        }
        return {};
    }

    std::vector<unsigned int> dist(n * n);
    unpackDistances(local.data(), dist, n, tileRows, tileCols, grid, coord);
    // Rank zero has the largest local tile rectangle, so reuse its receive buffer.
    for (int peer = 1; peer < grid[0] * grid[1]; ++peer) {
        const int peerCoord[2] = {peer / grid[1], peer % grid[1]};
        const int peerRows = localTileCount(tiles, grid[0], peerCoord[0]);
        const int peerCols = localTileCount(tiles, grid[1], peerCoord[1]);
        const size_t peerElements = static_cast<size_t>(peerRows) * peerCols * TileElements;
        for (size_t offset = 0; offset < peerElements;) {
            const int count = std::min<size_t>(peerElements - offset, INT_MAX);
            MPI_Recv(local.data() + offset, count, MPI_UNSIGNED_CHAR, peer, 0, active, MPI_STATUS_IGNORE);
            offset += count;
        }
        unpackDistances(local.data(), dist, n, peerRows, peerCols, grid, peerCoord);
    }
    return dist;
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

int runBenchmark(int argc, char** argv, int rank, int worldSize) {
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const long nodes = strtol(argv[++i], nullptr, 10);
            // Each panel is broadcast with an MPI int count.
            if (nodes < 0 || nodes > INT_MAX / Tile - Tile) {
                if (rank == 0) {
                    fprintf(stderr, "Number of nodes is out of range: %s\n", argv[i]);
                }
                return 1;
            }
            numNodes = static_cast<size_t>(nodes);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Assign tiles cyclically in both dimensions and omit empty process rows
    // and columns for graphs smaller than the requested process grid.
    const int tiles = (numNodes + Tile - 1) / Tile;
    int grid[2] = {0, 0};
    MPI_Dims_create(worldSize, 2, grid);
    grid[0] = std::min(grid[0], std::max(1, tiles));
    grid[1] = std::min(grid[1], std::max(1, tiles));

    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < grid[0] * grid[1] ? 0 : MPI_UNDEFINED, rank, &active);
    std::vector<unsigned int> result;
    if (active != MPI_COMM_NULL) {
        const int coord[2] = {rank / grid[1], rank % grid[1]};
        MPI_Comm row, col, node;
        MPI_Comm_split(active, coord[0], coord[1], &row);
        MPI_Comm_split(active, coord[1], coord[0], &col);
        MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_free(&node);

        int devices;
        CUDA_CHECK(cudaGetDeviceCount(&devices));
        if (devices == 0) {
            fprintf(stderr, "Rank %d: no CUDA device is visible\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
            return 1;
        }
        CUDA_CHECK(cudaSetDevice(localRank % devices));

        const int tileRows = localTileCount(tiles, grid[0], coord[0]);
        const int tileCols = localTileCount(tiles, grid[1], coord[1]);
        const size_t elements = static_cast<size_t>(tileRows) * tileCols * TileElements;
        Distance* dist = nullptr;
        unsigned int* path = nullptr;
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (elements) {
            const size_t bytes = elements * sizeof(Distance);
            Distance* initial;
            CUDA_CHECK(cudaMallocHost(&initial, bytes));
            initializeDistances(initial, numNodes, tileRows, tileCols, grid, coord);
            CUDA_CHECK(cudaMalloc(&dist, bytes));
            CUDA_CHECK(cudaMalloc(&path, elements * sizeof(unsigned int)));
            CUDA_CHECK(cudaMemcpyAsync(dist, initial, bytes, cudaMemcpyHostToDevice, stream));
            const int blocks = std::min<size_t>((elements + 255) / 256, 65535);
            initializePaths<<<blocks, 256, 0, stream>>>(path, elements, tileCols, grid[0], coord[0]);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFreeHost(initial));
        }

        if (rank == 0) {
            printf("Computing shortest paths...\n");
        }
        const double elapsed = tiles
            ? floydWarshall(dist, path, tiles, tileRows, tileCols, grid, coord, active, row, col, stream)
            : 0.0;
        double seconds = 0;
        MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, active);

        if (rank == 0) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / (seconds * 1e9) : 0.0);
        }
        if (tiles && (printResults || validate)) {
            result = gatherDistances(dist, numNodes, tiles, tileRows, tileCols, grid, coord, active, rank);
        }
        CUDA_CHECK(cudaFree(path));
        CUDA_CHECK(cudaFree(dist));
        CUDA_CHECK(cudaStreamDestroy(stream));
        MPI_Comm_free(&col);
        MPI_Comm_free(&row);
        MPI_Comm_free(&active);
    }

    int status = 0;
    if (rank == 0) {
        if (printResults) {
            print_results_int(result, "DistanceMatrix");
        }
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(result, numNodes);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            status = valid ? 0 : 1;
        }
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) {
            fprintf(stderr, "MPI does not provide MPI_THREAD_FUNNELED\n");
        }
        MPI_Finalize();
        return 1;
    }

    int status = 1;
    try {
        status = runBenchmark(argc, argv, rank, worldSize);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
