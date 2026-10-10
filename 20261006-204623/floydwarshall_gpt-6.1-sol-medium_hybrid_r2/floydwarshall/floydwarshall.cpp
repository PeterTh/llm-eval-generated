#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cerrno>
#include <climits>
#include <exception>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, 
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < numNodes * numNodes; ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = 0; i < numNodes; ++i) {
        dist[idx2(i, i, numNodes)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes) {
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}

// Rows are distributed in whole tiles. A panel contains the next TILE pivots.
// Snapshot each pivot row at its original scalar iteration, rather than using
// its final blocked value: this also preserves the original path tie semantics.
constexpr int TILE = 32;
constexpr int THREADS = 256;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Portable staging works with ordinary MPI, without requiring CUDA-aware MPI.
void broadcastPanel(unsigned int* panel, size_t count, int owner) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(panel + offset, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        offset += chunk;
    }
}

__global__ void pivotSnapshots(const unsigned int* panel, unsigned int* snapshots,
                               size_t n, size_t first, int width) {
    __shared__ unsigned int pivot[TILE * TILE];
    __shared__ unsigned int values[TILE * TILE];
    const size_t column = size_t(blockIdx.x) * TILE;
    const int tid = threadIdx.x;
    for (int cell = tid; cell < TILE * TILE; cell += THREADS) {
        int row = cell / TILE, col = cell % TILE;
        pivot[cell] = row < width && col < width
            ? panel[size_t(row) * n + first + col] : INF;
        values[cell] = row < width && column + col < n
            ? panel[size_t(row) * n + column + col] : INF;
    }
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        if (tid < TILE && column + tid < n)
            snapshots[size_t(k) * n + column + tid] = values[k * TILE + tid];
        unsigned int nextPivot[TILE * TILE / THREADS];
        unsigned int nextValues[TILE * TILE / THREADS];
        #pragma unroll
        for (int item = 0; item < TILE * TILE / THREADS; ++item) {
            int cell = tid + item * THREADS;
            int row = cell / TILE, col = cell % TILE;
            nextPivot[item] = min(pivot[cell], pivot[row * TILE + k] + pivot[k * TILE + col]);
            nextValues[item] = min(values[cell], pivot[row * TILE + k] + values[k * TILE + col]);
        }
        __syncthreads();
        #pragma unroll
        for (int item = 0; item < TILE * TILE / THREADS; ++item) {
            int cell = tid + item * THREADS;
            pivot[cell] = nextPivot[item];
            values[cell] = nextValues[item];
        }
        __syncthreads();
    }
}

__global__ void updateRows(unsigned int* dist, unsigned int* path,
                           const unsigned int* links, const unsigned int* snapshots,
                           size_t n, size_t rows, size_t first, int width,
                           size_t tileOffset, size_t skipTile) {
    __shared__ unsigned int right[TILE * TILE];
    __shared__ unsigned int pivot[TILE * TILE];
    size_t rowTile = size_t(blockIdx.y) + tileOffset;
    if (rowTile >= skipTile) ++rowTile;
    const size_t rowStart = rowTile * TILE;
    const size_t columnStart = size_t(blockIdx.x) * TILE;
    const int tid = threadIdx.x;
    unsigned int left[TILE * TILE / THREADS];
    unsigned int distance[TILE * TILE / THREADS];
    unsigned int intermediate[TILE * TILE / THREADS];
    #pragma unroll
    for (int item = 0; item < TILE * TILE / THREADS; ++item) {
        int cell = tid + item * THREADS;
        int row = cell / TILE, col = cell % TILE;
        left[item] = rowStart + row < rows && col < width
            ? links[(rowStart + row) * TILE + col] : INF;
        right[cell] = row < width && columnStart + col < n
            ? snapshots[size_t(row) * n + columnStart + col] : INF;
        pivot[cell] = row < width && col < width
            ? snapshots[size_t(row) * n + first + col] : INF;
        bool valid = rowStart + row < rows && columnStart + col < n;
        size_t index = (rowStart + row) * n + columnStart + col;
        distance[item] = valid ? dist[index] : INF;
        intermediate[item] = valid ? path[index] : 0;
    }
    __syncthreads();
    // Each warp owns a row, with four independent rows per warp. Keep
    // row-to-pivot distances in registers and broadcast d[i,k] with a shuffle;
    // immutable pivot snapshots require only the initial block barrier.
    for (int k = 0; k < width; ++k) {
        #pragma unroll
        for (int item = 0; item < TILE * TILE / THREADS; ++item) {
            int col = tid % TILE;
            unsigned int toPivot = __shfl_sync(0xffffffffu, left[item], k);
            unsigned int candidate = toPivot + right[k * TILE + col];
            if (candidate < distance[item]) {
                distance[item] = candidate;
                intermediate[item] = static_cast<unsigned int>(first + k);
            }
            left[item] = min(left[item], toPivot + pivot[k * TILE + col]);
        }
    }
    #pragma unroll
    for (int item = 0; item < TILE * TILE / THREADS; ++item) {
        int cell = tid + item * THREADS;
        size_t row = rowStart + cell / TILE, col = columnStart + cell % TILE;
        if (row < rows && col < n) {
            dist[row * n + col] = distance[item];
            path[row * n + col] = intermediate[item];
        }
    }
}

size_t tileBegin(size_t tiles, int rank, int ranks) {
    return tiles / ranks * rank + std::min(size_t(rank), tiles % ranks);
}

void transferRows(unsigned int* buffer, size_t count, int peer, bool send) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, size_t(INT_MAX)));
        if (send) mpiCheck(MPI_Send(buffer + offset, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD));
        else mpiCheck(MPI_Recv(buffer + offset, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        offset += chunk;
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path, size_t n,
                   int rank, int ranks, size_t begin, size_t rows) {
    if (n == 0) return;
    unsigned int *deviceDist = nullptr, *devicePath = nullptr;
    unsigned int *panel = nullptr, *snapshots = nullptr, *links = nullptr, *hostPanel = nullptr;
    const size_t bytes = rows * n * sizeof(unsigned int);
    if (rows) {
        CUDA_CHECK(cudaMalloc(&deviceDist, bytes));
        CUDA_CHECK(cudaMalloc(&devicePath, bytes));
        CUDA_CHECK(cudaMalloc(&links, rows * TILE * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(deviceDist, dist.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(devicePath, path.data(), bytes, cudaMemcpyHostToDevice));
    }
    if (ranks > 1) CUDA_CHECK(cudaMallocHost(&hostPanel, TILE * n * sizeof(unsigned int)));
    if (rows) {
        CUDA_CHECK(cudaMalloc(&panel, TILE * n * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&snapshots, TILE * n * sizeof(unsigned int)));
    }
    cudaStream_t computeStream, transferStream;
    cudaEvent_t pivotReady, panelConsumed;
    CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&transferStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&pivotReady, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&panelConsumed, cudaEventDisableTiming));
    const size_t tiles = (n + TILE - 1) / TILE;
    const size_t localTiles = (rows + TILE - 1) / TILE;
    int owner = 0;
    for (size_t block = 0; block < tiles; ++block) {
        while (block >= tileBegin(tiles, owner + 1, ranks)) ++owner;
        const size_t first = block * TILE;
        const int width = static_cast<int>(std::min(size_t(TILE), n - first));
        const size_t panelBytes = size_t(width) * n * sizeof(unsigned int);
        if (ranks > 1) {
            // Reuse staging only after the preceding asynchronous upload.
            if (block > 0 && rows) CUDA_CHECK(cudaEventSynchronize(panelConsumed));
            if (rank == owner) {
                if (block > 0) CUDA_CHECK(cudaStreamWaitEvent(transferStream, pivotReady, 0));
                CUDA_CHECK(cudaMemcpyAsync(hostPanel, deviceDist + (first - begin) * n,
                                           panelBytes, cudaMemcpyDeviceToHost, transferStream));
                CUDA_CHECK(cudaStreamSynchronize(transferStream));
            }
            // The owner finishes its next pivot rows first. Staging and MPI can
            // therefore overlap the remaining row tiles on every accelerator.
            broadcastPanel(hostPanel, size_t(width) * n, owner);
        }
        if (!rows) continue;
        if (rank == owner)
            CUDA_CHECK(cudaMemcpyAsync(panel, deviceDist + (first - begin) * n,
                                       panelBytes, cudaMemcpyDeviceToDevice, computeStream));
        else CUDA_CHECK(cudaMemcpyAsync(panel, hostPanel, panelBytes, cudaMemcpyHostToDevice, computeStream));
        CUDA_CHECK(cudaEventRecord(panelConsumed, computeStream));
        // Freeze row-to-pivot links before any output tile overwrites them.
        CUDA_CHECK(cudaMemcpy2DAsync(links, TILE * sizeof(unsigned int), deviceDist + first,
                                    n * sizeof(unsigned int), width * sizeof(unsigned int),
                                    rows, cudaMemcpyDeviceToDevice, computeStream));
        pivotSnapshots<<<static_cast<unsigned int>(tiles), THREADS, 0, computeStream>>>(panel, snapshots, n, first, width);
        CUDA_CHECK(cudaGetLastError());
        size_t priorityTile = localTiles;
        if (ranks > 1 && block + 1 < tiles) {
            int nextOwner = owner;
            while (block + 1 >= tileBegin(tiles, nextOwner + 1, ranks)) ++nextOwner;
            if (rank == nextOwner) {
                priorityTile = ((block + 1) * TILE - begin) / TILE;
                dim3 priorityGrid(static_cast<unsigned int>(tiles), 1);
                updateRows<<<priorityGrid, THREADS, 0, computeStream>>>(
                    deviceDist, devicePath, links, snapshots, n, rows, first, width,
                    priorityTile, localTiles);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaEventRecord(pivotReady, computeStream));
            }
        }
        const size_t remainingTiles = localTiles - (priorityTile < localTiles ? 1 : 0);
        if (remainingTiles) {
            dim3 grid(static_cast<unsigned int>(tiles), static_cast<unsigned int>(remainingTiles));
            updateRows<<<grid, THREADS, 0, computeStream>>>(
                deviceDist, devicePath, links, snapshots, n, rows, first, width,
                0, priorityTile);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(computeStream));
    if (rows) {
        CUDA_CHECK(cudaMemcpy(dist.data(), deviceDist, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), devicePath, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventDestroy(pivotReady));
    CUDA_CHECK(cudaEventDestroy(panelConsumed));
    CUDA_CHECK(cudaStreamDestroy(computeStream));
    CUDA_CHECK(cudaStreamDestroy(transferStream));
    CUDA_CHECK(cudaFree(deviceDist));
    CUDA_CHECK(cudaFree(devicePath));
    CUDA_CHECK(cudaFree(links));
    CUDA_CHECK(cudaFree(panel));
    CUDA_CHECK(cudaFree(snapshots));
    if (hostPanel) CUDA_CHECK(cudaFreeHost(hostPanel));
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

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    int argumentStatus = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            const char* value = argv[++i];
            char* end = nullptr;
            errno = 0;
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end || parsed > UINT_MAX)
                argumentStatus = 1;
            else numNodes = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) argumentStatus = 2;
        else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            argumentStatus = 1;
        }
    }
    if (argumentStatus) {
        if (rank == 0) printUsage(argv[0]);
        MPI_Finalize();
        return argumentStatus == 2 ? 0 : 1;
    }
    if (numNodes > 0 && numNodes > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / numNodes) {
        if (rank == 0) fprintf(stderr, "Matrix size overflows address space\n");
        MPI_Finalize();
        return 1;
    }
    // One rank per visible accelerator; local ranks also work when a scheduler
    // exposes only the assigned GPU to each process.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank = 0, deviceCount = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "Rank %d requires a CUDA accelerator\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));
    mpiCheck(MPI_Comm_free(&localComm));
    const size_t tiles = (numNodes + TILE - 1) / TILE;
    const size_t begin = std::min(numNodes, tileBegin(tiles, rank, ranks) * TILE);
    const size_t end = std::min(numNodes, tileBegin(tiles, rank + 1, ranks) * TILE);
    const size_t rows = end - begin;
    int status = 0;
    try {
        std::vector<unsigned int> full;
        std::vector<unsigned int> dist(rows * numNodes), path(rows * numNodes);
        if (rank == 0) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("MPI ranks: %d; OpenMP threads: %d; CUDA tile: %d\n", ranks, omp_get_max_threads(), TILE);
            printf("Initializing graph...\n");
            full.resize(numNodes * numNodes);
            initializeDistanceMatrix(full, numNodes, 1, MAX_DISTANCE);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < dist.size(); ++i) dist[i] = full[i];
            for (int peer = 1; peer < ranks; ++peer) {
                size_t peerBegin = std::min(numNodes, tileBegin(tiles, peer, ranks) * TILE);
                size_t peerEnd = std::min(numNodes, tileBegin(tiles, peer + 1, ranks) * TILE);
                if (peerEnd > peerBegin)
                    transferRows(full.data() + peerBegin * numNodes, (peerEnd - peerBegin) * numNodes, peer, true);
            }
            if (!validate && !printResults) std::vector<unsigned int>().swap(full);
        } else if (rows) transferRows(dist.data(), dist.size(), 0, false);
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < numNodes; ++col)
                path[row * numNodes + col] = static_cast<unsigned int>(col);
        if (rank == 0) printf("Computing shortest paths...\n");
        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        double start = MPI_Wtime();
        floydWarshall(dist, path, numNodes, rank, ranks, begin, rows);
        double elapsed = MPI_Wtime() - start, seconds = 0;
        mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        if (rank == 0) {
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0);
        }
        if (validate || printResults) {
            if (rank == 0) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < dist.size(); ++i) full[i] = dist[i];
                for (int peer = 1; peer < ranks; ++peer) {
                    size_t peerBegin = std::min(numNodes, tileBegin(tiles, peer, ranks) * TILE);
                    size_t peerEnd = std::min(numNodes, tileBegin(tiles, peer + 1, ranks) * TILE);
                    if (peerEnd > peerBegin)
                        transferRows(full.data() + peerBegin * numNodes, (peerEnd - peerBegin) * numNodes, peer, false);
                }
                if (printResults) print_results_int(full, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    status = validateResult(full, numNodes) ? 0 : 1;
                    printf("Validation: %s\n", status == 0 ? "PASSED" : "FAILED");
                }
            } else if (rows) transferRows(dist.data(), dist.size(), 0, true);
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return status;
}
