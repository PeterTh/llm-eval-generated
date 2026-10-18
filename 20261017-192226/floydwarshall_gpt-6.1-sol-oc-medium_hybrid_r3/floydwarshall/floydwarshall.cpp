#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <exception>
#include <limits>
#include <vector>
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
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              unsigned int& seed, const size_t firstRow) {
    if (numNodes == 0) return;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    
    for (size_t i = 0; i < dist.size(); ++i) {
        dist[i] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
    }
    
    // Set diagonal to zero (distance from node to itself is 0)
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dist.size() / numNodes; ++i) {
        dist[idx2(firstRow + i, i, numNodes)] = 0;
    }
}

namespace {
constexpr int TILE = 32;
constexpr int THREADS = 256;

void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s: %s\n", rank, operation,
                cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA_CHECK(call) cudaCheck((call), #call)

// Each rank owns contiguous complete tile rows. All device indexing uses size_t
// so matrices larger than 2^32 elements are supported.
__global__ void closePivot(unsigned int* matrix, size_t stride, int localPivot,
                           int pivot) {
    __shared__ unsigned int tile[TILE * TILE];
    const int t = threadIdx.x;
    for (int x = t; x < TILE * TILE; x += THREADS)
        tile[x] = matrix[(size_t(localPivot) * TILE + x / TILE) * stride
                         + size_t(pivot) * TILE + x % TILE];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        // Row/column k never change (nonnegative weights, zero diagonal).
        // Excluding them also prevents shared-memory read/write races.
        for (int x = t; x < TILE * TILE; x += THREADS) {
            const int i = x / TILE, j = x % TILE;
            if (i != k && j != k)
                tile[x] = min(tile[x], tile[i * TILE + k] + tile[k * TILE + j]);
        }
        __syncthreads();
    }
    for (int x = t; x < TILE * TILE; x += THREADS)
        matrix[(size_t(localPivot) * TILE + x / TILE) * stride
               + size_t(pivot) * TILE + x % TILE] = tile[x];
}

// Phase 2: update either a pivot-row tile (on its owner) or a pivot-column
// tile (on every rank). The closed diagonal tile is replicated on each GPU.
template<bool ROW>
__global__ void updatePanel(unsigned int* matrix, const unsigned int* diagonal,
                            size_t stride, int firstTile, int pivot) {
    const int tileIndex = blockIdx.x;
    const int globalTile = ROW ? tileIndex : firstTile + tileIndex;
    if (globalTile == pivot) return;
    const int tileRow = ROW ? pivot - firstTile : tileIndex;
    const int tileCol = ROW ? tileIndex : pivot;
    __shared__ unsigned int diag[TILE * TILE];
    __shared__ unsigned int work[TILE * TILE];
    const int t = threadIdx.x;
    for (int x = t; x < TILE * TILE; x += THREADS) {
        diag[x] = diagonal[x];
        work[x] = matrix[(size_t(tileRow) * TILE + x / TILE) * stride
                         + size_t(tileCol) * TILE + x % TILE];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int x = t; x < TILE * TILE; x += THREADS) {
            const int i = x / TILE, j = x % TILE;
            if (ROW) {
                if (i != k)
                    work[x] = min(work[x], diag[i * TILE + k] + work[k * TILE + j]);
            } else {
                if (j != k)
                    work[x] = min(work[x], work[i * TILE + k] + diag[k * TILE + j]);
            }
        }
        __syncthreads();
    }
    for (int x = t; x < TILE * TILE; x += THREADS)
        matrix[(size_t(tileRow) * TILE + x / TILE) * stride
               + size_t(tileCol) * TILE + x % TILE] = work[x];
}

// Phase 3 does the cubic work: coalesced loads/stores, two shared tiles and
// four register-resident outputs per thread. No per-k global synchronization.
__global__ void updateRemaining(unsigned int* matrix, const unsigned int* panel,
                                size_t stride, int firstTile, int pivot,
                                int tiles, int localTiles) {
    const size_t tileId = size_t(blockIdx.x) + size_t(blockIdx.y) * gridDim.x;
    const int tileRow = tileId / tiles, tileCol = tileId % tiles;
    if (tileRow >= localTiles || tileCol == pivot || firstTile + tileRow == pivot)
        return;
    __shared__ unsigned int column[TILE * TILE];
    __shared__ unsigned int row[TILE * TILE];
    unsigned int value[TILE * TILE / THREADS];
    const int t = threadIdx.x;
    #pragma unroll
    for (int q = 0; q < TILE * TILE / THREADS; ++q) {
        const int x = t + q * THREADS;
        const size_t offset = (size_t(tileRow) * TILE + x / TILE) * stride;
        column[x] = matrix[offset + size_t(pivot) * TILE + x % TILE];
        row[x] = panel[size_t(x / TILE) * stride + size_t(tileCol) * TILE + x % TILE];
        value[q] = matrix[offset + size_t(tileCol) * TILE + x % TILE];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int q = 0; q < TILE * TILE / THREADS; ++q) {
            const int x = t + q * THREADS;
            value[q] = min(value[q], column[(x / TILE) * TILE + k] + row[k * TILE + x % TILE]);
        }
    }
    #pragma unroll
    for (int q = 0; q < TILE * TILE / THREADS; ++q) {
        const int x = t + q * THREADS;
        matrix[(size_t(tileRow) * TILE + x / TILE) * stride
               + size_t(tileCol) * TILE + x % TILE] = value[q];
    }
}

// Chunk MPI operations to avoid the legacy MPI int-count limit. Host staging
// works with ordinary MPI installations as well as CUDA-aware implementations.
void broadcast(unsigned int* data, size_t count, int root, MPI_Comm comm) {
    while (count) {
        const int chunk = int(std::min(count, size_t(INT_MAX)));
        MPI_Bcast(data, chunk, MPI_UNSIGNED, root, comm);
        data += chunk;
        count -= chunk;
    }
}

void transfer(unsigned int* data, size_t count, int peer, bool send, MPI_Comm comm) {
    while (count) {
        const int chunk = int(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, comm);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

// Only distances are observable in this benchmark; the original unused path
// matrix is eliminated to save device memory and memory bandwidth.
double floydWarshall(std::vector<unsigned int>& dist, size_t n, bool gather,
                     MPI_Comm comm) {
    int rank, ranks;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    const int tiles = std::max(1, int((n + TILE - 1) / TILE));
    const size_t stride = size_t(tiles) * TILE;
    const int first = int(int64_t(tiles) * rank / ranks);
    const int last = int(int64_t(tiles) * (rank + 1) / ranks);
    const int localTiles = last - first;
    const size_t rows = std::min(n, size_t(last) * TILE) - size_t(first) * TILE;

    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&node);

    std::vector<unsigned int> packed(rows * n);
    if (rank == 0) {
        // Stream the original rand_r sequence in rank order. This preserves
        // every input bit without replicating a full graph on the root.
        unsigned int seed = 42;
        initializeDistanceMatrix(packed, n, 1, MAX_DISTANCE, seed, 0);
        for (int r = 1; r < ranks; ++r) {
            const size_t begin = size_t(int64_t(tiles) * r / ranks) * TILE;
            const size_t end = std::min(n, size_t(int64_t(tiles) * (r + 1) / ranks) * TILE);
            std::vector<unsigned int> chunk((end - begin) * n);
            initializeDistanceMatrix(chunk, n, 1, MAX_DISTANCE, seed, begin);
            transfer(chunk.data(), chunk.size(), r, true, comm);
        }
    } else {
        transfer(packed.data(), packed.size(), 0, false, comm);
    }
    const size_t elements = size_t(localTiles) * TILE * stride;
    unsigned int *host, *hostPanel, *device, *panel, *diagonal;
    CUDA_CHECK(cudaMallocHost(&host, elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hostPanel, TILE * stride * sizeof(unsigned int)));
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < size_t(localTiles) * TILE; ++i) {
        unsigned int* dest = host + i * stride;
        std::fill_n(dest, stride, INF);
        if (i < rows) std::copy_n(packed.data() + i * n, n, dest);
        else dest[size_t(first) * TILE + i] = 0;
    }
    std::vector<unsigned int>().swap(packed);
    CUDA_CHECK(cudaMalloc(&device, elements * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&panel, TILE * stride * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&diagonal, TILE * TILE * sizeof(unsigned int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMemcpyAsync(device, host, elements * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    MPI_Barrier(comm);
    if (rank == 0) printf("Computing shortest paths...\n");
    const double start = MPI_Wtime();
    for (int p = 0; p < tiles; ++p) {
        // Inverse of floor(tiles * rank / ranks), including unequal partitions.
        const int owner = int((int64_t(p + 1) * ranks - 1) / tiles);
        if (rank == owner) {
            closePivot<<<1, THREADS, 0, stream>>>(device, stride, p - first, p);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy2DAsync(diagonal, TILE * sizeof(unsigned int),
                device + size_t(p - first) * TILE * stride + size_t(p) * TILE,
                stride * sizeof(unsigned int), TILE * sizeof(unsigned int), TILE,
                cudaMemcpyDeviceToDevice, stream));
            updatePanel<true><<<tiles, THREADS, 0, stream>>>(device, diagonal, stride, first, p);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(ranks == 1 ? panel : hostPanel,
                device + size_t(p - first) * TILE * stride,
                TILE * stride * sizeof(unsigned int),
                ranks == 1 ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost, stream));
            if (ranks > 1) CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        // One panel broadcast per block of 32 intermediates; the closed
        // diagonal is embedded in that panel, so it needs no extra collective.
        if (ranks > 1) {
            broadcast(hostPanel, TILE * stride, owner, comm);
            CUDA_CHECK(cudaMemcpyAsync(panel, hostPanel, TILE * stride * sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
            if (rank != owner)
                CUDA_CHECK(cudaMemcpy2DAsync(diagonal, TILE * sizeof(unsigned int),
                    panel + size_t(p) * TILE, stride * sizeof(unsigned int),
                    TILE * sizeof(unsigned int), TILE, cudaMemcpyDeviceToDevice, stream));
        }
        updatePanel<false><<<localTiles, THREADS, 0, stream>>>(device, diagonal, stride, first, p);
        CUDA_CHECK(cudaGetLastError());
        const size_t blocks = size_t(localTiles) * tiles;
        const unsigned int gridX = unsigned(std::min(blocks, size_t(65535)));
        const unsigned int gridY = unsigned((blocks + gridX - 1) / gridX);
        updateRemaining<<<dim3(gridX, gridY), THREADS, 0, stream>>>(device, panel, stride, first, p, tiles, localTiles);
        CUDA_CHECK(cudaGetLastError());
        if (ranks > 1) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds;
    MPI_Allreduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
    if (gather) {
        CUDA_CHECK(cudaMemcpyAsync(host, device, elements * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        packed.resize(rows * n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < rows; ++i)
            std::copy_n(host + i * stride, n, packed.data() + i * n);
        if (rank == 0) {
            dist.resize(n * n);
            std::copy(packed.begin(), packed.end(), dist.begin());
            for (int r = 1; r < ranks; ++r) {
                const size_t begin = size_t(int64_t(tiles) * r / ranks) * TILE;
                const size_t end = std::min(n, size_t(int64_t(tiles) * (r + 1) / ranks) * TILE);
                transfer(dist.data() + begin * n, (end - begin) * n, r, false, comm);
            }
        } else transfer(packed.data(), packed.size(), 0, true, comm);
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(diagonal));
    CUDA_CHECK(cudaFree(panel));
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(hostPanel));
    CUDA_CHECK(cudaFreeHost(host));
    return seconds;
}
} // namespace

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
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* value = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || *value == '-' || end == value || *end ||
                parsed > size_t(INT_MAX) - TILE ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(unsigned int) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid number of nodes: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            numNodes = size_t(parsed);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    const int activeRanks = std::min(ranks, std::max(1, int((numNodes + TILE - 1) / TILE)));
    MPI_Comm active;
    MPI_Comm_split(MPI_COMM_WORLD, rank < activeRanks ? 0 : MPI_UNDEFINED, rank, &active);
    int status = 0;
    try {
        if (rank < activeRanks) {
            std::vector<unsigned int> dist;
            if (rank == 0) {
                printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
                printf("Number of nodes: %zu\n", numNodes);
                printf("Validation: %s\n", validate ? "enabled" : "disabled");
                printf("MPI ranks: %d; OpenMP threads per rank: %d; CUDA tile: %d\n",
                       activeRanks, omp_get_max_threads(), TILE);
                printf("Initializing graph...\n");
            }
            const double seconds = floydWarshall(dist, numNodes, validate || printResults, active);
            if (rank == 0) {
                printf("Computation time: %.3f ms\n", seconds * 1000.0);
                const double ops = double(numNodes) * numNodes * numNodes;
                printf("Performance: %.3f GOPS\n", ops / std::max(seconds, 1e-9) / 1e9);
    
                // Print results for external validation (integer hash-based).
                if (printResults) print_results_int(dist, "DistanceMatrix");
    
                // Validation
                if (validate) {
                    printf("Validating result...\n");
                    const bool valid = validateResult(dist, numNodes);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                    status = valid ? 0 : 1;
                }
            }
            MPI_Comm_free(&active);
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
