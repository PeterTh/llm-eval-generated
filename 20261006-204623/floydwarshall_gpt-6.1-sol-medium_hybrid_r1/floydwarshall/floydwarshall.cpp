#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;
constexpr int THREAD_ROWS = 8;
constexpr int ROWS_PER_THREAD = TILE / THREAD_ROWS;

inline constexpr size_t idx2(size_t i, size_t j, size_t n) noexcept {
    return j * n + i;
}

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI error: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: CUDA error: %s\n", rank, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Tile-aligned row stripes keep each diagonal tile on a single rank.
static int firstTile(int rank, int ranks, int tiles) {
    return static_cast<int>(static_cast<long long>(rank) * tiles / ranks);
}

#ifdef __GLIBC__
// glibc rand_r advances its 32-bit LCG three times per returned value.
// Jump to the seed for a row without generating any preceding rows. Unsigned
// wraparound is intentional and exactly matches rand_r's state arithmetic.
static unsigned int rowSeed(size_t values) {
    unsigned int mul = 1103515245u, add = 12345u, seed = 42u;
    size_t steps = values * 3;
    while (steps) {
        if (steps & 1) seed = mul * seed + add;
        add *= mul + 1u;
        mul *= mul;
        steps >>= 1;
    }
    return seed;
}
#endif

static void initializeStripe(unsigned int* matrix, size_t n, size_t pitch,
                             size_t firstRow, size_t rows, int rank, int ranks) {
#ifndef __GLIBC__
    // Preserve the native rand_r sequence on other libc implementations too.
    // Only seed metadata is passed along the rank chain, never a full matrix.
    unsigned int seed = 42u;
    if (rank) mpiCheck(MPI_Recv(&seed, 1, MPI_UNSIGNED, rank - 1, 10,
                               MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    std::vector<unsigned int> seeds(rows);
    for (size_t row = 0; row < rows; ++row) {
        seeds[row] = seed;
        if (firstRow + row < n)
            for (size_t j = 0; j < n; ++j) (void)rand_r(&seed);
    }
    if (rank + 1 < ranks)
        mpiCheck(MPI_Send(&seed, 1, MPI_UNSIGNED, rank + 1, 10, MPI_COMM_WORLD));
#else
    (void)rank;
    (void)ranks;
#endif
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < rows; ++row) {
        const size_t globalRow = firstRow + row;
#ifdef __GLIBC__
        unsigned int seed = rowSeed(globalRow * n);
#else
        unsigned int seed = seeds[row];
#endif
        unsigned int* output = matrix + row * pitch;
        if (globalRow < n) {
            for (size_t j = 0; j < n; ++j)
                output[j] = 1u + static_cast<unsigned int>(
                    static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
            std::fill(output + n, output + pitch, INF);
        } else {
            std::fill(output, output + pitch, INF);
        }
        output[globalRow] = 0;
    }
}

// Phase 1: close the diagonal tile. Barriers separate all reads and writes
// for each k, including the pivot row/column (no shared-memory data races).
__global__ void closeDiagonal(unsigned int* panel, size_t pitch, int tile) {
    __shared__ unsigned int diagonal[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int row = y + q * THREAD_ROWS;
        diagonal[row][x] = panel[row * pitch + tile * TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        unsigned int value[ROWS_PER_THREAD];
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int row = y + q * THREAD_ROWS;
            value[q] = min(diagonal[row][x], diagonal[row][k] + diagonal[k][x]);
        }
        __syncthreads();
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q)
            diagonal[y + q * THREAD_ROWS][x] = value[q];
        __syncthreads();
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int row = y + q * THREAD_ROWS;
        panel[row * pitch + tile * TILE + x] = diagonal[row][x];
    }
}

// Phase 2: multiply a panel tile by the closed diagonal in the min-plus
// semiring. Every operand is a read-only shared-memory snapshot.
template<bool ROW_PANEL>
__global__ void updatePanel(unsigned int* matrix, const unsigned int* panel,
                            size_t pitch, int pivot, int first) {
    const int tile = blockIdx.x;
    if ((ROW_PANEL ? tile : tile + first) == pivot) return;
    __shared__ unsigned int diagonal[TILE][TILE];
    __shared__ unsigned int target[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t baseRow = ROW_PANEL ? 0 : static_cast<size_t>(tile) * TILE;
    const size_t baseCol = static_cast<size_t>(ROW_PANEL ? tile : pivot) * TILE;
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int row = y + q * THREAD_ROWS;
        diagonal[row][x] = panel[row * pitch + pivot * TILE + x];
        target[row][x] = matrix[(baseRow + row) * pitch + baseCol + x];
    }
    __syncthreads();
    unsigned int value[ROWS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) value[q] = target[y + q * THREAD_ROWS][x];
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q) {
            const int row = y + q * THREAD_ROWS;
            const unsigned int candidate = ROW_PANEL
                ? diagonal[row][k] + target[k][x]
                : target[row][k] + diagonal[k][x];
            value[q] = min(value[q], candidate);
        }
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q)
        matrix[(baseRow + y + q * THREAD_ROWS) * pitch + baseCol + x] = value[q];
}

// Phase 3: the O(n^3) work. Reuse both panels in shared memory and keep four
// output cells per thread in registers; consecutive lanes access consecutive
// columns. Flatten the tile grid to avoid CUDA's small grid.y limit.
__global__ void updateTiles(unsigned int* matrix, const unsigned int* panel,
                            size_t pitch, int tiles, int pivot, int first) {
    const int tileRow = blockIdx.x / tiles;
    const int tileCol = blockIdx.x % tiles;
    if (tileRow + first == pivot || tileCol == pivot) return;
    __shared__ unsigned int left[TILE][TILE];
    __shared__ unsigned int right[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t baseRow = static_cast<size_t>(tileRow) * TILE;
    const size_t baseCol = static_cast<size_t>(tileCol) * TILE;
    unsigned int value[ROWS_PER_THREAD];
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q) {
        const int row = y + q * THREAD_ROWS;
        left[row][x] = matrix[(baseRow + row) * pitch + pivot * TILE + x];
        right[row][x] = panel[row * pitch + baseCol + x];
        value[q] = matrix[(baseRow + row) * pitch + baseCol + x];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        const unsigned int rightValue = right[k][x];
        #pragma unroll
        for (int q = 0; q < ROWS_PER_THREAD; ++q)
            value[q] = min(value[q], left[y + q * THREAD_ROWS][k] + rightValue);
    }
    #pragma unroll
    for (int q = 0; q < ROWS_PER_THREAD; ++q)
        matrix[(baseRow + y + q * THREAD_ROWS) * pitch + baseCol + x] = value[q];
}

static void floydWarshall(unsigned int* matrix, unsigned int* panel,
                          unsigned int* hostPanel, size_t pitch, int tiles,
                          int first, int localTiles, int rank, int ranks,
                          cudaStream_t stream) {
    const dim3 threads(TILE, THREAD_ROWS);
    const size_t panelElements = TILE * pitch;
    cudaEvent_t stagingConsumed = nullptr;
    bool copyPending = false;
    if (ranks > 1)
        cudaCheck(cudaEventCreateWithFlags(&stagingConsumed, cudaEventDisableTiming));
    for (int pivot = 0; pivot < tiles; ++pivot) {
        // The next broadcast may overlap the previous iteration's GPU work.
        // Only wait for the copy that consumed pinned staging, not all kernels.
        if (copyPending) {
            cudaCheck(cudaEventSynchronize(stagingConsumed));
            copyPending = false;
        }
        // Inverse of floor(rank * tiles / ranks), including empty stripes.
        const int owner = static_cast<int>((static_cast<long long>(pivot + 1) * ranks - 1) / tiles);
        if (rank == owner) {
            cudaCheck(cudaMemcpyAsync(panel, matrix + static_cast<size_t>(pivot - first) * TILE * pitch,
                                      panelElements * sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream));
            closeDiagonal<<<1, threads, 0, stream>>>(panel, pitch, pivot);
            updatePanel<true><<<tiles, threads, 0, stream>>>(panel, panel, pitch, pivot, 0);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpyAsync(matrix + static_cast<size_t>(pivot - first) * TILE * pitch,
                                      panel, panelElements * sizeof(unsigned int), cudaMemcpyDeviceToDevice, stream));
            if (ranks > 1) {
                cudaCheck(cudaMemcpyAsync(hostPanel, panel, panelElements * sizeof(unsigned int),
                                          cudaMemcpyDeviceToHost, stream));
                cudaCheck(cudaStreamSynchronize(stream));
            }
        }
        if (ranks > 1) {
            // Pinned staging works with any MPI, without a CUDA-aware MPI
            // dependency. Overlap the owner's column work with the broadcast.
            MPI_Request request;
            mpiCheck(MPI_Ibcast(hostPanel, static_cast<int>(panelElements), MPI_UNSIGNED,
                                owner, MPI_COMM_WORLD, &request));
            if (rank == owner && localTiles)
                updatePanel<false><<<localTiles, threads, 0, stream>>>(matrix, panel, pitch, pivot, first);
            mpiCheck(MPI_Wait(&request, MPI_STATUS_IGNORE));
            if (rank != owner) {
                cudaCheck(cudaMemcpyAsync(panel, hostPanel, panelElements * sizeof(unsigned int),
                                          cudaMemcpyHostToDevice, stream));
                cudaCheck(cudaEventRecord(stagingConsumed, stream));
                copyPending = true;
            }
        }
        if (localTiles) {
            if (ranks == 1 || rank != owner)
                updatePanel<false><<<localTiles, threads, 0, stream>>>(matrix, panel, pitch, pivot, first);
            updateTiles<<<static_cast<unsigned int>(static_cast<size_t>(localTiles) * tiles), threads, 0, stream>>>(
                matrix, panel, pitch, tiles, pivot, first);
            cudaCheck(cudaGetLastError());
        }
    }
    cudaCheck(cudaStreamSynchronize(stream));
    if (stagingConsumed) cudaCheck(cudaEventDestroy(stagingConsumed));
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
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) fprintf(stderr, "MPI_THREAD_FUNNELED is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    int status = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* argument = argv[++i];
            const unsigned long long value = strtoull(argument, &end, 10);
            // Panel broadcasts use an MPI int count; bound it before padding.
            if (errno || argument[0] == '-' || end == argument || *end ||
                value > static_cast<unsigned long long>(INT_MAX / TILE - TILE)) {
                if (!rank) fprintf(stderr, "Invalid or unsupported node count: %s\n", argument);
                status = 1;
                break;
            }
            numNodes = static_cast<size_t>(value);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            mpiCheck(MPI_Finalize());
            return 0;
        } else {
            if (!rank) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            status = 1;
            break;
        }
    }
    if (status) {
        mpiCheck(MPI_Finalize());
        return status;
    }

    try {
        MPI_Comm localComm;
        mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                                     MPI_INFO_NULL, &localComm));
        int localRank = 0;
        mpiCheck(MPI_Comm_rank(localComm, &localRank));
        int devices = 0;
        cudaCheck(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Rank %d: a CUDA GPU is required.\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Respect CUDA_VISIBLE_DEVICES; with several visible devices use one
        // per node-local rank. Launch one MPI rank per GPU for full utilization.
        cudaCheck(cudaSetDevice(localRank % devices));
        mpiCheck(MPI_Comm_free(&localComm));
        const size_t pitch = (numNodes + TILE - 1) / TILE * TILE;
        const int tiles = static_cast<int>(pitch / TILE);
        const int first = firstTile(rank, ranks, tiles);
        const int localTiles = firstTile(rank + 1, ranks, tiles) - first;
        const size_t rows = static_cast<size_t>(localTiles) * TILE;
        if ((pitch && rows > std::numeric_limits<size_t>::max() / pitch / sizeof(unsigned int)) ||
            static_cast<size_t>(localTiles) * tiles > INT_MAX ||
            (numNodes && numNodes > std::numeric_limits<size_t>::max() / numNodes / sizeof(unsigned int))) {
            if (!rank) fprintf(stderr, "Matrix exceeds supported address or launch limits.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        const size_t elements = rows * pitch;
        const size_t bytes = elements * sizeof(unsigned int);
        std::vector<unsigned int> host(elements);
        unsigned int *matrix = nullptr, *panel = nullptr, *hostPanel = nullptr;
        cudaStream_t stream;
        cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        if (elements) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&matrix), bytes));
        if (tiles) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&panel), TILE * pitch * sizeof(unsigned int)));
            if (ranks > 1)
                cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostPanel), TILE * pitch * sizeof(unsigned int)));
        }
        if (!rank) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Parallelism: %d MPI ranks, OpenMP, CUDA (32 x 32 tiles)\n", ranks);
            printf("Initializing graph...\n");
        }
        initializeStripe(host.data(), numNodes, pitch, static_cast<size_t>(first) * TILE,
                         rows, rank, ranks);
        if (elements) cudaCheck(cudaMemcpyAsync(matrix, host.data(), bytes, cudaMemcpyHostToDevice, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        if (!rank) printf("Computing shortest paths...\n");
        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        const double start = MPI_Wtime();
        // The original path matrix was never read or output. Its dead writes
        // are eliminated; the observable distance matrix is unchanged.
        floydWarshall(matrix, panel, hostPanel, pitch, tiles, first, localTiles, rank, ranks, stream);
        const double elapsed = MPI_Wtime() - start;
        double seconds = 0;
        mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
        if (!rank) {
            printf("Computation time: %.3f ms\n", seconds * 1000.0);
            const double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
        }
        // Keep the matrix distributed unless output or validation needs it.
        if (validate || printResults) {
            if (elements) cudaCheck(cudaMemcpyAsync(host.data(), matrix, bytes, cudaMemcpyDeviceToHost, stream));
            cudaCheck(cudaStreamSynchronize(stream));
            const size_t firstRow = std::min(static_cast<size_t>(first) * TILE, numNodes);
            const size_t validRows = std::min(rows, numNodes - firstRow);
            // Compact away padded columns in parallel. A separate buffer
            // prevents overlapping row copies from racing.
            std::vector<unsigned int> packed(validRows * numNodes);
            #pragma omp parallel for schedule(static)
            for (size_t row = 0; row < validRows; ++row)
                std::copy_n(host.data() + row * pitch, numNodes, packed.data() + row * numNodes);
            std::vector<unsigned int> dist;
            if (!rank) dist.resize(numNodes * numNodes);
            // Chunked gather supports matrices larger than MPI's int count.
            // Only the root holds the complete unpadded result.
            constexpr size_t CHUNK = 1u << 26;
            if (!rank) {
                std::copy(packed.begin(), packed.end(), dist.begin());
                for (int source = 1; source < ranks; ++source) {
                    const size_t begin = std::min(static_cast<size_t>(firstTile(source, ranks, tiles)) * TILE, numNodes) * numNodes;
                    const size_t end = std::min(static_cast<size_t>(firstTile(source + 1, ranks, tiles)) * TILE, numNodes) * numNodes;
                    for (size_t offset = begin; offset < end; offset += CHUNK) {
                        const int count = static_cast<int>(std::min(CHUNK, end - offset));
                        mpiCheck(MPI_Recv(dist.data() + offset, count, MPI_UNSIGNED, source, 20,
                                          MPI_COMM_WORLD, MPI_STATUS_IGNORE));
                    }
                }
            } else {
                for (size_t offset = 0; offset < packed.size(); offset += CHUNK) {
                    const int count = static_cast<int>(std::min(CHUNK, packed.size() - offset));
                    mpiCheck(MPI_Send(packed.data() + offset, count, MPI_UNSIGNED, 0, 20, MPI_COMM_WORLD));
                }
            }
            if (!rank) {
                if (printResults) print_results_int(dist, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    status = validateResult(dist, numNodes) ? 0 : 1;
                    printf("Validation: %s\n", status ? "FAILED" : "PASSED");
                }
            }
        }
        mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
        if (hostPanel) cudaCheck(cudaFreeHost(hostPanel));
        if (panel) cudaCheck(cudaFree(panel));
        if (matrix) cudaCheck(cudaFree(matrix));
        cudaCheck(cudaStreamDestroy(stream));
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return status;
}
