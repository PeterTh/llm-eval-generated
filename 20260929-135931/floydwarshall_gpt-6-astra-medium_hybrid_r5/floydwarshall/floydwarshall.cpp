#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;
constexpr int TILE = 32;
using Value = unsigned int;
inline size_t idx2(size_t i, size_t j, size_t n) { return j * n + i; }

static void cudaCheck(cudaError_t error, const char* where) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA(call) cudaCheck((call), #call)
static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, error);
}

// Save the pivot column as it existed at each scalar Floyd-Warshall step.
__global__ void diagonalSnapshots(const Value* matrix, Value* columns,
                                  size_t width, size_t row, size_t pivot) {
    __shared__ Value tile[TILE][TILE];
    int x = threadIdx.x;
    for (int y = threadIdx.y; y < TILE; y += blockDim.y)
        tile[y][x] = matrix[(row + y) * width + pivot + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        Value column[4], candidate[4];
        for (int t = 0; t < 4; ++t) {
            int y = threadIdx.y + t * 8;
            column[t] = tile[y][k];
            candidate[t] = column[t] + tile[k][x];
            if (x == 0) columns[k * TILE + y] = column[t];
        }
        __syncthreads();
        for (int t = 0; t < 4; ++t) {
            int y = threadIdx.y + t * 8;
            tile[y][x] = min(tile[y][x], candidate[t]);
        }
        __syncthreads();
    }
}

// Build a panel of pivot ROW snapshots, not the fully relaxed pivot rows.
// This retains strict-improvement/tie semantics of the original k loop.
__global__ void panelSnapshots(const Value* matrix, const Value* columns,
                               Value* panel, size_t width, size_t row) {
    __shared__ Value tile[TILE][TILE];
    __shared__ Value col[TILE][TILE];
    int x = threadIdx.x;
    size_t j = size_t(blockIdx.x) * TILE + x;
    for (int y = threadIdx.y; y < TILE; y += 8) {
        tile[y][x] = matrix[(row + y) * width + j];
        col[y][x] = columns[y * TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        Value pivot = tile[k][x];
        if (threadIdx.y == 0) panel[size_t(k) * width + j] = pivot;
        __syncthreads();
        for (int y = threadIdx.y; y < TILE; y += 8)
            tile[y][x] = min(tile[y][x], col[k][y] + pivot);
        __syncthreads();
    }
}

__global__ void relaxPanel(Value* matrix, Value* path, const Value* panel,
                           const Value* pivotColumns, size_t width, size_t pivot, size_t tileCount) {
    __shared__ Value right[TILE][TILE], square[TILE][TILE];
    int x = threadIdx.x;
    // A linear grid avoids CUDA's small grid.y limit on large matrices.
    size_t row = size_t(blockIdx.x) / tileCount * TILE;
    size_t j = (size_t(blockIdx.x) % tileCount) * TILE + x;
    Value value[4], via[4], left[4];
    for (int t = 0; t < 4; ++t) {
        int y = threadIdx.y + t * 8;
        left[t] = pivotColumns[(row + y) * TILE + x];
        right[y][x] = panel[size_t(y) * width + j];
        square[y][x] = panel[size_t(y) * width + pivot + x];
        value[t] = matrix[(row + y) * width + j];
        via[t] = path[(row + y) * width + j];
    }
    __syncthreads();
    // Each warp owns four rows in registers. Pivot-column
    // exchange stays within that warp, avoiding per-pivot block barriers.
    for (int k = 0; k < TILE; ++k) {
#pragma unroll
        for (int t = 0; t < 4; ++t) {
            Value dik = __shfl_sync(0xffffffffu, left[t], k);
            Value candidate = dik + right[k][x];
            if (candidate < value[t]) {
                value[t] = candidate;
                via[t] = static_cast<Value>(pivot + k);
            }
            left[t] = min(left[t], dik + square[k][x]);
        }
    }
    for (int t = 0; t < 4; ++t) {
        size_t offset = (row + threadIdx.y + t * 8) * width + j;
        matrix[offset] = value[t];
        path[offset] = via[t];
    }
}

// glibc rand_r advances its 32-bit LCG three times per returned value.
// Jump ahead lets OpenMP initialize independent rows with the exact seed-42 stream.
static unsigned int advanceSeed(unsigned int seed, size_t steps) {
    unsigned int a = 1103515245u, b = 12345u;
    while (steps) {
        if (steps & 1) seed = a * seed + b;
        b *= a + 1;
        a *= a;
        steps >>= 1;
    }
    return seed;
}

static void broadcastPanel(Value* data, size_t count, int owner, MPI_Comm comm) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        mpiCheck(MPI_Bcast(data, chunk, MPI_UNSIGNED, owner, comm));
        data += chunk;
        count -= chunk;
    }
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
    int provided, rank, ranks;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    size_t n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            unsigned long long parsed = strtoull(arg, &end, 10);
            if (*arg == '-' || !*arg || *end || parsed > UINT_MAX - TILE) {
                if (!rank) fprintf(stderr, "Invalid node count: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }
    const size_t tiles = (n + TILE - 1) / TILE;
    const size_t width = tiles * TILE;
    if (width && width > std::numeric_limits<size_t>::max() / width / sizeof(Value)) {
        if (!rank) fprintf(stderr, "Matrix size overflows address space\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    // Extra ranks are excluded from compute collectives when there are fewer row tiles.
    const int active = std::min(size_t(ranks), std::max(size_t(1), tiles));
    MPI_Comm comm;
    mpiCheck(MPI_Comm_split(MPI_COMM_WORLD, rank < active ? 0 : MPI_UNDEFINED, rank, &comm));
    int status = 0;
    if (rank < active) {
        MPI_Comm local;
        mpiCheck(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int localRank, devices;
        mpiCheck(MPI_Comm_rank(local, &localRank));
        CUDA(cudaGetDeviceCount(&devices));
        if (!devices) {
            fprintf(stderr, "Every compute rank requires a CUDA device\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        CUDA(cudaSetDevice(localRank % devices));
        MPI_Comm_free(&local);
        const size_t base = tiles / active, extra = tiles % active;
        const size_t firstTile = rank * base + std::min(size_t(rank), extra);
        const size_t localTiles = base + (size_t(rank) < extra);
        const size_t rows = localTiles * TILE, firstRow = firstTile * TILE;
        const size_t count = rows * width;
        if (localTiles * tiles > INT_MAX) {
            if (!rank) fprintf(stderr, "Matrix exceeds CUDA grid capacity\n");
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        if (!rank) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
            printf("MPI ranks: %d; OpenMP threads per rank: %d\n", active, omp_get_max_threads());
            printf("Initializing graph...\n");
        }
        try {
            std::vector<Value> host(count), hostPath(count);
#ifdef __GLIBC__
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i) {
                size_t global = firstRow + i;
                unsigned int seed = advanceSeed(42, 3 * global * n);
                for (size_t j = 0; j < width; ++j) {
                    Value v = INF;
                    if (global < n && j < n)
                        v = 1 + static_cast<Value>(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
                    host[i * width + j] = global == j ? 0 : v;
                    hostPath[i * width + j] = static_cast<Value>(global);
                }
            }
#else
            // Other libc implementations retain their own rand_r stream exactly.
            unsigned int seed = 42;
            for (size_t i = 0; i < std::min(firstRow, n) * n; ++i) rand_r(&seed);
            for (size_t i = 0; i < rows; ++i)
                for (size_t j = 0; j < width; ++j) {
                    size_t global = firstRow + i;
                    Value v = INF;
                    if (global < n && j < n)
                        v = 1 + static_cast<Value>(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
                    host[i * width + j] = global == j ? 0 : v;
                }
#pragma omp parallel for schedule(static)
            for (size_t i = 0; i < rows; ++i)
                std::fill_n(hostPath.data() + i * width, width, Value(firstRow + i));
#endif
            Value *matrix = nullptr, *path = nullptr, *panel = nullptr;
            Value *diagonal = nullptr, *columns = nullptr, *staging = nullptr;
            cudaStream_t stream;
            CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            // At least one element allows the original empty-graph case.
            CUDA(cudaMalloc(&matrix, std::max(size_t(1), count) * sizeof(Value)));
            CUDA(cudaMalloc(&path, std::max(size_t(1), count) * sizeof(Value)));
            CUDA(cudaMalloc(&panel, std::max(size_t(1), TILE * width) * sizeof(Value)));
            CUDA(cudaMalloc(&columns, std::max(size_t(1), rows * TILE) * sizeof(Value)));
            CUDA(cudaMalloc(&diagonal, TILE * TILE * sizeof(Value)));
            CUDA(cudaMallocHost(&staging, std::max(size_t(1), TILE * width) * sizeof(Value)));
            CUDA(cudaMemcpy(matrix, host.data(), count * sizeof(Value), cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(path, hostPath.data(), count * sizeof(Value), cudaMemcpyHostToDevice));
            if (!rank) printf("Computing shortest paths...\n");
            mpiCheck(MPI_Barrier(comm));
            const double start = MPI_Wtime();
            const dim3 threads(TILE, 8);
            for (size_t block = 0; block < tiles; ++block) {
                const size_t pivot = block * TILE;
                const size_t largeEnd = (base + 1) * extra;
                const int owner = block < largeEnd ? block / (base + 1)
                    : extra + (block - largeEnd) / base;
                if (rank == owner) {
                    diagonalSnapshots<<<1, threads, 0, stream>>>(matrix, diagonal, width, pivot - firstRow, pivot);
                    panelSnapshots<<<static_cast<unsigned int>(tiles), threads, 0, stream>>>(matrix, diagonal, panel, width, pivot - firstRow);
                    CUDA(cudaGetLastError());
                    if (active > 1)
                        CUDA(cudaMemcpyAsync(staging, panel, TILE * width * sizeof(Value), cudaMemcpyDeviceToHost, stream));
                }
                // Snapshot local pivot columns before any tile can overwrite them.
                CUDA(cudaMemcpy2DAsync(columns, TILE * sizeof(Value), matrix + pivot,
                    width * sizeof(Value), TILE * sizeof(Value), rows, cudaMemcpyDeviceToDevice, stream));
                if (active > 1) {
                    CUDA(cudaStreamSynchronize(stream));
                    // Pinned staging works with standard MPI; CUDA-aware MPI is not required.
                    broadcastPanel(staging, TILE * width, owner, comm);
                    if (rank != owner)
                        CUDA(cudaMemcpyAsync(panel, staging, TILE * width * sizeof(Value), cudaMemcpyHostToDevice, stream));
                }
                relaxPanel<<<static_cast<unsigned int>(localTiles * tiles), threads, 0, stream>>>(
                    matrix, path, panel, columns, width, pivot, tiles);
                CUDA(cudaGetLastError());
            }
            CUDA(cudaStreamSynchronize(stream));
            double elapsed = MPI_Wtime() - start, seconds;
            mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
            if (!rank) {
                printf("Computation time: %.3f ms\n", seconds * 1000.0);
                printf("Performance: %.3f GOPS\n", seconds > 0 ? double(n) * n * n / seconds / 1e9 : 0.0);
            }
            if (results || validate) {
                CUDA(cudaMemcpy(host.data(), matrix, count * sizeof(Value), cudaMemcpyDeviceToHost));
                // Small validation runs also check the otherwise unprinted path matrix
                // against the original scalar ordering, including strict ties.
                if (validate && n <= 128) {
                    std::vector<Value> reference(n * n), referencePath(n * n);
                    unsigned int seed = 42;
                    for (size_t i = 0; i < n * n; ++i) {
                        reference[i] = 1 + static_cast<Value>(double(MAX_DISTANCE) * rand_r(&seed) / double(RAND_MAX));
                        referencePath[i] = static_cast<Value>(i / n);
                    }
                    for (size_t i = 0; i < n; ++i) reference[i * n + i] = 0;
                    for (size_t k = 0; k < n; ++k)
                        for (size_t i = 0; i < n; ++i)
                            for (size_t j = 0; j < n; ++j) {
                                Value candidate = reference[i * n + k] + reference[k * n + j];
                                if (candidate < reference[i * n + j]) {
                                    reference[i * n + j] = candidate;
                                    referencePath[i * n + j] = static_cast<Value>(k);
                                }
                            }
                    CUDA(cudaMemcpy(hostPath.data(), path, count * sizeof(Value), cudaMemcpyDeviceToHost));
                    int mismatch = 0;
#pragma omp parallel for reduction(|:mismatch) schedule(static)
                    for (size_t i = 0; i < rows; ++i)
                        if (firstRow + i < n)
                            for (size_t j = 0; j < n; ++j) {
                                size_t localIndex = i * width + j;
                                size_t refIndex = (firstRow + i) * n + j;
                                mismatch |= host[localIndex] != reference[refIndex] ||
                                            hostPath[localIndex] != referencePath[refIndex];
                            }
                    mpiCheck(MPI_Allreduce(&mismatch, &status, 1, MPI_INT, MPI_MAX, comm));
                    if (!rank && status) fprintf(stderr, "Scalar distance/path comparison failed\n");
                }
                const size_t actualRows = std::min(rows, n > firstRow ? n - firstRow : size_t(0));
                std::vector<Value> packed(actualRows * n);
#pragma omp parallel for schedule(static)
                for (size_t i = 0; i < actualRows; ++i)
                    std::copy_n(host.data() + i * width, n, packed.data() + i * n);
                std::vector<Value> dist;
                if (!rank) dist.resize(n * n);
                // Chunked transfers avoid MPI's signed-int count/displacement limit.
                for (int source = 0; source < active; ++source) {
                    size_t begin = (source * base + std::min(size_t(source), extra)) * TILE;
                    size_t end = std::min(n, begin + (base + (size_t(source) < extra)) * TILE);
                    size_t total = (end > begin ? end - begin : 0) * n;
                    for (size_t offset = 0; offset < total;) {
                        int chunk = static_cast<int>(std::min(total - offset, size_t(INT_MAX)));
                        if (!rank && !source) std::copy_n(packed.data() + offset, chunk, dist.data() + offset);
                        else if (rank == source) mpiCheck(MPI_Send(packed.data() + offset, chunk, MPI_UNSIGNED, 0, 0, comm));
                        else if (!rank) mpiCheck(MPI_Recv(dist.data() + begin * n + offset, chunk, MPI_UNSIGNED, source, 0, comm, MPI_STATUS_IGNORE));
                        offset += chunk;
                    }
                }
                if (!rank) {
                    if (results) print_results_int(dist, "DistanceMatrix");
                    if (validate) {
                        printf("Validating result...\n");
                        status |= validateResult(dist, n) ? 0 : 1;
                        printf("Validation: %s\n", status ? "FAILED" : "PASSED");
                    }
                }
            }
            CUDA(cudaFreeHost(staging));
            CUDA(cudaFree(diagonal));
            CUDA(cudaFree(columns));
            CUDA(cudaFree(panel));
            CUDA(cudaFree(path));
            CUDA(cudaFree(matrix));
            CUDA(cudaStreamDestroy(stream));
        } catch (const std::exception& error) {
            fprintf(stderr, "Rank %d: %s\n", rank, error.what());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        MPI_Comm_free(&comm);
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    MPI_Finalize();
    return status;
}
