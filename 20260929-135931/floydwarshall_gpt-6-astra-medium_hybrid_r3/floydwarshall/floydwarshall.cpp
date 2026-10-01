#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, error);
    }
}
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Each rank owns complete, contiguous tiles of rows. Empty ranks are legal.
static size_t firstRow(int rank, int ranks, size_t n) {
    const size_t tiles = (n + TILE - 1) / TILE;
    return std::min(n, (tiles / ranks * rank +
                      std::min(static_cast<size_t>(rank), tiles % ranks)) * TILE);
}

// Jump ahead in the original rand_r stream, without replicating the matrix.
static unsigned int seedAt(size_t position) {
    unsigned int seed = 42;
#if defined(__GLIBC__)
    // glibc rand_r applies this 32-bit LCG three times per output.
    // Exponentiate its three-step affine map in O(log(position)).
    unsigned int a = 1, c = 0;
    for (int step = 0; step < 3; ++step) {
        a *= 1103515245u;
        c = c * 1103515245u + 12345u;
    }
    while (position) {
        if (position & 1) seed = a * seed + c;
        c *= a + 1;
        a *= a;
        position >>= 1;
    }
#else
    // Preserve the system libc's exact sequence on other platforms too.
    for (size_t i = 0; i < position; ++i) rand_r(&seed);
#endif
    return seed;
}

static void initialize(Value* dist, Value* path, size_t rows, size_t first, size_t n) {
    const size_t count = rows * n;
    #pragma omp parallel
    {
        const size_t tid = omp_get_thread_num(), threads = omp_get_num_threads();
        const size_t begin = count / threads * tid + std::min(tid, count % threads);
        const size_t end = begin + count / threads + (tid < count % threads);
        unsigned int seed = seedAt(first * n + begin);
        for (size_t i = begin; i < end; ++i) {
            dist[i] = 1 + static_cast<Value>(static_cast<double>(MAX_DISTANCE) * rand_r(&seed) / static_cast<double>(RAND_MAX));
            path[i] = static_cast<Value>(first + i / n);
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (size_t i = 0; i < rows; ++i) dist[i * n + first + i] = 0;
    }
}

// Save the column coefficients at each pivot's original scalar time step.
// Keeping these snapshots (rather than a fully closed pivot tile) preserves
// strict-improvement path labels as well as the final distances.
__global__ void pivotCore(const Value* dist, Value* coefficients,
                          size_t n, size_t localPivot, size_t base, int width) {
    __shared__ Value tile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    Value v = x < width && y < width ? dist[(localPivot + y) * n + base + x] : INF;
    tile[y][x] = v;
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        const Value a = tile[y][k], b = tile[k][x];
        if (x == k) coefficients[y * TILE + k] = a;
        __syncthreads();
        v = min(v, a + b);
        tile[y][x] = v;
        __syncthreads();
    }
}

// Materialize B pivot rows, each as it was immediately before its pivot.
__global__ void pivotRows(const Value* dist, const Value* coefficients, Value* pivots,
                          size_t n, size_t localPivot, int width) {
    __shared__ Value tile[TILE][TILE + 1];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t j = static_cast<size_t>(blockIdx.x) * TILE + x;
    Value v = y < width && j < n ? dist[(localPivot + y) * n + j] : INF;
    tile[y][x] = v;
    __syncthreads();
    for (int k = 0; k < width; ++k) {
        const Value b = tile[k][x];
        if (y == k && j < n) pivots[static_cast<size_t>(k) * n + j] = b;
        const Value a = coefficients[y * TILE + k];
        __syncthreads();
        v = min(v, a + b);
        tile[y][x] = v;
        __syncthreads();
    }
}

// A warp owns a row's B pivot columns; warp shuffles broadcast d[i][k].
__global__ void rowCoefficients(const Value* dist, const Value* pivots, Value* coefficients,
                                size_t n, size_t rows, size_t base, int width) {
    const int x = threadIdx.x;
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.y + threadIdx.y;
    Value v = i < rows && x < width ? dist[i * n + base + x] : INF;
    for (int k = 0; k < width; ++k) {
        const Value a = __shfl_sync(0xffffffffu, v, k);
        if (i < rows && x == k) coefficients[i * TILE + k] = a;
        if (x < width) v = min(v, a + pivots[static_cast<size_t>(k) * n + base + x]);
    }
}

// Reuse a 32x32 pair of panels for 1024 updates; each thread keeps four
// output elements in registers across all B ordered scalar pivots.
__global__ void updateRows(Value* dist, Value* path, const Value* pivots,
                           const Value* coefficients, size_t n, size_t rows,
                           size_t base, int width, size_t tileColumns, size_t totalTiles) {
    __shared__ Value left[TILE][TILE + 1], right[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    for (size_t tile = blockIdx.x; tile < totalTiles; tile += gridDim.x) {
        const size_t row = (tile / tileColumns) * TILE;
        const size_t j = (tile % tileColumns) * TILE + x;
        Value value[4], label[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const int r = y + t * 8;
            const size_t i = row + r;
            left[r][x] = i < rows && x < width ? coefficients[i * TILE + x] : INF;
            right[r][x] = r < width && j < n ? pivots[static_cast<size_t>(r) * n + j] : INF;
            value[t] = i < rows && j < n ? dist[i * n + j] : INF;
            label[t] = i < rows && j < n ? path[i * n + j] : 0;
        }
        __syncthreads();
        for (int k = 0; k < width; ++k) {
            const Value b = right[k][x];
            #pragma unroll
            for (int t = 0; t < 4; ++t) {
                const Value candidate = left[y + t * 8][k] + b;
                if (candidate < value[t]) {
                    value[t] = candidate;
                    label[t] = static_cast<Value>(base + k);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const size_t i = row + y + t * 8;
            if (i < rows && j < n) {
                dist[i * n + j] = value[t];
                path[i * n + j] = label[t];
            }
        }
        __syncthreads();
    }
}

// Chunk MPI counts to support matrices larger than INT_MAX elements.
static void broadcast(Value* data, size_t count, int root) {
    for (size_t offset = 0; offset < count;) {
        int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
        mpiCheck(MPI_Bcast(data + offset, chunk, MPI_UNSIGNED, root, MPI_COMM_WORLD));
        offset += chunk;
    }
}
static void gather(const Value* local, std::vector<Value>& result,
                   size_t n, int rank, int ranks) {
    for (int source = 0; source < ranks; ++source) {
        const size_t first = firstRow(source, ranks, n);
        const size_t count = (firstRow(source + 1, ranks, n) - first) * n;
        if (source == 0 && rank == 0) std::copy(local, local + count, result.data());
        if (source == 0 || (rank != 0 && rank != source)) continue;
        for (size_t offset = 0; offset < count;) {
            const int chunk = static_cast<int>(std::min(count - offset, static_cast<size_t>(INT_MAX)));
            if (rank == source) mpiCheck(MPI_Send(local + offset, chunk, MPI_UNSIGNED, 0, 0, MPI_COMM_WORLD));
            else mpiCheck(MPI_Recv(result.data() + first * n + offset, chunk, MPI_UNSIGNED,
                                   source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            offset += chunk;
        }
    }
}

static bool validateResult(const std::vector<Value>& dist, const std::vector<Value>& path, size_t n) {
    int valid = 1;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < n; ++i) if (dist[i * n + i] != 0) valid = 0;
    #pragma omp parallel for reduction(&:valid) schedule(static)
    for (size_t i = 0; i < std::min(n, size_t(10)); ++i)
        for (size_t j = 0; j < std::min(n, size_t(10)); ++j)
            for (size_t k = 0; k < n; ++k)
                if (dist[i * n + k] < INF && dist[k * n + j] < INF &&
                    dist[i * n + k] + dist[k * n + j] < dist[i * n + j]) valid = 0;
    // Small validation runs also check every distance and path against the
    // original scalar recurrence, including its tie-breaking semantics.
    if (n <= 128) {
        std::vector<Value> reference(n * n), referencePath(n * n);
        initialize(reference.data(), referencePath.data(), n, 0, n);
        for (size_t k = 0; k < n; ++k)
            for (size_t i = 0; i < n; ++i)
                for (size_t j = 0; j < n; ++j) {
                    const Value candidate = reference[i * n + k] + reference[k * n + j];
                    if (candidate < reference[i * n + j]) {
                        reference[i * n + j] = candidate;
                        referencePath[i * n + j] = static_cast<Value>(k);
                    }
                }
        if (dist != reference || path != referencePath) valid = 0;
    }
    return valid != 0;
}

static void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n", name);
    printf("  -n <num>     Number of nodes in the graph (default: 512)\n");
    printf("  -v           Enable validation\n  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

static int run(int argc, char** argv, int rank, int ranks) {
    size_t n = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            const unsigned long long parsed = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || end == arg || *end || parsed > UINT_MAX ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / sizeof(Value) / parsed)) {
                if (rank == 0) fprintf(stderr, "Invalid matrix size: %s\n", arg);
                return 1;
            }
            n = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) { if (rank == 0) printUsage(argv[0]); return 0; }
        else { if (rank == 0) printUsage(argv[0]); return 1; }
    }

    MPI_Comm node;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node));
    int localRank = 0, devices = 0;
    mpiCheck(MPI_Comm_rank(node, &localRank));
    cudaCheck(cudaGetDeviceCount(&devices));
    if (devices == 0) { fprintf(stderr, "A CUDA GPU is required on every rank.\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    cudaCheck(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&node));
    const size_t first = firstRow(rank, ranks, n);
    const size_t rows = firstRow(rank + 1, ranks, n) - first;
    const size_t count = rows * n;
    const size_t bytes = std::max(size_t(1), count) * sizeof(Value);
    Value *hostDist, *hostPath, *hostPivots;
    cudaCheck(cudaMallocHost(&hostDist, bytes));
    cudaCheck(cudaMallocHost(&hostPath, bytes));
    cudaCheck(cudaMallocHost(&hostPivots, std::max(size_t(1), 2 * TILE * n) * sizeof(Value)));
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n", n);
        printf("Validation: %s\nMPI ranks: %d\nInitializing graph...\n", validate ? "enabled" : "disabled", ranks);
    }
    initialize(hostDist, hostPath, rows, first, n);
    Value *dist, *path, *pivots, *coefficients, *core;
    cudaCheck(cudaMalloc(&dist, bytes));
    cudaCheck(cudaMalloc(&path, bytes));
    cudaCheck(cudaMalloc(&pivots, std::max(size_t(1), TILE * n) * sizeof(Value)));
    cudaCheck(cudaMalloc(&coefficients, std::max(size_t(1), rows * TILE) * sizeof(Value)));
    cudaCheck(cudaMalloc(&core, TILE * TILE * sizeof(Value)));
    cudaStream_t stream;
    cudaCheck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    cudaEvent_t uploaded[2];
    for (auto& event : uploaded) cudaCheck(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    cudaCheck(cudaMemcpyAsync(dist, hostDist, count * sizeof(Value), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaMemcpyAsync(path, hostPath, count * sizeof(Value), cudaMemcpyHostToDevice, stream));
    cudaCheck(cudaStreamSynchronize(stream));
    if (rank == 0) printf("Computing shortest paths...\n");
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    const double start = MPI_Wtime();
    int owner = 0;
    for (size_t base = 0; base < n; base += TILE) {
        while (base >= firstRow(owner + 1, ranks, n)) ++owner;
        const int width = static_cast<int>(std::min(size_t(TILE), n - base));
        const size_t panelCount = static_cast<size_t>(width) * n;
        const int slot = (base / TILE) % 2;
        Value* staging = hostPivots + static_cast<size_t>(slot) * TILE * n;
        if (ranks > 1) cudaCheck(cudaEventSynchronize(uploaded[slot]));
        if (rank == owner) {
            pivotCore<<<1, dim3(TILE, TILE), 0, stream>>>(dist, core, n, base - first, base, width);
            cudaCheck(cudaGetLastError());
            pivotRows<<<static_cast<unsigned>((n + TILE - 1) / TILE), dim3(TILE, TILE), 0, stream>>>
                (dist, core, pivots, n, base - first, width);
            cudaCheck(cudaGetLastError());
            if (ranks > 1) {
                cudaCheck(cudaMemcpyAsync(staging, pivots, panelCount * sizeof(Value), cudaMemcpyDeviceToHost, stream));
                cudaCheck(cudaStreamSynchronize(stream));
            }
        }
        if (ranks > 1) {
            // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
            // Double-buffer host staging so nonowners can communicate while
            // their previous update runs. Stream ordering protects device data.
            broadcast(staging, panelCount, owner);
            if (rank != owner) cudaCheck(cudaMemcpyAsync(pivots, staging, panelCount * sizeof(Value), cudaMemcpyHostToDevice, stream));
            cudaCheck(cudaEventRecord(uploaded[slot], stream));
        }
        if (rows) {
            rowCoefficients<<<static_cast<unsigned>((rows + 7) / 8), dim3(TILE, 8), 0, stream>>>
                (dist, pivots, coefficients, n, rows, base, width);
            cudaCheck(cudaGetLastError());
            const size_t columns = (n + TILE - 1) / TILE;
            const size_t tiles = ((rows + TILE - 1) / TILE) * columns;
            updateRows<<<static_cast<unsigned>(std::min(tiles, size_t(65535))), dim3(TILE, 8), 0, stream>>>
                (dist, path, pivots, coefficients, n, rows, base, width, columns, tiles);
            cudaCheck(cudaGetLastError());
        }
    }
    cudaCheck(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Performance: %.3f GOPS\n", seconds > 0 ? double(n) * n * n / seconds / 1e9 : 0);
    }
    int status = 0;
    if (printResults || validate) {
        cudaCheck(cudaMemcpyAsync(hostDist, dist, count * sizeof(Value), cudaMemcpyDeviceToHost, stream));
        if (validate && n <= 128) cudaCheck(cudaMemcpyAsync(hostPath, path, count * sizeof(Value), cudaMemcpyDeviceToHost, stream));
        cudaCheck(cudaStreamSynchronize(stream));
        std::vector<Value> result(rank == 0 ? n * n : 0), resultPath;
        gather(hostDist, result, n, rank, ranks);
        if (validate && n <= 128) {
            if (rank == 0) resultPath.resize(n * n);
            gather(hostPath, resultPath, n, rank, ranks);
        }
        if (rank == 0) {
            if (printResults) print_results_int(result, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(result, resultPath, n) ? 0 : 1;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
    }
    mpiCheck(MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD));
    for (auto event : uploaded) cudaCheck(cudaEventDestroy(event));
    cudaCheck(cudaStreamDestroy(stream));
    cudaCheck(cudaFree(core));
    cudaCheck(cudaFree(coefficients));
    cudaCheck(cudaFree(pivots));
    cudaCheck(cudaFree(path));
    cudaCheck(cudaFree(dist));
    cudaCheck(cudaFreeHost(hostPivots));
    cudaCheck(cudaFreeHost(hostPath));
    cudaCheck(cudaFreeHost(hostDist));
    return status;
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    mpiCheck(MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN));
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    int status = 1;
    try { status = run(argc, argv, rank, ranks); }
    catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Finalize());
    return status;
}
