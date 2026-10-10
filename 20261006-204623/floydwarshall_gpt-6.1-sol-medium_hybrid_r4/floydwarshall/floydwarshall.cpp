#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <limits>
#include <cerrno>
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
    #pragma omp parallel for schedule(static) if(numNodes * numNodes >= 262144)
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
        }
        path[idx2(j, j, numNodes)] = j;
    }
}


constexpr int TILE = 32;

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

static void mpiCheck(int error) {
    if (error != MPI_SUCCESS) {
        char message[MPI_MAX_ERROR_STRING];
        int length = 0;
        MPI_Error_string(error, message, &length);
        fprintf(stderr, "MPI: %.*s\n", length, message);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Snapshot each pivot row/column BEFORE its k update. This keeps even the
// path matrix identical to the original ordered, strict-improvement loop.
// Mode 0: diagonal; mode 1: pivot rows; mode 2: local pivot columns.
template<int Mode>
__global__ void preparePanel(unsigned* dist, unsigned* path, unsigned* panel,
                             unsigned* columns, size_t pitch, size_t firstRow,
                             size_t pivot, size_t pivotLocal) {
    __shared__ unsigned tile[TILE][TILE];
    __shared__ unsigned factors[TILE][TILE];
    const int x = threadIdx.x;
    const int y = threadIdx.y;
    size_t rowBase = Mode == 2 ? size_t(blockIdx.x) * TILE : pivotLocal;
    size_t colBase = Mode == 1 ? size_t(blockIdx.x) * TILE : pivot;
    if ((Mode == 1 && colBase == pivot) ||
        (Mode == 2 && firstRow + rowBase == pivot)) return;
    unsigned paths[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * 8;
        size_t offset = (rowBase + r) * pitch + colBase + x;
        tile[r][x] = dist[offset];
        paths[t] = path[offset];
        if (Mode == 1) factors[r][x] = columns[(rowBase + r) * TILE + x];
        if (Mode == 2) factors[r][x] = panel[r * pitch + pivot + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        if (Mode == 0 || Mode == 1) {
            if (y == 0) panel[k * pitch + colBase + x] = tile[k][x];
        }
        unsigned left[4];
        unsigned right = Mode == 2 ? factors[k][x] : tile[k][x];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int r = y + t * 8;
            left[t] = Mode == 1 ? factors[r][k] : tile[r][k];
            if ((Mode == 0 || Mode == 2) && x == 0)
                columns[(rowBase + r) * TILE + k] = left[t];
        }
        // All pivot operands must be read before any thread overwrites tile.
        __syncthreads();
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            int r = y + t * 8;
            unsigned candidate = left[t] + right;
            if (candidate < tile[r][x]) {
                tile[r][x] = candidate;
                paths[t] = static_cast<unsigned>(pivot + k);
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        size_t offset = (rowBase + y + t * 8) * pitch + colBase + x;
        dist[offset] = tile[y + t * 8][x];
        path[offset] = paths[t];
    }
}

__global__ void updateTiles(unsigned* __restrict__ dist, unsigned* __restrict__ path,
                            const unsigned* __restrict__ panel,
                            const unsigned* __restrict__ columns, size_t pitch,
                            size_t firstRow, size_t pivot) {
    size_t rowBase = size_t(blockIdx.y) * TILE;
    size_t colBase = size_t(blockIdx.x) * TILE;
    if (firstRow + rowBase == pivot || colBase == pivot) return;
    __shared__ unsigned left[TILE][TILE];
    __shared__ unsigned right[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    unsigned values[4], paths[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        int r = y + t * 8;
        left[r][x] = columns[(rowBase + r) * TILE + x];
        right[r][x] = panel[r * pitch + colBase + x];
        size_t offset = (rowBase + r) * pitch + colBase + x;
        values[t] = dist[offset];
        paths[t] = path[offset];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned kj = right[k][x];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            unsigned candidate = left[y + t * 8][k] + kj;
            if (candidate < values[t]) {
                values[t] = candidate;
                paths[t] = static_cast<unsigned>(pivot + k);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        size_t offset = (rowBase + y + t * 8) * pitch + colBase + x;
        dist[offset] = values[t];
        path[offset] = paths[t];
    }
}

// Chunked transfers avoid MPI's signed-int count limit on large matrices.
static void transfer(unsigned* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(1) << 26));
        if (send) mpiCheck(MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD));
        else mpiCheck(MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        data += chunk;
        count -= chunk;
    }
}

static void broadcastPanel(unsigned* panel, size_t count, int owner) {
    while (count) {
        int chunk = static_cast<int>(std::min(count, size_t(1) << 26));
        mpiCheck(MPI_Bcast(panel, chunk, MPI_UNSIGNED, owner, MPI_COMM_WORLD));
        panel += chunk;
        count -= chunk;
    }
}

void floydWarshall(std::vector<unsigned>& dist, std::vector<unsigned>& path,
                   size_t n, int rank, int ranks, double& seconds, bool gather) {
    if (!n) {
        mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
        seconds = 0.0;
        return;
    }
    const size_t blocks = (n + TILE - 1) / TILE;
    const size_t pitch = blocks * TILE;
    const size_t begin = blocks * rank / ranks;
    const size_t end = blocks * (rank + 1) / ranks;
    const size_t rows = (end - begin) * TILE;
    const size_t elements = rows * pitch;
    std::vector<unsigned> local(elements);
    // Only root uses full graph storage. Padding cannot improve real paths.
    for (int peer = 0; peer < ranks; ++peer) {
        size_t first = blocks * peer / ranks * TILE;
        size_t last = blocks * (peer + 1) / ranks * TILE;
        if (rank == 0) {
            std::vector<unsigned> packed((last - first) * pitch, INF);
            #pragma omp parallel for schedule(static) if((last - first) * pitch >= 262144)
            for (size_t i = first; i < last; ++i) {
                if (i < n) std::copy_n(dist.data() + i * n, n, packed.data() + (i - first) * pitch);
                packed[(i - first) * pitch + i] = 0;
            }
            if (peer == 0) local.swap(packed);
            else transfer(packed.data(), packed.size(), peer, true);
        } else if (rank == peer) transfer(local.data(), elements, 0, false);
    }
    unsigned *deviceDist = nullptr, *devicePath = nullptr, *devicePanel = nullptr;
    unsigned *deviceColumns = nullptr, *hostPanel = nullptr;
    CUDA(cudaMalloc(&deviceDist, std::max(elements, size_t(1)) * sizeof(unsigned)));
    CUDA(cudaMalloc(&devicePath, std::max(elements, size_t(1)) * sizeof(unsigned)));
    CUDA(cudaMalloc(&deviceColumns, std::max(rows * TILE, size_t(1)) * sizeof(unsigned)));
    CUDA(cudaMalloc(&devicePanel, TILE * pitch * sizeof(unsigned)));
    CUDA(cudaMallocHost(&hostPanel, TILE * pitch * sizeof(unsigned)));
    cudaStream_t stream;
    cudaEvent_t panelCopied;
    CUDA(cudaEventCreateWithFlags(&panelCopied, cudaEventDisableTiming));
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA(cudaMemcpyAsync(deviceDist, local.data(), elements * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    CUDA(cudaStreamSynchronize(stream));
    #pragma omp parallel for schedule(static) if(elements >= 262144)
    for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < pitch; ++j) local[i * pitch + j] = static_cast<unsigned>(begin * TILE + i);
    CUDA(cudaMemcpyAsync(devicePath, local.data(), elements * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
    CUDA(cudaStreamSynchronize(stream));
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    const dim3 threads(TILE, 8);
    for (size_t block = 0; block < blocks; ++block) {
        // The floor partition may leave empty ranks when ranks > blocks.
        int owner = static_cast<int>(((block + 1) * ranks - 1) / blocks);
        size_t pivot = block * TILE;
        if (rank == owner) {
            size_t pivotLocal = pivot - begin * TILE;
            preparePanel<0><<<1, threads, 0, stream>>>(deviceDist, devicePath, devicePanel, deviceColumns,
                                                     pitch, begin * TILE, pivot, pivotLocal);
            preparePanel<1><<<static_cast<unsigned>(blocks), threads, 0, stream>>>(deviceDist, devicePath, devicePanel,
                                        deviceColumns, pitch, begin * TILE, pivot, pivotLocal);
            CUDA(cudaGetLastError());
            CUDA(cudaMemcpyAsync(hostPanel, devicePanel, TILE * pitch * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
            CUDA(cudaStreamSynchronize(stream));
        }
        broadcastPanel(hostPanel, TILE * pitch, owner);
        if (rank != owner) {
            CUDA(cudaMemcpyAsync(devicePanel, hostPanel, TILE * pitch * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
            CUDA(cudaEventRecord(panelCopied, stream));
        }
        if (rows) {
            preparePanel<2><<<static_cast<unsigned>(end - begin), threads, 0, stream>>>(deviceDist, devicePath,
                                devicePanel, deviceColumns, pitch, begin * TILE, pivot, 0);
            updateTiles<<<dim3(static_cast<unsigned>(blocks), static_cast<unsigned>(end - begin)), threads, 0, stream>>>(
                                deviceDist, devicePath, devicePanel, deviceColumns, pitch, begin * TILE, pivot);
            CUDA(cudaGetLastError());
        }
        // Reuse staging once its copy finishes, allowing the next broadcast
        // to overlap trailing tile updates. The stream orders device accesses.
        if (rank != owner) CUDA(cudaEventSynchronize(panelCopied));
    }
    CUDA(cudaStreamSynchronize(stream));
    double elapsed = MPI_Wtime() - start;
    mpiCheck(MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (gather) {
        // Gather both matrices to retain the original function's semantics.
        for (int which = 0; which < 2; ++which) {
            CUDA(cudaMemcpy(local.data(), which ? devicePath : deviceDist, elements * sizeof(unsigned), cudaMemcpyDeviceToHost));
            for (int peer = 0; peer < ranks; ++peer) {
                size_t first = blocks * peer / ranks * TILE;
                size_t last = blocks * (peer + 1) / ranks * TILE;
                if (rank == 0) {
                    std::vector<unsigned> received;
                    unsigned* source = local.data();
                    if (peer) {
                        received.resize((last - first) * pitch);
                        transfer(received.data(), received.size(), peer, false);
                        source = received.data();
                    }
                    auto& target = which ? path : dist;
                    #pragma omp parallel for schedule(static) if((last - first) * pitch >= 262144)
                    for (size_t i = first; i < std::min(last, n); ++i)
                        std::copy_n(source + (i - first) * pitch, n, target.data() + i * n);
                } else if (rank == peer) transfer(local.data(), elements, 0, true);
            }
        }
    }
    CUDA(cudaEventDestroy(panelCopied));
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFreeHost(hostPanel));
    CUDA(cudaFree(deviceColumns));
    CUDA(cudaFree(devicePanel));
    CUDA(cudaFree(devicePath));
    CUDA(cudaFree(deviceDist));
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
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks));
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* value = argv[++i];
            unsigned long long parsed = strtoull(value, &end, 10);
            if (errno || value[0] == '-' || end == value || *end ||
                parsed > std::numeric_limits<unsigned>::max() - TILE ||
                (parsed && parsed > std::numeric_limits<size_t>::max() / parsed / sizeof(unsigned))) {
                if (!rank) fprintf(stderr, "Invalid number of nodes: %s\n", value);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(parsed);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) printResults = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // Bind one process to each visible local accelerator, honoring CUDA_VISIBLE_DEVICES.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm));
    int localRank = 0, localSize = 1, devices = 0;
    mpiCheck(MPI_Comm_rank(localComm, &localRank));
    mpiCheck(MPI_Comm_size(localComm, &localSize));
    if (!std::getenv("OMP_NUM_THREADS"))
        omp_set_num_threads(std::max(1, std::min(8, omp_get_num_procs() / localSize)));
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA accelerator is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    mpiCheck(MPI_Comm_free(&localComm));
    int result = 0;
    try {
        std::vector<unsigned> dist, path;
        if (!rank) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
            dist.resize(numNodes * numNodes);
            path.resize(numNodes * numNodes);
            initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
            initializePathMatrix(path, numNodes);
            printf("Computing shortest paths...\n");
        }
        double seconds = 0;
        floydWarshall(dist, path, numNodes, rank, ranks, seconds, validate || printResults);
        if (!rank) {
            printf("Computation time: %ld ms\n", static_cast<long>(seconds * 1000));
            double ops = static_cast<double>(numNodes) * numNodes * numNodes;
            printf("Performance: %.3f GOPS\n", seconds > 0 ? ops / seconds / 1e9 : 0.0);
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                result = validateResult(dist, numNodes) ? 0 : 1;
                printf("Validation: %s\n", result ? "FAILED" : "PASSED");
            }
        }
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    mpiCheck(MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD));
    mpiCheck(MPI_Finalize());
    return result;
}
