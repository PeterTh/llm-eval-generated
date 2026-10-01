#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cerrno>
#include <climits>
#include <limits>
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

// A tile is also the communication batch. Rows are distributed in whole tiles.
constexpr int TILE = 32;

static void cudaCheck(cudaError_t status, const char* expression) {
    if (status != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", expression, cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}
#define CUDA(call) cudaCheck((call), #call)

// Save each pivot-column value BEFORE processing that pivot. Unlike ordinary
// blocked Floyd-Warshall, these snapshots preserve the serial path tie-breaking.
__global__ void pivotCoefficients(const unsigned* dist, unsigned* coefficients,
                                  size_t stride, size_t localPivot, size_t pivot) {
    __shared__ unsigned d[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    for (int r = y; r < TILE; r += 8)
        d[r][x] = dist[(localPivot + r) * stride + pivot + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int r = y; r < TILE; r += 8) {
            if (x == k) coefficients[r * TILE + k] = d[r][k];
            // Row k and column k stay unchanged during this iteration.
            if (r > k && x != k) d[r][x] = min(d[r][x], d[r][k] + d[k][x]);
        }
        __syncthreads();
    }
}

__global__ void pivotSnapshots(const unsigned* dist, const unsigned* coefficients,
                               unsigned* panel, size_t stride, size_t localPivot) {
    __shared__ unsigned rows[TILE][TILE];
    __shared__ unsigned c[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t col = size_t(blockIdx.x) * TILE + x;
    for (int r = y; r < TILE; r += 8) {
        rows[r][x] = dist[(localPivot + r) * stride + col];
        c[r][x] = coefficients[r * TILE + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int r = y; r < TILE; r += 8)
            if (r > k) rows[r][x] = min(rows[r][x], c[r][k] + rows[k][x]);
        __syncthreads();
    }
    for (int r = y; r < TILE; r += 8)
        panel[size_t(r) * stride + col] = rows[r][x];
}

__global__ void localCoefficients(const unsigned* dist, const unsigned* panel,
                                  unsigned* left, size_t stride, size_t pivot) {
    __shared__ unsigned columns[TILE][TILE];
    __shared__ unsigned diagonal[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t base = size_t(blockIdx.x) * TILE;
    for (int r = y; r < TILE; r += 8) {
        columns[r][x] = dist[(base + r) * stride + pivot + x];
        diagonal[r][x] = panel[size_t(r) * stride + pivot + x];
    }
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        for (int r = y; r < TILE; r += 8)
            if (x > k)
                columns[r][x] = min(columns[r][x], columns[r][k] + diagonal[k][x]);
        __syncthreads();
    }
    for (int r = y; r < TILE; r += 8)
        left[(base + r) * TILE + x] = columns[r][x];
}

__global__ void updateTiles(unsigned* dist, unsigned* path, const unsigned* left,
                            const unsigned* panel, size_t stride, size_t pivot) {
    __shared__ unsigned a[TILE][TILE];
    __shared__ unsigned b[TILE][TILE];
    const int x = threadIdx.x, y = threadIdx.y;
    const size_t row = size_t(blockIdx.y) * TILE;
    const size_t col = size_t(blockIdx.x) * TILE + x;
    unsigned value[4], via[4];
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const int r = y + 8 * t;
        a[r][x] = left[(row + r) * TILE + x];
        b[r][x] = panel[size_t(r) * stride + col];
        value[t] = dist[(row + r) * stride + col];
        via[t] = path[(row + r) * stride + col];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            const unsigned candidate = a[y + 8 * t][k] + b[k][x];
            if (candidate < value[t]) {
                value[t] = candidate;
                via[t] = static_cast<unsigned>(pivot + k);
            }
        }
    }
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        dist[(row + y + 8 * t) * stride + col] = value[t];
        path[(row + y + 8 * t) * stride + col] = via[t];
    }
}

// Chunk messages to avoid MPI's signed-int count limit, including large graphs.
static void transfer(unsigned* data, size_t count, int peer, bool send) {
    while (count) {
        const int chunk = static_cast<int>(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

static size_t firstTile(int rank, int ranks, size_t tiles) {
    return (tiles / ranks) * rank + std::min(size_t(rank), tiles % ranks);
}

static double floydWarshall(std::vector<unsigned>& dist, size_t n, int rank, int ranks) {
    if (!n) { MPI_Barrier(MPI_COMM_WORLD); return 0.0; }
    const size_t tiles = (n + TILE - 1) / TILE, stride = tiles * TILE;
    const size_t begin = firstTile(rank, ranks, tiles) * TILE;
    const size_t rows = (firstTile(rank + 1, ranks, tiles) * TILE) - begin;
    const size_t elements = rows * stride;
    std::vector<unsigned> local(elements), paths(elements);
    // Pack/distribute once; only rank zero holds the full unpadded matrix.
    for (int peer = 0; peer < ranks; ++peer) {
        const size_t start = firstTile(peer, ranks, tiles) * TILE;
        const size_t length = (firstTile(peer + 1, ranks, tiles) * TILE) - start;
        if (rank == 0) {
            std::vector<unsigned> packed(peer ? length * stride : 0);
            unsigned* target = peer ? packed.data() : local.data();
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < length; ++r) {
                std::fill_n(target + r * stride, stride, INF);
                if (start + r < n)
                    std::copy_n(dist.data() + (start + r) * n, n, target + r * stride);
                target[r * stride + start + r] = 0;
            }
            if (peer) transfer(target, length * stride, peer, true);
        } else if (rank == peer) transfer(local.data(), elements, 0, false);
    }
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r)
        std::fill_n(paths.data() + r * stride, stride, static_cast<unsigned>(begin + r));

    unsigned *deviceDist = nullptr, *devicePath = nullptr, *panel = nullptr;
    unsigned *left = nullptr, *coefficients = nullptr, *hostPanel = nullptr;
    CUDA(cudaMalloc(&deviceDist, std::max(size_t(1), elements) * sizeof(unsigned)));
    CUDA(cudaMalloc(&devicePath, std::max(size_t(1), elements) * sizeof(unsigned)));
    CUDA(cudaMalloc(&left, std::max(size_t(1), rows * TILE) * sizeof(unsigned)));
    CUDA(cudaMalloc(&coefficients, TILE * TILE * sizeof(unsigned)));
    CUDA(cudaMalloc(&panel, TILE * stride * sizeof(unsigned)));
    CUDA(cudaMallocHost(&hostPanel, TILE * stride * sizeof(unsigned)));
    if (elements) {
        CUDA(cudaMemcpy(deviceDist, local.data(), elements * sizeof(unsigned), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(devicePath, paths.data(), elements * sizeof(unsigned), cudaMemcpyHostToDevice));
    }
    cudaStream_t stream;
    CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    int owner = 0;
    const dim3 threads(TILE, 8);
    for (size_t tile = 0; tile < tiles; ++tile) {
        while (tile >= firstTile(owner + 1, ranks, tiles)) ++owner;
        const size_t pivot = tile * TILE;
        if (rank == owner) {
            pivotCoefficients<<<1, threads, 0, stream>>>(deviceDist, coefficients, stride, pivot - begin, pivot);
            pivotSnapshots<<<static_cast<unsigned>(tiles), threads, 0, stream>>>(deviceDist, coefficients, panel, stride, pivot - begin);
            if (ranks > 1)
                CUDA(cudaMemcpyAsync(hostPanel, panel, TILE * stride * sizeof(unsigned), cudaMemcpyDeviceToHost, stream));
        }
        if (ranks > 1) {
            // Pinned staging works with ordinary MPI; CUDA-aware MPI is not required.
            CUDA(cudaStreamSynchronize(stream));
            for (size_t offset = 0; offset < TILE * stride;) {
                const int count = static_cast<int>(std::min(TILE * stride - offset, size_t(INT_MAX)));
                MPI_Bcast(hostPanel + offset, count, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
                offset += count;
            }
            if (rank != owner)
                CUDA(cudaMemcpyAsync(panel, hostPanel, TILE * stride * sizeof(unsigned), cudaMemcpyHostToDevice, stream));
        }
        if (rows) {
            localCoefficients<<<static_cast<unsigned>(rows / TILE), threads, 0, stream>>>(deviceDist, panel, left, stride, pivot);
            updateTiles<<<dim3(static_cast<unsigned>(tiles), static_cast<unsigned>(rows / TILE)), threads, 0, stream>>>(deviceDist, devicePath, left, panel, stride, pivot);
        }
        CUDA(cudaGetLastError());
    }
    CUDA(cudaStreamSynchronize(stream));
    const double elapsed = MPI_Wtime() - start;
    double seconds = 0;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (elements) CUDA(cudaMemcpy(local.data(), deviceDist, elements * sizeof(unsigned), cudaMemcpyDeviceToHost));
    for (int peer = 0; peer < ranks; ++peer) {
        const size_t startRow = firstTile(peer, ranks, tiles) * TILE;
        const size_t length = firstTile(peer + 1, ranks, tiles) * TILE - startRow;
        if (rank == 0) {
            std::vector<unsigned> packed(peer ? length * stride : 0);
            unsigned* source = peer ? packed.data() : local.data();
            if (peer) transfer(source, length * stride, peer, false);
            #pragma omp parallel for schedule(static)
            for (size_t r = 0; r < length; ++r)
                if (startRow + r < n)
                    std::copy_n(source + r * stride, n, dist.data() + (startRow + r) * n);
        } else if (rank == peer) transfer(local.data(), elements, 0, true);
    }
    CUDA(cudaStreamDestroy(stream));
    CUDA(cudaFreeHost(hostPanel));
    CUDA(cudaFree(panel));
    CUDA(cudaFree(coefficients));
    CUDA(cudaFree(left));
    CUDA(cudaFree(devicePath));
    CUDA(cudaFree(deviceDist));
    return seconds;
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
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, ranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (!rank) fprintf(stderr, "MPI_THREAD_FUNNELED support is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    size_t numNodes = 512;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            const char* arg = argv[++i];
            errno = 0;
            const unsigned long long value = strtoull(arg, &end, 10);
            if (errno || arg == end || *end || *arg == '-' ||
                value > UINT_MAX - TILE ||
                (value && value > std::numeric_limits<size_t>::max() / sizeof(unsigned) / value)) {
                if (!rank) fprintf(stderr, "Invalid graph size: %s\n", arg);
                MPI_Finalize();
                return 1;
            }
            numNodes = static_cast<size_t>(value);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) printResults = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (!rank) { fprintf(stderr, "Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            MPI_Finalize();
            return 1;
        }
    }
    // Map ranks by their node-local index, respecting CUDA_VISIBLE_DEVICES.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, devices = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "Rank %d: a CUDA device is required\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&nodeComm);
    int status = 0;
    try {
        std::vector<unsigned> dist;
        if (!rank) {
            printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
            printf("Number of nodes: %zu\n", numNodes);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("Initializing graph...\n");
            dist.resize(numNodes * numNodes);
            initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
            printf("Computing shortest paths...\n");
        }
        const double seconds = floydWarshall(dist, numNodes, rank, ranks);
        if (!rank) {
            printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000.0));
            printf("Performance: %.3f GOPS\n", seconds > 0 ? double(numNodes) * numNodes * numNodes / seconds / 1e9 : 0.0);
            if (printResults) print_results_int(dist, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(dist, numNodes) ? 0 : 1;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Rank %d: %s\n", rank, e.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
