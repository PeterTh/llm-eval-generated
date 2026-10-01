#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <climits>
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
    for (size_t j = 0; j < numNodes; ++j) {
        for (size_t i = 0; i < numNodes; ++i) {
            path[idx2(i, j, numNodes)] = j;
            path[idx2(j, i, numNodes)] = i;
        }
        path[idx2(j, j, numNodes)] = j;
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

// Keep the original k ordering inside each panel. Using fully closed pivot
// tiles instead would change which strict improvement is recorded in path.
constexpr int TILE = 32;

void cudaCheck(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(result));
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
}
#define CUDA(call) cudaCheck((call), #call)

__global__ void diagonalSnapshots(const unsigned* dist, unsigned* diagonal,
                                  size_t pitch, size_t pivotRow, size_t pivot) {
    __shared__ unsigned tile[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    tile[y][x] = dist[(pivotRow + y) * pitch + pivot + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        if (x == k) diagonal[y * TILE + k] = tile[y][k];
        // Neither the pivot row nor column changes (nonnegative weights and
        // zero diagonal), so readers never race with writers within a step.
        if (x != k && y != k)
            tile[y][x] = min(tile[y][x], tile[y][k] + tile[k][x]);
        __syncthreads();
    }
}

__global__ void rowSnapshots(const unsigned* dist, const unsigned* diagonal,
                             unsigned* panel, size_t pitch, size_t pivotRow) {
    __shared__ unsigned tile[TILE][TILE];
    __shared__ unsigned left[TILE][TILE];
    int x = threadIdx.x, y = threadIdx.y;
    size_t col = size_t(blockIdx.x) * TILE + x;
    tile[y][x] = dist[(pivotRow + y) * pitch + col];
    left[y][x] = diagonal[y * TILE + x];
    __syncthreads();
    for (int k = 0; k < TILE; ++k) {
        if (y == k) panel[size_t(k) * pitch + col] = tile[y][x];
        if (y != k) tile[y][x] = min(tile[y][x], left[y][k] + tile[k][x]);
        __syncthreads();
    }
}

__global__ void columnSnapshots(const unsigned* dist, const unsigned* panel,
                                unsigned* left, size_t pitch, size_t rows,
                                size_t pivot) {
    int lane = threadIdx.x;
    size_t row = size_t(blockIdx.x) * blockDim.y + threadIdx.y;
    if (row >= rows) return; // Entire warp is inactive.
    unsigned value = dist[row * pitch + pivot + lane];
    for (int k = 0; k < TILE; ++k) {
        unsigned atK = __shfl_sync(0xffffffffu, value, k);
        if (lane == k) left[row * TILE + k] = atK;
        value = min(value, atK + panel[size_t(k) * pitch + pivot + lane]);
    }
}

__global__ void updateTiles(unsigned* dist, unsigned* path,
                            const unsigned* panel, const unsigned* left,
                            size_t pitch, size_t tileOffset, size_t pivot) {
    __shared__ unsigned a[TILE][TILE], b[TILE][TILE];
    size_t tile = tileOffset + blockIdx.x;
    size_t rowBase = (tile / (pitch / TILE)) * TILE;
    size_t col = (tile % (pitch / TILE)) * TILE + threadIdx.x;
    unsigned values[4], paths[4];
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        int y = threadIdx.y + r * 8;
        a[y][threadIdx.x] = left[(rowBase + y) * TILE + threadIdx.x];
        b[y][threadIdx.x] = panel[size_t(y) * pitch + col];
        values[r] = dist[(rowBase + y) * pitch + col];
        paths[r] = path[(rowBase + y) * pitch + col];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < TILE; ++k) {
        unsigned right = b[k][threadIdx.x];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            unsigned candidate = a[threadIdx.y + r * 8][k] + right;
            if (candidate < values[r]) {
                values[r] = candidate;
                paths[r] = unsigned(pivot + k);
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        size_t index = (rowBase + threadIdx.y + r * 8) * pitch + col;
        dist[index] = values[r];
        path[index] = paths[r];
    }
}

// Chunk transfers to avoid MPI's signed-int count limit for large matrices.
void transfer(unsigned* data, size_t count, int peer, bool send) {
    while (count) {
        int chunk = int(std::min(count, size_t(INT_MAX)));
        if (send) MPI_Send(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD);
        else MPI_Recv(data, chunk, MPI_UNSIGNED, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        data += chunk;
        count -= chunk;
    }
}

int run(int argc, char** argv, int rank, int ranks) {
    size_t n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            char* end = nullptr;
            errno = 0;
            const char* arg = argv[++i];
            unsigned long long parsed = strtoull(arg, &end, 10);
            if (errno || *arg == '-' || !*arg || *end || parsed > UINT_MAX - TILE) {
                if (!rank) fprintf(stderr, "Invalid node count: %s\n", arg);
                return 1;
            }
            n = size_t(parsed);
        } else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else if (!strcmp(argv[i], "-h")) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) printUsage(argv[0]);
            return 1;
        }
    }
    size_t tiles = (n + TILE - 1) / TILE, pitch = tiles * TILE;
    if (pitch && pitch > std::numeric_limits<size_t>::max() / pitch / sizeof(unsigned)) {
        if (!rank) fprintf(stderr, "Matrix size overflows address space\n");
        return 1;
    }
    // Contiguous block rows; extra ranks can participate with empty partitions.
    auto firstTile = [=](int r) { return tiles / ranks * r + std::min(size_t(r), tiles % ranks); };
    size_t first = firstTile(rank) * TILE;
    size_t rows = (firstTile(rank + 1) - firstTile(rank)) * TILE;
    size_t realRows = first < n ? std::min(rows, n - first) : 0;
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    CUDA(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "A CUDA device is required on every participating node\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    CUDA(cudaSetDevice(localRank % devices));
    MPI_Comm_free(&local);

    std::vector<unsigned> full;
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\nValidation: %s\n", n, validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
        full.resize(n * n);
        initializeDistanceMatrix(full, n, 1, MAX_DISTANCE);
    }
    std::vector<unsigned> compact(realRows * n);
    if (!rank) {
        for (int r = 1; r < ranks; ++r) {
            size_t begin = firstTile(r) * TILE;
            size_t length = begin < n ? std::min((firstTile(r + 1) - firstTile(r)) * TILE, n - begin) : 0;
            if (length) transfer(full.data() + begin * n, length * n, r, true);
        }
        std::copy_n(full.data(), compact.size(), compact.data());
        std::vector<unsigned>().swap(full);
    } else if (!compact.empty()) transfer(compact.data(), compact.size(), 0, false);
    std::vector<unsigned> host(rows * pitch), hostPath(rows * pitch);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < pitch; ++j) {
            host[i * pitch + j] = i < realRows && j < n ? compact[i * n + j] : (first + i == j ? 0 : INF);
            hostPath[i * pitch + j] = unsigned(first + i);
        }
    }
    std::vector<unsigned>().swap(compact);
    unsigned *dist = nullptr, *path = nullptr, *panel, *left = nullptr, *diagonal, *staging;
    if (rows) {
        CUDA(cudaMalloc(&dist, rows * pitch * sizeof(unsigned)));
        CUDA(cudaMalloc(&path, rows * pitch * sizeof(unsigned)));
        CUDA(cudaMalloc(&left, rows * TILE * sizeof(unsigned)));
        CUDA(cudaMemcpy(dist, host.data(), host.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(path, hostPath.data(), hostPath.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
    }
    std::vector<unsigned>().swap(hostPath);
    std::vector<unsigned>().swap(host);
    CUDA(cudaMalloc(&panel, std::max(size_t(1), TILE * pitch) * sizeof(unsigned)));
    CUDA(cudaMalloc(&diagonal, TILE * TILE * sizeof(unsigned)));
    CUDA(cudaMallocHost(&staging, std::max(size_t(1), TILE * pitch) * sizeof(unsigned)));
    MPI_Barrier(MPI_COMM_WORLD);
    if (!rank) printf("Computing shortest paths...\n");
    double start = MPI_Wtime();
    int owner = 0;
    for (size_t pivot = 0; pivot < pitch; pivot += TILE) {
        while (pivot >= firstTile(owner + 1) * TILE) ++owner;
        if (rank == owner) {
            diagonalSnapshots<<<1, dim3(TILE, TILE)>>>(dist, diagonal, pitch, pivot - first, pivot);
            rowSnapshots<<<unsigned(tiles), dim3(TILE, TILE)>>>(dist, diagonal, panel, pitch, pivot - first);
            CUDA(cudaGetLastError());
        }
        // Portable pinned-host transport works with MPI implementations that
        // do not support device pointers. Single-rank runs stay on the GPU.
        if (ranks > 1) {
            if (rank == owner) CUDA(cudaMemcpy(staging, panel, TILE * pitch * sizeof(unsigned), cudaMemcpyDeviceToHost));
            for (size_t offset = 0; offset < TILE * pitch;) {
                int count = int(std::min(TILE * pitch - offset, size_t(INT_MAX)));
                MPI_Bcast(staging + offset, count, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
                offset += count;
            }
            if (rank != owner) CUDA(cudaMemcpy(panel, staging, TILE * pitch * sizeof(unsigned), cudaMemcpyHostToDevice));
        }
        if (rows) {
            columnSnapshots<<<unsigned((rows + 7) / 8), dim3(TILE, 8)>>>(dist, panel, left, pitch, rows, pivot);
            size_t count = rows / TILE * tiles;
            for (size_t offset = 0; offset < count; offset += 65535) {
                unsigned blocks = unsigned(std::min(count - offset, size_t(65535)));
                updateTiles<<<blocks, dim3(TILE, 8)>>>(dist, path, panel, left, pitch, offset, pivot);
            }
            CUDA(cudaGetLastError());
        }
    }
    CUDA(cudaDeviceSynchronize());
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (!rank) {
        printf("Computation time: %lld ms\n", static_cast<long long>(seconds * 1000));
        printf("Performance: %.3f GOPS\n", double(n) * n * n / seconds / 1e9);
    }
    int status = 0;
    if (results || validate) {
        host.resize(rows * pitch);
        if (rows) CUDA(cudaMemcpy(host.data(), dist, host.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
        compact.resize(realRows * n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < realRows; ++i)
            std::copy_n(host.data() + i * pitch, n, compact.data() + i * n);
        if (!rank) {
            full.resize(n * n);
            std::copy(compact.begin(), compact.end(), full.begin());
            for (int r = 1; r < ranks; ++r) {
                size_t begin = firstTile(r) * TILE;
                size_t length = begin < n ? std::min((firstTile(r + 1) - firstTile(r)) * TILE, n - begin) : 0;
                if (length) transfer(full.data() + begin * n, length * n, r, false);
            }
            if (results) print_results_int(full, "DistanceMatrix");
            if (validate) {
                printf("Validating result...\n");
                status = validateResult(full, n) ? 0 : 1;
                printf("Validation: %s\n", status ? "FAILED" : "PASSED");
            }
        } else if (!compact.empty()) transfer(compact.data(), compact.size(), 0, true);
    }
    CUDA(cudaFree(dist));
    CUDA(cudaFree(path));
    CUDA(cudaFree(left));
    CUDA(cudaFree(panel));
    CUDA(cudaFree(diagonal));
    CUDA(cudaFreeHost(staging));
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 2);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result = 1;
    try {
        result = run(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    MPI_Finalize();
    return result;
}
