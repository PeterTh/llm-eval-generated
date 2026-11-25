#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define OMPI_SKIP_MPICXX 1
#define MPICH_SKIP_MPICXX 1
#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include "../common/results_output.hpp"

using Value = unsigned int;
constexpr Value INF = 1000000000;
constexpr Value MAX_DISTANCE = 200;
constexpr int Tile = 32;
// Every real edge starts at most MAX_DISTANCE + 1 and can only decrease.
// The extra value represents padded vertices, including in byte-sized panels.
using PanelValue = unsigned char;
constexpr Value Padding = UCHAR_MAX;
static_assert(MAX_DISTANCE + 1 < Padding);

inline size_t idx2(size_t col, size_t row, size_t pitch) {
    return row * pitch + col;
}

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T>
T* deviceArray(size_t count) {
    T* ptr;
    check(cudaMalloc(&ptr, count * sizeof(T)));
    return ptr;
}

// Record the pivot row and column as they appeared at each scalar k.
// Using the final closed tile here could change the intermediate path indices.
__global__ void diagonal(Value* dist, Value* path, PanelValue* panel, PanelValue* left,
                         size_t pitch, int origin) {
    __shared__ Value row[Tile], col[Tile];
    int x = threadIdx.x, y = threadIdx.y;
    size_t pos = y * pitch + origin + x;
    Value d = dist[pos], p = path[pos];
    for (int k = 0; k < Tile; ++k) {
        if (y == k) {
            row[x] = d;
            panel[pos] = d;
        }
        if (x == k) {
            col[y] = d;
            left[y * Tile + x] = d;
        }
        __syncthreads();
        Value candidate = col[y] + row[x];
        if (candidate < d) { d = candidate; p = origin + k; }
        __syncthreads();
    }
    dist[pos] = d;
    path[pos] = p;
}

// Sweep each tile of the pivot row or column. The transcribed pivot
// values also supply the independent off-diagonal tile products.
template<bool Row>
__global__ void updatePanel(Value* dist, Value* path, PanelValue* panel, PanelValue* left,
                            size_t pitch, int origin, int skip) {
    int tile = blockIdx.x;
    if (tile == skip) return;
    int x = threadIdx.x, y = threadIdx.y;
    size_t i = Row ? y : size_t(tile) * Tile + y;
    int j = (Row ? tile * Tile : origin) + x;
    size_t pos = i * pitch + j;
    __shared__ Value diag[Tile][Tile + 1], pivot[Tile];
    diag[y][x] = Row ? left[y * Tile + x] : panel[y * pitch + origin + x];
    Value d = dist[pos], p = path[pos];
    for (int k = 0; k < Tile; ++k) {
        if (Row && y == k) {
            pivot[x] = d;
            panel[pos] = d;
        }
        if (!Row && x == k) {
            pivot[y] = d;
            left[i * Tile + k] = d;
        }
        __syncthreads();
        Value candidate = Row ? diag[y][k] + pivot[x] : pivot[y] + diag[k][x];
        if (candidate < d) { d = candidate; p = origin + k; }
        __syncthreads();
    }
    dist[pos] = d;
    path[pos] = p;
}

__global__ void updateTiles(Value* dist, Value* path, const PanelValue* panel,
                            const PanelValue* left, size_t pitch, int origin, int skip) {
    if (blockIdx.x == origin / Tile || int(blockIdx.y) == skip) return;
    constexpr int Rows = 4;
    __shared__ Value a[Tile][Tile + 1], b[Tile][Tile];
    int x = threadIdx.x, y = threadIdx.y;
    size_t i = size_t(blockIdx.y) * Tile + y;
    int j = blockIdx.x * Tile + x;
    Value d[Rows], p[Rows];
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        int offset = r * (Tile / Rows);
        a[y + offset][x] = left[(i + offset) * Tile + x];
        b[y + offset][x] = panel[(y + offset) * pitch + j];
        d[r] = dist[(i + offset) * pitch + j];
        p[r] = UINT_MAX;
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < Tile; ++k) {
        #pragma unroll
        for (int r = 0; r < Rows; ++r) {
            Value candidate = a[y + r * (Tile / Rows)][k] + b[k][x];
            if (candidate < d[r]) { d[r] = candidate; p[r] = origin + k; }
        }
    }
    #pragma unroll
    for (int r = 0; r < Rows; ++r) {
        size_t pos = (i + r * (Tile / Rows)) * pitch + j;
        if (p[r] != UINT_MAX) {
            dist[pos] = d[r];
            path[pos] = p[r];
        }
    }
}

__global__ void initPaths(Value* path, size_t pitch, int first) {
    size_t i = size_t(blockIdx.y) * Tile;
    size_t j = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < pitch) {
        for (int r = 0; r < Tile; ++r) path[(i + r) * pitch + j] = first + i + r;
    }
}

// On glibc each rand_r draw advances its 32-bit LCG three times.
// Exponentiate the affine transformation to start each thread independently.
Value rowSeed(size_t draws) {
    Value seed = 42;
#ifdef __GLIBC__
    Value a = 1103515245u, b = 12345u;
    draws *= 3;
    while (draws) {
        if (draws & 1) seed = a * seed + b;
        b *= a + 1;
        a *= a;
        draws >>= 1;
    }
#else
    while (draws--) rand_r(&seed);
#endif
    return seed;
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

struct Partition {
    int tiles, ranks;
    int first(int rank) const {
        return rank * (tiles / ranks) + std::min(rank, tiles % ranks);
    }
};

void compute(Value* dist, Value* path, size_t pitch, Partition layout, int rank,
             MPI_Comm comm) {
    int first = layout.first(rank);
    int local = layout.first(rank + 1) - first;
    size_t count = pitch * Tile;
    PanelValue* panel = deviceArray<PanelValue>(count);
    PanelValue* left = deviceArray<PanelValue>(size_t(local) * Tile * Tile);
    PanelValue* staging = nullptr;
    if (layout.ranks > 1) check(cudaMallocHost(&staging, count * sizeof(PanelValue)));

    dim3 tile(Tile, Tile);
    int owner = 0;
    for (int t = 0; t < layout.tiles; ++t) {
        if (t == layout.first(owner + 1)) ++owner;
        int origin = t * Tile;
        int skip = rank == owner ? t - first : -1;
        if (rank == owner) {
            size_t offset = size_t(skip) * Tile * pitch;
            PanelValue* diagLeft = left + size_t(skip) * Tile * Tile;
            diagonal<<<1, tile>>>(dist + offset, path + offset, panel, diagLeft, pitch, origin);
            updatePanel<true><<<layout.tiles, tile>>>(dist + offset, path + offset,
                                                      panel, diagLeft, pitch, origin, t);
        }
        if (layout.ranks > 1) {
            if (rank == owner)
                check(cudaMemcpy(staging, panel, count * sizeof(PanelValue), cudaMemcpyDeviceToHost));
            for (size_t pos = 0; pos < count; ) {
                int chunk = std::min<size_t>(INT_MAX, count - pos);
                MPI_Bcast(staging + pos, chunk, MPI_UNSIGNED_CHAR, owner, comm);
                pos += chunk;
            }
            if (rank != owner)
                check(cudaMemcpy(panel, staging, count * sizeof(PanelValue), cudaMemcpyHostToDevice));
        }
        updatePanel<false><<<local, tile>>>(dist, path, panel, left, pitch, origin, skip);
        updateTiles<<<dim3(layout.tiles, local), dim3(Tile, Tile / 4)>>>(
            dist, path, panel, left, pitch, origin, skip);
        check(cudaGetLastError());
    }
    check(cudaDeviceSynchronize());
    if (staging) check(cudaFreeHost(staging));
    check(cudaFree(left));
    check(cudaFree(panel));
}

int run(int argc, char** argv, int rank, int ranks) {
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            char* end;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (*end || end == argv[i] || errno == ERANGE || value < 0 || value > INT_MAX - Tile) {
                if (!rank) fprintf(stderr, "Invalid node count: %s\n", argv[i]);
                return 1;
            }
            n = int(value);
        } else if (strcmp(argv[i], "-v") == 0) validate = true;
        else if (strcmp(argv[i], "-r") == 0) results = true;
        else if (strcmp(argv[i], "-h") == 0) {
            if (!rank) printUsage(argv[0]);
            return 0;
        } else {
            if (!rank) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); }
            return 1;
        }
    }
    int tiles = (n + Tile - 1) / Tile;
    Partition layout{tiles, std::min(ranks, std::max(1, tiles))};
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < layout.ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    if (comm == MPI_COMM_NULL) return 0;
    MPI_Comm node;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
    int localRank, devices;
    MPI_Comm_rank(node, &localRank);
    MPI_Comm_free(&node);
    check(cudaGetDeviceCount(&devices));
    if (!devices) { fprintf(stderr, "CUDA device required\n"); MPI_Abort(comm, 1); }
    check(cudaSetDevice(localRank % devices));

    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %d\n", n);
        printf("Validation: %s\nInitializing graph...\n", validate ? "enabled" : "disabled");
    }
    size_t pitch = size_t(tiles) * Tile;
    int first = layout.first(rank) * Tile;
    int paddedRows = layout.first(rank + 1) * Tile - first;
    int rows = std::min(n - first, paddedRows);
    size_t elements = size_t(paddedRows) * pitch;
    std::vector<Value> host(elements);
    #pragma omp parallel num_threads(std::max(1, std::min(rows, omp_get_max_threads())))
    {
        int thread = omp_get_thread_num();
        int threads = omp_get_num_threads();
        int begin = size_t(rows) * thread / threads;
        int end = size_t(rows) * (thread + 1) / threads;
        Value seed = rowSeed(size_t(first + begin) * n);
        for (int i = begin; i < end; ++i) {
            Value* row = host.data() + i * pitch;
            for (int j = 0; j < n; ++j)
                row[j] = 1 + Value(double(MAX_DISTANCE) * rand_r(&seed) / RAND_MAX);
            row[first + i] = 0;
            std::fill(row + n, row + pitch, Padding);
        }
    }
    std::fill(host.begin() + rows * pitch, host.end(), Padding);
    Value *dist = nullptr, *path = nullptr;
    if (n) {
        dist = deviceArray<Value>(elements);
        path = deviceArray<Value>(elements);
        check(cudaMemcpy(dist, host.data(), elements * sizeof(Value), cudaMemcpyHostToDevice));
        initPaths<<<dim3((pitch + 255) / 256, paddedRows / Tile), 256>>>(path, pitch, first);
        check(cudaDeviceSynchronize());
    }
    if (!rank) printf("Computing shortest paths...\n");
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    if (n) compute(dist, path, pitch, layout, rank, comm);
    double elapsed = MPI_Wtime() - start, seconds;
    MPI_Reduce(&elapsed, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (!rank) {
        printf("Computation time: %.3f ms\n", seconds * 1000);
        printf("Performance: %.3f GOPS\n", seconds ? double(n) * n * n / seconds / 1e9 : 0.);
    }
    int failed = 0;
    if (results || validate) {
        if (n) check(cudaMemcpy2D(host.data(), size_t(n) * sizeof(Value), dist, pitch * sizeof(Value),
                                  size_t(n) * sizeof(Value), rows, cudaMemcpyDeviceToHost));
        if (!rank) host.resize(size_t(n) * n);
        // Gather packed row slabs in chunks to respect MPI's int count limit.
        for (int src = 1; src < layout.ranks; ++src) {
            if (rank && rank != src) continue;
            size_t begin = size_t(layout.first(src)) * Tile * n;
            size_t end = size_t(std::min(n, layout.first(src + 1) * Tile)) * n;
            for (size_t offset = begin; offset < end; ) {
                int chunk = std::min<size_t>(INT_MAX, end - offset);
                if (!rank) MPI_Recv(host.data() + offset, chunk, MPI_UNSIGNED, src, 0, comm, MPI_STATUS_IGNORE);
                else MPI_Send(host.data() + (offset - begin), chunk, MPI_UNSIGNED, 0, 0, comm);
                offset += chunk;
            }
        }
        if (!rank && results) print_results_int(host, "DistanceMatrix");
        if (!rank && validate) {
            printf("Validating result...\n");
            failed = !validateResult(host, n);
            printf("Validation: %s\n", failed ? "FAILED" : "PASSED");
        }
    }
    check(cudaFree(path));
    check(cudaFree(dist));
    MPI_Comm_free(&comm);
    return failed;
}

int main(int argc, char** argv) {
    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int status = run(argc, argv, rank, ranks);
    MPI_Finalize();
    return status;
}
