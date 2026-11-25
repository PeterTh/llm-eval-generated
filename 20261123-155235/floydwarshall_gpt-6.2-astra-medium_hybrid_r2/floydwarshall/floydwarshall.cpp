#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned INF = 1000000000;
// Every real pair starts with a direct edge of weight at most 201.
// Keep stored distances in one byte and add candidates in unsigned registers.
using Distance = unsigned char;
constexpr Distance padding = UCHAR_MAX;
constexpr int tile = 64;
constexpr int lanesY = 8;

inline size_t idx2(size_t column, size_t row, size_t stride) {
    return row * stride + column;
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

void checkCuda(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// glibc rand_r advances its 32-bit linear congruential state three times per
// sample. Compose powers of that transformation to start any row independently.
unsigned rowSeed(size_t offset) {
    unsigned seed = 42;
#ifdef __GLIBC__
    unsigned factor = 1103515245u, add = 12345u;
    for (size_t steps = 3 * offset; steps; steps >>= 1) {
        if (steps & 1) seed = factor * seed + add;
        add *= factor + 1;
        factor *= factor;
    }
#else
    for (size_t i = 0; i < offset; ++i) rand_r(&seed);
#endif
    return seed;
}

void initialize(Distance* data, size_t first, size_t rows, size_t n, size_t stride) {
    // Each thread consumes one contiguous slice of the original random stream.
    int threads = std::min<size_t>(omp_get_max_threads(), rows);
    #pragma omp parallel num_threads(threads)
    {
        size_t begin = rows * omp_get_thread_num() / omp_get_num_threads();
        size_t end = rows * (omp_get_thread_num() + 1) / omp_get_num_threads();
        unsigned seed = rowSeed(std::min(n, first + begin) * n);
        for (size_t r = begin; r < end; ++r) {
            Distance* row = data + r * stride;
            size_t count = first + r < n ? n : 0;
            for (size_t j = 0; j < count; ++j)
                row[j] = 1 + static_cast<unsigned>(200.0 * rand_r(&seed) / RAND_MAX);
            std::fill(row + count, row + stride, padding);
            row[first + r] = 0;
        }
    }
}

__global__ void closeDiagonal(Distance* diag, size_t stride) {
    __shared__ unsigned d[tile][tile];
    int x = threadIdx.x, y = threadIdx.y;
    for (int i = y; i < tile; i += lanesY)
        d[i][x] = diag[i * stride + x];
    __syncthreads();
    for (int k = 0; k < tile; ++k) {
        // The pivot row and column are immutable during this iteration.
        for (int i = y; i < tile; i += lanesY)
            if (i != k && x != k) d[i][x] = min(d[i][x], d[i][k] + d[k][x]);
        __syncthreads();
    }
    for (int i = y; i < tile; i += lanesY)
        diag[i * stride + x] = d[i][x];
}

// Each block computes one min-plus tile product. Once the diagonal tile
// is closed, both panel updates and the trailing update have this form.
enum Phase { PivotRow, PivotColumn, Trailing };
template<Phase phase>
__global__ void update(Distance* local, const Distance* panel, size_t stride,
                       int pivot, int skipRow) {
    int tr = blockIdx.y;
    if (phase != PivotRow && tr >= skipRow) ++tr;
    int tc = blockIdx.x;
    if (phase != PivotColumn && tc >= pivot) ++tc;
    if (phase == PivotColumn) tc = pivot;
    size_t r = size_t(tr) * tile * stride;
    size_t col = size_t(tc) * tile, pc = size_t(pivot) * tile;
    Distance* result = local + r + col;
    const Distance* left = local + r + pc;
    const Distance* right = panel + col;
    int x = threadIdx.x, y = threadIdx.y;
    __shared__ unsigned a[tile][tile], b[tile][tile];
    unsigned best[tile / lanesY];
    #pragma unroll
    for (int t = 0; t < tile / lanesY; ++t) {
        size_t offset = (y + t * lanesY) * stride + x;
        a[y + t * lanesY][x] = left[offset];
        b[y + t * lanesY][x] = right[offset];
        best[t] = result[offset];
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < tile; ++k) {
        unsigned step = b[k][x];
        #pragma unroll
        for (int t = 0; t < tile / lanesY; ++t)
            best[t] = min(best[t], a[y + t * lanesY][k] + step);
    }
    #pragma unroll
    for (int t = 0; t < tile / lanesY; ++t)
        result[(y + t * lanesY) * stride + x] = best[t];
}

int firstTile(int rank, int tiles, int ranks) {
    return rank * (tiles / ranks) + std::min(rank, tiles % ranks);
}

void solve(Distance* local, Distance* remote, Distance* staging, size_t stride,
           int tiles, int rank, int ranks, MPI_Comm comm) {
    dim3 threads(tile, lanesY);
    int first = firstTile(rank, tiles, ranks);
    int rows = firstTile(rank + 1, tiles, ranks) - first;
    for (int owner = 0; owner < ranks; ++owner) {
        int begin = firstTile(owner, tiles, ranks);
        int end = firstTile(owner + 1, tiles, ranks);
        for (int pivot = begin; pivot < end; ++pivot) {
            Distance* panel = remote;
            int skip = rows;
            if (owner == rank) {
                skip = pivot - first;
                panel = local + size_t(skip) * tile * stride;
                closeDiagonal<<<1, threads>>>(panel + size_t(pivot) * tile, stride);
                if (tiles > 1)
                    update<PivotRow><<<tiles - 1, threads>>>(panel, panel, stride, pivot, 0);
            }
            size_t count = tile * stride;
            auto updateLocalRows = [&] {
                int remaining = rows - (owner == rank);
                if (remaining) {
                    update<PivotColumn><<<dim3(1, remaining), threads>>>(local, panel, stride, pivot, skip);
                    update<Trailing><<<dim3(tiles - 1, remaining), threads>>>(local, panel, stride, pivot, skip);
                }
            };
            if (owner == rank) {
                if (ranks > 1)
                    checkCuda(cudaMemcpy(staging, panel, count * sizeof(Distance), cudaMemcpyDeviceToHost));
                // The GPU can update the owner's rows while its CPU broadcasts.
                updateLocalRows();
            }
            if (ranks > 1) {
                for (size_t off = 0; off < count; off += INT_MAX)
                    MPI_Bcast(staging + off, std::min<size_t>(count - off, INT_MAX),
                              MPI_UNSIGNED_CHAR, owner, comm);
                if (owner != rank) {
                    checkCuda(cudaMemcpy(remote, staging, count * sizeof(Distance), cudaMemcpyHostToDevice));
                    updateLocalRows();
                }
            }
        }
    }
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int rank, world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    size_t n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) n = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate = true;
        else if (!strcmp(argv[i], "-r")) results = true;
        else {
            bool help = !strcmp(argv[i], "-h");
            if (!rank) {
                if (!help) printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return help ? 0 : 1;
        }
    }
    if (n > INT_MAX) MPI_Abort(MPI_COMM_WORLD, 1);
    int tiles = (n + tile - 1) / tile;
    int ranks = std::min(world, std::max(1, tiles));
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, rank < ranks ? 0 : MPI_UNDEFINED, rank, &comm);
    int valid = 1;
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\nNumber of nodes: %zu\n", n);
        printf("Validation: %s\nInitializing graph...\n", validate ? "enabled" : "disabled");
    }
    if (rank < ranks) {
        MPI_Comm node;
        MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank, devices;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_free(&node);
        checkCuda(cudaGetDeviceCount(&devices));
        if (!devices) MPI_Abort(MPI_COMM_WORLD, 1);
        checkCuda(cudaSetDevice(localRank % devices));

        size_t stride = size_t(tiles) * tile;
        size_t first = size_t(firstTile(rank, tiles, ranks)) * tile;
        size_t end = size_t(firstTile(rank + 1, tiles, ranks)) * tile;
        size_t rows = end - first;
        Distance *local = nullptr, *remote = nullptr, *staging = nullptr;
        if (rows) {
            std::vector<Distance> input(rows * stride);
            initialize(input.data(), first, rows, n, stride);
            size_t bytes = input.size() * sizeof(Distance);
            checkCuda(cudaMalloc(&local, bytes));
            checkCuda(cudaMemcpy(local, input.data(), bytes, cudaMemcpyHostToDevice));
        }
        if (ranks > 1) {
            checkCuda(cudaMalloc(&remote, tile * stride * sizeof(Distance)));
            checkCuda(cudaMallocHost(&staging, tile * stride * sizeof(Distance)));
        }
        checkCuda(cudaDeviceSynchronize());
        if (!rank) printf("Computing shortest paths...\n");
        MPI_Barrier(comm);
        double start = MPI_Wtime();
        solve(local, remote, staging, stride, tiles, rank, ranks, comm);
        double elapsed = MPI_Wtime() - start, longest;
        MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        if (!rank) {
            printf("Computation time: %lld ms\n", static_cast<long long>(longest * 1000));
            printf("Performance: %.3f GOPS\n", double(n) * n * n / longest / 1e9);
        }
        if (results || validate) {
            size_t count = (std::min(end, n) - first) * n;
            std::vector<unsigned> dist(rank ? count : n * n);
            if (count) {
                std::vector<Distance> packed(count);
                checkCuda(cudaMemcpy2D(packed.data(), n * sizeof(Distance), local, stride * sizeof(Distance),
                                       n * sizeof(Distance), count / n, cudaMemcpyDeviceToHost));
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < count; ++i) dist[i] = packed[i];
            }
            if (rank) {
                for (size_t off = 0; off < count; off += INT_MAX)
                    MPI_Send(dist.data() + off, std::min<size_t>(count - off, INT_MAX), MPI_UNSIGNED, 0, 0, comm);
            } else {
                for (int peer = 1; peer < ranks; ++peer) {
                    size_t off = size_t(firstTile(peer, tiles, ranks)) * tile * n;
                    size_t stop = std::min(n, size_t(firstTile(peer + 1, tiles, ranks)) * tile) * n;
                    for (; off < stop; off += INT_MAX)
                        MPI_Recv(dist.data() + off, std::min<size_t>(stop - off, INT_MAX),
                                 MPI_UNSIGNED, peer, 0, comm, MPI_STATUS_IGNORE);
                }
                if (results) print_results_int(dist, "DistanceMatrix");
                if (validate) {
                    printf("Validating result...\n");
                    valid = validateResult(dist, n);
                    printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
                }
            }
        }
        if (local) checkCuda(cudaFree(local));
        if (remote) checkCuda(cudaFree(remote));
        if (staging) checkCuda(cudaFreeHost(staging));
        MPI_Comm_free(&comm);
    }
    MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return valid ? 0 : 1;
}
