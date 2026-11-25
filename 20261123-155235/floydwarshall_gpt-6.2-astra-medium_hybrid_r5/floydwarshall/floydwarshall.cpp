#include <cuda_runtime.h>
#include <mpi.h>
#if defined(OPEN_MPI) && __has_include(<mpi-ext.h>)
#include <mpi-ext.h>
#endif
#include <omp.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

using Word = unsigned int;
constexpr Word INF = 1000000000;
constexpr int B = 64;
constexpr int Y = 8;

void check(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(status));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

template<class T>
void deviceAlloc(T*& ptr, size_t count) {
    check(cudaMalloc(&ptr, count * sizeof(T)));
}

// Keep the pivot column as it stood immediately before each scalar k step.
__global__ void diagonalHistory(const Word* tile, size_t stride, Word* history) {
    __shared__ Word d[B][B];
    int x = threadIdx.x;
    for (int y = threadIdx.y; y < B; y += Y)
        d[y][x] = tile[y * stride + x];
    __syncthreads();
    for (int k = 0; k < B; ++k) {
        for (int y = threadIdx.y; y < B; y += Y) {
            Word left = d[y][k];
            if (x == k) history[y * B + k] = left;
            if (y != k && x != k)
                d[y][x] = min(d[y][x], left + d[k][x]);
        }
        __syncthreads();
    }
}

// Independently evolve each tile of the pivot block row, exporting row k
// before relaxing through k. Using these snapshots preserves strict-tie paths.
__global__ void rowHistory(const Word* rows, size_t stride, const Word* diagonal,
                           Word* panel) {
    __shared__ Word d[B][B];
    __shared__ Word left[B][B];
    int x = threadIdx.x;
    size_t col = size_t(blockIdx.x) * B + x;
    for (int y = threadIdx.y; y < B; y += Y) {
        d[y][x] = rows[y * stride + col];
        left[y][x] = diagonal[y * B + x];
    }
    __syncthreads();
    for (int k = 0; k < B; ++k) {
        Word pivot = d[k][x];
        if (threadIdx.y == 0) panel[k * stride + col] = pivot;
        for (int y = threadIdx.y; y < B; y += Y)
            if (y != k) d[y][x] = min(d[y][x], left[y][k] + pivot);
        __syncthreads();
    }
}

__global__ void columnHistory(const Word* column, size_t stride, const Word* panel,
                              Word* history) {
    __shared__ Word d[B][B];
    __shared__ Word top[B][B];
    int x = threadIdx.x;
    size_t row = size_t(blockIdx.x) * B;
    for (int y = threadIdx.y; y < B; y += Y) {
        d[y][x] = column[(row + y) * stride + x];
        top[y][x] = panel[y * stride + x];
    }
    __syncthreads();
    for (int k = 0; k < B; ++k) {
        for (int y = threadIdx.y; y < B; y += Y) {
            Word pivot = d[y][k];
            if (x == k) history[(row + y) * B + x] = pivot;
            else d[y][x] = min(d[y][x], pivot + top[k][x]);
        }
        __syncthreads();
    }
}

__global__ void relax(Word* dist, Word* path, size_t stride, int firstRow, int pivot,
                      const Word* column, const Word* panel, unsigned skipTile) {
    __shared__ Word left[B][B];
    __shared__ Word top[B][B];
    int x = threadIdx.x;
    size_t row = (blockIdx.y + (blockIdx.y >= skipTile)) * B;
    size_t col = size_t(blockIdx.x) * B + x;
    for (int y = threadIdx.y; y < B; y += Y) {
        left[y][x] = column[(row + y) * B + x];
        top[y][x] = panel[y * stride + col];
    }
    __syncthreads();
    #pragma unroll
    for (int r = 0; r < B / Y; ++r) {
        int y = threadIdx.y + r * Y;
        size_t index = (row + y) * stride + col;
        Word best = dist[index];
        int via = -1;
        #pragma unroll
        for (int k = 0; k < B; ++k) {
            Word candidate = left[y][k] + top[k][x];
            if (candidate < best) {
                best = candidate;
                via = pivot + k;
            }
        }
        dist[index] = best;
        if (via >= 0) path[index] = via;
        else if (pivot == 0) path[index] = firstRow + row + y;
    }
}

// glibc rand_r advances its 32-bit LCG three times per result. Compose
// affine transforms by squaring so every row starts at the serial seed.
Word rowSeed(size_t offset) {
    Word seed = 42;
#ifdef __GLIBC__
    Word a = 1103515245u, b = 12345u;
    Word step = a * a * a, shift = (a * a + a + 1u) * b;
    while (offset) {
        if (offset & 1) seed = step * seed + shift;
        shift *= step + 1u;
        step *= step;
        offset >>= 1;
    }
#else
    while (offset--) rand_r(&seed);
#endif
    return seed;
}

bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
    // Basic sanity checks
    
    // 1. Diagonal should be zero
    for (size_t i = 0; i < numNodes; ++i) {
        if (dist[i * numNodes + i] != 0) {
            printf("Validation failed: diagonal element [%zu,%zu] is not zero\n", i, i);
            return false;
        }
    }
    
    // 2. Triangle inequality: dist[i][k] + dist[k][j] >= dist[i][j]
    // Check a sample of paths to avoid O(n^3) validation time
    for (size_t i = 0; i < std::min(numNodes, static_cast<size_t>(10)); ++i) {
        for (size_t j = 0; j < std::min(numNodes, static_cast<size_t>(10)); ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[i * numNodes + j];
                const unsigned int distIK = dist[i * numNodes + k];
                const unsigned int distKJ = dist[k * numNodes + j];
                
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

int partitionStart(int part, int total, int parts) {
    return part * (total / parts) + std::min(part, total % parts);
}

bool run(int n, bool validate, bool results, int localRank, MPI_Comm comm) {
    int rank, ranks, devices;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &ranks);
    check(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check(cudaSetDevice(localRank % devices));

    int tiles = (n + B - 1) / B;
    size_t stride = size_t(tiles) * B;
    int first = partitionStart(rank, tiles, ranks) * B;
    int end = partitionStart(rank + 1, tiles, ranks) * B;
    int rows = end - first;
    size_t count = size_t(rows) * stride;
    Word *dist = nullptr, *path = nullptr, *column = nullptr, *diagonal = nullptr;
    Word *panel = nullptr, *staging = nullptr;
    if (rank == 0) printf("Initializing graph...\n");
    if (n) {
        deviceAlloc(dist, count);
        deviceAlloc(path, count);
        deviceAlloc(column, size_t(rows) * B);
        deviceAlloc(diagonal, B * B);
        deviceAlloc(panel, (ranks > 1 ? 2 : 1) * B * stride);
        bool deviceMPI = false;
#if defined(OMPI_HAVE_MPI_EXT_CUDA) && OMPI_HAVE_MPI_EXT_CUDA
        deviceMPI = MPIX_Query_cuda_support() != 0;
#endif
        if (ranks > 1 && !deviceMPI)
            check(cudaMallocHost(&staging, B * stride * sizeof(Word)));

        std::vector<Word> input(count);
        #pragma omp parallel num_threads(std::min(rows, omp_get_max_threads()))
        {
            int thread = omp_get_thread_num(), team = omp_get_num_threads();
            int beginRow = first + partitionStart(thread, rows, team);
            int endRow = first + partitionStart(thread + 1, rows, team);
            Word seed = rowSeed(size_t(std::min(beginRow, n)) * n);
            for (int i = beginRow; i < endRow; ++i) {
                Word* dst = input.data() + size_t(i - first) * stride;
                int valid = i < n ? n : 0;
                for (int j = 0; j < valid; ++j)
                    dst[j] = 1u + static_cast<Word>(200.0 * rand_r(&seed) / RAND_MAX);
                std::fill(dst + valid, dst + stride, INF);
                dst[i] = 0;
            }
        }
        check(cudaMemcpy(dist, input.data(), count * sizeof(Word), cudaMemcpyHostToDevice));
    }
    // While the devices update the bulk of their slabs, the next owner
    // prepares and broadcasts its pivot panel on an independent stream.
    cudaStream_t exchange = nullptr;
    cudaEvent_t pivotReady{}, uploaded{}, consumed[2]{};
    if (ranks > 1) {
        check(cudaStreamCreateWithFlags(&exchange, cudaStreamNonBlocking));
        for (auto e : {&pivotReady, &uploaded, &consumed[0], &consumed[1]})
            check(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    }
    if (rank == 0) printf("Computing shortest paths...\n");
    check(cudaDeviceSynchronize());
    MPI_Barrier(comm);
    double start = MPI_Wtime();
    dim3 threads(B, Y);
    int owner = 0;
    for (int t = 0; t < tiles; ++t) {
        if (t == partitionStart(owner + 1, tiles, ranks)) ++owner;
        int pivot = t * B;
        size_t panelSize = B * stride;
        int slot = ranks > 1 ? t % 2 : 0;
        Word* current = panel + slot * panelSize;
        if (ranks > 1) check(cudaStreamWaitEvent(exchange, consumed[slot], 0));
        if (rank == owner) {
            if (ranks > 1) check(cudaStreamWaitEvent(exchange, pivotReady, 0));
            const Word* pivotRows = dist + size_t(pivot - first) * stride;
            diagonalHistory<<<1, threads, 0, exchange>>>(pivotRows + pivot, stride, diagonal);
            rowHistory<<<tiles, threads, 0, exchange>>>(pivotRows, stride, diagonal, current);
            if (staging)
                check(cudaMemcpyAsync(staging, current, panelSize * sizeof(Word),
                                      cudaMemcpyDeviceToHost, exchange));
        }
        if (ranks > 1) {
            // Finish the outgoing copy (or the last use of staging on a receiver).
            check(cudaStreamSynchronize(exchange));
            for (size_t offset = 0; offset < panelSize;) {
                int chunk = std::min<size_t>(panelSize - offset, INT_MAX);
                MPI_Bcast((staging ? staging : current) + offset, chunk, MPI_UNSIGNED, owner, comm);
                offset += chunk;
            }
            if (staging && rank != owner)
                check(cudaMemcpyAsync(current, staging, panelSize * sizeof(Word),
                                      cudaMemcpyHostToDevice, exchange));
            check(cudaEventRecord(uploaded, exchange));
            check(cudaStreamWaitEvent(nullptr, uploaded, 0));
        }
        int localTiles = rows / B;
        columnHistory<<<localTiles, threads>>>(dist + pivot, stride, current + pivot, column);
        int skip = localTiles;
        int nextRow = pivot + B;
        if (ranks > 1 && nextRow >= first && nextRow < end) {
            skip = (nextRow - first) / B;
            size_t offset = size_t(skip) * B;
            relax<<<tiles, threads>>>(dist + offset * stride, path + offset * stride,
                                      stride, nextRow, pivot, column + offset * B, current, 1);
            check(cudaEventRecord(pivotReady));
            --localTiles;
        }
        if (localTiles)
            relax<<<dim3(tiles, localTiles), threads>>>(dist, path, stride, first, pivot,
                                                        column, current, skip);
        if (ranks > 1) check(cudaEventRecord(consumed[slot]));
        check(cudaGetLastError());
    }
    check(cudaDeviceSynchronize());
    double localTime = MPI_Wtime() - start, elapsed = 0;
    MPI_Reduce(&localTime, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    if (rank == 0) {
        printf("Computation time: %lld ms\n", static_cast<long long>(elapsed * 1000));
        printf("Performance: %.3f GOPS\n", elapsed ? double(n) * n * n / elapsed / 1e9 : 0);
    }
    bool valid = true;
    if (results || validate) {
        std::vector<Word> full(rank == 0 ? size_t(n) * n : 0);
        if (n) {
            int actualRows = std::min(end, n) - first;
            std::vector<Word> local(rank ? size_t(actualRows) * n : 0);
            Word* dst = rank ? local.data() : full.data();
            size_t width = size_t(n) * sizeof(Word);
            check(cudaMemcpy2D(dst, width, dist, stride * sizeof(Word), width,
                               actualRows, cudaMemcpyDeviceToHost));
            MPI_Datatype rowType;
            MPI_Type_contiguous(n, MPI_UNSIGNED, &rowType);
            MPI_Type_commit(&rowType);
            std::vector<int> counts(ranks), offsets(ranks);
            for (int r = 0; r < ranks; ++r) {
                offsets[r] = partitionStart(r, tiles, ranks) * B;
                counts[r] = std::min(n, partitionStart(r + 1, tiles, ranks) * B) - offsets[r];
            }
            MPI_Gatherv(rank ? dst : MPI_IN_PLACE, actualRows, rowType, full.data(),
                        counts.data(), offsets.data(), rowType, 0, comm);
            MPI_Type_free(&rowType);
        }
        if (!rank && results) print_results_int(full, "DistanceMatrix");
        if (!rank && validate) {
            printf("Validating result...\n");
            valid = validateResult(full, n);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
        }
    }
    if (ranks > 1) {
        for (auto e : {pivotReady, uploaded, consumed[0], consumed[1]})
            check(cudaEventDestroy(e));
        check(cudaStreamDestroy(exchange));
        if (staging) check(cudaFreeHost(staging));
    }
    for (auto p : {dist, path, panel, diagonal, column}) check(cudaFree(p));
    return valid;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);
    int n = 512;
    bool validate = false, results = false;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            n = atoi(argv[++i]);
            if (n < 0 || n > INT_MAX - B) {
                if (!rank) fprintf(stderr, "Invalid number of nodes\n");
                MPI_Finalize();
                return 1;
            }
        } else if (!strcmp(argv[i], "-v")) validate = true;
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
    if (!rank) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %d\n", n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    MPI_Comm active;
    int workers = std::max(1, (n + B - 1) / B);
    MPI_Comm_split(MPI_COMM_WORLD, rank < workers ? 0 : MPI_UNDEFINED, rank, &active);
    int status = 0;
    if (active != MPI_COMM_NULL) {
        MPI_Comm node;
        MPI_Comm_split_type(active, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node);
        int localRank;
        MPI_Comm_rank(node, &localRank);
        MPI_Comm_free(&node);
        try {
            status = run(n, validate, results, localRank, active) ? 0 : 1;
        } catch (const std::exception& e) {
            fprintf(stderr, "Rank %d: %s\n", rank, e.what());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        MPI_Comm_free(&active);
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
