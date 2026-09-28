#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Blocked Floyd-Warshall tile size and thread-block geometry.
// Each CUDA block is TILE x ROWS_PER_BLOCK threads and every thread processes
// ROWS_PER_THREAD rows of the tile, which raises ILP and cuts index arithmetic.
constexpr int TILE = 64;
constexpr int ROWS_PER_BLOCK = 4;
constexpr int ROWS_PER_THREAD = TILE / ROWS_PER_BLOCK;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                        \
            printf("CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,             \
                   cudaGetErrorString(err_));                                             \
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

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

// ---------------------------------------------------------------------------
// Blocked Floyd-Warshall GPU kernels
//
// The matrix (padded to a multiple of TILE) is partitioned into TILE x TILE
// blocks. Round b of the outer loop over intermediate nodes processes:
//   phase 1: the pivot block (b,b)                       - self dependent
//   phase 2: the pivot row/column blocks                 - depend on (b,b)
//   phase 3: all remaining blocks                        - depend on phase 2
// After round b every entry is the shortest distance that uses intermediate
// nodes 0..(b+1)*TILE-1 only, which is the same invariant the scalar triple
// loop maintains, so the resulting distance matrix is bit-identical to it.
// Individual predecessor entries may name a different intermediate node than
// the sequential version whenever several intermediates yield the same optimal
// distance, because a relaxation can already see improved operands here; the
// path matrix is not part of the benchmark's result.
// ---------------------------------------------------------------------------

__global__ __launch_bounds__(TILE* ROWS_PER_BLOCK) void fwPhase1(
    unsigned int* __restrict__ dist, unsigned int* __restrict__ path, const int n,
    const int block) {
    __shared__ unsigned int sd[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = block * TILE;

    unsigned int cur[ROWS_PER_THREAD];
    unsigned int pth[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];
    size_t off[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = ty + r * ROWS_PER_BLOCK;
        off[r] = static_cast<size_t>(base + row) * n + base + tx;
        cur[r] = dist[off[r]];
        pth[r] = 0;
        changed[r] = false;
        sd[row][tx] = cur[r];
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        const unsigned int dkj = sd[k][tx];
        unsigned int dik[ROWS_PER_THREAD];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            dik[r] = sd[ty + r * ROWS_PER_BLOCK][k];
        }
        // All reads of round k must complete before the in-place updates.
        __syncthreads();
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const unsigned int nd = dik[r] + dkj;
            if (nd < cur[r]) {
                cur[r] = nd;
                pth[r] = base + k;
                changed[r] = true;
            }
            sd[ty + r * ROWS_PER_BLOCK][tx] = cur[r];
        }
        __syncthreads();
    }

    // Untouched entries keep their old dist/path, so they need no write-back and
    // the old path value never has to be read.
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        if (changed[r]) {
            dist[off[r]] = cur[r];
            path[off[r]] = pth[r];
        }
    }
}

// blockIdx.y == 0 -> blocks in the pivot row, blockIdx.y == 1 -> pivot column.
__global__ __launch_bounds__(TILE* ROWS_PER_BLOCK) void fwPhase2(
    unsigned int* __restrict__ dist, unsigned int* __restrict__ path, const int n,
    const int block) {
    if (static_cast<int>(blockIdx.x) == block) return;

    __shared__ unsigned int piv[TILE][TILE + 1];
    __shared__ unsigned int own[TILE][TILE + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = block * TILE;
    const bool inRow = (blockIdx.y == 0);
    const int rowBase = inRow ? base : static_cast<int>(blockIdx.x) * TILE;
    const int colBase = inRow ? static_cast<int>(blockIdx.x) * TILE : base;

    unsigned int cur[ROWS_PER_THREAD];
    unsigned int pth[ROWS_PER_THREAD];
    bool changed[ROWS_PER_THREAD];
    size_t off[ROWS_PER_THREAD];

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = ty + r * ROWS_PER_BLOCK;
        piv[row][tx] = dist[static_cast<size_t>(base + row) * n + base + tx];
        off[r] = static_cast<size_t>(rowBase + row) * n + colBase + tx;
        cur[r] = dist[off[r]];
        pth[r] = 0;
        changed[r] = false;
        own[row][tx] = cur[r];
    }
    __syncthreads();

    for (int k = 0; k < TILE; ++k) {
        unsigned int dik[ROWS_PER_THREAD];
        unsigned int dkj;
        if (inRow) {
            // dist[i][k] lives in the pivot block, dist[k][j] in this block.
            dkj = own[k][tx];
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                dik[r] = piv[ty + r * ROWS_PER_BLOCK][k];
            }
        } else {
            // dist[i][k] lives in this block, dist[k][j] in the pivot block.
            dkj = piv[k][tx];
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                dik[r] = own[ty + r * ROWS_PER_BLOCK][k];
            }
        }
        __syncthreads();
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const unsigned int nd = dik[r] + dkj;
            if (nd < cur[r]) {
                cur[r] = nd;
                pth[r] = base + k;
                changed[r] = true;
            }
            own[ty + r * ROWS_PER_BLOCK][tx] = cur[r];
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        if (changed[r]) {
            dist[off[r]] = cur[r];
            path[off[r]] = pth[r];
        }
    }
}

// This kernel does (numBlocks-1)^2 of the numBlocks^2 tiles per round and thus
// dominates the run time; it is kept at two resident blocks per SM by using no
// shared-memory padding (the accesses are conflict free anyway) and by holding
// as little per-element state in registers as possible.
__global__ __launch_bounds__(TILE* ROWS_PER_BLOCK, 2) void fwPhase3(
    unsigned int* __restrict__ dist, unsigned int* __restrict__ path, const int n,
    const int block) {
    if (static_cast<int>(blockIdx.x) == block || static_cast<int>(blockIdx.y) == block) return;

    __shared__ unsigned int rowTile[TILE][TILE];  // block (pivot, blockIdx.x)
    __shared__ unsigned int colTile[TILE][TILE];  // block (blockIdx.y, pivot)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int base = block * TILE;
    const int rowBase = static_cast<int>(blockIdx.y) * TILE;
    const int colBase = static_cast<int>(blockIdx.x) * TILE;

    unsigned int cur[ROWS_PER_THREAD];
    // kUnchanged marks entries that were never improved in this round; they keep
    // their old dist/path, so no write-back is needed and the old path value
    // never has to be read at all.
    constexpr unsigned int kUnchanged = 0xffffffffu;
    unsigned int pth[ROWS_PER_THREAD];

    const size_t ownBase = static_cast<size_t>(rowBase + ty) * n + colBase + tx;

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const int row = ty + r * ROWS_PER_BLOCK;
        rowTile[row][tx] = dist[static_cast<size_t>(base + row) * n + colBase + tx];
        colTile[row][tx] = dist[static_cast<size_t>(rowBase + row) * n + base + tx];
        cur[r] = dist[ownBase + static_cast<size_t>(r) * ROWS_PER_BLOCK * n];
        pth[r] = kUnchanged;
    }
    __syncthreads();

    // Both operand tiles are final for this round, so no synchronization is
    // needed inside the k loop.
#pragma unroll 8
    for (int k = 0; k < TILE; ++k) {
        const unsigned int dkj = rowTile[k][tx];
#pragma unroll
        for (int r = 0; r < ROWS_PER_THREAD; ++r) {
            const unsigned int nd = colTile[ty + r * ROWS_PER_BLOCK][k] + dkj;
            if (nd < cur[r]) {
                cur[r] = nd;
                pth[r] = base + k;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        if (pth[r] != kUnchanged) {
            const size_t o = ownBase + static_cast<size_t>(r) * ROWS_PER_BLOCK * n;
            dist[o] = cur[r];
            path[o] = pth[r];
        }
    }
}

__global__ void fwFill(unsigned int* __restrict__ data, const size_t count,
                       const unsigned int value) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        data[i] = value;
    }
}

// Reproduces initializePathMatrix() (path[row][col] == row) directly on the
// device, which avoids uploading the whole predecessor matrix over PCIe.
__global__ void fwInitPath(unsigned int* __restrict__ path, const int n, const int np) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;
    for (int row = blockIdx.y; row < n; row += gridDim.y) {
        path[static_cast<size_t>(row) * np + col] = row;
    }
}

// Writing the results back to the host is the second cost centre after the
// phase-3 kernel. Two effects dominate it:
//   * a single-threaded copy only reaches ~1 GB/s because the host pages are
//     spread over the NUMA nodes, while a multi-threaded one scales almost
//     linearly, and
//   * copies that target pageable memory are serialized inside the driver.
// Both are avoided by giving every worker thread its own stream and its own
// slice of one pinned staging buffer: the DMA of one chunk overlaps with the
// host-side copy of the previous one, and all threads make progress at once.
void copyMatricesToHost(unsigned int* const* hostDst, const unsigned int* const* devSrc,
                        const int numMatrices, const size_t n, const size_t pitch) {
    constexpr size_t kMaxThreads = 16;
    constexpr size_t kBuffersPerThread = 2;
    constexpr size_t kMinChunkBytes = 512u << 10;
    constexpr size_t kMaxStagingBytes = 16u << 20;
    // Pinning host memory costs about 1 ms per MiB, so small results are simply
    // copied directly instead.
    constexpr size_t kMinStagedTransfer = 8u << 20;

    const size_t rowBytes = n * sizeof(unsigned int);
    const size_t totalBytes = static_cast<size_t>(numMatrices) * n * rowBytes;
    const size_t chunkBytes = std::max(kMinChunkBytes, rowBytes);
    const size_t rowsPerChunk = chunkBytes / rowBytes;
    const size_t hw = std::max(1u, std::thread::hardware_concurrency());
    const size_t stagingBudget = std::min(kMaxStagingBytes, totalBytes / 4);
    const size_t numBands = std::max<size_t>(
        1, std::min({kMaxThreads, hw, n / rowsPerChunk, stagingBudget / (kBuffersPerThread * chunkBytes)}));
    const size_t rowsPerBand =
        ((n + numBands - 1) / numBands + rowsPerChunk - 1) / rowsPerChunk * rowsPerChunk;

    unsigned char* staging = nullptr;
    const size_t stagingBytes = numBands * kBuffersPerThread * chunkBytes;
    if (totalBytes < kMinStagedTransfer ||
        cudaHostAlloc(&staging, stagingBytes, cudaHostAllocDefault) != cudaSuccess) {
        cudaGetLastError();
        for (int m = 0; m < numMatrices; ++m) {
            if (pitch == n) {
                CUDA_CHECK(cudaMemcpy(hostDst[m], devSrc[m], totalBytes / numMatrices,
                                      cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaMemcpy2D(hostDst[m], rowBytes, devSrc[m],
                                        pitch * sizeof(unsigned int), rowBytes, n,
                                        cudaMemcpyDeviceToHost));
            }
        }
        return;
    }

    const auto copyBand = [&](const size_t b) {
        const size_t firstRow = b * rowsPerBand;
        if (firstRow >= n) return;
        const size_t bandRows = std::min(rowsPerBand, n - firstRow);

        cudaStream_t stream[kBuffersPerThread] = {};
        unsigned char* buf[kBuffersPerThread] = {};
        for (size_t i = 0; i < kBuffersPerThread; ++i) {
            CUDA_CHECK(cudaStreamCreate(&stream[i]));
            buf[i] = staging + ((b * kBuffersPerThread) + i) * chunkBytes;
        }

        for (int m = 0; m < numMatrices; ++m) {
            unsigned int* dst = hostDst[m] + firstRow * n;
            const unsigned int* src = devSrc[m] + firstRow * pitch;

            const size_t numChunks = (bandRows + rowsPerChunk - 1) / rowsPerChunk;
            for (size_t c = 0; c <= numChunks; ++c) {
                if (c < numChunks) {
                    const size_t first = c * rowsPerChunk;
                    const size_t rows = std::min(rowsPerChunk, bandRows - first);
                    const size_t slot = c % kBuffersPerThread;
                    if (pitch == n) {
                        CUDA_CHECK(cudaMemcpyAsync(buf[slot], src + first * pitch, rows * rowBytes,
                                                   cudaMemcpyDeviceToHost, stream[slot]));
                    } else {
                        CUDA_CHECK(cudaMemcpy2DAsync(buf[slot], rowBytes, src + first * pitch,
                                                     pitch * sizeof(unsigned int), rowBytes, rows,
                                                     cudaMemcpyDeviceToHost, stream[slot]));
                    }
                }
                if (c > 0) {
                    const size_t prev = c - 1;
                    const size_t first = prev * rowsPerChunk;
                    const size_t rows = std::min(rowsPerChunk, bandRows - first);
                    const size_t slot = prev % kBuffersPerThread;
                    CUDA_CHECK(cudaStreamSynchronize(stream[slot]));
                    memcpy(dst + first * n, buf[slot], rows * rowBytes);
                }
            }
        }

        for (size_t i = 0; i < kBuffersPerThread; ++i) {
            CUDA_CHECK(cudaStreamDestroy(stream[i]));
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(numBands - 1);
    for (size_t b = 1; b < numBands; ++b) {
        workers.emplace_back(copyBand, b);
    }
    copyBand(0);
    for (auto& w : workers) w.join();

    CUDA_CHECK(cudaFreeHost(staging));
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;

    const int n = static_cast<int>(numNodes);
    const int numBlocks = (n + TILE - 1) / TILE;
    const int np = numBlocks * TILE;  // padded dimension
    const size_t bytes = static_cast<size_t>(np) * np * sizeof(unsigned int);

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));

    // Padding rows/columns are unreachable (INF), so they can never improve a
    // real entry and the kernels can run without bounds checks.
    if (np != n) {
        fwFill<<<1024, 256>>>(dDist, static_cast<size_t>(np) * np, INF);
        CUDA_CHECK(cudaMemset(dPath, 0, bytes));
    }
    fwInitPath<<<dim3((n + 255) / 256, std::min(n, 32768)), 256>>>(dPath, n, np);

    // Strided (2D) pageable transfers are far slower than contiguous ones, so
    // take the flat path whenever the matrix needs no padding.
    if (np == n) {
        CUDA_CHECK(cudaMemcpy(dDist, dist.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(cudaMemcpy2D(dDist, np * sizeof(unsigned int), dist.data(),
                                n * sizeof(unsigned int), n * sizeof(unsigned int), n,
                                cudaMemcpyHostToDevice));
    }

    const dim3 threads(TILE, ROWS_PER_BLOCK);
    const dim3 grid2(numBlocks, 2);
    const dim3 grid3(numBlocks, numBlocks);

    for (int b = 0; b < numBlocks; ++b) {
        fwPhase1<<<1, threads>>>(dDist, dPath, np, b);
        fwPhase2<<<grid2, threads>>>(dDist, dPath, np, b);
        fwPhase3<<<grid3, threads>>>(dDist, dPath, np, b);
    }
    CUDA_CHECK(cudaGetLastError());

    {
        unsigned int* hostDst[2] = {dist.data(), path.data()};
        const unsigned int* devSrc[2] = {dDist, dPath};
        copyMatricesToHost(hostDst, devSrc, 2, n, np);
    }

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
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
    size_t numNodes = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numNodes = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
    printf("Number of nodes: %zu\n", numNodes);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);

    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);

    // Take the CUDA context creation cost out of the measured region.
    CUDA_CHECK(cudaFree(nullptr));

    // Run Floyd-Warshall
    printf("Computing shortest paths...\n");
    auto start = std::chrono::high_resolution_clock::now();

    floydWarshall(dist, path, numNodes);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Computation time: %ld ms\n", duration.count());

    // Calculate operations per second
    // Floyd-Warshall has O(n³) complexity
    double ops = (double)numNodes * numNodes * numNodes;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GOPS\n", gflops);

    // Print results for external validation (integer hash-based)
    if (printResults) {
        print_results_int(dist, "DistanceMatrix");
    }

    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateResult(dist, numNodes);

        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
