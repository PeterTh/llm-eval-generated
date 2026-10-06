#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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

#define CUDA_CHECK(call)                                                            \
    do {                                                                            \
        cudaError_t err_ = (call);                                                  \
        if (err_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),   \
                    __FILE__, __LINE__);                                            \
            exit(1);                                                                \
        }                                                                           \
    } while (0)

// Blocked (tiled) Floyd-Warshall on the GPU.
//
// To reproduce the path matrix of the classic algorithm exactly (it records the
// last k that strictly improved an entry), every entry carries a pair
// (dist, hop) ordered lexicographically, where hop = 1 + the largest
// intermediate node on the path (0 = direct edge). The classic algorithm's final
// path[i][j] is the smallest k such that a shortest i->j path only uses
// intermediates <= k, which is exactly the lexicographic minimum computed here,
// independent of the order in which the blocked algorithm relaxes entries.
//
// The pair is packed into one 32-bit key: key = (dist << S) | hop with
// 2^S > padded node count. Entries never increase, so every key stays bounded by
// the largest initial weight, which is checked to fit. Relaxing through k with
// A = key(i,k), B = key(k,j) and hi(x) = x with the hop bits cleared:
//     candidate = max(A + hi(B), hi(A) + B, hi(A) + hi(B) + (k + 1))
// i.e. (dist_ik + dist_kj, max(hop_ik, hop_kj, k + 1)).
constexpr int TILE = 64;        // tile size and k-block size
constexpr int P12_THREADS = 32; // phases 1/2: 32x32 threads, 2x2 entries each
constexpr int P12_ITEMS = TILE / P12_THREADS;
constexpr int P3_THREADS = 16;  // phase 3: 16x16 threads, P3_TM x P3_TN entries each
constexpr int P3_TM = 4;
constexpr int P3_TN = 8;
constexpr int P3_ROWS = P3_THREADS * P3_TM;  // output region of a phase-3 block
constexpr int P3_COLS = P3_THREADS * P3_TN;
constexpr size_t P3_SMEM = (P3_ROWS * TILE + TILE * P3_COLS) * sizeof(unsigned int);
// Padded matrix size granularity
constexpr int PAD_GRAN = (P3_ROWS > P3_COLS ? P3_ROWS : P3_COLS);
static_assert(PAD_GRAN % TILE == 0 && PAD_GRAN % P3_ROWS == 0 && PAD_GRAN % P3_COLS == 0,
              "tile sizes must nest");
static_assert(P3_TN % 4 == 0, "phase 3 uses 128-bit accesses along rows");

// In-tile Floyd-Warshall over the TILE k values of the current block.
// A(i,k) comes from sLeft, B(k,j) from sTop; sOwn is updated in place.
// At step k, row k and column k of the tile being updated cannot change
// (key(k,k) = 0), so a single barrier per step is sufficient.
__device__ __forceinline__ void closeTile(unsigned int (*sOwn)[TILE],
                                          const unsigned int (*sLeft)[TILE],
                                          const unsigned int (*sTop)[TILE],
                                          const int base, const unsigned int hiMask) {
    const int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int cur[P12_ITEMS][P12_ITEMS];
#pragma unroll
    for (int a = 0; a < P12_ITEMS; ++a)
#pragma unroll
        for (int b = 0; b < P12_ITEMS; ++b)
            cur[a][b] = sOwn[ty + a * P12_THREADS][tx + b * P12_THREADS];

#pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        const unsigned int kk = base + k + 1;
        unsigned int av[P12_ITEMS], bv[P12_ITEMS];
#pragma unroll
        for (int a = 0; a < P12_ITEMS; ++a) av[a] = sLeft[ty + a * P12_THREADS][k];
#pragma unroll
        for (int b = 0; b < P12_ITEMS; ++b) bv[b] = sTop[k][tx + b * P12_THREADS];
#pragma unroll
        for (int a = 0; a < P12_ITEMS; ++a) {
            const unsigned int ah = av[a] & hiMask;
#pragma unroll
            for (int b = 0; b < P12_ITEMS; ++b) {
                const unsigned int bh = bv[b] & hiMask;
                const unsigned int cand = max(max(av[a] + bh, ah + bv[b]), ah + bh + kk);
                if (cand < cur[a][b]) {
                    cur[a][b] = cand;
                    sOwn[ty + a * P12_THREADS][tx + b * P12_THREADS] = cand;
                }
            }
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void loadTile(unsigned int (*s)[TILE], const unsigned int* __restrict__ key,
                                         const int n, const int row0, const int col0) {
    const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
    for (int a = 0; a < P12_ITEMS; ++a)
#pragma unroll
        for (int b = 0; b < P12_ITEMS; ++b) {
            const int r = ty + a * P12_THREADS, c = tx + b * P12_THREADS;
            s[r][c] = key[(size_t)(row0 + r) * n + col0 + c];
        }
}

__device__ __forceinline__ void storeTile(const unsigned int (*s)[TILE], unsigned int* __restrict__ key,
                                          const int n, const int row0, const int col0) {
    const int tx = threadIdx.x, ty = threadIdx.y;
#pragma unroll
    for (int a = 0; a < P12_ITEMS; ++a)
#pragma unroll
        for (int b = 0; b < P12_ITEMS; ++b) {
            const int r = ty + a * P12_THREADS, c = tx + b * P12_THREADS;
            key[(size_t)(row0 + r) * n + col0 + c] = s[r][c];
        }
}

// Phase 1: close the diagonal tile (kb, kb).
__global__ void __launch_bounds__(P12_THREADS * P12_THREADS)
fwPhase1(unsigned int* __restrict__ key, const int n, const int kb, const unsigned int hiMask) {
    __shared__ unsigned int sD[TILE][TILE];
    const int base = kb * TILE;
    loadTile(sD, key, n, base, base);
    __syncthreads();
    closeTile(sD, sD, sD, base, hiMask);
    storeTile(sD, key, n, base, base);
}

// Phase 2: update the tiles in block-row kb (blockIdx.y == 0) and
// block-column kb (blockIdx.y == 1) using the closed diagonal tile.
__global__ void __launch_bounds__(P12_THREADS * P12_THREADS)
fwPhase2(unsigned int* __restrict__ key, const int n, const int kb, const unsigned int hiMask) {
    const int b = blockIdx.x;
    if (b == kb) return;
    __shared__ unsigned int sDiag[TILE][TILE];
    __shared__ unsigned int sD[TILE][TILE];
    const int base = kb * TILE;
    const bool rowPanel = (blockIdx.y == 0);
    const int row0 = rowPanel ? base : b * TILE;
    const int col0 = rowPanel ? b * TILE : base;

    loadTile(sDiag, key, n, base, base);
    loadTile(sD, key, n, row0, col0);
    __syncthreads();
    if (rowPanel) closeTile(sD, sDiag, sD, base, hiMask);  // D[i][j] vs Diag[i][k] + D[k][j]
    else          closeTile(sD, sD, sDiag, base, hiMask);  // D[i][j] vs D[i][k] + Diag[k][j]
    storeTile(sD, key, n, row0, col0);
}

// Phase 3: update all entries outside block-row/column kb using the (final)
// row and column panels of block kb. Each thread block covers a
// P3_ROWS x P3_COLS region; each thread owns P3_TM rows (strided by
// P3_THREADS) x P3_TN contiguous columns. Entries inside block-row/column kb
// are computed but not stored: they are final and serve as the read-only panels.
__global__ void __launch_bounds__(P3_THREADS * P3_THREADS)
fwPhase3(unsigned int* __restrict__ key, const int n, const int kb, const unsigned int hiMask) {
    const int base = kb * TILE;
    const int row0 = blockIdx.y * P3_ROWS;
    const int col0 = blockIdx.x * P3_COLS;
    const bool rowsHitK = (unsigned int)(base - row0) < (unsigned int)P3_ROWS;
    const bool colsHitK = (unsigned int)(base - col0) < (unsigned int)P3_COLS;
    if (rowsHitK && P3_ROWS == TILE) return;
    if (colsHitK && P3_COLS == TILE) return;

    extern __shared__ unsigned int smem[];
    // Column panel, with the intermediate node k folded into the hop bits.
    unsigned int (*sA)[TILE] = reinterpret_cast<unsigned int (*)[TILE]>(smem);
    // Row panel.
    unsigned int (*sB)[P3_COLS] = reinterpret_cast<unsigned int (*)[P3_COLS]>(smem + P3_ROWS * TILE);
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * P3_THREADS + tx;

#pragma unroll 4
    for (int e = tid; e < P3_ROWS * TILE; e += P3_THREADS * P3_THREADS) {
        const int r = e / TILE, c = e % TILE;
        const unsigned int a = key[(size_t)(row0 + r) * n + base + c];
        const unsigned int ah = a & hiMask;
        sA[r][c] = max(a, ah + (unsigned int)(base + c + 1));
    }
#pragma unroll 4
    for (int e = tid; e < TILE * P3_COLS / 4; e += P3_THREADS * P3_THREADS) {
        const int r = e / (P3_COLS / 4), c = (e % (P3_COLS / 4)) * 4;
        *reinterpret_cast<uint4*>(&sB[r][c]) =
            *reinterpret_cast<const uint4*>(&key[(size_t)(base + r) * n + col0 + c]);
    }

    const int cbase = col0 + tx * P3_TN;
    unsigned int cur[P3_TM][P3_TN];
#pragma unroll
    for (int a = 0; a < P3_TM; ++a)
#pragma unroll
        for (int b = 0; b < P3_TN; b += 4) {
            const uint4 v = *reinterpret_cast<const uint4*>(
                &key[(size_t)(row0 + ty + a * P3_THREADS) * n + cbase + b]);
            cur[a][b] = v.x; cur[a][b + 1] = v.y; cur[a][b + 2] = v.z; cur[a][b + 3] = v.w;
        }
    __syncthreads();

#pragma unroll 4
    for (int k = 0; k < TILE; ++k) {
        unsigned int av[P3_TM], ah[P3_TM];
        unsigned int bv[P3_TN], bh[P3_TN];
#pragma unroll
        for (int a = 0; a < P3_TM; ++a) {
            av[a] = sA[ty + a * P3_THREADS][k];
            ah[a] = av[a] & hiMask;
        }
#pragma unroll
        for (int b = 0; b < P3_TN; b += 4) {
            const uint4 v = *reinterpret_cast<const uint4*>(&sB[k][tx * P3_TN + b]);
            bv[b] = v.x; bv[b + 1] = v.y; bv[b + 2] = v.z; bv[b + 3] = v.w;
        }
#pragma unroll
        for (int b = 0; b < P3_TN; ++b) bh[b] = bv[b] & hiMask;
#pragma unroll
        for (int a = 0; a < P3_TM; ++a)
#pragma unroll
            for (int b = 0; b < P3_TN; ++b)
                cur[a][b] = min(cur[a][b], max(av[a] + bh[b], ah[a] + bv[b]));
    }

#pragma unroll
    for (int a = 0; a < P3_TM; ++a) {
        const int r = row0 + ty + a * P3_THREADS;
        if (rowsHitK && (unsigned int)(r - base) < (unsigned int)TILE) continue;
#pragma unroll
        for (int b = 0; b < P3_TN; b += 4) {
            // TILE is a multiple of 4, so a 4-wide group lies entirely inside or outside the band
            if (colsHitK && (unsigned int)(cbase + b - base) < (unsigned int)TILE) continue;
            *reinterpret_cast<uint4*>(&key[(size_t)r * n + cbase + b]) =
                make_uint4(cur[a][b], cur[a][b + 1], cur[a][b + 2], cur[a][b + 3]);
        }
    }
}

// Largest initial distance (bounds every key during the computation).
__global__ void fwMaxWeight(const unsigned int* __restrict__ src, const size_t count,
                            unsigned int* __restrict__ result) {
    unsigned int m = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count;
         i += (size_t)gridDim.x * blockDim.x)
        m = max(m, src[i]);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) m = max(m, __shfl_xor_sync(0xffffffffu, m, off));
    if ((threadIdx.x & 31) == 0) atomicMax(result, m);
}

// Build the padded key matrix. Padding nodes are only connected through edges
// of weight padDist > maxWeight, so they never lie on a shortest path between
// real nodes.
__global__ void fwPadInit(unsigned int* __restrict__ key, const unsigned int* __restrict__ src,
                          const int n, const int np, const int shift,
                          const unsigned int* __restrict__ maxWeight) {
    const unsigned int padDist = *maxWeight + 1;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= np) return;
    unsigned int v;
    if (r < n && c < n) v = src[(size_t)r * n + c];
    else v = (r == c) ? 0u : padDist;
    key[(size_t)r * np + c] = v << shift;
}

// Unpack the final distance and path matrices.
__global__ void fwExtract(const unsigned int* __restrict__ key, unsigned int* __restrict__ outDist,
                          unsigned int* __restrict__ path, const int n, const int np,
                          const int shift) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y;
    if (c >= n) return;
    const unsigned int v = key[(size_t)r * np + c];
    const size_t o = (size_t)r * n + c;
    outDist[o] = v >> shift;
    const unsigned int h = v & ((1u << shift) - 1u);
    if (h != 0) path[o] = h - 1;
}

// Create the CUDA context and load/configure all kernels (outside the timed region).
void floydWarshallInit() {
    CUDA_CHECK(cudaFree(nullptr));
    CUDA_CHECK(cudaFuncSetAttribute(fwPhase3, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)P3_SMEM));
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase1));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase2));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fwMaxWeight));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPadInit));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, fwExtract));
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const int n = static_cast<int>(numNodes);
    const int np = ((n + PAD_GRAN - 1) / PAD_GRAN) * PAD_GRAN;
    int shift = 0;
    while ((1ull << shift) <= (unsigned long long)np) ++shift;
    const unsigned int hiMask = ~((1u << shift) - 1u);
    const size_t count = numNodes * numNodes;
    const size_t bytes = count * sizeof(unsigned int);
    const size_t kbytes = (size_t)np * np * sizeof(unsigned int);

    unsigned int *dIn = nullptr, *dPath = nullptr, *dKey = nullptr, *dMax = nullptr;
    CUDA_CHECK(cudaMalloc(&dIn, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMalloc(&dKey, kbytes));
    CUDA_CHECK(cudaMalloc(&dMax, sizeof(unsigned int)));
    cudaStream_t stream, copyStream;
    cudaEvent_t pathReady;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&pathReady, cudaEventDisableTiming));

    CUDA_CHECK(cudaMemcpyAsync(dIn, dist.data(), bytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemsetAsync(dMax, 0, sizeof(unsigned int), stream));
    fwMaxWeight<<<1024, 256, 0, stream>>>(dIn, count, dMax);

    const dim3 lin(256);
    fwPadInit<<<dim3((np + 255) / 256, np), lin, 0, stream>>>(dKey, dIn, n, np, shift, dMax);
    CUDA_CHECK(cudaGetLastError());

    const int nb = np / TILE;
    const dim3 p12Threads(P12_THREADS, P12_THREADS);
    const dim3 p3Threads(P3_THREADS, P3_THREADS);
    for (int kb = 0; kb < nb; ++kb) {
        fwPhase1<<<1, p12Threads, 0, stream>>>(dKey, np, kb, hiMask);
        fwPhase2<<<dim3(nb, 2), p12Threads, 0, stream>>>(dKey, np, kb, hiMask);
        fwPhase3<<<dim3(np / P3_COLS, np / P3_ROWS), p3Threads, P3_SMEM, stream>>>(dKey, np, kb, hiMask);
    }
    CUDA_CHECK(cudaGetLastError());

    // The initial path matrix is only needed at the end: upload it while the
    // GPU is computing.
    CUDA_CHECK(cudaMemcpyAsync(dPath, path.data(), bytes, cudaMemcpyHostToDevice, copyStream));
    CUDA_CHECK(cudaEventRecord(pathReady, copyStream));
    CUDA_CHECK(cudaStreamWaitEvent(stream, pathReady, 0));

    fwExtract<<<dim3((n + 255) / 256, n), lin, 0, stream>>>(dKey, dIn, dPath, n, np, shift);
    CUDA_CHECK(cudaGetLastError());

    unsigned int maxWeight = 0;
    CUDA_CHECK(cudaMemcpyAsync(&maxWeight, dMax, sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(dist.data(), dIn, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(path.data(), dPath, bytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Every key is <= (padDist << shift) | hopBits and every candidate is
    // <= ((2 * padDist) << shift) | hopBits (padDist = maxWeight + 1); both
    // must have fit in 32 bits for the packed computation to be exact.
    const unsigned long long padDist = (unsigned long long)maxWeight + 1;
    if (((2 * padDist + 1) << shift) > (1ull << 32)) {
        fprintf(stderr, "Edge weights too large for packed 32-bit keys (max weight %u, %d nodes)\n",
                maxWeight, n);
        exit(1);
    }

    CUDA_CHECK(cudaEventDestroy(pathReady));
    CUDA_CHECK(cudaStreamDestroy(copyStream));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(dIn));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dKey));
    CUDA_CHECK(cudaFree(dMax));
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
    
    // Initialize the CUDA context and kernels outside the timed region (before
    // the host buffers are allocated, which makes pageable transfers faster)
    floydWarshallInit();

    // Allocate matrices
    std::vector<unsigned int> dist(numNodes * numNodes);
    std::vector<unsigned int> path(numNodes * numNodes);
    
    // Initialize
    printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    initializePathMatrix(path, numNodes);
    

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
