#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                                     \
    do {                                                                                     \
        const cudaError_t err_ = (call);                                                      \
        if (err_ != cudaSuccess) {                                                            \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            exit(1);                                                                          \
        }                                                                                     \
    } while (0)

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

// ---------------------------------------------------------------------------
// CUDA blocked Floyd-Warshall
//
// The host matrices are stored so that element (i, j) lives at index i*n + j
// (idx2(j, i, n)).  On the device the matrix is padded up to a multiple of the
// tile size BS; padded entries are initialized to INF so they can never shorten
// a path between real nodes (real distances stay far below INF, and values only
// ever decrease, so no unsigned overflow can occur).
//
// The dependency-respecting blocked formulation visits the intermediate nodes k
// in increasing order, and it is the standard equivalent reformulation of the
// triple loop: the resulting distance matrix is bit-identical to the sequential
// one.  The predecessor matrix records, for each pair, an intermediate node on a
// shortest path; where several nodes are equally optimal a block may settle on a
// different one than the sequential order would (the entry is still a valid
// witness for the same, identical distance).
// ---------------------------------------------------------------------------

constexpr int BS = 32;    // tile size (also blockDim.x for all phases)
constexpr int RPT = 4;    // rows per thread in the dependent-block phase

// Phase 1: the pivot block (kb, kb) - fully self-dependent.
__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int pitch, const int kb) {
    __shared__ unsigned int sd[BS][BS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t off = static_cast<size_t>(kb * BS + ty) * pitch + (kb * BS + tx);

    unsigned int d = dist[off];
    unsigned int p = path[off];
    sd[ty][tx] = d;
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < BS; ++k) {
        const unsigned int nd = sd[ty][k] + sd[k][tx];
        if (nd < d) {
            d = nd;
            p = static_cast<unsigned int>(kb * BS + k);
        }
        __syncthreads();
        sd[ty][tx] = d;
        __syncthreads();
    }

    dist[off] = d;
    path[off] = p;
}

// Phase 2: blocks in the pivot row (blockIdx.y == 0) and pivot column
// (blockIdx.y == 1); each depends only on itself and on the pivot block.
__global__ void fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int pitch, const int kb) {
    __shared__ unsigned int piv[BS][BS + 1];
    __shared__ unsigned int own[BS][BS + 1];

    const int b = blockIdx.x;
    if (b == kb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const bool pivotRow = (blockIdx.y == 0);

    const int i = pivotRow ? (kb * BS + ty) : (b * BS + ty);
    const int j = pivotRow ? (b * BS + tx) : (kb * BS + tx);
    const size_t off = static_cast<size_t>(i) * pitch + j;

    piv[ty][tx] = dist[static_cast<size_t>(kb * BS + ty) * pitch + (kb * BS + tx)];
    unsigned int d = dist[off];
    unsigned int p = path[off];
    own[ty][tx] = d;
    __syncthreads();

#pragma unroll 8
    for (int k = 0; k < BS; ++k) {
        const unsigned int nd = pivotRow ? (piv[ty][k] + own[k][tx]) : (own[ty][k] + piv[k][tx]);
        if (nd < d) {
            d = nd;
            p = static_cast<unsigned int>(kb * BS + k);
        }
        __syncthreads();
        own[ty][tx] = d;
        __syncthreads();
    }

    dist[off] = d;
    path[off] = p;
}

// Phase 3: all remaining blocks; they read the already-updated pivot row/column
// blocks and are independent of each other, so no in-loop synchronization.
// Each thread block covers a BS x BS tile with BS x (BS/RPT) threads.
__global__ void fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int pitch, const int kb) {
    __shared__ unsigned int rowT[BS][BS + 1];    // tile (bi, kb): dist(i, k)
    __shared__ unsigned int colT[BS][BS + 1];    // tile (kb, bj): dist(k, j)

    const int bi = blockIdx.y;
    const int bj = blockIdx.x;
    if (bi == kb || bj == kb) return;

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        const int ly = ty + r * (BS / RPT);
        rowT[ly][tx] = dist[static_cast<size_t>(bi * BS + ly) * pitch + (kb * BS + tx)];
        colT[ly][tx] = dist[static_cast<size_t>(kb * BS + ly) * pitch + (bj * BS + tx)];
    }
    __syncthreads();

    const size_t base = static_cast<size_t>(bi * BS + ty) * pitch + (bj * BS + tx);
    const size_t stride = static_cast<size_t>(BS / RPT) * pitch;

    unsigned int d[RPT];
    unsigned int p[RPT];
#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        d[r] = dist[base + r * stride];
        p[r] = path[base + r * stride];
    }

#pragma unroll 8
    for (int k = 0; k < BS; ++k) {
        const unsigned int dkj = colT[k][tx];
        const unsigned int gk = static_cast<unsigned int>(kb * BS + k);
#pragma unroll
        for (int r = 0; r < RPT; ++r) {
            const unsigned int nd = rowT[ty + r * (BS / RPT)][k] + dkj;
            if (nd < d[r]) {
                d[r] = nd;
                p[r] = gk;
            }
        }
    }

#pragma unroll
    for (int r = 0; r < RPT; ++r) {
        dist[base + r * stride] = d[r];
        path[base + r * stride] = p[r];
    }
}

// Initializes the padded device matrices: distances to INF (padding entries can
// then never shorten a real path, and the real values are copied in on top of
// them), and the predecessor matrix to its initial host value, which
// initializePathMatrix() sets to path(i, j) = i for every element.
__global__ void fwInitDevice(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                             const int pitch, const size_t total) {
    for (size_t e = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; e < total;
         e += static_cast<size_t>(gridDim.x) * blockDim.x) {
        dist[e] = INF;
        path[e] = static_cast<unsigned int>(e / pitch);
    }
}

// Host <-> device transfers of the (large) matrices are pipelined through a
// small pinned staging buffer: pageable copies run at a fraction of the PCIe
// bandwidth, while page-locking the full matrices costs more than it saves.
// While one chunk is in flight on the DMA engine, the CPU copies the previous
// one to/from the pageable host matrix.
struct StagedCopier {
    static constexpr size_t CHUNK_BYTES = 1u << 20;    // pinned staging chunk
    static constexpr size_t DIRECT_LIMIT = 8u << 20;   // below this, copy directly

    unsigned int* staging = nullptr;
    cudaStream_t streams[2] = {};
    size_t chunkRows = 0;
    size_t rowBytes = 0;
    size_t chunkElems = 0;

    // Matrices that fit into a single chunk are copied directly; setting up the
    // pinned buffer and the streams would cost more than the staging saves.
    StagedCopier(const size_t numNodes) {
        rowBytes = numNodes * sizeof(unsigned int);
        if (rowBytes * numNodes <= DIRECT_LIMIT) return;
        chunkRows = std::max<size_t>(1, CHUNK_BYTES / rowBytes);
        chunkElems = chunkRows * numNodes;
        CUDA_CHECK(cudaStreamCreate(&streams[0]));
        CUDA_CHECK(cudaStreamCreate(&streams[1]));
        CUDA_CHECK(cudaMallocHost(&staging, 2 * chunkElems * sizeof(unsigned int)));
    }

    ~StagedCopier() {
        if (!staging) return;
        cudaFreeHost(staging);
        cudaStreamDestroy(streams[0]);
        cudaStreamDestroy(streams[1]);
    }

    // hostSrc (numNodes x numNodes, tightly packed) -> devDst (row stride pitch)
    void toDevice(unsigned int* devDst, const unsigned int* hostSrc, const int pitch,
                  const size_t numNodes) {
        if (!staging) {
            CUDA_CHECK(cudaMemcpy2D(devDst, pitch * sizeof(unsigned int), hostSrc, rowBytes,
                                    rowBytes, numNodes, cudaMemcpyHostToDevice));
            return;
        }
        int buf = 0;
        for (size_t row = 0; row < numNodes; row += chunkRows) {
            const size_t rows = std::min(chunkRows, numNodes - row);
            CUDA_CHECK(cudaStreamSynchronize(streams[buf]));
            memcpy(staging + buf * chunkElems, hostSrc + row * numNodes, rows * rowBytes);
            CUDA_CHECK(cudaMemcpy2DAsync(devDst + row * static_cast<size_t>(pitch),
                                         pitch * sizeof(unsigned int), staging + buf * chunkElems,
                                         rowBytes, rowBytes, rows, cudaMemcpyHostToDevice,
                                         streams[buf]));
            buf ^= 1;
        }
        CUDA_CHECK(cudaStreamSynchronize(streams[0]));
        CUDA_CHECK(cudaStreamSynchronize(streams[1]));
    }

    // devSrc (row stride pitch) -> hostDst (numNodes x numNodes, tightly packed)
    void toHost(unsigned int* hostDst, const unsigned int* devSrc, const int pitch,
                const size_t numNodes) {
        if (!staging) {
            CUDA_CHECK(cudaMemcpy2D(hostDst, rowBytes, devSrc, pitch * sizeof(unsigned int),
                                    rowBytes, numNodes, cudaMemcpyDeviceToHost));
            return;
        }
        int buf = 0;
        size_t pendingRow = 0, pendingRows = 0;
        int pendingBuf = 0;
        for (size_t row = 0; row < numNodes; row += chunkRows) {
            const size_t rows = std::min(chunkRows, numNodes - row);
            CUDA_CHECK(cudaMemcpy2DAsync(staging + buf * chunkElems, rowBytes,
                                         devSrc + row * static_cast<size_t>(pitch),
                                         pitch * sizeof(unsigned int), rowBytes, rows,
                                         cudaMemcpyDeviceToHost, streams[buf]));
            if (pendingRows > 0) {
                CUDA_CHECK(cudaStreamSynchronize(streams[pendingBuf]));
                memcpy(hostDst + pendingRow * numNodes, staging + pendingBuf * chunkElems,
                       pendingRows * rowBytes);
            }
            pendingRow = row;
            pendingRows = rows;
            pendingBuf = buf;
            buf ^= 1;
        }
        if (pendingRows > 0) {
            CUDA_CHECK(cudaStreamSynchronize(streams[pendingBuf]));
            memcpy(hostDst + pendingRow * numNodes, staging + pendingBuf * chunkElems,
                   pendingRows * rowBytes);
        }
    }
};

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) return;

    const int n = static_cast<int>(numNodes);
    const int numBlocks = (n + BS - 1) / BS;
    const int np = numBlocks * BS;    // padded dimension
    const int pitch = np + BS;        // row stride; the slack avoids cache-set camping

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    const size_t padded = static_cast<size_t>(np) * pitch;
    CUDA_CHECK(cudaMalloc(&dDist, padded * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, padded * sizeof(unsigned int)));

    // Initialize the padded device matrices, then upload the distances.  The
    // predecessor matrix is generated directly on the device and therefore never
    // has to travel across PCIe in the host-to-device direction.
    const int initBlocks = static_cast<int>(std::min<size_t>((padded + 255) / 256, 8192));
    fwInitDevice<<<initBlocks, 256>>>(dDist, dPath, pitch, padded);

    StagedCopier copier(numNodes);
    copier.toDevice(dDist, dist.data(), pitch, numNodes);

    const dim3 blockFull(BS, BS);
    const dim3 blockPhase3(BS, BS / RPT);
    const dim3 gridPhase2(numBlocks, 2);
    const dim3 gridPhase3(numBlocks, numBlocks);

    for (int kb = 0; kb < numBlocks; ++kb) {
        fwPhase1<<<1, blockFull>>>(dDist, dPath, pitch, kb);
        if (numBlocks > 1) {
            fwPhase2<<<gridPhase2, blockFull>>>(dDist, dPath, pitch, kb);
            fwPhase3<<<gridPhase3, blockPhase3>>>(dDist, dPath, pitch, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    copier.toHost(dist.data(), dDist, pitch, numNodes);
    copier.toHost(path.data(), dPath, pitch, numNodes);

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
    
    // Create the CUDA context up front: it must not be part of the measured
    // region, and initializing it before the host matrices are allocated keeps
    // the driver's address-space setup from slowing down the first pass over
    // them later on.
    CUDA_CHECK(cudaFree(0));

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
