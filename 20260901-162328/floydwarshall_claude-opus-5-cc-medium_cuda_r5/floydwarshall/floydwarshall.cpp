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

// Tile size of the blocked (tiled) GPU Floyd-Warshall algorithm
constexpr int BS = 32;
// Rows processed per thread in the (dominant) phase 3 kernel
constexpr int ROWS_PER_THREAD = 4;
constexpr int P3_BLOCK_Y = BS / ROWS_PER_THREAD;

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                             \
            printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorName(err_), __FILE__, __LINE__, \
                   cudaGetErrorString(err_));                                                  \
            exit(1);                                                                           \
        }                                                                                      \
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
// GPU kernels: blocked Floyd-Warshall (Venkataraman et al.)
//
// The matrices are addressed as dist[i * n + j] (== dist[idx2(j, i, n)] of the
// host indexing scheme), i.e. row-major with i the source and j the target
// node. The matrices passed to the kernels are padded to a multiple of BS;
// padding entries are INF (0 on the diagonal), which can never relax a real
// entry, so the computed distances are identical to the serial algorithm.
// ---------------------------------------------------------------------------

// Phase 1: the pivot (diagonal) block depends only on itself.
__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int n, const int base) {
    __shared__ unsigned int sd[BS][BS + 1];
    __shared__ unsigned int sp[BS][BS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t gidx = (size_t)(base + ty) * n + (base + tx);

    sd[ty][tx] = dist[gidx];
    sp[ty][tx] = path[gidx];
    __syncthreads();

    // Row/column k of the tile are invariant during step k, so a single
    // barrier per iteration is sufficient.
    #pragma unroll 8
    for (int k = 0; k < BS; ++k) {
        const unsigned int nd = sd[ty][k] + sd[k][tx];
        if (nd < sd[ty][tx]) {
            sd[ty][tx] = nd;
            sp[ty][tx] = base + k;
        }
        __syncthreads();
    }

    dist[gidx] = sd[ty][tx];
    path[gidx] = sp[ty][tx];
}

// Phase 2: blocks in the pivot row and pivot column, each depending on the
// pivot block and on itself. blockIdx.y == 0 -> pivot row, 1 -> pivot column.
__global__ void fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int n, const int base) {
    const int blk = blockIdx.x;
    if (blk == base / BS) {
        return;
    }

    __shared__ unsigned int piv[BS][BS + 1];
    __shared__ unsigned int sd[BS][BS + 1];
    __shared__ unsigned int sp[BS][BS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const bool pivotRow = (blockIdx.y == 0);

    const int i = pivotRow ? (base + ty) : (blk * BS + ty);
    const int j = pivotRow ? (blk * BS + tx) : (base + tx);
    const size_t gidx = (size_t)i * n + j;

    piv[ty][tx] = dist[(size_t)(base + ty) * n + (base + tx)];
    sd[ty][tx] = dist[gidx];
    sp[ty][tx] = path[gidx];
    __syncthreads();

    #pragma unroll 8
    for (int k = 0; k < BS; ++k) {
        // dist[i][k] and dist[k][j]: one operand comes from the pivot tile,
        // the other from this tile (depending on the orientation).
        const unsigned int nd = pivotRow ? (piv[ty][k] + sd[k][tx]) : (sd[ty][k] + piv[k][tx]);
        if (nd < sd[ty][tx]) {
            sd[ty][tx] = nd;
            sp[ty][tx] = base + k;
        }
        __syncthreads();
    }

    dist[gidx] = sd[ty][tx];
    path[gidx] = sp[ty][tx];
}

// Phase 3: all remaining blocks, depending only on phase 2 results.
__global__ void fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int n, const int base) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int pivBlk = base / BS;
    if (bx == pivBlk || by == pivBlk) {
        return;
    }

    __shared__ unsigned int a[BS][BS + 1];  // dist[i][base..base+BS)
    __shared__ unsigned int b[BS][BS + 1];  // dist[base..base+BS)[j]

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int i0 = by * BS;
    const int j = bx * BS + tx;

    #pragma unroll
    for (int r = 0; r < BS; r += P3_BLOCK_Y) {
        a[ty + r][tx] = dist[(size_t)(i0 + ty + r) * n + (base + tx)];
        b[ty + r][tx] = dist[(size_t)(base + ty + r) * n + j];
    }
    __syncthreads();

    #pragma unroll
    for (int r = 0; r < BS; r += P3_BLOCK_Y) {
        const size_t gidx = (size_t)(i0 + ty + r) * n + j;
        unsigned int d = dist[gidx];
        unsigned int p = 0;
        bool updated = false;

        #pragma unroll
        for (int k = 0; k < BS; ++k) {
            const unsigned int nd = a[ty + r][k] + b[k][tx];
            if (nd < d) {
                d = nd;
                p = base + k;
                updated = true;
            }
        }

        if (updated) {
            dist[gidx] = d;
            path[gidx] = p;
        }
    }
}

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const int n = (int)numNodes;
    const int numBlocks = (n + BS - 1) / BS;
    const int np = numBlocks * BS;  // padded dimension
    const size_t paddedElems = (size_t)np * np;

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, paddedElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&dPath, paddedElems * sizeof(unsigned int)));

    if (np == n) {
        CUDA_CHECK(cudaMemcpy(dDist, dist.data(), paddedElems * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data(), paddedElems * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    } else {
        // Fill the padding with INF (never relaxes anything), then upload the
        // real rows. INF + INF does not overflow an unsigned int.
        std::vector<unsigned int> padRow(np, INF);
        for (int i = n; i < np; ++i) {
            padRow[i] = 0u;  // diagonal entry of this padded row
            CUDA_CHECK(cudaMemcpy(dDist + (size_t)i * np, padRow.data(),
                                  np * sizeof(unsigned int), cudaMemcpyHostToDevice));
            padRow[i] = INF;
        }
        CUDA_CHECK(cudaMemset(dPath, 0, paddedElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy2D(dDist, np * sizeof(unsigned int), dist.data(),
                                n * sizeof(unsigned int), n * sizeof(unsigned int), n,
                                cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy2D(dPath, np * sizeof(unsigned int), path.data(),
                                n * sizeof(unsigned int), n * sizeof(unsigned int), n,
                                cudaMemcpyHostToDevice));
        // Padding columns of the real rows
        std::vector<unsigned int> padTail(np - n, INF);
        for (int i = 0; i < n; ++i) {
            CUDA_CHECK(cudaMemcpy(dDist + (size_t)i * np + n, padTail.data(),
                                  (np - n) * sizeof(unsigned int), cudaMemcpyHostToDevice));
        }
    }

    const dim3 blockDim2D(BS, BS);
    const dim3 blockDim3(BS, P3_BLOCK_Y);
    const dim3 gridPhase2(numBlocks, 2);
    const dim3 gridPhase3(numBlocks, numBlocks);

    for (int blk = 0; blk < numBlocks; ++blk) {
        const int base = blk * BS;
        fwPhase1<<<1, blockDim2D>>>(dDist, dPath, np, base);
        if (numBlocks > 1) {
            fwPhase2<<<gridPhase2, blockDim2D>>>(dDist, dPath, np, base);
            fwPhase3<<<gridPhase3, blockDim3>>>(dDist, dPath, np, base);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    if (np == n) {
        CUDA_CHECK(cudaMemcpy(dist.data(), dDist, paddedElems * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), dPath, paddedElems * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    } else {
        CUDA_CHECK(cudaMemcpy2D(dist.data(), n * sizeof(unsigned int), dDist,
                                np * sizeof(unsigned int), n * sizeof(unsigned int), n,
                                cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy2D(path.data(), n * sizeof(unsigned int), dPath,
                                np * sizeof(unsigned int), n * sizeof(unsigned int), n,
                                cudaMemcpyDeviceToHost));
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

    // Initialize the CUDA context up front so that the measured region only
    // contains the actual solver (transfers included).
    CUDA_CHECK(cudaFree(0));

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
