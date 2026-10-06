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
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                        \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

constexpr int TILE = 64;              // tile edge (elements)
constexpr int TPB = 16;               // threads per tile edge
constexpr int EPT = TILE / TPB;       // elements per thread per dimension

// Phase 1: the diagonal tile (kb, kb). Row/column k inside the tile never
// changes during step k (dist[k][k] == 0), so one barrier per step suffices.
__global__ void __launch_bounds__(TPB * TPB)
fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int pitch, const int kb) {
    __shared__ unsigned int sD[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int base = kb * TILE;
    unsigned int* gD = dist + (size_t)base * pitch + base;
    unsigned int* gP = path + (size_t)base * pitch + base;

    unsigned int p[EPT][EPT];
#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int i = ty + r * TPB, j = tx + c * TPB;
            sD[i][j] = gD[(size_t)i * pitch + j];
            p[r][c] = gP[(size_t)i * pitch + j];
        }
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
#pragma unroll
        for (int r = 0; r < EPT; ++r)
#pragma unroll
            for (int c = 0; c < EPT; ++c) {
                const int i = ty + r * TPB, j = tx + c * TPB;
                const unsigned int nd = sD[i][kk] + sD[kk][j];
                if (nd < sD[i][j]) {
                    sD[i][j] = nd;
                    p[r][c] = base + kk;
                }
            }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int i = ty + r * TPB, j = tx + c * TPB;
            gD[(size_t)i * pitch + j] = sD[i][j];
            gP[(size_t)i * pitch + j] = p[r][c];
        }
}

// Phase 2: tiles in row kb (blockIdx.y == 0) and column kb (blockIdx.y == 1).
__global__ void __launch_bounds__(TPB * TPB)
fwPhase2(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int pitch, const int kb) {
    int ob = blockIdx.x;
    if (ob >= kb) ++ob;   // skip the diagonal tile
    const bool isRow = (blockIdx.y == 0);
    const int bi = isRow ? kb : ob;
    const int bj = isRow ? ob : kb;

    __shared__ unsigned int sDiag[TILE][TILE];
    __shared__ unsigned int sC[TILE][TILE];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int kbase = kb * TILE;
    const unsigned int* gDiag = dist + (size_t)kbase * pitch + kbase;
    unsigned int* gC = dist + (size_t)(bi * TILE) * pitch + bj * TILE;
    unsigned int* gP = path + (size_t)(bi * TILE) * pitch + bj * TILE;

    unsigned int p[EPT][EPT];
#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int i = ty + r * TPB, j = tx + c * TPB;
            sDiag[i][j] = gDiag[(size_t)i * pitch + j];
            sC[i][j] = gC[(size_t)i * pitch + j];
            p[r][c] = gP[(size_t)i * pitch + j];
        }
    __syncthreads();

    if (isRow) {
        // dist[i][j] = min(dist[i][j], diag[i][kk] + C[kk][j])
        for (int kk = 0; kk < TILE; ++kk) {
#pragma unroll
            for (int r = 0; r < EPT; ++r)
#pragma unroll
                for (int c = 0; c < EPT; ++c) {
                    const int i = ty + r * TPB, j = tx + c * TPB;
                    const unsigned int nd = sDiag[i][kk] + sC[kk][j];
                    if (nd < sC[i][j]) {
                        sC[i][j] = nd;
                        p[r][c] = kbase + kk;
                    }
                }
            __syncthreads();
        }
    } else {
        // dist[i][j] = min(dist[i][j], C[i][kk] + diag[kk][j])
        for (int kk = 0; kk < TILE; ++kk) {
#pragma unroll
            for (int r = 0; r < EPT; ++r)
#pragma unroll
                for (int c = 0; c < EPT; ++c) {
                    const int i = ty + r * TPB, j = tx + c * TPB;
                    const unsigned int nd = sC[i][kk] + sDiag[kk][j];
                    if (nd < sC[i][j]) {
                        sC[i][j] = nd;
                        p[r][c] = kbase + kk;
                    }
                }
            __syncthreads();
        }
    }

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int i = ty + r * TPB, j = tx + c * TPB;
            gC[(size_t)i * pitch + j] = sC[i][j];
            gP[(size_t)i * pitch + j] = p[r][c];
        }
}

// Phase 3: all remaining tiles. Depends only on the (already final) row-kb
// and column-kb tiles, so the update is fully register-resident.
__global__ void __launch_bounds__(TPB * TPB)
fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
         const int pitch, const int kb) {
    const int bi = blockIdx.y, bj = blockIdx.x;
    if (bi == kb || bj == kb) return;

    __shared__ unsigned int sA[TILE][TILE];   // dist[bi-tile][kb-tile]
    __shared__ unsigned int sB[TILE][TILE];   // dist[kb-tile][bj-tile]
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int kbase = kb * TILE;
    const unsigned int* gA = dist + (size_t)(bi * TILE) * pitch + kbase;
    const unsigned int* gB = dist + (size_t)kbase * pitch + bj * TILE;
    unsigned int* gC = dist + (size_t)(bi * TILE) * pitch + bj * TILE;
    unsigned int* gP = path + (size_t)(bi * TILE) * pitch + bj * TILE;

    unsigned int d[EPT][EPT];
#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            const int i = ty + r * TPB, j = tx + c * TPB;
            sA[i][j] = gA[(size_t)i * pitch + j];
            sB[i][j] = gB[(size_t)i * pitch + j];
            d[r][c] = gC[(size_t)i * pitch + j];
        }
    __syncthreads();

    // Track the best intermediate offset in the tile; only write path if changed.
    int best[EPT][EPT];
#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) best[r][c] = -1;

#pragma unroll 8
    for (int kk = 0; kk < TILE; ++kk) {
        unsigned int a[EPT], b[EPT];
#pragma unroll
        for (int r = 0; r < EPT; ++r) a[r] = sA[ty + r * TPB][kk];
#pragma unroll
        for (int c = 0; c < EPT; ++c) b[c] = sB[kk][tx + c * TPB];
#pragma unroll
        for (int r = 0; r < EPT; ++r)
#pragma unroll
            for (int c = 0; c < EPT; ++c) {
                const unsigned int nd = a[r] + b[c];
                if (nd < d[r][c]) {
                    d[r][c] = nd;
                    best[r][c] = kk;
                }
            }
    }

#pragma unroll
    for (int r = 0; r < EPT; ++r)
#pragma unroll
        for (int c = 0; c < EPT; ++c) {
            if (best[r][c] >= 0) {
                const int i = ty + r * TPB, j = tx + c * TPB;
                gC[(size_t)i * pitch + j] = d[r][c];
                gP[(size_t)i * pitch + j] = kbase + best[r][c];
            }
        }
}

// Marks path entries the device has not modified (no valid node index).
constexpr unsigned int PATH_UNCHANGED = 0xFFFFFFFFu;

// Initialise padding of the device distance matrix: padding nodes are
// unreachable (INF), padded diagonal is 0 so "dist[k][k] == 0" holds everywhere.
__global__ void fwPadInit(unsigned int* __restrict__ dist, const int n, const int pitch) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= pitch || j >= pitch) return;
    if (i < n && j < n) return;
    dist[(size_t)i * pitch + j] = (i == j) ? 0u : INF;
}

// Fill entries of the (padded) device path matrix that were never improved
// with their original values (dense n x n layout).
__global__ void fwPathMerge(unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ orig,
                            const int n, const int pitch) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= n || j >= n) return;
    const size_t o = (size_t)i * pitch + j;
    if (path[o] == PATH_UNCHANGED) path[o] = orig[(size_t)i * n + j];
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) return;
    const int n = static_cast<int>(numNodes);
    const int nb = (n + TILE - 1) / TILE;
    const int pitch = nb * TILE;
    const size_t bytes = (size_t)pitch * pitch * sizeof(unsigned int);
    const size_t denseBytes = numNodes * numNodes * sizeof(unsigned int);
    const size_t rowBytes = numNodes * sizeof(unsigned int);
    const size_t pitchBytes = (size_t)pitch * sizeof(unsigned int);

    cudaStream_t sCompute, sCopy;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCompute, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
    cudaEvent_t pathUploaded;
    CUDA_CHECK(cudaEventCreateWithFlags(&pathUploaded, cudaEventDisableTiming));

    unsigned int *dDist = nullptr, *dPath = nullptr, *dPathOrig = nullptr;
    CUDA_CHECK(cudaMalloc(&dDist, bytes));
    CUDA_CHECK(cudaMalloc(&dPath, bytes));
    CUDA_CHECK(cudaMalloc(&dPathOrig, denseBytes));

    const dim3 eb(32, 8);
    if (pitch != n) {
        const dim3 g((pitch + 31) / 32, (pitch + 7) / 8);
        fwPadInit<<<g, eb, 0, sCompute>>>(dDist, n, pitch);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemsetAsync(dPath, 0xFF, bytes, sCompute));
    CUDA_CHECK(cudaMemcpy2DAsync(dDist, pitchBytes, dist.data(), rowBytes, rowBytes, numNodes,
                                 cudaMemcpyHostToDevice, sCompute));

    // The kernels only ever write path entries (never read original values),
    // so the original path matrix is uploaded concurrently with the
    // computation and merged in at the end. A host thread issues the copy
    // because pageable-memory copies block the calling thread.
    std::thread pathUploader([&] {
        CUDA_CHECK(cudaMemcpyAsync(dPathOrig, path.data(), denseBytes,
                                   cudaMemcpyHostToDevice, sCopy));
        CUDA_CHECK(cudaEventRecord(pathUploaded, sCopy));
    });

    const dim3 block(TPB, TPB);
    for (int kb = 0; kb < nb; ++kb) {
        fwPhase1<<<1, block, 0, sCompute>>>(dDist, dPath, pitch, kb);
        if (nb > 1) {
            fwPhase2<<<dim3(nb - 1, 2), block, 0, sCompute>>>(dDist, dPath, pitch, kb);
            fwPhase3<<<dim3(nb, nb), block, 0, sCompute>>>(dDist, dPath, pitch, kb);
        }
    }
    CUDA_CHECK(cudaGetLastError());

    pathUploader.join();
    CUDA_CHECK(cudaStreamWaitEvent(sCompute, pathUploaded, 0));
    {
        const dim3 g((n + 31) / 32, (n + 7) / 8);
        fwPathMerge<<<g, eb, 0, sCompute>>>(dPath, dPathOrig, n, pitch);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaMemcpy2DAsync(dist.data(), rowBytes, dDist, pitchBytes, rowBytes, numNodes,
                                 cudaMemcpyDeviceToHost, sCompute));
    CUDA_CHECK(cudaMemcpy2DAsync(path.data(), rowBytes, dPath, pitchBytes, rowBytes, numNodes,
                                 cudaMemcpyDeviceToHost, sCompute));
    CUDA_CHECK(cudaStreamSynchronize(sCompute));

    CUDA_CHECK(cudaFree(dDist));
    CUDA_CHECK(cudaFree(dPath));
    CUDA_CHECK(cudaFree(dPathOrig));
    CUDA_CHECK(cudaEventDestroy(pathUploaded));
    CUDA_CHECK(cudaStreamDestroy(sCompute));
    CUDA_CHECK(cudaStreamDestroy(sCopy));
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
    
    // Initialise the CUDA context (and load kernels) up front so it is not
    // part of the measurement
    CUDA_CHECK(cudaFree(nullptr));
    {
        cudaFuncAttributes attr;
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase1));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase2));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPhase3));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPadInit));
        CUDA_CHECK(cudaFuncGetAttributes(&attr, fwPathMerge));
    }

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
