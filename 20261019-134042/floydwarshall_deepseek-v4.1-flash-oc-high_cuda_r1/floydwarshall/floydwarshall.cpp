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

constexpr int TILE = 32;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            exit(1);                                                            \
        }                                                                       \
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

__global__ void fwDiagonalKernel(unsigned int* __restrict__ dist,
                                 unsigned int* __restrict__ path,
                                 const int n, const int k0) {
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int s = ty * TILE + tx;
    const int g = (k0 + ty) * n + (k0 + tx);

    __shared__ unsigned int sd[TILE * TILE];
    __shared__ unsigned int sp[TILE * TILE];

    sd[s] = dist[g];
    sp[s] = path[g];
    __syncthreads();

    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int nd = sd[kk * TILE + tx] + sd[ty * TILE + kk];
        if (nd < sd[s]) {
            sd[s] = nd;
            sp[s] = static_cast<unsigned int>(k0 + kk);
        }
        __syncthreads();
    }

    dist[g] = sd[s];
    path[g] = sp[s];
}

__global__ void fwPivotKernel(unsigned int* __restrict__ dist,
                              unsigned int* __restrict__ path,
                              const int n, const int kb) {
    const int k0 = kb * TILE;
    int b = blockIdx.x;
    if (b >= kb) {
        ++b;
    }

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int s = ty * TILE + tx;

    __shared__ unsigned int sD[TILE * TILE];
    __shared__ unsigned int sT[TILE * TILE];

    sD[s] = dist[(k0 + ty) * n + (k0 + tx)];

    const int i = (blockIdx.y == 0) ? (b * TILE + tx) : (k0 + tx);
    const int j = (blockIdx.y == 0) ? (k0 + ty) : (b * TILE + ty);
    const int g = j * n + i;

    unsigned int best = dist[g];
    unsigned int bestPath = path[g];
    sT[s] = best;
    __syncthreads();

    if (blockIdx.y == 0) {
        for (int kk = 0; kk < TILE; ++kk) {
            const unsigned int nd = sT[kk * TILE + tx] + sD[ty * TILE + kk];
            if (nd < best) {
                best = nd;
                bestPath = static_cast<unsigned int>(k0 + kk);
            }
        }
    } else {
        for (int kk = 0; kk < TILE; ++kk) {
            const unsigned int nd = sD[kk * TILE + tx] + sT[ty * TILE + kk];
            if (nd < best) {
                best = nd;
                bestPath = static_cast<unsigned int>(k0 + kk);
            }
        }
    }

    dist[g] = best;
    path[g] = bestPath;
}

__global__ void fwUpdateKernel(unsigned int* __restrict__ dist,
                               unsigned int* __restrict__ path,
                               const int n, const int kb) {
    const int k0 = kb * TILE;
    int br = blockIdx.x;
    if (br >= kb) {
        ++br;
    }
    int bc = blockIdx.y;
    if (bc >= kb) {
        ++bc;
    }

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int s = ty * TILE + tx;

    const int i = br * TILE + tx;
    const int j = bc * TILE + ty;
    const int g = j * n + i;

    __shared__ unsigned int sT[TILE * TILE];
    __shared__ unsigned int sP[TILE * TILE];
    __shared__ unsigned int sQ[TILE * TILE];

    sT[s] = dist[g];
    sP[s] = dist[(k0 + ty) * n + (br * TILE + tx)];
    sQ[s] = dist[(bc * TILE + ty) * n + (k0 + tx)];
    __syncthreads();

    unsigned int best = sT[s];
    unsigned int bestPath = path[g];

    for (int kk = 0; kk < TILE; ++kk) {
        const unsigned int nd = sP[kk * TILE + tx] + sQ[ty * TILE + kk];
        if (nd < best) {
            best = nd;
            bestPath = static_cast<unsigned int>(k0 + kk);
        }
    }

    dist[g] = best;
    path[g] = bestPath;
}

void floydWarshall(std::vector<unsigned int>& dist, 
                   std::vector<unsigned int>& path, 
                   const size_t numNodes) {
    if (numNodes == 0) {
        return;
    }

    const int n = static_cast<int>(numNodes);
    const int nPad = ((n + TILE - 1) / TILE) * TILE;
    const size_t paddedSize = static_cast<size_t>(nPad) * nPad;

    std::vector<unsigned int> hDist(paddedSize, INF);
    std::vector<unsigned int> hPath(paddedSize, 0);

    const size_t rowBytes = numNodes * sizeof(unsigned int);
    for (size_t j = 0; j < numNodes; ++j) {
        std::memcpy(&hDist[j * nPad], &dist[j * numNodes], rowBytes);
        std::memcpy(&hPath[j * nPad], &path[j * numNodes], rowBytes);
    }
    for (int i = n; i < nPad; ++i) {
        hDist[static_cast<size_t>(i) * nPad + i] = 0;
    }

    unsigned int* dDist = nullptr;
    unsigned int* dPath = nullptr;
    const size_t matrixBytes = paddedSize * sizeof(unsigned int);
    CUDA_CHECK(cudaMalloc(&dDist, matrixBytes));
    CUDA_CHECK(cudaMalloc(&dPath, matrixBytes));
    CUDA_CHECK(cudaMemcpy(dDist, hDist.data(), matrixBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dPath, hPath.data(), matrixBytes, cudaMemcpyHostToDevice));

    const int numBlocks = nPad / TILE;
    const dim3 block(TILE, TILE);

    for (int kb = 0; kb < numBlocks; ++kb) {
        fwDiagonalKernel<<<1, block>>>(dDist, dPath, nPad, kb * TILE);
        if (numBlocks > 1) {
            const dim3 pivotGrid(numBlocks - 1, 2);
            fwPivotKernel<<<pivotGrid, block>>>(dDist, dPath, nPad, kb);
            const dim3 updateGrid(numBlocks - 1, numBlocks - 1);
            fwUpdateKernel<<<updateGrid, block>>>(dDist, dPath, nPad, kb);
        }
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hDist.data(), dDist, matrixBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hPath.data(), dPath, matrixBytes, cudaMemcpyDeviceToHost));

    for (size_t j = 0; j < numNodes; ++j) {
        std::memcpy(&dist[j * numNodes], &hDist[j * nPad], rowBytes);
        std::memcpy(&path[j * numNodes], &hPath[j * nPad], rowBytes);
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
