#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Tile edge for blocked Floyd-Warshall on the GPU
constexpr int TB = 32;
constexpr int TROWS = 8;  // thread rows per block in the update kernel (each thread does TB/TROWS rows)

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// GPU kernels. Matrices are row-major with row stride N (padded size);
// element (row r, col c) lives at r*N + c, same as idx2(c, r, N).
// ---------------------------------------------------------------------------

// Phase 1: update the diagonal tile (kb,kb). `d`/`p` point to the first row of block-row kb.
__global__ void fwDiagKernel(unsigned int* d, unsigned int* p, size_t N, int kb) {
    __shared__ unsigned int s[TB][TB];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t off = (size_t)ty * N + (size_t)kb * TB + tx;
    unsigned int v = d[off];
    unsigned int pv = p[off];
    s[ty][tx] = v;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TB; ++k) {
        const unsigned int nd = s[ty][k] + s[k][tx];
        if (nd < v) {
            v = nd;
            pv = kb * TB + k;
            s[ty][tx] = v;
        }
        __syncthreads();
    }
    d[off] = v;
    p[off] = pv;
}

// Phase 2a: update row panel tiles (kb, J), J != kb. `d`/`p` point to first row of block-row kb.
__global__ void fwRowKernel(unsigned int* d, unsigned int* p, size_t N, int kb) {
    const int J = blockIdx.x;
    if (J == kb) return;
    __shared__ unsigned int sd[TB][TB];
    __shared__ unsigned int sr[TB][TB];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t off = (size_t)ty * N + (size_t)J * TB + tx;
    unsigned int v = d[off];
    unsigned int pv = p[off];
    sd[ty][tx] = d[(size_t)ty * N + (size_t)kb * TB + tx];
    sr[ty][tx] = v;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TB; ++k) {
        const unsigned int nd = sd[ty][k] + sr[k][tx];
        if (nd < v) {
            v = nd;
            pv = kb * TB + k;
            sr[ty][tx] = v;
        }
        __syncthreads();
    }
    d[off] = v;
    p[off] = pv;
}

// Phase 2b: update column panel tiles (I, kb), I != kb, for locally owned block rows.
// `panel` is the (final) pivot block-row kb (TB x N).
__global__ void fwColKernel(unsigned int* d, unsigned int* p, const unsigned int* panel,
                            size_t N, int kb, int firstBlock) {
    const int lI = blockIdx.x;
    if (firstBlock + lI == kb) return;
    __shared__ unsigned int sd[TB][TB];
    __shared__ unsigned int sc[TB][TB];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const size_t off = ((size_t)lI * TB + ty) * N + (size_t)kb * TB + tx;
    unsigned int v = d[off];
    unsigned int pv = p[off];
    sd[ty][tx] = panel[(size_t)ty * N + (size_t)kb * TB + tx];
    sc[ty][tx] = v;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TB; ++k) {
        const unsigned int nd = sc[ty][k] + sd[k][tx];
        if (nd < v) {
            v = nd;
            pv = kb * TB + k;
            sc[ty][tx] = v;
        }
        __syncthreads();
    }
    d[off] = v;
    p[off] = pv;
}

// Phase 3: update all remaining tiles (I, J), I != kb, J != kb.
__global__ void __launch_bounds__(TB * TROWS)
fwUpdateKernel(unsigned int* __restrict__ d, unsigned int* __restrict__ p,
               const unsigned int* __restrict__ panel, size_t N, int kb, int firstBlock) {
    const int J = blockIdx.x;
    const int lI = blockIdx.y;
    if (J == kb || firstBlock + lI == kb) return;
    constexpr int R = TB / TROWS;
    __shared__ unsigned int cs[TB][TB];
    __shared__ unsigned int rs[TB][TB];
    const int tx = threadIdx.x, ty = threadIdx.y;
    unsigned int v[R], pv[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const int row = ty + r * TROWS;
        const size_t g = ((size_t)lI * TB + row) * N + (size_t)J * TB + tx;
        v[r] = d[g];
        pv[r] = p[g];
        cs[row][tx] = d[((size_t)lI * TB + row) * N + (size_t)kb * TB + tx];
        rs[row][tx] = panel[(size_t)row * N + (size_t)J * TB + tx];
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < TB; ++k) {
        const unsigned int rv = rs[k][tx];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const unsigned int nd = cs[ty + r * TROWS][k] + rv;
            if (nd < v[r]) {
                v[r] = nd;
                pv[r] = kb * TB + k;
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const size_t g = ((size_t)lI * TB + ty + r * TROWS) * N + (size_t)J * TB + tx;
        d[g] = v[r];
        p[g] = pv[r];
    }
}

// ---------------------------------------------------------------------------
// Initialization (same values as the serial original)
// ---------------------------------------------------------------------------

// Fills the locally owned rows [rowBegin,rowEnd) of the padded N x N matrix.
// The pseudo-random stream is sequential, so every rank walks the whole stream
// but keeps only its own rows.
void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes, const size_t N,
                              const size_t rowBegin, const size_t rowEnd,
                              const unsigned int rangeMin, const unsigned int rangeMax) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < dist.size(); ++i) dist[i] = INF;

    const size_t last = std::min(rowEnd, numNodes);
    for (size_t i = 0; i < last; ++i) {
        if (i < rowBegin) {
            for (size_t j = 0; j < numNodes; ++j) rand_r(&seed);
        } else {
            unsigned int* row = &dist[(i - rowBegin) * N];
            for (size_t j = 0; j < numNodes; ++j)
                row[j] = rangeMin + (unsigned int)(range * rand_r(&seed) / (double)RAND_MAX);
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = rowBegin; i < rowEnd; ++i) dist[(i - rowBegin) * N + i] = 0;
}

// path[row][col] = row (as in the original initialization)
void initializePathMatrix(std::vector<unsigned int>& path, const size_t N,
                          const size_t rowBegin, const size_t rowEnd) {
    #pragma omp parallel for schedule(static)
    for (size_t r = rowBegin; r < rowEnd; ++r)
        for (size_t c = 0; c < N; ++c) path[(r - rowBegin) * N + c] = (unsigned int)r;
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

// Block-row distribution of nb block rows over P ranks
static inline int blocksFirst(int r, int nb, int P) { return r * (nb / P) + std::min(r, nb % P); }
static inline int blocksCount(int r, int nb, int P) { return nb / P + (r < nb % P ? 1 : 0); }

// Distributed, GPU-accelerated blocked Floyd-Warshall.
// Each rank owns a contiguous range of tile rows and drives one GPU. Per round kb the
// owner of the pivot tile row computes the diagonal and row panels and broadcasts the
// panel; all ranks then update their column tiles and the remaining tiles on the GPU.
void floydWarshallHybrid(unsigned int* dDist, unsigned int* dPath, const size_t N, const int rank,
                         const int P) {
    const int nb = (int)(N / TB);
    const int myFirst = blocksFirst(rank, nb, P);
    const int myCount = blocksCount(rank, nb, P);

    unsigned int* dPanel = nullptr;
    unsigned int* hPanel = nullptr;
    const size_t panelElems = (size_t)TB * N;
    CUDA_CHECK(cudaMalloc(&dPanel, panelElems * sizeof(unsigned int)));
    CUDA_CHECK(cudaMallocHost(&hPanel, panelElems * sizeof(unsigned int)));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    const dim3 tile(TB, TB);
    const dim3 upd(TB, TROWS);

    for (int kb = 0; kb < nb; ++kb) {
        int owner = 0;
        while (blocksFirst(owner, nb, P) + blocksCount(owner, nb, P) <= kb) ++owner;

        const unsigned int* panel;
        if (rank == owner) {
            const size_t rowOff = (size_t)(kb - myFirst) * TB * N;
            unsigned int* pd = dDist + rowOff;
            unsigned int* pp = dPath + rowOff;
            fwDiagKernel<<<1, tile, 0, stream>>>(pd, pp, N, kb);
            fwRowKernel<<<nb, tile, 0, stream>>>(pd, pp, N, kb);
            CUDA_CHECK(cudaMemcpyAsync(hPanel, pd, panelElems * sizeof(unsigned int),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            panel = pd;
        } else {
            panel = dPanel;
        }
        if (P > 1) {
            MPI_Bcast(hPanel, (int)panelElems, MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            if (rank != owner)
                CUDA_CHECK(cudaMemcpyAsync(dPanel, hPanel, panelElems * sizeof(unsigned int),
                                           cudaMemcpyHostToDevice, stream));
        }
        if (myCount > 0) {
            fwColKernel<<<myCount, tile, 0, stream>>>(dDist, dPath, panel, N, kb, myFirst);
            fwUpdateKernel<<<dim3(nb, myCount), upd, 0, stream>>>(dDist, dPath, panel, N, kb, myFirst);
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(hPanel));
    CUDA_CHECK(cudaFree(dPanel));
}

// Gather the padded distributed matrix onto rank 0 as an unpadded n x n matrix.
void gatherMatrix(const std::vector<unsigned int>& local, std::vector<unsigned int>& out,
                  const size_t n, const size_t N, const int rank, const int P) {
    const int nb = (int)(N / TB);
    const size_t chunk = (size_t)TB * N;
    if (rank != 0) {
        const int cnt = blocksCount(rank, nb, P);
        for (int b = 0; b < cnt; ++b)
            MPI_Send(&local[(size_t)b * chunk], (int)chunk, MPI_UNSIGNED, 0, 100 + b, MPI_COMM_WORLD);
        return;
    }
    std::vector<unsigned int> tmp(chunk);
    for (int r = 0; r < P; ++r) {
        const int first = blocksFirst(r, nb, P), cnt = blocksCount(r, nb, P);
        for (int b = 0; b < cnt; ++b) {
            const unsigned int* src;
            if (r == 0) {
                src = &local[(size_t)b * chunk];
            } else {
                MPI_Recv(tmp.data(), (int)chunk, MPI_UNSIGNED, r, 100 + b, MPI_COMM_WORLD,
                         MPI_STATUS_IGNORE);
                src = tmp.data();
            }
            const size_t row0 = (size_t)(first + b) * TB;
            for (size_t i = 0; i < (size_t)TB && row0 + i < n; ++i)
                std::memcpy(&out[(row0 + i) * n], src + i * N, n * sizeof(unsigned int));
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, P;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // One GPU per rank, chosen by node-local rank
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lr = 0, ndev = 0;
        MPI_Comm_rank(local, &lr);
        MPI_Comm_free(&local);
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lr % ndev));
    }
    
    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", P, omp_get_max_threads());
    }

    const size_t N = (numNodes + TB - 1) / TB * TB;  // padded size
    const int nb = (int)(N / TB);
    const int myFirst = blocksFirst(rank, nb, P);
    const int myCount = blocksCount(rank, nb, P);
    const size_t rowBegin = (size_t)myFirst * TB;
    const size_t rowEnd = (size_t)(myFirst + myCount) * TB;
    const size_t localElems = (rowEnd - rowBegin) * N;
    
    // Allocate matrices (local block rows only)
    std::vector<unsigned int> dist(localElems);
    std::vector<unsigned int> path(localElems);
    
    // Initialize
    if (rank == 0) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, N, rowBegin, rowEnd, 1, MAX_DISTANCE);
    initializePathMatrix(path, N, rowBegin, rowEnd);

    unsigned int *dDist = nullptr, *dPath = nullptr;
    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&dDist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dPath, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(dDist, dist.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dPath, path.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run Floyd-Warshall
    if (rank == 0) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    
    floydWarshallHybrid(dDist, dPath, N, rank, P);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (localElems > 0) {
        CUDA_CHECK(cudaMemcpy(dist.data(), dDist, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(path.data(), dPath, localElems * sizeof(unsigned int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(dDist));
        CUDA_CHECK(cudaFree(dPath));
    }

    std::vector<unsigned int> fullDist, fullPath;
    if (rank == 0) {
        fullDist.resize(numNodes * numNodes);
        fullPath.resize(numNodes * numNodes);
    }
    gatherMatrix(dist, fullDist, numNodes, N, rank, P);
    gatherMatrix(path, fullPath, numNodes, N, rank, P);

    int status = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());
        
        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
        
        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(fullDist, "DistanceMatrix");
        }
        
        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(fullDist, numNodes);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Finalize();
    return status;
}
