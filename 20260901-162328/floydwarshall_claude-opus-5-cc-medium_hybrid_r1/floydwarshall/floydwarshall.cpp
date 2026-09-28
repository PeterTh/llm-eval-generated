#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Blocking factor of the tiled (blocked) Floyd-Warshall formulation.
// Every CUDA block processes a BS x BS tile with 32x32 threads (4 elements/thread).
constexpr int BS = 64;
constexpr int TPB_X = 32;
constexpr int TPB_Y = 32;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t err_ = (call);                                                                 \
        if (err_ != cudaSuccess) {                                                                 \
            printf("CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__);      \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                          \
        }                                                                                          \
    } while (0)

// ---------------------------------------------------------------------------
// CUDA kernels: blocked Floyd-Warshall (dependent / partially-dependent /
// independent phases).  The matrix is padded to a multiple of BS and
// distributed over MPI ranks by contiguous row blocks; each rank drives one
// GPU and owns "localRows" rows of the global N x N matrix.
// ---------------------------------------------------------------------------

// Phase 1: the diagonal block [k0,k0+BS) x [k0,k0+BS) is closed against itself.
__global__ void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                         const int N, const int rowOff, const int k0) {
    __shared__ unsigned int sd[BS][BS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r0 = ty, r1 = ty + TPB_Y;
    const int c0 = tx, c1 = tx + TPB_X;

    const size_t base = static_cast<size_t>(rowOff) * N + k0;
    unsigned int p00 = path[base + static_cast<size_t>(r0) * N + c0];
    unsigned int p01 = path[base + static_cast<size_t>(r0) * N + c1];
    unsigned int p10 = path[base + static_cast<size_t>(r1) * N + c0];
    unsigned int p11 = path[base + static_cast<size_t>(r1) * N + c1];
    const unsigned int op00 = p00, op01 = p01, op10 = p10, op11 = p11;

    sd[r0][c0] = dist[base + static_cast<size_t>(r0) * N + c0];
    sd[r0][c1] = dist[base + static_cast<size_t>(r0) * N + c1];
    sd[r1][c0] = dist[base + static_cast<size_t>(r1) * N + c0];
    sd[r1][c1] = dist[base + static_cast<size_t>(r1) * N + c1];
    __syncthreads();

    for (int k = 0; k < BS; ++k) {
        const unsigned int a0 = sd[r0][k], a1 = sd[r1][k];
        const unsigned int b0 = sd[k][c0], b1 = sd[k][c1];
        __syncthreads();

        unsigned int nd = a0 + b0;
        if (nd < sd[r0][c0]) { sd[r0][c0] = nd; p00 = k0 + k; }
        nd = a0 + b1;
        if (nd < sd[r0][c1]) { sd[r0][c1] = nd; p01 = k0 + k; }
        nd = a1 + b0;
        if (nd < sd[r1][c0]) { sd[r1][c0] = nd; p10 = k0 + k; }
        nd = a1 + b1;
        if (nd < sd[r1][c1]) { sd[r1][c1] = nd; p11 = k0 + k; }
        __syncthreads();
    }

    dist[base + static_cast<size_t>(r0) * N + c0] = sd[r0][c0];
    dist[base + static_cast<size_t>(r0) * N + c1] = sd[r0][c1];
    dist[base + static_cast<size_t>(r1) * N + c0] = sd[r1][c0];
    dist[base + static_cast<size_t>(r1) * N + c1] = sd[r1][c1];
    if (p00 != op00) path[base + static_cast<size_t>(r0) * N + c0] = p00;
    if (p01 != op01) path[base + static_cast<size_t>(r0) * N + c1] = p01;
    if (p10 != op10) path[base + static_cast<size_t>(r1) * N + c0] = p10;
    if (p11 != op11) path[base + static_cast<size_t>(r1) * N + c1] = p11;
}

// Phase 2a: the panel rows [k0,k0+BS) x all columns, using the closed diagonal
// block as the left operand.  Executed by the rank owning the panel rows.
__global__ void fwPhase2Row(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                            const int N, const int rowOff, const int k0) {
    const int colBase = blockIdx.x * BS;
    if (colBase == k0) return; // handled by phase 1

    __shared__ unsigned int sdiag[BS][BS + 1];
    __shared__ unsigned int sd[BS][BS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r0 = ty, r1 = ty + TPB_Y;
    const int c0 = tx, c1 = tx + TPB_X;

    const size_t dbase = static_cast<size_t>(rowOff) * N + k0;
    sdiag[r0][c0] = dist[dbase + static_cast<size_t>(r0) * N + c0];
    sdiag[r0][c1] = dist[dbase + static_cast<size_t>(r0) * N + c1];
    sdiag[r1][c0] = dist[dbase + static_cast<size_t>(r1) * N + c0];
    sdiag[r1][c1] = dist[dbase + static_cast<size_t>(r1) * N + c1];

    const size_t tbase = static_cast<size_t>(rowOff) * N + colBase;
    sd[r0][c0] = dist[tbase + static_cast<size_t>(r0) * N + c0];
    sd[r0][c1] = dist[tbase + static_cast<size_t>(r0) * N + c1];
    sd[r1][c0] = dist[tbase + static_cast<size_t>(r1) * N + c0];
    sd[r1][c1] = dist[tbase + static_cast<size_t>(r1) * N + c1];

    unsigned int p00 = path[tbase + static_cast<size_t>(r0) * N + c0];
    unsigned int p01 = path[tbase + static_cast<size_t>(r0) * N + c1];
    unsigned int p10 = path[tbase + static_cast<size_t>(r1) * N + c0];
    unsigned int p11 = path[tbase + static_cast<size_t>(r1) * N + c1];
    const unsigned int op00 = p00, op01 = p01, op10 = p10, op11 = p11;
    __syncthreads();

    for (int k = 0; k < BS; ++k) {
        const unsigned int a0 = sdiag[r0][k], a1 = sdiag[r1][k];
        const unsigned int b0 = sd[k][c0], b1 = sd[k][c1];
        __syncthreads();

        unsigned int nd = a0 + b0;
        if (nd < sd[r0][c0]) { sd[r0][c0] = nd; p00 = k0 + k; }
        nd = a0 + b1;
        if (nd < sd[r0][c1]) { sd[r0][c1] = nd; p01 = k0 + k; }
        nd = a1 + b0;
        if (nd < sd[r1][c0]) { sd[r1][c0] = nd; p10 = k0 + k; }
        nd = a1 + b1;
        if (nd < sd[r1][c1]) { sd[r1][c1] = nd; p11 = k0 + k; }
        __syncthreads();
    }

    dist[tbase + static_cast<size_t>(r0) * N + c0] = sd[r0][c0];
    dist[tbase + static_cast<size_t>(r0) * N + c1] = sd[r0][c1];
    dist[tbase + static_cast<size_t>(r1) * N + c0] = sd[r1][c0];
    dist[tbase + static_cast<size_t>(r1) * N + c1] = sd[r1][c1];
    if (p00 != op00) path[tbase + static_cast<size_t>(r0) * N + c0] = p00;
    if (p01 != op01) path[tbase + static_cast<size_t>(r0) * N + c1] = p01;
    if (p10 != op10) path[tbase + static_cast<size_t>(r1) * N + c0] = p10;
    if (p11 != op11) path[tbase + static_cast<size_t>(r1) * N + c1] = p11;
}

// Phase 2b: all local rows (except the panel rows) x panel columns, using the
// broadcast panel as the right operand.  Sequential in k because the left
// operand dist[i][k] lives inside the updated tile.
__global__ void fwPhase2Col(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                            const unsigned int* __restrict__ panel, const int N, const int k0,
                            const int blkOff, const int skipA, const int skipB) {
    const int blk = blockIdx.x + blkOff;
    if (blk == skipA || blk == skipB) return;
    const int rowBase = blk * BS;

    __shared__ unsigned int sdiag[BS][BS + 1];
    __shared__ unsigned int sd[BS][BS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int r0 = ty, r1 = ty + TPB_Y;
    const int c0 = tx, c1 = tx + TPB_X;

    sdiag[r0][c0] = panel[static_cast<size_t>(r0) * N + k0 + c0];
    sdiag[r0][c1] = panel[static_cast<size_t>(r0) * N + k0 + c1];
    sdiag[r1][c0] = panel[static_cast<size_t>(r1) * N + k0 + c0];
    sdiag[r1][c1] = panel[static_cast<size_t>(r1) * N + k0 + c1];

    const size_t tbase = static_cast<size_t>(rowBase) * N + k0;
    sd[r0][c0] = dist[tbase + static_cast<size_t>(r0) * N + c0];
    sd[r0][c1] = dist[tbase + static_cast<size_t>(r0) * N + c1];
    sd[r1][c0] = dist[tbase + static_cast<size_t>(r1) * N + c0];
    sd[r1][c1] = dist[tbase + static_cast<size_t>(r1) * N + c1];

    unsigned int p00 = path[tbase + static_cast<size_t>(r0) * N + c0];
    unsigned int p01 = path[tbase + static_cast<size_t>(r0) * N + c1];
    unsigned int p10 = path[tbase + static_cast<size_t>(r1) * N + c0];
    unsigned int p11 = path[tbase + static_cast<size_t>(r1) * N + c1];
    const unsigned int op00 = p00, op01 = p01, op10 = p10, op11 = p11;
    __syncthreads();

    for (int k = 0; k < BS; ++k) {
        const unsigned int a0 = sd[r0][k], a1 = sd[r1][k];
        const unsigned int b0 = sdiag[k][c0], b1 = sdiag[k][c1];
        __syncthreads();

        unsigned int nd = a0 + b0;
        if (nd < sd[r0][c0]) { sd[r0][c0] = nd; p00 = k0 + k; }
        nd = a0 + b1;
        if (nd < sd[r0][c1]) { sd[r0][c1] = nd; p01 = k0 + k; }
        nd = a1 + b0;
        if (nd < sd[r1][c0]) { sd[r1][c0] = nd; p10 = k0 + k; }
        nd = a1 + b1;
        if (nd < sd[r1][c1]) { sd[r1][c1] = nd; p11 = k0 + k; }
        __syncthreads();
    }

    dist[tbase + static_cast<size_t>(r0) * N + c0] = sd[r0][c0];
    dist[tbase + static_cast<size_t>(r0) * N + c1] = sd[r0][c1];
    dist[tbase + static_cast<size_t>(r1) * N + c0] = sd[r1][c0];
    dist[tbase + static_cast<size_t>(r1) * N + c1] = sd[r1][c1];
    if (p00 != op00) path[tbase + static_cast<size_t>(r0) * N + c0] = p00;
    if (p01 != op01) path[tbase + static_cast<size_t>(r0) * N + c1] = p01;
    if (p10 != op10) path[tbase + static_cast<size_t>(r1) * N + c0] = p10;
    if (p11 != op11) path[tbase + static_cast<size_t>(r1) * N + c1] = p11;
}

// Phase 3: the bulk of the work.  All local rows (minus the panel rows) x all
// columns (minus the panel columns).  Both operands are final, so the k loop
// carries no dependency and runs entirely out of shared memory / registers.
// Register blocking: P3_RY x 2 elements per thread keeps the shared-memory
// load count per min-plus operation well below one.
constexpr int P3_TX = 32;
constexpr int P3_TY = 8;
constexpr int P3_RY = BS / P3_TY; // rows per thread
constexpr unsigned int NO_UPDATE = 0xffffffffu;

__global__ __launch_bounds__(P3_TX* P3_TY) void fwPhase3(unsigned int* __restrict__ dist,
                                                         unsigned int* __restrict__ path,
                                                         const unsigned int* __restrict__ panel,
                                                         const int N, const int k0,
                                                         const int blkOff, const int skipA,
                                                         const int skipB) {
    const int blk = blockIdx.y + blkOff;
    const int colBase = blockIdx.x * BS;
    if (colBase == k0 || blk == skipA || blk == skipB) return;
    const int rowBase = blk * BS;

    __shared__ unsigned int sik[BS][BS + 1];
    __shared__ unsigned int skj[BS][BS + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int c0 = tx, c1 = tx + P3_TX;

    const size_t ikbase = static_cast<size_t>(rowBase) * N + k0;
    const size_t tbase = static_cast<size_t>(rowBase) * N + colBase;

    unsigned int d0[P3_RY], d1[P3_RY], p0[P3_RY], p1[P3_RY];

#pragma unroll
    for (int i = 0; i < P3_RY; ++i) {
        const int r = ty + i * P3_TY;
        sik[r][c0] = dist[ikbase + static_cast<size_t>(r) * N + c0];
        sik[r][c1] = dist[ikbase + static_cast<size_t>(r) * N + c1];
        skj[r][c0] = panel[static_cast<size_t>(r) * N + colBase + c0];
        skj[r][c1] = panel[static_cast<size_t>(r) * N + colBase + c1];
        d0[i] = dist[tbase + static_cast<size_t>(r) * N + c0];
        d1[i] = dist[tbase + static_cast<size_t>(r) * N + c1];
        p0[i] = NO_UPDATE;
        p1[i] = NO_UPDATE;
    }
    __syncthreads();

#pragma unroll 4
    for (int k = 0; k < BS; ++k) {
        const unsigned int b0 = skj[k][c0], b1 = skj[k][c1];
        const unsigned int pk = static_cast<unsigned int>(k0 + k);
#pragma unroll
        for (int i = 0; i < P3_RY; ++i) {
            const unsigned int a = sik[ty + i * P3_TY][k];
            unsigned int nd = a + b0;
            if (nd < d0[i]) { d0[i] = nd; p0[i] = pk; }
            nd = a + b1;
            if (nd < d1[i]) { d1[i] = nd; p1[i] = pk; }
        }
    }

#pragma unroll
    for (int i = 0; i < P3_RY; ++i) {
        const int r = ty + i * P3_TY;
        if (p0[i] != NO_UPDATE) {
            dist[tbase + static_cast<size_t>(r) * N + c0] = d0[i];
            path[tbase + static_cast<size_t>(r) * N + c0] = p0[i];
        }
        if (p1[i] != NO_UPDATE) {
            dist[tbase + static_cast<size_t>(r) * N + c1] = d1[i];
            path[tbase + static_cast<size_t>(r) * N + c1] = p1[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

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
    bool ok = true;
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
#pragma omp parallel for collapse(2) schedule(static) reduction(&& : ok)
    for (size_t i = 0; i < lim; ++i) {
        for (size_t j = 0; j < lim; ++j) {
            for (size_t k = 0; k < numNodes; ++k) {
                const unsigned int distIJ = dist[idx2(j, i, numNodes)];
                const unsigned int distIK = dist[idx2(k, i, numNodes)];
                const unsigned int distKJ = dist[idx2(j, k, numNodes)];

                // Check for overflow before addition
                if (distIK < INF && distKJ < INF) {
                    if (distIK + distKJ < distIJ) {
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n",
                               i, j, k);
                        ok = false;
                    }
                }
            }
        }
    }

    return ok;
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (numNodes == 0) {
        if (rank == 0) printf("Number of nodes must be positive\n");
        MPI_Finalize();
        return 1;
    }

    // Bind one GPU per rank (round robin over the node-local devices).
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) printf("No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    // Leaders (one per node) carry the inter-node part of the panel exchange;
    // inside a node the panel is shared through an MPI-3 shared memory window.
    MPI_Comm leaderComm;
    MPI_Comm_split(MPI_COMM_WORLD, localRank == 0 ? 0 : MPI_UNDEFINED, rank, &leaderComm);
    int myLeaderId = -1;
    if (localRank == 0) MPI_Comm_rank(leaderComm, &myLeaderId);
    MPI_Bcast(&myLeaderId, 1, MPI_INT, 0, nodeComm);
    std::vector<int> leaderOfRank(nranks);
    MPI_Allgather(&myLeaderId, 1, MPI_INT, leaderOfRank.data(), 1, MPI_INT, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads: %d, CUDA devices/node: %d\n", nranks,
               omp_get_max_threads(), deviceCount);
    }

    // Padded problem size: multiple of the blocking factor.
    const int N = static_cast<int>(((numNodes + BS - 1) / BS) * BS);
    const int numBlocks = N / BS;

    // Contiguous block-row distribution.
    std::vector<int> blkStart(nranks + 1, 0);
    for (int r = 0; r < nranks; ++r) {
        const int base = numBlocks / nranks;
        const int rem = numBlocks % nranks;
        blkStart[r + 1] = blkStart[r] + base + (r < rem ? 1 : 0);
    }
    const int localBlocks = blkStart[rank + 1] - blkStart[rank];
    const int rowStart = blkStart[rank] * BS;
    const int localRows = localBlocks * BS;

    // Rows of the (unpadded) input this rank is responsible for.
    const size_t realStart = std::min(static_cast<size_t>(rowStart), numNodes);
    const size_t realEnd = std::min(static_cast<size_t>(rowStart + localRows), numNodes);
    const size_t realRows = realEnd - realStart;

    // Rank 0 keeps the full unpadded matrix (input generation + final output).
    std::vector<unsigned int> dist;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
        printf("Initializing graph...\n");
        initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);
    }

    // Scatter the input row blocks.
    std::vector<int> counts(nranks), displs(nranks);
    for (int r = 0; r < nranks; ++r) {
        const size_t s = std::min(static_cast<size_t>(blkStart[r] * BS), numNodes);
        const size_t e = std::min(static_cast<size_t>(blkStart[r + 1] * BS), numNodes);
        counts[r] = static_cast<int>((e - s) * numNodes);
        displs[r] = static_cast<int>(s * numNodes);
    }
    std::vector<unsigned int> rawRows(realRows * numNodes);
    MPI_Scatterv(rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED,
                 rawRows.data(), static_cast<int>(realRows * numNodes), MPI_UNSIGNED, 0,
                 MPI_COMM_WORLD);

    // Build the padded local tiles.  Padding nodes are isolated (INF), which
    // leaves the result for the real nodes untouched.
    unsigned int* h_local = nullptr;
    unsigned int* h_panel[2] = {nullptr, nullptr};
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_panel[2] = {nullptr, nullptr};
    const size_t localElems = static_cast<size_t>(localRows) * N;
    const size_t panelBytes = static_cast<size_t>(BS) * N * sizeof(unsigned int);

    // Double buffered panel staging area, shared by all ranks on this node, so
    // the node-local part of the panel exchange needs no data movement at all.
    MPI_Win panelWin;
    unsigned int* winBase = nullptr;
    MPI_Win_allocate_shared(localRank == 0 ? static_cast<MPI_Aint>(2 * panelBytes) : 0,
                            sizeof(unsigned int), MPI_INFO_NULL, nodeComm, &winBase, &panelWin);
    {
        MPI_Aint winSize = 0;
        int dispUnit = 0;
        MPI_Win_shared_query(panelWin, 0, &winSize, &dispUnit, &winBase);
    }
    h_panel[0] = winBase;
    h_panel[1] = winBase + static_cast<size_t>(BS) * N;
    // Page-lock the window so the panel copies are asynchronous and fast.
    cudaHostRegister(winBase, 2 * panelBytes, cudaHostRegisterDefault);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, panelWin);

    CUDA_CHECK(cudaMalloc(&d_panel[0], panelBytes));
    CUDA_CHECK(cudaMalloc(&d_panel[1], panelBytes));

    // Compute stream plus a dedicated copy stream, so that staging the next
    // k-panel over PCIe overlaps with the current panel's update kernels.
    cudaStream_t stream, cstream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUDA_CHECK(cudaStreamCreate(&cstream));
    cudaEvent_t evPanelUsed[2], evPanelReady[2], evPanelFree[2], evPanelMade, evD2H;
    CUDA_CHECK(cudaEventCreateWithFlags(&evPanelMade, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&evD2H, cudaEventDisableTiming));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evPanelUsed[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evPanelReady[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evPanelFree[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(evPanelUsed[b], cstream));
        CUDA_CHECK(cudaEventRecord(evPanelFree[b], stream));
    }

    if (localRows > 0) {
        CUDA_CHECK(cudaHostAlloc(&h_local, localElems * sizeof(unsigned int), cudaHostAllocDefault));
        std::vector<unsigned int> h_path(localElems);

#pragma omp parallel for schedule(static)
        for (int li = 0; li < localRows; ++li) {
            const size_t gi = static_cast<size_t>(rowStart) + li;
            unsigned int* drow = h_local + static_cast<size_t>(li) * N;
            unsigned int* prow = h_path.data() + static_cast<size_t>(li) * N;
            if (gi < numNodes) {
                memcpy(drow, rawRows.data() + (gi - realStart) * numNodes,
                       numNodes * sizeof(unsigned int));
                for (int j = static_cast<int>(numNodes); j < N; ++j) drow[j] = INF;
            } else {
                for (int j = 0; j < N; ++j) drow[j] = INF;
                drow[gi] = 0;
            }
            // Original initializePathMatrix() leaves path[i][j] == i.
            for (int j = 0; j < N; ++j) prow[j] = static_cast<unsigned int>(gi);
        }

        CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMemcpy(d_dist, h_local, localElems * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, h_path.data(), localElems * sizeof(unsigned int),
                              cudaMemcpyHostToDevice));
    }
    std::vector<unsigned int>().swap(rawRows);

    if (rank == 0) printf("Computing shortest paths...\n");

    // Owner rank of every block row.
    std::vector<int> blockOwner(numBlocks);
    for (int r = 0; r < nranks; ++r) {
        for (int b = blkStart[r]; b < blkStart[r + 1]; ++b) blockOwner[b] = r;
    }

    const dim3 threads(TPB_X, TPB_Y);
    const dim3 threads3(P3_TX, P3_TY);

    // Phases 1 and 2a close the panel k-block "kb"; only its owner runs them.
    auto makePanelKernels = [&](int kb) {
        const int panelRowOff = kb * BS - rowStart;
        fwPhase1<<<1, threads, 0, stream>>>(d_dist, d_path, N, panelRowOff, kb * BS);
        fwPhase2Row<<<numBlocks, threads, 0, stream>>>(d_dist, d_path, N, panelRowOff, kb * BS);
        CUDA_CHECK(cudaEventRecord(evPanelMade, stream));
    };

    // Downloads the finished panel into the node-shared staging buffer.
    auto downloadPanel = [&](int kb, unsigned int* hbuf) {
        CUDA_CHECK(cudaStreamWaitEvent(cstream, evPanelMade, 0));
        CUDA_CHECK(cudaMemcpyAsync(hbuf, d_dist + static_cast<size_t>(kb * BS - rowStart) * N,
                                   panelBytes, cudaMemcpyDeviceToHost, cstream));
        CUDA_CHECK(cudaEventRecord(evD2H, cstream));
        CUDA_CHECK(cudaEventSynchronize(evD2H));
    };

    // Publishes the panel produced by "owner" to every rank: node-local ranks
    // simply read the shared window, the node leaders exchange it across nodes.
    auto sharePanel = [&](int owner, unsigned int* hbuf) {
        MPI_Win_sync(panelWin);
        MPI_Barrier(nodeComm);
        if (localRank == 0 && nranks > nodeSize) {
            MPI_Bcast(hbuf, BS * N, MPI_UNSIGNED, leaderOfRank[owner], leaderComm);
            MPI_Win_sync(panelWin);
        }
        MPI_Barrier(nodeComm);
    };

    // Makes panel "kb" available in the (double buffered) device panel slot on
    // the copy stream, so it proceeds while the compute stream is still busy.
    // The owner already has the rows in device memory and copies them locally.
    auto stagePanel = [&](int kb, unsigned int* hbuf) {
        const int pb = kb & 1;
        CUDA_CHECK(cudaStreamWaitEvent(cstream, evPanelFree[pb], 0));
        if (blockOwner[kb] == rank) {
            CUDA_CHECK(cudaMemcpyAsync(d_panel[pb],
                                       d_dist + static_cast<size_t>(kb * BS - rowStart) * N,
                                       panelBytes, cudaMemcpyDeviceToDevice, cstream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(d_panel[pb], hbuf, panelBytes, cudaMemcpyHostToDevice,
                                       cstream));
        }
        CUDA_CHECK(cudaEventRecord(evPanelUsed[pb], cstream));
        CUDA_CHECK(cudaEventRecord(evPanelReady[pb], cstream));
    };

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    // Prologue: publish the first panel.
    if (rank == blockOwner[0]) {
        makePanelKernels(0);
        downloadPanel(0, h_panel[0]);
    }
    sharePanel(blockOwner[0], h_panel[0]);
    if (localBlocks > 0) stagePanel(0, h_panel[0]);

    int cur = 0;
    for (int kb = 0; kb < numBlocks; ++kb) {
        const int k0 = kb * BS;
        const int nxt = cur ^ 1;
        const int pb = kb & 1;
        const int panelBlk = (blockOwner[kb] == rank) ? kb - blkStart[rank] : -1;
        const int nextOwner = (kb + 1 < numBlocks) ? blockOwner[kb + 1] : -1;
        const int nextBlk = (nextOwner == rank) ? kb + 1 - blkStart[rank] : -1;

        if (localBlocks > 0) CUDA_CHECK(cudaStreamWaitEvent(stream, evPanelReady[pb], 0));

        // Give the next panel's owner a head start: update just those rows and
        // close the next panel, so its exchange overlaps the bulk update below.
        if (nextBlk >= 0) {
            fwPhase2Col<<<1, threads, 0, stream>>>(d_dist, d_path, d_panel[pb], N, k0, nextBlk, -1,
                                                   -1);
            fwPhase3<<<dim3(numBlocks, 1), threads3, 0, stream>>>(d_dist, d_path, d_panel[pb], N, k0,
                                                                  nextBlk, -1, -1);
            makePanelKernels(kb + 1);
        }

        // Bulk update of the remaining local row blocks for this k-panel.  It is
        // enqueued before the panel exchange so that the exchange overlaps with
        // these kernels instead of stalling the GPU.
        if (localBlocks > 0) {
            fwPhase2Col<<<localBlocks, threads, 0, stream>>>(d_dist, d_path, d_panel[pb], N, k0, 0,
                                                             panelBlk, nextBlk);
            fwPhase3<<<dim3(numBlocks, localBlocks), threads3, 0, stream>>>(
                d_dist, d_path, d_panel[pb], N, k0, 0, panelBlk, nextBlk);
            CUDA_CHECK(cudaEventRecord(evPanelFree[pb], stream));
        }

        if (nextOwner >= 0) {
            // Nobody may still be uploading out of the slot about to be rewritten.
            CUDA_CHECK(cudaEventSynchronize(evPanelUsed[nxt]));
            MPI_Barrier(nodeComm);
            if (nextBlk >= 0) downloadPanel(kb + 1, h_panel[nxt]);
            sharePanel(nextOwner, h_panel[nxt]);
        }

        cur = nxt;
        if (nextOwner >= 0 && localBlocks > 0) stagePanel(kb + 1, h_panel[cur]);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Collect the result on rank 0.
    if (localRows > 0) {
        CUDA_CHECK(cudaMemcpy(h_local, d_dist, localElems * sizeof(unsigned int),
                              cudaMemcpyDeviceToHost));
    }
    std::vector<unsigned int> sendRows(realRows * numNodes);
#pragma omp parallel for schedule(static)
    for (size_t li = 0; li < realRows; ++li) {
        memcpy(sendRows.data() + li * numNodes,
               h_local + (realStart - rowStart + li) * static_cast<size_t>(N),
               numNodes * sizeof(unsigned int));
    }
    MPI_Gatherv(sendRows.data(), static_cast<int>(realRows * numNodes), MPI_UNSIGNED,
                rank == 0 ? dist.data() : nullptr, counts.data(), displs.data(), MPI_UNSIGNED, 0,
                MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (d_dist) CUDA_CHECK(cudaFree(d_dist));
    if (d_path) CUDA_CHECK(cudaFree(d_path));
    CUDA_CHECK(cudaFree(d_panel[0]));
    CUDA_CHECK(cudaFree(d_panel[1]));
    if (h_local) CUDA_CHECK(cudaFreeHost(h_local));
    MPI_Win_unlock_all(panelWin);
    cudaHostUnregister(winBase);
    MPI_Win_free(&panelWin);
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaEventDestroy(evPanelUsed[b]));
        CUDA_CHECK(cudaEventDestroy(evPanelReady[b]));
        CUDA_CHECK(cudaEventDestroy(evPanelFree[b]));
    }
    CUDA_CHECK(cudaEventDestroy(evPanelMade));
    CUDA_CHECK(cudaEventDestroy(evD2H));
    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaStreamDestroy(cstream));

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
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                status = 1;
            }
        }
    }

    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
