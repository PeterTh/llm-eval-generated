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

// ---------------------------------------------------------------------------
// Hybrid MPI + OpenMP + CUDA blocked Floyd-Warshall
//
//  * CUDA   : the O(n^3) kernel is executed on the GPUs using the classic
//             three-phase blocked Floyd-Warshall formulation (tiles of
//             BS x BS held in shared memory, register blocking per thread).
//  * MPI    : the distance matrix is distributed by block rows over the
//             ranks (one GPU per rank).  For every pivot block k the owner
//             of block row k computes the pivot tile and the pivot row panel
//             and broadcasts it to all other ranks.  A one step look-ahead on
//             a separate CUDA stream produces the panel for step k+1 while the
//             independent tiles of step k are still being computed, so the
//             (pipelined, non-blocking) broadcast is hidden behind the GPU work.
//  * OpenMP : host side work (deterministic matrix generation and validation)
//             is threaded.
//
// The reported time covers the distributed solve including all communication
// it requires; gathering the distributed result on rank 0 for printing is
// output marshalling and is done after the timer, just like the (untimed)
// initialisation and printing of the original program.
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err__ = (call);                                                                              \
        if (err__ != cudaSuccess) {                                                                                    \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__);               \
            fflush(stderr);                                                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// Tile size and thread geometry.  Every thread owns an RY x RX register tile.
constexpr int BS = 64;
constexpr int TX = 32;
constexpr int TY = 16;
constexpr int RY = BS / TY; // 4
constexpr int RX = BS / TX; // 2
constexpr unsigned int NO_PATH = 0xFFFFFFFFu;
// Number of pivot panel buffers kept in flight (look-ahead pipeline depth).
constexpr int NSLOT = 3;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// Phase 1: the dependent pivot tile (k,k) updates itself.
// `dist`/`path` point at the top left corner of the pivot tile, row stride nPad.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(TX* TY) void fwPhase1(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                                   const size_t nPad, const unsigned int kBase) {
    __shared__ unsigned int s[BS][BS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    unsigned int d[RY][RX];
    unsigned int p[RY][RX];

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const unsigned int v = dist[static_cast<size_t>(ty + a * TY) * nPad + (tx + b * TX)];
            s[ty + a * TY][tx + b * TX] = v;
            d[a][b] = v;
            p[a][b] = NO_PATH;
        }
    }

    // Row kk and column kk of the tile are provably invariant during
    // iteration kk (the tile diagonal is zero), hence a single barrier
    // per iteration is sufficient.
    for (int kk = 0; kk < BS; ++kk) {
        __syncthreads();
        unsigned int rv[RX];
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            rv[b] = s[kk][tx + b * TX];
        }
#pragma unroll
        for (int a = 0; a < RY; ++a) {
            const unsigned int c = s[ty + a * TY][kk];
#pragma unroll
            for (int b = 0; b < RX; ++b) {
                const unsigned int nd = c + rv[b];
                if (nd < d[a][b]) {
                    d[a][b] = nd;
                    s[ty + a * TY][tx + b * TX] = nd;
                    p[a][b] = kBase + static_cast<unsigned int>(kk);
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const size_t off = static_cast<size_t>(ty + a * TY) * nPad + (tx + b * TX);
            dist[off] = d[a][b];
            if (p[a][b] != NO_PATH) {
                path[off] = p[a][b];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2a: the tiles of the pivot block row (k, j), j != k.
// `dist`/`path` point at the first row of the pivot block row (column 0).
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(TX* TY) void fwPhase2Row(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                                      const size_t nPad, const int kb) {
    const int bj = blockIdx.x;
    if (bj == kb) {
        return;
    }

    __shared__ unsigned int piv[BS][BS];
    __shared__ unsigned int s[BS][BS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t pivCol = static_cast<size_t>(kb) * BS;
    const size_t blkCol = static_cast<size_t>(bj) * BS;

    unsigned int d[RY][RX];
    unsigned int p[RY][RX];

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const size_t r = static_cast<size_t>(ty + a * TY);
            piv[ty + a * TY][tx + b * TX] = dist[r * nPad + pivCol + (tx + b * TX)];
            const unsigned int v = dist[r * nPad + blkCol + (tx + b * TX)];
            s[ty + a * TY][tx + b * TX] = v;
            d[a][b] = v;
            p[a][b] = NO_PATH;
        }
    }

    const unsigned int kBase = static_cast<unsigned int>(kb) * BS;
    for (int kk = 0; kk < BS; ++kk) {
        __syncthreads();
        unsigned int rv[RX];
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            rv[b] = s[kk][tx + b * TX];
        }
#pragma unroll
        for (int a = 0; a < RY; ++a) {
            const unsigned int c = piv[ty + a * TY][kk];
#pragma unroll
            for (int b = 0; b < RX; ++b) {
                const unsigned int nd = c + rv[b];
                if (nd < d[a][b]) {
                    d[a][b] = nd;
                    s[ty + a * TY][tx + b * TX] = nd;
                    p[a][b] = kBase + static_cast<unsigned int>(kk);
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const size_t off = static_cast<size_t>(ty + a * TY) * nPad + blkCol + (tx + b * TX);
            dist[off] = d[a][b];
            if (p[a][b] != NO_PATH) {
                path[off] = p[a][b];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 2b: the tiles of the pivot block column (i, k), i != k.
// `dist`/`path` point at the first locally owned row of the launch range,
// `panel` holds the (already updated) pivot block row.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(TX* TY) void fwPhase2Col(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                                      const unsigned int* __restrict__ panel, const size_t nPad,
                                                      const int kb, const int gbStart, const int skipA, const int skipB) {
    const int bi = blockIdx.x;
    const int gi = gbStart + bi;
    if (gi == kb || gi == skipA || gi == skipB) {
        return;
    }

    __shared__ unsigned int piv[BS][BS];
    __shared__ unsigned int s[BS][BS];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t pivCol = static_cast<size_t>(kb) * BS;
    const size_t rowBase = static_cast<size_t>(bi) * BS;

    unsigned int d[RY][RX];
    unsigned int p[RY][RX];

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const size_t r = static_cast<size_t>(ty + a * TY);
            piv[ty + a * TY][tx + b * TX] = panel[r * nPad + pivCol + (tx + b * TX)];
            const unsigned int v = dist[(rowBase + r) * nPad + pivCol + (tx + b * TX)];
            s[ty + a * TY][tx + b * TX] = v;
            d[a][b] = v;
            p[a][b] = NO_PATH;
        }
    }

    const unsigned int kBase = static_cast<unsigned int>(kb) * BS;
    for (int kk = 0; kk < BS; ++kk) {
        __syncthreads();
        unsigned int rv[RX];
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            rv[b] = piv[kk][tx + b * TX];
        }
#pragma unroll
        for (int a = 0; a < RY; ++a) {
            const unsigned int c = s[ty + a * TY][kk];
#pragma unroll
            for (int b = 0; b < RX; ++b) {
                const unsigned int nd = c + rv[b];
                if (nd < d[a][b]) {
                    d[a][b] = nd;
                    s[ty + a * TY][tx + b * TX] = nd;
                    p[a][b] = kBase + static_cast<unsigned int>(kk);
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const size_t off = (rowBase + static_cast<size_t>(ty + a * TY)) * nPad + pivCol + (tx + b * TX);
            dist[off] = d[a][b];
            if (p[a][b] != NO_PATH) {
                path[off] = p[a][b];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Phase 3: all remaining (independent) tiles (i, j), i != k, j != k.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(TX* TY) void fwPhase3(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                                                   const unsigned int* __restrict__ panel, const size_t nPad,
                                                   const int kb, const int gbStart, const int skipA, const int skipB) {
    const int bj = blockIdx.x;
    const int bi = blockIdx.y;
    const int gi = gbStart + bi;
    if (bj == kb || gi == kb || gi == skipA || gi == skipB) {
        return;
    }

    __shared__ unsigned int srow[BS][BS]; // tile (k, bj)
    __shared__ unsigned int scol[BS][BS]; // tile (gi, k)

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const size_t pivCol = static_cast<size_t>(kb) * BS;
    const size_t blkCol = static_cast<size_t>(bj) * BS;
    const size_t rowBase = static_cast<size_t>(bi) * BS;

    unsigned int d[RY][RX];
    unsigned int p[RY][RX];

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const size_t r = static_cast<size_t>(ty + a * TY);
            srow[ty + a * TY][tx + b * TX] = panel[r * nPad + blkCol + (tx + b * TX)];
            scol[ty + a * TY][tx + b * TX] = dist[(rowBase + r) * nPad + pivCol + (tx + b * TX)];
            d[a][b] = dist[(rowBase + r) * nPad + blkCol + (tx + b * TX)];
            p[a][b] = NO_PATH;
        }
    }
    __syncthreads();

    const unsigned int kBase = static_cast<unsigned int>(kb) * BS;
#pragma unroll 8
    for (int kk = 0; kk < BS; ++kk) {
        unsigned int rv[RX];
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            rv[b] = srow[kk][tx + b * TX];
        }
#pragma unroll
        for (int a = 0; a < RY; ++a) {
            const unsigned int c = scol[ty + a * TY][kk];
#pragma unroll
            for (int b = 0; b < RX; ++b) {
                const unsigned int nd = c + rv[b];
                if (nd < d[a][b]) {
                    d[a][b] = nd;
                    p[a][b] = kBase + static_cast<unsigned int>(kk);
                }
            }
        }
    }

#pragma unroll
    for (int a = 0; a < RY; ++a) {
#pragma unroll
        for (int b = 0; b < RX; ++b) {
            const size_t off = (rowBase + static_cast<size_t>(ty + a * TY)) * nPad + blkCol + (tx + b * TX);
            dist[off] = d[a][b];
            if (p[a][b] != NO_PATH) {
                path[off] = p[a][b];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Deterministic parallel reproduction of the original serial rand_r() stream.
// The generator is a 32 bit LCG advanced three times per rand_r() call, so an
// arbitrary position in the stream can be reached with a logarithmic jump.
// ---------------------------------------------------------------------------
constexpr unsigned int LCG_A = 1103515245u;
constexpr unsigned int LCG_C = 12345u;

static inline int randRepl(unsigned int& state) noexcept {
    unsigned int next = state;
    int result;
    next = next * LCG_A + LCG_C;
    result = static_cast<int>((next / 65536u) % 2048u);
    next = next * LCG_A + LCG_C;
    result <<= 10;
    result ^= static_cast<int>((next / 65536u) % 1024u);
    next = next * LCG_A + LCG_C;
    result <<= 10;
    result ^= static_cast<int>((next / 65536u) % 1024u);
    state = next;
    return result;
}

static inline unsigned int lcgJump(const unsigned int s0, unsigned long long steps) noexcept {
    unsigned int accA = 1u, accC = 0u;
    unsigned int a = LCG_A, c = LCG_C;
    while (steps) {
        if (steps & 1ull) {
            accC = accC * a + c;
            accA = accA * a;
        }
        c = a * c + c;
        a = a * a;
        steps >>= 1;
    }
    return accA * s0 + accC;
}

// Fill the locally owned block rows [rowBegin, rowEnd) of the padded matrix.
static void initializeLocalMatrices(unsigned int* dist, unsigned int* path, const size_t rowBegin, const size_t rowEnd,
                                    const size_t numNodes, const size_t nPad, const unsigned int rangeMin,
                                    const unsigned int rangeMax) {
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;

#pragma omp parallel for schedule(static)
    for (long long rr = static_cast<long long>(rowBegin); rr < static_cast<long long>(rowEnd); ++rr) {
        const size_t r = static_cast<size_t>(rr);
        unsigned int* drow = dist + (r - rowBegin) * nPad;
        unsigned int* prow = path + (r - rowBegin) * nPad;

        if (r < numNodes) {
            unsigned int state = lcgJump(42u, 3ull * static_cast<unsigned long long>(r) * numNodes);
            for (size_t j = 0; j < numNodes; ++j) {
                drow[j] = rangeMin + static_cast<unsigned int>(range * randRepl(state) / static_cast<double>(RAND_MAX));
            }
            for (size_t j = numNodes; j < nPad; ++j) {
                drow[j] = INF;
            }
            drow[r] = 0; // distance from node to itself is 0
        } else {
            // Padding rows: isolated nodes, they can never improve a real path.
            for (size_t j = 0; j < nPad; ++j) {
                drow[j] = INF;
            }
            drow[r] = 0;
        }

        // The original path initialisation assigns the row index to every entry.
        const unsigned int rowValue = static_cast<unsigned int>(r);
        for (size_t j = 0; j < nPad; ++j) {
            prow[j] = rowValue;
        }
    }
}

static bool validateResult(const std::vector<unsigned int>& dist, const size_t numNodes) {
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
    const size_t lim = std::min(numNodes, static_cast<size_t>(10));
    bool ok = true;
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
                        printf("Validation failed: triangle inequality violated at [%zu,%zu,%zu]\n", i, j, k);
                        ok = false;
                        break;
                    }
                }
            }
        }
    }

    return ok;
}

static void printUsage(const char* progName) {
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

    int rank = 0, nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

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
            if (rank == 0) {
                printUsage(argv[0]);
            }
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
        if (rank == 0) {
            printf("Number of nodes must be greater than zero\n");
        }
        MPI_Finalize();
        return 1;
    }

    // ---- one GPU per rank -------------------------------------------------
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, GPUs/node: %d, OpenMP threads: %d\n", nprocs, deviceCount, omp_get_max_threads());
    }

    // ---- block row decomposition -----------------------------------------
    const int nb = static_cast<int>((numNodes + BS - 1) / BS);
    const size_t nPad = static_cast<size_t>(nb) * BS;

    std::vector<int> rbBegin(nprocs + 1);
    for (int r = 0; r <= nprocs; ++r) {
        rbBegin[r] = static_cast<int>((static_cast<long long>(nb) * r) / nprocs);
    }
    std::vector<int> ownerOf(nb);
    for (int r = 0; r < nprocs; ++r) {
        for (int b = rbBegin[r]; b < rbBegin[r + 1]; ++b) {
            ownerOf[b] = r;
        }
    }

    const int rbStart = rbBegin[rank];
    const int rbEnd = rbBegin[rank + 1];
    const int localNb = rbEnd - rbStart;
    const size_t localRows = static_cast<size_t>(localNb) * BS;
    const size_t localElems = localRows * nPad;

    // ---- allocate & initialise -------------------------------------------
    unsigned int* d_dist = nullptr;
    unsigned int* d_path = nullptr;
    unsigned int* d_panel[NSLOT] = {nullptr, nullptr, nullptr};
    unsigned int* h_panel[NSLOT] = {nullptr, nullptr, nullptr};

    if (localElems > 0) {
        CUDA_CHECK(cudaMalloc(&d_dist, localElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&d_path, localElems * sizeof(unsigned int)));
    }
    const size_t panelElems = static_cast<size_t>(BS) * nPad;
    for (int i = 0; i < NSLOT; ++i) {
        CUDA_CHECK(cudaMalloc(&d_panel[i], panelElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaHostAlloc(&h_panel[i], panelElems * sizeof(unsigned int), cudaHostAllocDefault));
    }

    if (rank == 0) {
        printf("Initializing graph...\n");
    }

    if (localElems > 0) {
        std::vector<unsigned int> hostDist(localElems);
        std::vector<unsigned int> hostPath(localElems);
        initializeLocalMatrices(hostDist.data(), hostPath.data(), static_cast<size_t>(rbStart) * BS,
                                static_cast<size_t>(rbEnd) * BS, numNodes, nPad, 1, MAX_DISTANCE);
        CUDA_CHECK(cudaMemcpy(d_dist, hostDist.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_path, hostPath.data(), localElems * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

    // Two streams: `sPanel` carries the latency critical pivot panel chain,
    // `sComp` carries the bulk of the independent tile updates.  Both run
    // concurrently so that the panel for step k+1 is produced (and broadcast)
    // while the GPU is still busy with the independent tiles of step k.
    int prioLow = 0, prioHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    cudaStream_t sComp, sPanel;
    CUDA_CHECK(cudaStreamCreateWithPriority(&sComp, cudaStreamNonBlocking, prioLow));
    CUDA_CHECK(cudaStreamCreateWithPriority(&sPanel, cudaStreamNonBlocking, prioHigh));

    cudaEvent_t evPanel[NSLOT]; // pivot panel is resident in d_panel[slot]
    cudaEvent_t evComp[NSLOT];  // sComp is done reading d_panel[slot]
    cudaEvent_t evFree[NSLOT];  // h_panel[slot] is no longer read by the GPU
    cudaEvent_t evRow[2];       // look-ahead pivot row is up to date
    for (int i = 0; i < NSLOT; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evPanel[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evComp[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&evFree[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(evComp[i], sComp));
        CUDA_CHECK(cudaEventRecord(evFree[i], sComp));
    }
    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventCreateWithFlags(&evRow[i], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(evRow[i], sComp));
    }

    const dim3 threads(TX, TY);
    const size_t panelBytes = panelElems * sizeof(unsigned int);

    // Updates the local block rows [gbFirst, gbFirst+nRows) with respect to
    // pivot block row `kbCur` (phase 2b + phase 3).  Rows `skipA`/`skipB` (and
    // the pivot row itself) are left untouched.
    auto updateRows = [&](cudaStream_t s, const int gbFirst, const int nRows, const int kbCur, const int skipA,
                          const int skipB, const unsigned int* panel) {
        if (nRows <= 0) {
            return;
        }
        const size_t off = static_cast<size_t>(gbFirst - rbStart) * BS * nPad;
        fwPhase2Col<<<nRows, threads, 0, s>>>(d_dist + off, d_path + off, panel, nPad, kbCur, gbFirst, skipA, skipB);
        fwPhase3<<<dim3(nb, nRows), threads, 0, s>>>(d_dist + off, d_path + off, panel, nPad, kbCur, gbFirst, skipA,
                                                     skipB);
    };

    // Computes the pivot tile and pivot row panel of block row `kbCur`
    // (phase 1 + phase 2a) and stages it into d_panel/h_panel.
    auto buildPanel = [&](const int kbCur, const int slot) {
        const size_t off = static_cast<size_t>(kbCur - rbStart) * BS * nPad;
        const size_t pivOff = off + static_cast<size_t>(kbCur) * BS;
        fwPhase1<<<1, threads, 0, sPanel>>>(d_dist + pivOff, d_path + pivOff, nPad,
                                            static_cast<unsigned int>(kbCur) * BS);
        fwPhase2Row<<<nb, threads, 0, sPanel>>>(d_dist + off, d_path + off, nPad, kbCur);
        CUDA_CHECK(cudaMemcpyAsync(d_panel[slot], d_dist + off, panelBytes, cudaMemcpyDeviceToDevice, sPanel));
        if (nprocs > 1) {
            CUDA_CHECK(cudaMemcpyAsync(h_panel[slot], d_panel[slot], panelBytes, cudaMemcpyDeviceToHost, sPanel));
            CUDA_CHECK(cudaStreamSynchronize(sPanel));
        }
    };

    // Non-blocking binomial tree broadcast of a pivot panel.  MPI_Ibcast of the
    // available implementation is markedly slower than plain point-to-point
    // messages for multi-megabyte panels, so the tree is spelled out here: the
    // transfer is started, the GPU work of the current step is enqueued and the
    // tree is only drained afterwards, which hides it behind the computation.
    // The panel is split into chunks so that an inner node of the tree can
    // already forward the leading part of a panel while its tail is still
    // arriving (software pipelining of the tree).
    const int bcastChunks =
        std::max<int>(1, std::min<size_t>(8, panelBytes / (static_cast<size_t>(1) << 20)));
    std::vector<int> bcastChildren;
    std::vector<MPI_Request> bcastRecv(bcastChunks, MPI_REQUEST_NULL);
    std::vector<MPI_Request> bcastSend;
    unsigned int* bcastBuf = nullptr;
    bool bcastActive = false;

    const auto chunkBegin = [&](const int c) { return (panelElems * c) / bcastChunks; };

    auto bcastStart = [&](unsigned int* buf, const int root) {
        bcastBuf = buf;
        bcastActive = true;
        bcastChildren.clear();
        bcastSend.clear();

        const int vrank = (rank - root + nprocs) % nprocs;
        int src = -1;
        int mask = 1;
        while (mask < nprocs) {
            if (vrank & mask) {
                src = (vrank - mask + root) % nprocs;
                break;
            }
            mask <<= 1;
        }
        for (mask >>= 1; mask > 0; mask >>= 1) {
            if (vrank + mask < nprocs) {
                bcastChildren.push_back((vrank + mask + root) % nprocs);
            }
        }

        bcastSend.reserve(bcastChildren.size() * bcastChunks);
        if (src >= 0) {
            for (int c = 0; c < bcastChunks; ++c) {
                MPI_Irecv(buf + chunkBegin(c), static_cast<int>(chunkBegin(c + 1) - chunkBegin(c)), MPI_UNSIGNED, src,
                          c, MPI_COMM_WORLD, &bcastRecv[c]);
            }
        } else { // root: forward immediately
            for (int c = 0; c < bcastChunks; ++c) {
                for (const int child : bcastChildren) {
                    bcastSend.emplace_back();
                    MPI_Isend(buf + chunkBegin(c), static_cast<int>(chunkBegin(c + 1) - chunkBegin(c)), MPI_UNSIGNED,
                              child, c, MPI_COMM_WORLD, &bcastSend.back());
                }
            }
        }
    };

    auto bcastFinish = [&]() {
        if (!bcastActive) {
            return;
        }
        if (bcastRecv[0] != MPI_REQUEST_NULL) {
            for (int c = 0; c < bcastChunks; ++c) {
                MPI_Wait(&bcastRecv[c], MPI_STATUS_IGNORE);
                for (const int child : bcastChildren) {
                    bcastSend.emplace_back();
                    MPI_Isend(bcastBuf + chunkBegin(c), static_cast<int>(chunkBegin(c + 1) - chunkBegin(c)),
                              MPI_UNSIGNED, child, c, MPI_COMM_WORLD, &bcastSend.back());
                }
            }
        }
        MPI_Waitall(static_cast<int>(bcastSend.size()), bcastSend.data(), MPI_STATUSES_IGNORE);
        bcastActive = false;
    };

    if (rank == 0) {
        printf("Computing shortest paths...\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto start = std::chrono::high_resolution_clock::now();

    // Prologue: make pivot panel 0 available everywhere.
    {
        const int owner = ownerOf[0];
        if (rank == owner) {
            buildPanel(0, 0);
        }
        if (nprocs > 1) {
            MPI_Bcast(h_panel[0], static_cast<int>(panelElems), MPI_UNSIGNED, owner, MPI_COMM_WORLD);
            if (rank != owner) {
                CUDA_CHECK(cudaMemcpyAsync(d_panel[0], h_panel[0], panelBytes, cudaMemcpyHostToDevice, sPanel));
                CUDA_CHECK(cudaEventRecord(evFree[0], sPanel));
            }
        }
        CUDA_CHECK(cudaEventRecord(evPanel[0], sPanel));
    }

    for (int kb = 0; kb < nb; ++kb) {
        const int cur = kb % NSLOT;
        const int nxt = (kb + 1) % NSLOT;
        const int nextOwner = (kb + 1 < nb) ? ownerOf[kb + 1] : -1;
        const bool ownPivotNext = (nextOwner == rank);                         // pivot row of step kb+1
        const bool ownPivotAfter = (kb + 2 < nb) && (ownerOf[kb + 2] == rank); // look-ahead row
        if (kb + 1 < nb) {
            // Slot `nxt` was last consumed by sComp during step kb-2.
            CUDA_CHECK(cudaStreamWaitEvent(sPanel, evComp[nxt], 0));
            if (ownPivotNext) {
                // Latency critical chain: bring the next pivot row up to date,
                // factor it and hand it to MPI as early as possible.
                CUDA_CHECK(cudaStreamWaitEvent(sPanel, evRow[(kb + 1) & 1], 0));
                updateRows(sPanel, kb + 1, 1, kb, -1, -1, d_panel[cur]);
                buildPanel(kb + 1, nxt);
            } else if (nprocs > 1) {
                CUDA_CHECK(cudaEventSynchronize(evFree[nxt]));
            }
            if (nprocs > 1) {
                bcastStart(h_panel[nxt], nextOwner);
            }
        }

        // Bulk work of this step.  The row that becomes the pivot row two steps
        // from now is issued first so that the look-ahead chain can start early.
        CUDA_CHECK(cudaStreamWaitEvent(sComp, evPanel[cur], 0));
        const int nextPivotRow = ownPivotNext ? kb + 1 : -1;
        const int aheadRow = ownPivotAfter ? kb + 2 : -1;
        if (ownPivotAfter) {
            updateRows(sComp, kb + 2, 1, kb, -1, -1, d_panel[cur]);
            CUDA_CHECK(cudaEventRecord(evRow[kb & 1], sComp));
        }
        updateRows(sComp, rbStart, localNb, kb, nextPivotRow, aheadRow, d_panel[cur]);
        CUDA_CHECK(cudaEventRecord(evComp[cur], sComp));

        if (kb + 1 < nb) {
            if (nprocs > 1) {
                bcastFinish();
                if (!ownPivotNext) {
                    CUDA_CHECK(cudaMemcpyAsync(d_panel[nxt], h_panel[nxt], panelBytes, cudaMemcpyHostToDevice, sPanel));
                    CUDA_CHECK(cudaEventRecord(evFree[nxt], sPanel));
                }
            }
            CUDA_CHECK(cudaEventRecord(evPanel[nxt], sPanel));
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(sPanel));
    CUDA_CHECK(cudaStreamSynchronize(sComp));
    MPI_Barrier(MPI_COMM_WORLD);

    const auto end = std::chrono::high_resolution_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // ---- collect the distance matrix on rank 0 ----------------------------
    const size_t localRealBegin = std::min(static_cast<size_t>(rbStart) * BS, numNodes);
    const size_t localRealEnd = std::min(static_cast<size_t>(rbEnd) * BS, numNodes);
    const size_t localRealRows = localRealEnd - localRealBegin;

    std::vector<unsigned int> dist;
    std::vector<unsigned int> sendBuf;
    if (rank == 0) {
        dist.resize(numNodes * numNodes);
    }
    if (localRealRows > 0) {
        unsigned int* dst = (rank == 0) ? dist.data() + localRealBegin * numNodes : nullptr;
        if (rank != 0) {
            sendBuf.resize(localRealRows * numNodes);
            dst = sendBuf.data();
        }
        CUDA_CHECK(cudaMemcpy2D(dst, numNodes * sizeof(unsigned int), d_dist, nPad * sizeof(unsigned int),
                                numNodes * sizeof(unsigned int), localRealRows, cudaMemcpyDeviceToHost));
    }

    if (nprocs > 1) {
        MPI_Datatype rowType;
        MPI_Type_contiguous(static_cast<int>(numNodes), MPI_UNSIGNED, &rowType);
        MPI_Type_commit(&rowType);
        if (rank == 0) {
            std::vector<MPI_Request> reqs;
            for (int r = 1; r < nprocs; ++r) {
                const size_t rb = std::min(static_cast<size_t>(rbBegin[r]) * BS, numNodes);
                const size_t re = std::min(static_cast<size_t>(rbBegin[r + 1]) * BS, numNodes);
                if (re > rb) {
                    reqs.emplace_back();
                    MPI_Irecv(dist.data() + rb * numNodes, static_cast<int>(re - rb), rowType, r, 0, MPI_COMM_WORLD,
                              &reqs.back());
                }
            }
            MPI_Waitall(static_cast<int>(reqs.size()), reqs.data(), MPI_STATUSES_IGNORE);
        } else if (localRealRows > 0) {
            MPI_Send(sendBuf.data(), static_cast<int>(localRealRows), rowType, 0, 0, MPI_COMM_WORLD);
        }
        MPI_Type_free(&rowType);
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        const double ops = (double)numNodes * numNodes * numNodes;
        const double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);

        // Print results for external validation (integer hash-based)
        if (printResults) {
            print_results_int(dist, "DistanceMatrix");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            const bool valid = validateResult(dist, numNodes);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    for (int i = 0; i < 2; ++i) {
        CUDA_CHECK(cudaEventDestroy(evPanel[i]));
        CUDA_CHECK(cudaEventDestroy(evRow[i]));
        CUDA_CHECK(cudaEventDestroy(evFree[i]));
        CUDA_CHECK(cudaFree(d_panel[i]));
        CUDA_CHECK(cudaFreeHost(h_panel[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(sComp));
    CUDA_CHECK(cudaStreamDestroy(sPanel));
    if (d_dist) {
        CUDA_CHECK(cudaFree(d_dist));
    }
    if (d_path) {
        CUDA_CHECK(cudaFree(d_path));
    }

    MPI_Finalize();
    return exitCode;
}
