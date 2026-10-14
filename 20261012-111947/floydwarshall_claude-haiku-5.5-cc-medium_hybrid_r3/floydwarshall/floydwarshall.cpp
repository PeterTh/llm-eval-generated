#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Blocked Floyd-Warshall tile edge, and CUDA thread-block edge.
// Each thread owns a (B/TB) x (B/TB) register sub-tile of every tile.
constexpr int B = 64;
constexpr int TB = 16;
constexpr int RT = B / TB;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err_));                                    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
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

// In-place Floyd-Warshall over one B x B tile. The k loop runs in ascending order and
// each element is relaxed with strict '<', so it sees the same relaxations as the
// sequential algorithm. Row k and column k never change during step k (their diagonal
// entry is 0), so the operands can be read while other threads update the tile.
//   MODE 0: diagonal tile       T(i,j) = min(T(i,j), T(i,k) + T(k,j))
//   MODE 1: row panel (K,J)     T(i,j) = min(T(i,j), D(i,k) + T(k,j))   D = final diagonal tile
//   MODE 2: column panel (I,K)  T(i,j) = min(T(i,j), T(i,k) + D(k,j))   D = final diagonal tile
// One CTA per tile: tile base = tileOffset + blockIdx.x * ctaStride, skipCta is never run.
template <int MODE>
__global__ void fwPanelKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                              size_t ldd, const unsigned int* __restrict__ diag,
                              size_t tileOffset, size_t ctaStride, int skipCta,
                              unsigned int kBase) {
    if (static_cast<int>(blockIdx.x) == skipCta) return;

    const size_t off = tileOffset + blockIdx.x * ctaStride;
    unsigned int* T = dist + off;
    unsigned int* P = path + off;

    __shared__ unsigned int Ts[B][B + 1];  // Ts[j][i] = T(i,j)
    __shared__ unsigned int Ds[B][B + 1];  // Ds[j][i] = D(i,j), MODE 1 and 2 only

    for (int j = threadIdx.y; j < B; j += TB) {
        for (int i = threadIdx.x; i < B; i += TB) {
            Ts[j][i] = T[j * ldd + i];
            if (MODE != 0) Ds[j][i] = diag[j * B + i];
        }
    }

    for (int k = 0; k < B; ++k) {
        __syncthreads();
        for (int j = threadIdx.y; j < B; j += TB) {
            const unsigned int rkj = (MODE == 2) ? Ds[j][k] : Ts[j][k];
            for (int i = threadIdx.x; i < B; i += TB) {
                const unsigned int lik = (MODE == 1) ? Ds[k][i] : Ts[k][i];
                const unsigned int cand = lik + rkj;
                if (cand < Ts[j][i]) {
                    Ts[j][i] = cand;
                    P[j * ldd + i] = kBase + k;
                }
            }
        }
    }
    __syncthreads();

    for (int j = threadIdx.y; j < B; j += TB) {
        for (int i = threadIdx.x; i < B; i += TB) {
            T[j * ldd + i] = Ts[j][i];
        }
    }
}

// Min-plus update of a tile that is neither in block row K nor block column K:
// T(i,j) = min over k in block K of A(i,k) + Bt(k,j), k ascending with strict '<'.
// A is the column panel (I,K), Bt is the row panel (K,J); both are final for this K.
// Grid: x = local block column, y = device-local block row.
__global__ void fwUpdateKernel(unsigned int* __restrict__ dist, unsigned int* __restrict__ path,
                               size_t ldd, const unsigned int* __restrict__ colPanel, size_t ldc,
                               const unsigned int* __restrict__ rowPanel,
                               int K, int r, int c, int pr, int pc, int G, int t,
                               unsigned int kBase) {
    const int lc = blockIdx.x;
    const int q = blockIdx.y;
    const int li = q * G + t;                       // rank-local block row
    if (li * pr + r == K || lc * pc + c == K) return;

    __shared__ unsigned int As[B][B + 1];  // As[k][i] = A(i,k)
    __shared__ unsigned int Bs[B][B + 1];  // Bs[j][k] = Bt(k,j)

    for (int k = threadIdx.y; k < B; k += TB) {
        for (int i = threadIdx.x; i < B; i += TB) {
            As[k][i] = colPanel[k * ldc + li * B + i];
        }
    }
    for (int j = threadIdx.y; j < B; j += TB) {
        for (int k = threadIdx.x; k < B; k += TB) {
            Bs[j][k] = rowPanel[static_cast<size_t>(lc) * B * B + j * B + k];
        }
    }
    __syncthreads();

    const size_t tileOff = static_cast<size_t>(lc) * B * ldd + static_cast<size_t>(q) * B;
    unsigned int* T = dist + tileOff;
    unsigned int* P = path + tileOff;

    unsigned int acc[RT][RT];
    unsigned int pacc[RT][RT];
    for (int a = 0; a < RT; ++a) {
        for (int b = 0; b < RT; ++b) {
            const size_t e = (threadIdx.y + TB * b) * ldd + (threadIdx.x + TB * a);
            acc[a][b] = T[e];
            pacc[a][b] = P[e];
        }
    }

    for (int k = 0; k < B; ++k) {
        unsigned int aik[RT];
        unsigned int bkj[RT];
        for (int a = 0; a < RT; ++a) aik[a] = As[k][threadIdx.x + TB * a];
        for (int b = 0; b < RT; ++b) bkj[b] = Bs[threadIdx.y + TB * b][k];
        for (int a = 0; a < RT; ++a) {
            for (int b = 0; b < RT; ++b) {
                const unsigned int cand = aik[a] + bkj[b];
                if (cand < acc[a][b]) {
                    acc[a][b] = cand;
                    pacc[a][b] = kBase + k;
                }
            }
        }
    }

    for (int a = 0; a < RT; ++a) {
        for (int b = 0; b < RT; ++b) {
            const size_t e = (threadIdx.y + TB * b) * ldd + (threadIdx.x + TB * a);
            T[e] = acc[a][b];
            P[e] = pacc[a][b];
        }
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

// Distributed blocked Floyd-Warshall. On return, rank 0 holds the result in dist
// (original column-major layout). Returns the computation time in ms on rank 0.
//
// Decomposition:
//  * MPI: ranks form a pr x pc grid. Tile (I,J) of B x B lives on rank (I % pr, J % pc).
//  * OpenMP: each rank runs G threads, one per GPU it drives. A rank's local tile rows are
//    dealt cyclically over its GPUs (local block row li lives on GPU li % G).
//  * CUDA: each GPU runs the blocked phases on its own tiles.
// Per block iteration K:
//  1. diagonal tile is updated on its owner GPU, staged through host memory and broadcast;
//  2. row panel (K,*) is updated in process row K%pr and broadcast down each process column,
//     column panel (*,K) is updated in process column K%pc and broadcast along each process row;
//  3. all other tiles get a min-plus update from the two panels.
static long floydWarshallHybrid(std::vector<unsigned int>& dist, const size_t n,
                                const int rank, const int size) {
    omp_set_dynamic(0);

    // GPUs per rank: the node's devices are split evenly across its ranks.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int ranksOnNode = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &ranksOnNode);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    int G = std::max(1, numDevices / ranksOnNode);
    MPI_Allreduce(MPI_IN_PLACE, &G, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);

    // Process grid: pr is the largest divisor of size not exceeding sqrt(size).
    int pr = 1;
    for (int d = 1; d * d <= size; ++d) {
        if (size % d == 0) pr = d;
    }
    const int pc = size / pr;
    const int myR = rank / pc;
    const int myC = rank % pc;

    // Tile geometry. The block count is padded so every rank and GPU gets an equal share.
    // Padding nodes are isolated (INF to everything, 0 to themselves), so real distances
    // are unaffected.
    const int numRealBlocks = static_cast<int>((n + B - 1) / B);
    const int L = std::lcm(pr * G, pc);
    const int NB = (numRealBlocks + L - 1) / L * L;
    const int Lr = NB / pr;                     // rank-local block rows
    const int Lc = NB / pc;                     // rank-local block columns
    const int Ld = Lr / G;                      // device-local block rows
    const size_t LDd = static_cast<size_t>(Ld) * B;     // device leading dimension
    const size_t LDr = static_cast<size_t>(Lr) * B;     // rank leading dimension
    const size_t LcB = static_cast<size_t>(Lc) * B;
    const size_t devElems = LDd * LcB;
    const size_t rankElems = LDr * LcB;

    MPI_Comm colComm;   // same process column: ranks indexed by myR
    MPI_Comm rowComm;   // same process row: ranks indexed by myC
    MPI_Comm_split(MPI_COMM_WORLD, myC, myR, &colComm);
    MPI_Comm_split(MPI_COMM_WORLD, myR, myC, &rowComm);

    // Host staging buffers, shared by the threads of this rank.
    std::vector<unsigned int> hDiag(static_cast<size_t>(B) * B);
    std::vector<unsigned int> hRow(static_cast<size_t>(LcB) * B);
    std::vector<unsigned int> hCol(static_cast<size_t>(LDr) * B);
    std::vector<unsigned int> hLocal(rankElems);
    std::vector<unsigned int> gathered;
    if (rank == 0) gathered.resize(rankElems * size);

    std::chrono::steady_clock::time_point t0;
    std::chrono::steady_clock::time_point t1;

    #pragma omp parallel num_threads(G)
    {
        const int t = omp_get_thread_num();
        if (omp_get_num_threads() != G) {
            fprintf(stderr, "Expected %d OpenMP threads, got %d\n", G, omp_get_num_threads());
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const int dev = (localRank * G + t) % numDevices;
        CUDA_CHECK(cudaSetDevice(dev));

        unsigned int* dDist = nullptr;
        unsigned int* dPath = nullptr;
        unsigned int* dDiag = nullptr;
        unsigned int* dRow = nullptr;
        unsigned int* dCol = nullptr;
        CUDA_CHECK(cudaMalloc(&dDist, devElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dPath, devElems * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dDiag, static_cast<size_t>(B) * B * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dRow, LcB * B * sizeof(unsigned int)));
        CUDA_CHECK(cudaMalloc(&dCol, LDr * B * sizeof(unsigned int)));

        // Scatter the initial graph into this GPU's tiles. Path starts as j, the same
        // value the sequential initializePathMatrix produces.
        {
            std::vector<unsigned int> lDist(devElems);
            std::vector<unsigned int> lPath(devElems);
            for (size_t cj = 0; cj < LcB; ++cj) {
                const size_t j = (cj / B * pc + myC) * B + cj % B;
                for (size_t ri = 0; ri < LDd; ++ri) {
                    const size_t i = ((ri / B) * G + t) * pr * B + myR * B + ri % B;
                    lDist[cj * LDd + ri] = (i < n && j < n) ? dist[idx2(i, j, n)]
                                                            : (i == j ? 0u : INF);
                    lPath[cj * LDd + ri] = static_cast<unsigned int>(j);
                }
            }
            CUDA_CHECK(cudaMemcpy(dDist, lDist.data(), devElems * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dPath, lPath.data(), devElems * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));
        }

        #pragma omp barrier
        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t0 = std::chrono::steady_clock::now();
        }
        #pragma omp barrier

        const size_t bytesW = B * sizeof(unsigned int);

        for (int K = 0; K < NB; ++K) {
            const int rr = K % pr;                  // owner process row of block K
            const int cc = K % pc;                  // owner process column of block K
            const int kr = K / pr;                  // local block row of K on its owner
            const int kc = K / pc;                  // local block column of K on its owner
            const int tr = kr % G;                  // GPU holding local block row kr
            const int qr = kr / G;                  // device-local block row of kr
            const unsigned int kBase = static_cast<unsigned int>(K) * B;

            // 1. Diagonal tile on its owner GPU, broadcast to every rank.
            if (myR == rr && myC == cc && t == tr) {
                const size_t off = static_cast<size_t>(kc) * B * LDd + static_cast<size_t>(qr) * B;
                fwPanelKernel<0><<<1, dim3(TB, TB)>>>(dDist, dPath, LDd, nullptr, off, 0, -1, kBase);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy2D(hDiag.data(), bytesW, dDist + off, LDd * sizeof(unsigned int),
                                        bytesW, B, cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier
            #pragma omp master
            MPI_Bcast(hDiag.data(), B * B, MPI_UNSIGNED, rr * pc + cc, MPI_COMM_WORLD);
            #pragma omp barrier
            CUDA_CHECK(cudaMemcpy(dDiag, hDiag.data(), static_cast<size_t>(B) * B * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

            // 2a. Row panel (K,*) on process row rr, broadcast down each process column.
            if (myR == rr && t == tr) {
                fwPanelKernel<1><<<Lc, dim3(TB, TB)>>>(dDist, dPath, LDd, dDiag,
                                                       static_cast<size_t>(qr) * B,
                                                       static_cast<size_t>(B) * LDd,
                                                       myC == cc ? kc : -1, kBase);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy2D(hRow.data(), bytesW, dDist + static_cast<size_t>(qr) * B,
                                        LDd * sizeof(unsigned int), bytesW, LcB,
                                        cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier
            #pragma omp master
            MPI_Bcast(hRow.data(), static_cast<int>(LcB * B), MPI_UNSIGNED, rr, colComm);
            #pragma omp barrier
            CUDA_CHECK(cudaMemcpy(dRow, hRow.data(), LcB * B * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

            // 2b. Column panel (*,K) on process column cc, broadcast along each process row.
            if (myC == cc) {
                const size_t off = static_cast<size_t>(kc) * B * LDd;
                const int skip = (myR == rr && t == tr) ? qr : -1;
                fwPanelKernel<2><<<Ld, dim3(TB, TB)>>>(dDist, dPath, LDd, dDiag, off, B, skip, kBase);
                CUDA_CHECK(cudaGetLastError());
                for (int q = 0; q < Ld; ++q) {
                    const int li = q * G + t;
                    CUDA_CHECK(cudaMemcpy2D(hCol.data() + static_cast<size_t>(li) * B,
                                            LDr * sizeof(unsigned int),
                                            dDist + off + static_cast<size_t>(q) * B,
                                            LDd * sizeof(unsigned int), bytesW, B,
                                            cudaMemcpyDeviceToHost));
                }
            }
            #pragma omp barrier
            #pragma omp master
            MPI_Bcast(hCol.data(), static_cast<int>(LDr * B), MPI_UNSIGNED, cc, rowComm);
            #pragma omp barrier
            CUDA_CHECK(cudaMemcpy(dCol, hCol.data(), LDr * B * sizeof(unsigned int),
                                  cudaMemcpyHostToDevice));

            // 3. Every remaining tile on this GPU.
            fwUpdateKernel<<<dim3(Lc, Ld), dim3(TB, TB)>>>(dDist, dPath, LDd, dCol, LDr, dRow,
                                                           K, myR, myC, pr, pc, G, t, kBase);
            CUDA_CHECK(cudaGetLastError());
        }

        CUDA_CHECK(cudaDeviceSynchronize());
        #pragma omp barrier
        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            t1 = std::chrono::steady_clock::now();
        }
        #pragma omp barrier

        // Gather this GPU's tiles into the rank-local buffer (rank-local block row = q*G + t).
        for (int q = 0; q < Ld; ++q) {
            CUDA_CHECK(cudaMemcpy2D(hLocal.data() + static_cast<size_t>(q * G + t) * B,
                                    LDr * sizeof(unsigned int),
                                    dDist + static_cast<size_t>(q) * B,
                                    LDd * sizeof(unsigned int), bytesW, LcB,
                                    cudaMemcpyDeviceToHost));
        }
        #pragma omp barrier
        #pragma omp master
        MPI_Gather(hLocal.data(), static_cast<int>(rankElems), MPI_UNSIGNED,
                   gathered.data(), static_cast<int>(rankElems), MPI_UNSIGNED, 0, MPI_COMM_WORLD);

        CUDA_CHECK(cudaFree(dDist));
        CUDA_CHECK(cudaFree(dPath));
        CUDA_CHECK(cudaFree(dDiag));
        CUDA_CHECK(cudaFree(dRow));
        CUDA_CHECK(cudaFree(dCol));
    }

    if (rank == 0) {
        for (int s = 0; s < size; ++s) {
            const int sR = s / pc;
            const int sC = s % pc;
            const unsigned int* src = gathered.data() + static_cast<size_t>(s) * rankElems;
            #pragma omp parallel for schedule(static)
            for (size_t cj = 0; cj < LcB; ++cj) {
                const size_t j = (cj / B * pc + sC) * B + cj % B;
                if (j >= n) continue;
                for (size_t ri = 0; ri < LDr; ++ri) {
                    const size_t i = (ri / B * pr + sR) * B + ri % B;
                    if (i < n) dist[idx2(i, j, n)] = src[cj * LDr + ri];
                }
            }
        }
    }

    MPI_Comm_free(&colComm);
    MPI_Comm_free(&rowComm);

    if (rank != 0) return 0;
    return std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI_THREAD_FUNNELED is required\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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

    if (rank == 0) {
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Initializing graph...\n");
    }

    // Every rank generates the same initial graph, so each can scatter its own tiles.
    std::vector<unsigned int> dist(numNodes * numNodes);
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE);

    if (rank == 0) printf("Computing shortest paths...\n");
    long durationMs = 0;
    if (numNodes > 0) {
        durationMs = floydWarshallHybrid(dist, numNodes, rank, size);
    }

    int exitCode = 0;
    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
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
                exitCode = 1;
            }
        }
    }

    MPI_Finalize();
    return exitCode;
}
