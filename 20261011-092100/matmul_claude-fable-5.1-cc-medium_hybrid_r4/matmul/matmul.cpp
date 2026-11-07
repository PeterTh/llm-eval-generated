// Hybrid MPI + OpenMP + CUDA dense matrix multiplication benchmark.
//
// Decomposition
//   * MPI   : C's rows are block-distributed over the ranks. Every rank
//             generates its own slab of A and the complete B locally from the
//             deterministic init formula, so no input data is ever
//             communicated. Rank 0 gathers C only when it is needed for
//             output or validation.
//   * CUDA  : each rank binds to one GPU (local rank modulo device count). B
//             is resident on the device; A row chunks are streamed in through
//             pinned host memory on two streams (double buffered) and the
//             resulting C rows are streamed back.
//   * OpenMP: one thread drives the GPU pipeline while all other threads run a
//             packed, register-blocked AVX DGEMM on the host. The GPU claims
//             row chunks from the top of the slab and CPU threads claim 2D
//             tiles from the bottom, through one shared atomic, so the work
//             split balances itself dynamically for whatever CPU/GPU ratio a
//             node has. Matrix initialisation is also OpenMP parallel.
//
// Numerics: every C element is computed as one sequential chain of
// round(sum + round(a*b)) over k on both the CPU and the GPU (no FMA
// contraction), which is exactly what the original scalar loop produces.
// Results are therefore bit-identical to the original regardless of which
// device computed which row.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h>

#include "../common/results_output.hpp"

// ----------------------------------------------------------------------------
// Error handling helpers
// ----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        const cudaError_t err__ = (call);                                                \
        if (err__ != cudaSuccess) {                                                      \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,   \
                    cudaGetErrorString(err__));                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

#define MPI_CHECK(call)                                                                  \
    do {                                                                                 \
        const int err__ = (call);                                                        \
        if (err__ != MPI_SUCCESS) {                                                      \
            fprintf(stderr, "MPI error %d in %s at %s:%d\n", err__, #call, __FILE__,     \
                    __LINE__);                                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

// Generate pseudo-random values for matrix initialization
constexpr double getPseudoRndValue(const size_t N, const size_t i, const size_t j) noexcept {
    return (((i + 1) * (i + j + 1) * 1299709) % (N * N)) / static_cast<double>(N * N);
}

// Initialise rows [rowBegin, rowBegin + numRows) of an N x N matrix into a
// slab buffer (slab row r corresponds to global row rowBegin + r).
void initMatrixRows(double* slab, const size_t N, const size_t rowBegin, const size_t numRows) {
#pragma omp parallel for schedule(static)
    for (size_t r = 0; r < numRows; ++r) {
        const size_t i = rowBegin + r;
        double* row = slab + r * N;
        for (size_t j = 0; j < N; ++j) {
            row[j] = getPseudoRndValue(N, i, j);
        }
    }
}

// ----------------------------------------------------------------------------
// CUDA kernel: C[M x N] = A[M x K] * B[K x N], row major, leading dimension N
// for B and C, K for A (here K == N). 64x64 output tile per block, 16-wide k
// slices, 16x16 threads each owning a 4x4 register tile. Accumulation per
// element is a strictly sequential, non-fused multiply/add chain over k.
// ----------------------------------------------------------------------------
namespace {
constexpr int TILE_M = 64;
constexpr int TILE_N = 64;
constexpr int TILE_K = 16;
constexpr int THR_X = 16;
constexpr int THR_Y = 16;
constexpr int MICRO_M = TILE_M / THR_Y;  // 4
constexpr int MICRO_N = TILE_N / THR_X;  // 4
}  // namespace

__global__ void __launch_bounds__(THR_X* THR_Y)
    dgemmKernel(const double* __restrict__ A, const double* __restrict__ B, double* __restrict__ C,
                const int M, const int N, const int K) {
    __shared__ double As[TILE_K][TILE_M + 1];
    __shared__ double Bs[TILE_K][TILE_N + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * THR_X + tx;
    const int rowBase = blockIdx.y * TILE_M;
    const int colBase = blockIdx.x * TILE_N;

    double acc[MICRO_M][MICRO_N];
#pragma unroll
    for (int i = 0; i < MICRO_M; ++i)
#pragma unroll
        for (int j = 0; j < MICRO_N; ++j) acc[i][j] = 0.0;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        // Cooperative load of the A tile (TILE_M x TILE_K): 256 threads, 1024 elements.
#pragma unroll
        for (int e = tid; e < TILE_M * TILE_K; e += THR_X * THR_Y) {
            const int r = e / TILE_K;
            const int kk = e % TILE_K;
            const int gr = rowBase + r;
            const int gk = k0 + kk;
            As[kk][r] = (gr < M && gk < K) ? A[static_cast<size_t>(gr) * K + gk] : 0.0;
        }
        // Cooperative load of the B tile (TILE_K x TILE_N), coalesced along N.
#pragma unroll
        for (int e = tid; e < TILE_K * TILE_N; e += THR_X * THR_Y) {
            const int kk = e / TILE_N;
            const int c = e % TILE_N;
            const int gk = k0 + kk;
            const int gc = colBase + c;
            Bs[kk][c] = (gk < K && gc < N) ? B[static_cast<size_t>(gk) * N + gc] : 0.0;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < TILE_K; ++kk) {
            double a[MICRO_M];
            double b[MICRO_N];
#pragma unroll
            for (int i = 0; i < MICRO_M; ++i) a[i] = As[kk][ty + i * THR_Y];
#pragma unroll
            for (int j = 0; j < MICRO_N; ++j) b[j] = Bs[kk][tx + j * THR_X];
#pragma unroll
            for (int i = 0; i < MICRO_M; ++i)
#pragma unroll
                for (int j = 0; j < MICRO_N; ++j)
                    acc[i][j] = __dadd_rn(acc[i][j], __dmul_rn(a[i], b[j]));
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < MICRO_M; ++i) {
        const int gr = rowBase + ty + i * THR_Y;
        if (gr >= M) continue;
#pragma unroll
        for (int j = 0; j < MICRO_N; ++j) {
            const int gc = colBase + tx + j * THR_X;
            if (gc < N) C[static_cast<size_t>(gr) * N + gc] = acc[i][j];
        }
    }
}

// ----------------------------------------------------------------------------
// Host DGEMM (one OpenMP thread per call), Goto/BLIS style:
//   * A rows and B panels are packed into contiguous, zero-padded micro panels
//     so the inner kernel streams both operands sequentially (no cache set
//     aliasing for power-of-two N) and needs no edge special cases.
//   * 6 x 8 register tile (12 AVX accumulators), k-blocked so the B micro
//     panel lives in L1 and the packed A block in L2.
// Per element the accumulation is a sequential non-fused multiply/add chain
// over k (same as the GPU and the original); the host is compiled with
// -ffp-contract=off so "c + a*b" stays two roundings.
// ----------------------------------------------------------------------------
namespace {
typedef double v4d __attribute__((vector_size(32)));

constexpr size_t CPU_MR = 6;    // rows per micro tile
constexpr size_t CPU_NR = 8;    // columns per micro tile (2 x v4d)
constexpr size_t CPU_KC = 512;  // k block: B micro panel = 32 KiB, bounds C traffic

inline v4d loadu4(const double* p) {
    v4d v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
inline void storeu4(double* p, const v4d v) { std::memcpy(p, &v, sizeof(v)); }
inline v4d bcast4(const double x) { return v4d{x, x, x, x}; }

constexpr size_t CPU_MC = 72;   // rows per packed A block = 288 KiB (L2)

// Per-thread packing workspace.
struct CpuWorkspace {
    std::vector<double> Ap;  // packed A block: CPU_MC/MR micro panels of MR x kc
    std::vector<double> Bp;  // packed B panel: ceil(nc/NR) micro panels of kc x NR
    void reserve(const size_t maxCols) {
        Ap.resize(CPU_MC * CPU_KC);
        Bp.resize((maxCols + CPU_NR - 1) / CPU_NR * CPU_NR * CPU_KC);
    }
};

// Pack A[0:rows, 0:kc] (leading dimension lda) into MR-row micro panels,
// k-interleaved: Ap[panel][k][r]. Rows past `rows` are zero.
inline void packA(double* __restrict__ Ap, const double* __restrict__ A, const size_t lda,
                  const size_t rows, const size_t kc) {
    for (size_t i0 = 0; i0 < rows; i0 += CPU_MR) {
        const size_t mr = std::min(CPU_MR, rows - i0);
        for (size_t k = 0; k < kc; ++k) {
            size_t r = 0;
            for (; r < mr; ++r) Ap[k * CPU_MR + r] = A[(i0 + r) * lda + k];
            for (; r < CPU_MR; ++r) Ap[k * CPU_MR + r] = 0.0;
        }
        Ap += CPU_MR * kc;
    }
}

// Pack B[0:kc, 0:nc] (leading dimension ldb) into NR-column micro panels:
// Bp[panel][k][c]. Columns past `nc` are zero.
inline void packB(double* __restrict__ Bp, const double* __restrict__ B, const size_t ldb,
                  const size_t kc, const size_t nc) {
    for (size_t j0 = 0; j0 < nc; j0 += CPU_NR) {
        const size_t nr = std::min(CPU_NR, nc - j0);
        for (size_t k = 0; k < kc; ++k) {
            const double* src = B + k * ldb + j0;
            double* dst = Bp + k * CPU_NR;
            size_t c = 0;
            for (; c < nr; ++c) dst[c] = src[c];
            for (; c < CPU_NR; ++c) dst[c] = 0.0;
        }
        Bp += CPU_NR * kc;
    }
}

// C[6x8] (+)= Ap[kc x 6] * Bp[kc x 8]. `first` means C holds no partial sums
// yet (k block 0), otherwise the existing values are continued.
// Kept out of line on purpose: inlined into the blocked loop nest the compiler
// runs out of vector registers and re-reads B operands from memory, which
// costs ~30% of the kernel's throughput.
__attribute__((noinline)) void microKernel6x8(const double* __restrict__ Ap, const double* __restrict__ Bp,
                           const size_t kc, double* __restrict__ C, const size_t ldc,
                           const bool first) {
    v4d c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
    if (first) {
        c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = bcast4(0.0);
    } else {
        c00 = loadu4(C + 0 * ldc), c01 = loadu4(C + 0 * ldc + 4);
        c10 = loadu4(C + 1 * ldc), c11 = loadu4(C + 1 * ldc + 4);
        c20 = loadu4(C + 2 * ldc), c21 = loadu4(C + 2 * ldc + 4);
        c30 = loadu4(C + 3 * ldc), c31 = loadu4(C + 3 * ldc + 4);
        c40 = loadu4(C + 4 * ldc), c41 = loadu4(C + 4 * ldc + 4);
        c50 = loadu4(C + 5 * ldc), c51 = loadu4(C + 5 * ldc + 4);
    }
    for (size_t k = 0; k < kc; ++k) {
        const v4d b0 = loadu4(Bp);
        const v4d b1 = loadu4(Bp + 4);
        v4d a;
        a = bcast4(Ap[0]);
        c00 = c00 + a * b0;
        c01 = c01 + a * b1;
        a = bcast4(Ap[1]);
        c10 = c10 + a * b0;
        c11 = c11 + a * b1;
        a = bcast4(Ap[2]);
        c20 = c20 + a * b0;
        c21 = c21 + a * b1;
        a = bcast4(Ap[3]);
        c30 = c30 + a * b0;
        c31 = c31 + a * b1;
        a = bcast4(Ap[4]);
        c40 = c40 + a * b0;
        c41 = c41 + a * b1;
        a = bcast4(Ap[5]);
        c50 = c50 + a * b0;
        c51 = c51 + a * b1;
        Ap += CPU_MR;
        Bp += CPU_NR;
    }
    storeu4(C + 0 * ldc, c00), storeu4(C + 0 * ldc + 4, c01);
    storeu4(C + 1 * ldc, c10), storeu4(C + 1 * ldc + 4, c11);
    storeu4(C + 2 * ldc, c20), storeu4(C + 2 * ldc + 4, c21);
    storeu4(C + 3 * ldc, c30), storeu4(C + 3 * ldc + 4, c31);
    storeu4(C + 4 * ldc, c40), storeu4(C + 4 * ldc + 4, c41);
    storeu4(C + 5 * ldc, c50), storeu4(C + 5 * ldc + 4, c51);
}

// Edge tile: compute the full (zero padded) 6x8 tile into a scratch buffer and
// copy back only the valid mValid x nValid part.
inline void microKernelEdge(const double* __restrict__ Ap, const double* __restrict__ Bp,
                            const size_t kc, double* __restrict__ C, const size_t ldc,
                            const bool first, const size_t mValid, const size_t nValid) {
    double tmp[CPU_MR * CPU_NR];
    if (!first) {
        for (size_t i = 0; i < mValid; ++i)
            for (size_t j = 0; j < nValid; ++j) tmp[i * CPU_NR + j] = C[i * ldc + j];
    }
    // Padded rows/cols of tmp are never read back, their content is irrelevant.
    for (size_t i = 0; i < CPU_MR; ++i)
        for (size_t j = 0; j < CPU_NR; ++j)
            if (first || i >= mValid || j >= nValid) tmp[i * CPU_NR + j] = 0.0;
    microKernel6x8(Ap, Bp, kc, tmp, CPU_NR, false);
    for (size_t i = 0; i < mValid; ++i)
        for (size_t j = 0; j < nValid; ++j) C[i * ldc + j] = tmp[i * CPU_NR + j];
}
}  // namespace

// Compute the C tile rows [r0, r1) x cols [j0, j1) of C = A_slab * B on the
// host (single thread). BLIS-style loop nest: for each k block the B panel
// (kc x cols) is packed once and reused by all row blocks; each packed A block
// (CPU_MC x kc) lives in L2 and is swept once per B micro panel (L1).
void cpuMultiplyTile(const double* __restrict__ A, const double* __restrict__ B,
                     double* __restrict__ C, const size_t N, const size_t r0, const size_t r1,
                     const size_t j0, const size_t j1, CpuWorkspace& ws) {
    const size_t rows = r1 - r0;
    const size_t cols = j1 - j0;

    for (size_t k0 = 0; k0 < N; k0 += CPU_KC) {
        const size_t kc = std::min(CPU_KC, N - k0);
        const bool first = (k0 == 0);
        packB(ws.Bp.data(), B + k0 * N + j0, N, kc, cols);
        for (size_t i0 = 0; i0 < rows; i0 += CPU_MC) {
            const size_t mc = std::min(CPU_MC, rows - i0);
            packA(ws.Ap.data(), A + (r0 + i0) * N + k0, N, mc, kc);
            for (size_t jt = 0; jt < cols; jt += CPU_NR) {
                const size_t nValid = std::min(CPU_NR, cols - jt);
                const double* Bp = ws.Bp.data() + (jt / CPU_NR) * CPU_NR * kc;
                for (size_t it = 0; it < mc; it += CPU_MR) {
                    const size_t mValid = std::min(CPU_MR, mc - it);
                    const double* Ap = ws.Ap.data() + (it / CPU_MR) * CPU_MR * kc;
                    double* Ct = C + (r0 + i0 + it) * N + j0 + jt;
                    if (mValid == CPU_MR && nValid == CPU_NR) {
                        microKernel6x8(Ap, Bp, kc, Ct, N, first);
                    } else {
                        microKernelEdge(Ap, Bp, kc, Ct, N, first, mValid, nValid);
                    }
                }
            }
        }
    }
}

// ----------------------------------------------------------------------------
// Hybrid GPU + CPU multiplication of one rank's row slab.
// ----------------------------------------------------------------------------
struct GpuContext {
    int device = -1;
    double* dB = nullptr;
    double* dA[2] = {nullptr, nullptr};
    double* dC[2] = {nullptr, nullptr};
    cudaStream_t stream[2] = {nullptr, nullptr};
    size_t maxChunkRows = 0;  // capacity of each dA/dC buffer in rows
};

// Two-ended row claim shared by the GPU and the CPU workers, packed into one
// 64-bit atomic: rows [0, top) are claimed by the GPU, rows [bottom, slabRows)
// by the CPU. The GPU grabs row chunks from the top, CPU threads grab whole
// row blocks from the bottom (and split them into column tiles between
// themselves); work ends when the two frontiers meet. No static split ratio
// is needed, so the balance adapts to whatever CPU/GPU ratio a node has.
namespace {
inline uint64_t packState(const uint32_t top, const uint32_t bottom) {
    return (static_cast<uint64_t>(top) << 32) | bottom;
}
inline uint32_t stateTop(const uint64_t s) { return static_cast<uint32_t>(s >> 32); }
inline uint32_t stateBottom(const uint64_t s) { return static_cast<uint32_t>(s & 0xffffffffu); }
}  // namespace

void matrixMultiplySlab(const double* A, const double* B, double* C, const size_t N,
                        const size_t slabRows, GpuContext& gpu) {
    if (slabRows == 0) return;

    std::atomic<uint64_t> state{packState(0, static_cast<uint32_t>(slabRows))};
    const int gridX = static_cast<int>((N + TILE_N - 1) / TILE_N);

    // GPU chunking: chunks are sized from the measured throughput to hold
    // about targetSec of work, bounding the end-of-slab imbalance.
    constexpr double targetSec = 0.08;
    const size_t gpuMinRows = std::min<size_t>(slabRows, 2 * TILE_M);

    // CPU tiling: row blocks of cpuBlockRows (claimed from the bottom), each
    // split into tileCols-wide column tiles that are handed out one per
    // thread. Tall blocks and wide tiles keep the arithmetic intensity of a
    // tile high (A rows + B columns + the per-k-block C update are all
    // streamed from L3/DRAM), which matters more than fine load balance.
    const size_t cpuBlockRows = (slabRows >= 2048) ? 384 : 192;  // multiples of CPU_MR (6)
    const size_t nBlocks = (slabRows + cpuBlockRows - 1) / cpuBlockRows;
    const size_t tileCols = std::min<size_t>(N, 512);
    const size_t nColTiles = (N + tileCols - 1) / tileCols;
    std::atomic<size_t> nextUnit{0};

#pragma omp parallel
    {
        const int tid = omp_get_thread_num();

        if (tid == 0) {
            // ---------------- GPU driver thread ----------------
            CUDA_CHECK(cudaSetDevice(gpu.device));
            int buf = 0;
            bool inFlight[2] = {false, false};
            size_t rowsInFlight[2] = {0, 0};
            size_t rowsDone = 0;
            size_t chunk = gpuMinRows;
            const auto t0 = std::chrono::steady_clock::now();
            while (true) {
                // Claim [r0, r0 + rows) from the top.
                size_t r0 = 0, rows = 0;
                {
                    uint64_t s = state.load(std::memory_order_relaxed);
                    while (true) {
                        const size_t top = stateTop(s), bottom = stateBottom(s);
                        const size_t newTop = std::min(top + chunk, bottom);
                        if (newTop == top) break;
                        if (state.compare_exchange_weak(s, packState(static_cast<uint32_t>(newTop),
                                                                     static_cast<uint32_t>(bottom)),
                                                        std::memory_order_acq_rel)) {
                            r0 = top;
                            rows = newTop - top;
                            break;
                        }
                    }
                }
                if (rows == 0) break;

                // Make sure the buffer we are about to reuse is free.
                if (inFlight[buf]) {
                    CUDA_CHECK(cudaStreamSynchronize(gpu.stream[buf]));
                    inFlight[buf] = false;
                    rowsDone += rowsInFlight[buf];
                }
                cudaStream_t s = gpu.stream[buf];
                CUDA_CHECK(cudaMemcpyAsync(gpu.dA[buf], A + r0 * N, rows * N * sizeof(double),
                                           cudaMemcpyHostToDevice, s));
                const dim3 block(THR_X, THR_Y);
                const dim3 grid(gridX, static_cast<unsigned>((rows + TILE_M - 1) / TILE_M));
                dgemmKernel<<<grid, block, 0, s>>>(gpu.dA[buf], gpu.dB, gpu.dC[buf],
                                                   static_cast<int>(rows), static_cast<int>(N),
                                                   static_cast<int>(N));
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpyAsync(C + r0 * N, gpu.dC[buf], rows * N * sizeof(double),
                                           cudaMemcpyDeviceToHost, s));
                inFlight[buf] = true;
                rowsInFlight[buf] = rows;
                buf ^= 1;

                // Re-estimate the chunk size from the measured throughput.
                if (rowsDone > 0) {
                    const double elapsed =
                        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                    const double rowsPerSec = static_cast<double>(rowsDone) / std::max(elapsed, 1e-6);
                    const double want = rowsPerSec * targetSec;
                    chunk = static_cast<size_t>(
                        std::min<double>(want, static_cast<double>(gpu.maxChunkRows)));
                    chunk = std::max(chunk, gpuMinRows);
                    chunk = (chunk + TILE_M - 1) / TILE_M * TILE_M;
                    chunk = std::min(chunk, gpu.maxChunkRows);
                }
            }
            for (int b = 0; b < 2; ++b) {
                if (inFlight[b]) CUDA_CHECK(cudaStreamSynchronize(gpu.stream[b]));
            }
        } else {
            // ---------------- CPU worker threads ----------------
            CpuWorkspace ws;
            ws.reserve(tileCols);
            while (true) {
                const size_t unit = nextUnit.fetch_add(1, std::memory_order_relaxed);
                const size_t blk = unit / nColTiles;
                if (blk >= nBlocks) break;
                const size_t blockEnd = slabRows - blk * cpuBlockRows;
                const size_t blockStart = blockEnd - std::min(cpuBlockRows, blockEnd);

                // Reserve the block against the GPU frontier: bottom becomes
                // max(blockStart, top). Idempotent, so every thread working on
                // this block observes the same effective start row.
                size_t effStart = 0;
                bool nonEmpty = true;
                {
                    uint64_t s = state.load(std::memory_order_acquire);
                    while (true) {
                        const size_t top = stateTop(s), bottom = stateBottom(s);
                        if (bottom <= blockStart) {
                            effStart = blockStart;
                            break;
                        }
                        const size_t newBottom = std::max(blockStart, top);
                        if (newBottom == bottom) {
                            effStart = bottom;
                            nonEmpty = bottom < blockEnd;
                            break;
                        }
                        if (state.compare_exchange_weak(
                                s, packState(static_cast<uint32_t>(top), static_cast<uint32_t>(newBottom)),
                                std::memory_order_acq_rel)) {
                            effStart = newBottom;
                            nonEmpty = newBottom < blockEnd;
                            break;
                        }
                    }
                }
                // The GPU has already passed this block, and therefore every
                // block above it as well: nothing left for the CPU.
                if (!nonEmpty) break;

                const size_t j0 = (unit % nColTiles) * tileCols;
                const size_t j1 = std::min(N, j0 + tileCols);
                cpuMultiplyTile(A, B, C, N, effStart, blockEnd, j0, j1, ws);
            }
        }
    }
}

// Simple validation: compute a few elements on rank 0 and compare.
// A's values are regenerated from the deterministic init formula (identical to
// the values the slabs were initialised with), so the full A is not needed.
bool validateResult(const std::vector<double>& B, const std::vector<double>& C, const size_t N) {
    // Check a few random positions
    constexpr size_t checkPoints[] = {0, 1, 2, 3, 4};

    for (size_t pi = 0; pi < 5; ++pi) {
        for (size_t pj = 0; pj < 5; ++pj) {
            const size_t i = checkPoints[pi] % N;
            const size_t j = checkPoints[pj] % N;

            double expected = 0.0;
            for (size_t k = 0; k < N; ++k) {
                expected += getPseudoRndValue(N, i, k) * B[k * N + j];
            }

            const double actual = C[i * N + j];
            const double relError = std::abs((actual - expected) / (expected + 1e-10));

            if (relError > 1e-6) {
                printf("Validation failed at (%zu, %zu): expected %.10f, got %.10f (error: %.10e)\n",
                       i, j, expected, actual, relError);
                return false;
            }
        }
    }

    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size N (computes NxN * NxN) (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_CHECK(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided));
    int rank = 0, nRanks = 1;
    MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    size_t N = 512;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            N = atoi(argv[++i]);
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

    // ---- GPU selection: one device per rank, round robin over local ranks ----
    MPI_Comm nodeComm;
    MPI_CHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                  &nodeComm));
    int localRank = 0, localSize = 1;
    MPI_CHECK(MPI_Comm_rank(nodeComm, &localRank));
    MPI_CHECK(MPI_Comm_size(nodeComm, &localSize));
    MPI_CHECK(MPI_Comm_free(&nodeComm));

    // Default thread count when OMP_NUM_THREADS is not set: if the launcher
    // already confined this rank to a subset of the node's CPUs, use that
    // subset; otherwise share the node's CPUs evenly among the local ranks.
    if (getenv("OMP_NUM_THREADS") == nullptr && localSize > 1) {
        const int inMask = omp_get_num_procs();
        const long online = sysconf(_SC_NPROCESSORS_ONLN);
        const int places = omp_get_num_places();  // e.g. cores when OMP_PLACES=cores
        int threads = (places > 0) ? places : inMask;
        if (online > 0 && inMask >= online) threads = std::max(1, threads / localSize);
        omp_set_num_threads(threads);
    }

    int deviceCount = 0;
    const cudaError_t devErr = cudaGetDeviceCount(&deviceCount);
    if (devErr != cudaSuccess || deviceCount == 0) {
        fprintf(stderr, "Rank %d: no CUDA device available (%s)\n", rank,
                cudaGetErrorString(devErr));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    GpuContext gpu;
    gpu.device = localRank % deviceCount;
    CUDA_CHECK(cudaSetDevice(gpu.device));
    CUDA_CHECK(cudaFree(0));  // establish the context now, outside the timed region

    if (rank == 0) {
        printf("Matrix Multiplication Benchmark\n");
        printf("Matrix size: %zu x %zu\n", N, N);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d, GPUs/node: %d\n", nRanks,
               omp_get_max_threads(), deviceCount);
    }

    // ---- Row distribution ----
    const size_t base = N / nRanks;
    const size_t rem = N % nRanks;
    auto rowsOf = [&](int r) -> size_t { return base + (static_cast<size_t>(r) < rem ? 1 : 0); };
    auto firstRowOf = [&](int r) -> size_t {
        return static_cast<size_t>(r) * base + std::min<size_t>(static_cast<size_t>(r), rem);
    };
    const size_t myRows = rowsOf(rank);
    const size_t myFirstRow = firstRowOf(rank);

    // Allocate matrices: A and C slabs (pinned for async GPU transfers), full B.
    std::vector<double> B(N * N);
    double* Aslab = nullptr;
    double* Cslab = nullptr;
    const size_t slabBytes = std::max<size_t>(1, myRows * N) * sizeof(double);
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&Aslab), slabBytes, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&Cslab), slabBytes, cudaHostAllocDefault));

    // Initialize matrices
    if (rank == 0) printf("Initializing matrices...\n");
    initMatrixRows(Aslab, N, myFirstRow, myRows);
    initMatrixRows(B.data(), N, 0, N);

    // ---- Device buffers: full B, double-buffered A/C chunks ----
    {
        size_t freeMem = 0, totalMem = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        const size_t bBytes = N * N * sizeof(double);
        if (bBytes + 4 * N * sizeof(double) > freeMem) {
            fprintf(stderr, "Rank %d: matrix B (%zu MiB) does not fit on GPU %d\n", rank,
                    bBytes >> 20, gpu.device);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        // Chunk buffer capacity: up to ~128 MiB per A/C chunk buffer, but at
        // most the slab, at least one GPU tile, and never more than fits in
        // the remaining device memory (4 buffers: 2 x A, 2 x C).
        const size_t rowBytes = std::max<size_t>(1, N) * sizeof(double);
        size_t chunk = std::max<size_t>(TILE_M, (128u << 20) / rowBytes);
        chunk = std::min(chunk, std::max<size_t>(1, myRows));
        const size_t avail = (freeMem - bBytes) / 4;
        chunk = std::min(chunk, std::max<size_t>(1, avail / rowBytes));
        gpu.maxChunkRows = chunk;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.dB), bBytes));
        for (int b = 0; b < 2; ++b) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.dA[b]), chunk * N * sizeof(double)));
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gpu.dC[b]), chunk * N * sizeof(double)));
            CUDA_CHECK(cudaStreamCreateWithFlags(&gpu.stream[b], cudaStreamNonBlocking));
        }
        CUDA_CHECK(cudaMemcpy(gpu.dB, B.data(), bBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Perform matrix multiplication
    if (rank == 0) printf("Computing matrix multiplication...\n");
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto start = std::chrono::high_resolution_clock::now();

    matrixMultiplySlab(Aslab, B.data(), Cslab, N, myRows, gpu);

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    long localMs = static_cast<long>(duration.count());
    long maxMs = localMs;
    MPI_CHECK(MPI_Reduce(&localMs, &maxMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Computation time: %ld ms\n", maxMs);

        // Calculate GFLOPS
        double gflops = (2.0 * N * N * N) / (maxMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // ---- Gather C on rank 0 when it is needed for output / validation ----
    std::vector<double> C;
    if (printResults || validate) {
        MPI_Datatype rowType;
        MPI_CHECK(MPI_Type_contiguous(static_cast<int>(N), MPI_DOUBLE, &rowType));
        MPI_CHECK(MPI_Type_commit(&rowType));
        std::vector<int> counts(nRanks), displs(nRanks);
        for (int r = 0; r < nRanks; ++r) {
            counts[r] = static_cast<int>(rowsOf(r));
            displs[r] = static_cast<int>(firstRowOf(r));
        }
        if (rank == 0) C.resize(N * N);
        MPI_CHECK(MPI_Gatherv(Cslab, static_cast<int>(myRows), rowType, rank == 0 ? C.data() : nullptr,
                              counts.data(), displs.data(), rowType, 0, MPI_COMM_WORLD));
        MPI_CHECK(MPI_Type_free(&rowType));
    }

    int exitCode = 0;
    if (rank == 0) {
        // Print results for external validation
        if (printResults) {
            print_results(C, "MatrixC");
        }

        // Validation
        if (validate) {
            printf("Validating result...\n");
            bool valid = validateResult(B, C, N);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_CHECK(MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD));

    // ---- Cleanup ----
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamDestroy(gpu.stream[b]));
        CUDA_CHECK(cudaFree(gpu.dA[b]));
        CUDA_CHECK(cudaFree(gpu.dC[b]));
    }
    CUDA_CHECK(cudaFree(gpu.dB));
    CUDA_CHECK(cudaFreeHost(Aslab));
    CUDA_CHECK(cudaFreeHost(Cslab));

    MPI_Finalize();
    return exitCode;
}
