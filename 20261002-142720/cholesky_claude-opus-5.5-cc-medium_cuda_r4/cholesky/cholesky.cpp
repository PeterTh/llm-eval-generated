#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Cholesky decomposition on the GPU (CUDA, blocked right-looking algorithm with lookahead)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err__ = (call);                                                   \
        if (err__ != cudaSuccess) {                                                   \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err__),    \
                    __FILE__, __LINE__);                                              \
            exit(1);                                                                  \
        }                                                                             \
    } while (0)

constexpr int NB = 64;           // panel width (block size of the factorization)
constexpr int TILE = 64;         // GEMM output tile size (large problems)
constexpr int TILE_S = 32;       // GEMM output tile size (small problems, more parallelism)
constexpr long long MIN_BLOCKS = 4 * 82; // use small tiles if large tiles give fewer blocks
constexpr int KC = 16;           // GEMM k-chunk
constexpr int GEMM_THREADS = 256;
constexpr int DIAG_THREADS = 256;
constexpr int TRSM_ROWS = 16;    // rows per TRSM block
constexpr int TRSM_TPR = 8;      // threads per row in TRSM
constexpr unsigned long long NO_ERROR = ~0ull;
constexpr size_t FIRST_CHUNK = 512; // rows in the first host-to-device chunk (chunks then double)
constexpr int MAX_CHUNKS = 64;

struct GpuContext {
    size_t n = 0;
    double* dA = nullptr;        // matrix being factorized
    double* dWork = nullptr;     // scratch n x n buffer (generation / validation)
    unsigned long long* dInfo = nullptr; // index of first non-positive pivot
    // 0: trailing updates, 1: panel factorization, 2: copies back, 3: copies in, 4: catch-up of new rows
    cudaStream_t s[5]{};
    cudaEvent_t evUpd{}, evDiag{}, evPanel{}, evCatch{};
    cudaEvent_t evChunk[MAX_CHUNKS]{};
    bool hostRegistered = false;
    void* hostPtr = nullptr;
};

// C[i][j] (op)= sum_k P[i][k] * Q[j][k] for i < M, j < N (row-major, leading dim ld / ldc).
// SUBTRACT: C -= acc, else C = acc. LOWER: only j <= i + diagOff (local coordinates) is written.
// TRI_GRID: grid enumerates the lower-triangular tiles of an M x M tile matrix. TM: output tile size.
template <int TM, bool SUBTRACT, bool LOWER, bool TRI_GRID>
__global__ void __launch_bounds__(GEMM_THREADS)
gemmNT(double* __restrict__ C, size_t ldc, const double* __restrict__ P,
       const double* __restrict__ Q, size_t ld, int M, int N, int K, int diagOff = 0) {
    int ti, tj;
    const long long bid = blockIdx.x;
    if (TRI_GRID) {
        long long r = (long long)((sqrt(8.0 * (double)bid + 1.0) - 1.0) * 0.5);
        while (r * (r + 1) / 2 > bid) --r;
        while ((r + 1) * (r + 2) / 2 <= bid) ++r;
        ti = (int)r;
        tj = (int)(bid - r * (r + 1) / 2);
    } else {
        const int tm = (M + TM - 1) / TM;
        ti = (int)(bid % tm);
        tj = (int)(bid / tm);
    }
    const int i0 = ti * TM, j0 = tj * TM;
    if (LOWER && j0 > i0 + TM - 1 + diagOff) return; // tile entirely above the diagonal

    __shared__ double As[KC][TM + 1];
    __shared__ double Bs[KC][TM + 1];

    const int tid = threadIdx.x;
    const int tx = tid % 16, ty = tid / 16;
    constexpr int RT = TM / 16; // outputs per thread in each dimension
    // For SUBTRACT, the accumulator starts from C (loaded up front so the latency is hidden)
    double acc[RT][RT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int i = i0 + ty + 16 * r;
#pragma unroll
        for (int c = 0; c < RT; ++c) {
            const int j = j0 + tx + 16 * c;
            acc[r][c] = (SUBTRACT && i < M && j < N) ? C[(size_t)i * ldc + j] : 0.0;
        }
    }

    for (int k0 = 0; k0 < K; k0 += KC) {
#pragma unroll
        for (int r = 0; r < (TM * KC) / GEMM_THREADS; ++r) {
            const int idx = tid + r * GEMM_THREADS;
            const int row = idx / KC, col = idx % KC;
            const int gk = k0 + col;
            const int gi = i0 + row, gj = j0 + row;
            As[col][row] = (gi < M && gk < K) ? P[(size_t)gi * ld + gk] : 0.0;
            Bs[col][row] = (gj < N && gk < K) ? Q[(size_t)gj * ld + gk] : 0.0;
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            double a[RT], b[RT];
#pragma unroll
            for (int r = 0; r < RT; ++r) a[r] = As[kk][ty + 16 * r];
#pragma unroll
            for (int c = 0; c < RT; ++c) b[c] = Bs[kk][tx + 16 * c];
#pragma unroll
            for (int r = 0; r < RT; ++r)
#pragma unroll
                for (int c = 0; c < RT; ++c) acc[r][c] = fma(SUBTRACT ? -a[r] : a[r], b[c], acc[r][c]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int i = i0 + ty + 16 * r;
        if (i >= M) continue;
#pragma unroll
        for (int c = 0; c < RT; ++c) {
            const int j = j0 + tx + 16 * c;
            if (j >= N || (LOWER && j > i + diagOff)) continue;
            C[(size_t)i * ldc + j] = acc[r][c];
        }
    }
}

// Zero the strict upper triangular part of rows [r0, r0 + gridDim.y) of A
__global__ void zeroUpper(double* __restrict__ A, size_t n, size_t r0) {
    const size_t i = r0 + blockIdx.y;
    for (size_t j = i + 1 + blockIdx.x * (size_t)blockDim.x + threadIdx.x; j < n;
         j += (size_t)gridDim.x * blockDim.x) {
        A[i * n + j] = 0.0;
    }
}

// Factorize the b x b diagonal block starting at (kb, kb) in place (lower triangle).
__global__ void __launch_bounds__(DIAG_THREADS)
factorDiag(double* __restrict__ A, size_t n, size_t kb, int b, unsigned long long* info) {
    __shared__ double S[NB][NB + 1];
    __shared__ bool failed;
    __shared__ double invDiag;
    // Lower-triangular (i, l) pairs ordered by decreasing l: at step j the pairs with l > j
    // form the prefix of length (NB-1-j)(NB-j)/2, so no FP64 work is wasted on masked lanes
    __shared__ unsigned short pairs[NB * (NB + 1) / 2];
    const int tid = threadIdx.x;
    if (*info != NO_ERROR) return; // an earlier pivot already failed

    for (int idx = tid; idx < b * b; idx += DIAG_THREADS) {
        const int i = idx / b, j = idx % b;
        if (j <= i) S[i][j] = A[(kb + i) * n + kb + j];
    }
    if (tid < NB) {
        const int l = tid, start = (NB - 1 - l) * (NB - l) / 2;
        for (int i = l; i < NB; ++i) pairs[start + i - l] = (unsigned short)((i << 8) | l);
    }
    if (tid == 0) failed = false;
    __syncthreads();

    for (int j = 0; j < b; ++j) {
        if (tid == 0) {
            const double val = S[j][j];
            if (val <= 0.0) {
                failed = true;
                *info = kb + j;
            } else {
                S[j][j] = sqrt(val);
                invDiag = 1.0 / S[j][j];
            }
        }
        __syncthreads();
        if (failed) return;
        if (j + 1 + tid < b) S[j + 1 + tid][j] *= invDiag;
        __syncthreads();
        // Rank-1 update of the remaining lower triangle
        const int cnt = (NB - 1 - j) * (NB - j) / 2;
        for (int idx = tid; idx < cnt; idx += DIAG_THREADS) {
            const int i = pairs[idx] >> 8, l = pairs[idx] & 0xFF;
            if (i < b) S[i][l] -= S[i][j] * S[l][j];
        }
        __syncthreads();
    }

    for (int idx = tid; idx < b * b; idx += DIAG_THREADS) {
        const int i = idx / b, j = idx % b;
        if (j <= i) A[(kb + i) * n + kb + j] = S[i][j];
    }
}

// Panel triangular solve: for rows i in [r0, r1), A[i, kb:kb+b] = A[i, kb:kb+b] * L11^{-T}
// TRSM_TPR threads cooperate on one row (split k-sum, reduced with warp shuffles).
__global__ void __launch_bounds__(TRSM_ROWS * TRSM_TPR)
trsmPanel(double* __restrict__ A, size_t n, size_t kb, int b, size_t r0, size_t r1) {
    extern __shared__ double smem[];
    double* L = smem;                       // NB x (NB+1), column NB holds 1 / L[j][j]
    double* X = smem + NB * (NB + 1);       // TRSM_ROWS x (NB+1)
    constexpr int LD = NB + 1;
    const int tid = threadIdx.x;
    const size_t rowStart = r0 + (size_t)blockIdx.x * TRSM_ROWS;
    const int rows = (r1 - rowStart < (size_t)TRSM_ROWS) ? (int)(r1 - rowStart) : TRSM_ROWS;

    for (int idx = tid; idx < b * b; idx += blockDim.x) {
        const int i = idx / b, j = idx % b;
        L[i * LD + j] = (j <= i) ? A[(kb + i) * n + kb + j] : 0.0;
    }
    for (int j = tid; j < b; j += blockDim.x) L[j * LD + NB] = 1.0 / A[(kb + j) * n + kb + j];
    for (int idx = tid; idx < rows * b; idx += blockDim.x) {
        const int i = idx / b, j = idx % b;
        X[i * LD + j] = A[(rowStart + i) * n + kb + j];
    }
    __syncthreads();

    const int r = tid / TRSM_TPR, lane = tid % TRSM_TPR;
    double* x = X + r * LD;
    for (int j = 0; j < b; ++j) {
        double sum = 0.0;
        for (int k = lane; k < j; k += TRSM_TPR) sum = fma(x[k], L[j * LD + k], sum);
#pragma unroll
        for (int off = TRSM_TPR / 2; off > 0; off /= 2) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == (j % TRSM_TPR)) x[j] = (x[j] - sum) * L[j * LD + NB];
        __syncwarp();
    }
    __syncthreads();

    for (int idx = tid; idx < rows * b; idx += blockDim.x) {
        const int i = idx / b, j = idx % b;
        A[(rowStart + i) * n + kb + j] = X[i * LD + j];
    }
}

static size_t trsmSmemBytes() { return sizeof(double) * (NB + TRSM_ROWS) * (NB + 1); }

void gpuInit(GpuContext& ctx, std::vector<double>& A, const size_t n) {
    ctx.n = n;
    const size_t bytes = std::max<size_t>(n * n, 1) * sizeof(double);
    CUDA_CHECK(cudaMalloc(&ctx.dA, bytes));
    CUDA_CHECK(cudaMalloc(&ctx.dWork, bytes));
    CUDA_CHECK(cudaMalloc(&ctx.dInfo, sizeof(unsigned long long)));
    // The panel factorization is on the critical path: give its stream the highest priority
    int prioLow = 0, prioHigh = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prioLow, &prioHigh));
    for (int i = 0; i < 5; ++i)
        CUDA_CHECK(cudaStreamCreateWithPriority(&ctx.s[i], cudaStreamNonBlocking, i == 1 ? prioHigh : prioLow));
    for (cudaEvent_t* e : {&ctx.evUpd, &ctx.evDiag, &ctx.evPanel, &ctx.evCatch})
        CUDA_CHECK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
    for (auto& e : ctx.evChunk) CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    CUDA_CHECK(cudaFuncSetAttribute(trsmPanel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    (int)trsmSmemBytes()));
    // Load all kernels now: with lazy module loading, the first launch of a kernel would otherwise
    // wait for all previously queued work (including the host-to-device copies) to finish
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, zeroUpper));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, factorDiag));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, gemmNT<TILE, true, true, true>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, gemmNT<TILE, true, true, false>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, gemmNT<TILE_S, true, true, true>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, gemmNT<TILE_S, true, true, false>));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, gemmNT<TILE, false, false, false>));
    // Pin the host matrix for fast asynchronous transfers
    if (n > 0 && cudaHostRegister(A.data(), n * n * sizeof(double), cudaHostRegisterDefault) == cudaSuccess) {
        ctx.hostRegistered = true;
        ctx.hostPtr = A.data();
    } else {
        cudaGetLastError(); // clear error, fall back to pageable copies
    }
}

void gpuFree(GpuContext& ctx) {
    if (ctx.hostRegistered) cudaHostUnregister(ctx.hostPtr);
    for (cudaEvent_t e : {ctx.evUpd, ctx.evDiag, ctx.evPanel, ctx.evCatch}) cudaEventDestroy(e);
    for (auto& e : ctx.evChunk) cudaEventDestroy(e);
    for (auto& s : ctx.s) cudaStreamDestroy(s);
    cudaFree(ctx.dInfo);
    cudaFree(ctx.dWork);
    cudaFree(ctx.dA);
}

// Update A[r0:r0+m, c0:c0+w] -= A[r0:r0+m, kb:kb+b] * A[c0:c0+w, kb:kb+b]^T (lower part only).
// square: r0 == c0 and m == w, only lower-triangular tiles are launched.
static void launchUpdate(GpuContext& ctx, cudaStream_t s, size_t kb, int b, size_t r0, size_t c0,
                         int m, int w, bool square) {
    const size_t n = ctx.n;
    double* C = ctx.dA + r0 * n + c0;
    const double* P = ctx.dA + r0 * n + kb;
    const double* Q = ctx.dA + c0 * n + kb;
    const int diagOff = (int)((long long)r0 - (long long)c0);
    auto blocks = [&](int t) {
        const long long tm = (m + t - 1) / t, tn = (w + t - 1) / t;
        return square ? tm * (tm + 1) / 2 : tm * tn;
    };
    if (blocks(TILE) >= MIN_BLOCKS) {
        if (square)
            gemmNT<TILE, true, true, true><<<(unsigned)blocks(TILE), GEMM_THREADS, 0, s>>>(C, n, P, Q, n, m, m, b);
        else
            gemmNT<TILE, true, true, false><<<(unsigned)blocks(TILE), GEMM_THREADS, 0, s>>>(C, n, P, Q, n, m, w, b,
                                                                                           diagOff);
    } else {
        if (square)
            gemmNT<TILE_S, true, true, true><<<(unsigned)blocks(TILE_S), GEMM_THREADS, 0, s>>>(C, n, P, Q, n, m, m, b);
        else
            gemmNT<TILE_S, true, true, false><<<(unsigned)blocks(TILE_S), GEMM_THREADS, 0, s>>>(C, n, P, Q, n, m, w,
                                                                                               b, diagOff);
    }
    CUDA_CHECK(cudaGetLastError());
}

static void launchTrsm(GpuContext& ctx, cudaStream_t s, size_t kb, int b, size_t r0, size_t r1) {
    trsmPanel<<<(unsigned)((r1 - r0 + TRSM_ROWS - 1) / TRSM_ROWS), TRSM_ROWS * TRSM_TPR, trsmSmemBytes(), s>>>(
        ctx.dA, ctx.n, kb, b, r0, r1);
    CUDA_CHECK(cudaGetLastError());
}

// Apply the finished panel at kb to rows [r0, r1) that joined the factorization late (catch-up).
static void catchUpStep(GpuContext& ctx, cudaStream_t s, size_t kb, int b, size_t r0, size_t r1) {
    launchTrsm(ctx, s, kb, b, r0, r1);
    const size_t kn = kb + b;
    launchUpdate(ctx, s, kb, b, r0, kn, (int)(r1 - r0), (int)(r1 - kn), false);
}

// Blocked right-looking Cholesky on the GPU.
// The matrix is streamed to the device in row chunks of growing size. The factorization proceeds
// on the rows already present (stream 0 / 1, with one panel of lookahead), while the next chunk is
// brought up to date with the finished panels concurrently (stream 4). Finished block rows are
// copied back while the factorization continues (stream 2).
bool choleskyDecomposition(std::vector<double>& A, const size_t n, GpuContext& ctx) {
    // A is stored in row-major order
    if (n == 0) return true;
    cudaStream_t sUpd = ctx.s[0], sPan = ctx.s[1], sOut = ctx.s[2], sIn = ctx.s[3], sCat = ctx.s[4];
    double* dA = ctx.dA;
    const size_t rowBytes = n * sizeof(double);

    // Chunk boundaries (multiples of NB, doubling in size)
    std::vector<size_t> R{0};
    for (size_t r = std::min(n, FIRST_CHUNK); ; r = std::min(n, 2 * r)) {
        if ((int)R.size() == MAX_CHUNKS) r = n;
        R.push_back(r);
        if (r == n) break;
    }
    const int C = (int)R.size() - 1;

    // Host-to-device: only the lower triangular part (columns < R[c+1]) of each chunk is needed
    for (int c = 0; c < C; ++c) {
        const size_t r0 = R[c], r1 = R[c + 1];
        // Issued in pieces of NB rows: the driver may otherwise delay work queued behind a huge copy
        for (size_t r = r0; r < r1; r += NB) {
            CUDA_CHECK(cudaMemcpy2DAsync(dA + r * n, rowBytes, A.data() + r * n, rowBytes, r1 * sizeof(double),
                                         std::min<size_t>(NB, r1 - r), cudaMemcpyHostToDevice, sIn));
        }
        CUDA_CHECK(cudaEventRecord(ctx.evChunk[c], sIn));
    }
    // Clear the upper triangular part of a chunk once it has arrived
    auto zeroChunk = [&](int c, cudaStream_t s) {
        CUDA_CHECK(cudaStreamWaitEvent(s, ctx.evChunk[c], 0));
        dim3 grid((unsigned)std::min<size_t>((n + 255) / 256, 64), (unsigned)(R[c + 1] - R[c]));
        zeroUpper<<<grid, 256, 0, s>>>(dA, n, R[c]);
        CUDA_CHECK(cudaGetLastError());
    };
    CUDA_CHECK(cudaMemsetAsync(ctx.dInfo, 0xFF, sizeof(unsigned long long), sPan)); // NO_ERROR
    zeroChunk(0, sPan);
    CUDA_CHECK(cudaEventRecord(ctx.evUpd, sPan));
    CUDA_CHECK(cudaStreamWaitEvent(sUpd, ctx.evUpd, 0));

    for (int c = 0; c < C; ++c) {
        const size_t rEnd = R[c + 1];
        const bool next = c + 1 < C;
        if (c > 0) {
            // Rows of this chunk have been caught up with all previous panels
            CUDA_CHECK(cudaStreamWaitEvent(sUpd, ctx.evCatch, 0));
            CUDA_CHECK(cudaStreamWaitEvent(sPan, ctx.evCatch, 0));
        }
        if (next) {
            // Bring the next chunk up to date with all panels finished before this chunk
            zeroChunk(c + 1, sCat);
            if (c > 0) CUDA_CHECK(cudaStreamWaitEvent(sCat, ctx.evPanel, 0));
            for (size_t kb = 0; kb < R[c]; kb += NB)
                catchUpStep(ctx, sCat, kb, NB, R[c + 1], R[c + 2]);
        }

        for (size_t kb = R[c]; kb < rEnd; kb += NB) {
            const int b = (int)std::min<size_t>(NB, rEnd - kb);
            const size_t kn = kb + b;

            // Panel factorization (stream 1), waits for the update of this panel's columns
            if (kb > R[c]) CUDA_CHECK(cudaStreamWaitEvent(sPan, ctx.evUpd, 0));
            factorDiag<<<1, DIAG_THREADS, 0, sPan>>>(dA, n, kb, b, ctx.dInfo);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(ctx.evDiag, sPan));

            // Rows kb..kb+b are final now: copy them back asynchronously
            CUDA_CHECK(cudaStreamWaitEvent(sOut, ctx.evDiag, 0));
            CUDA_CHECK(cudaMemcpyAsync(A.data() + kb * n, dA + kb * n, (size_t)b * rowBytes,
                                       cudaMemcpyDeviceToHost, sOut));

            const int m = (int)(rEnd - kn);
            if (m > 0) launchTrsm(ctx, sPan, kb, b, kn, rEnd);
            CUDA_CHECK(cudaEventRecord(ctx.evPanel, sPan));

            // Apply this panel to the next chunk as soon as it is finished
            if (next) {
                CUDA_CHECK(cudaStreamWaitEvent(sCat, ctx.evPanel, 0));
                catchUpStep(ctx, sCat, kb, b, R[c + 1], R[c + 2]);
            }

            if (m == 0) continue;
            // Trailing update (stream 0): first the next panel's columns (lookahead), then the rest
            CUDA_CHECK(cudaStreamWaitEvent(sUpd, ctx.evPanel, 0));
            const int bn = std::min(NB, m);
            launchUpdate(ctx, sUpd, kb, b, kn, kn, m, bn, false);
            CUDA_CHECK(cudaEventRecord(ctx.evUpd, sUpd));
            if (m > bn) launchUpdate(ctx, sUpd, kb, b, kn + bn, kn + bn, m - bn, m - bn, true);
        }
        if (next) CUDA_CHECK(cudaEventRecord(ctx.evCatch, sCat));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    unsigned long long info = NO_ERROR;
    CUDA_CHECK(cudaMemcpy(&info, ctx.dInfo, sizeof(info), cudaMemcpyDeviceToHost));
    if (info != NO_ERROR) {
        // Matrix is not positive definite
        printf("Error: Matrix is not positive definite at diagonal element %llu\n", info);
        return false;
    }
    return true;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n, GpuContext& ctx) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    if (n == 0) return;

    std::vector<double> B(n * n);
    unsigned int seed = 42;
    
    // Generate random matrix B
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
    
    // Compute A = B * B^T on the GPU
    CUDA_CHECK(cudaMemcpy(ctx.dWork, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
    const long long t = (n + TILE - 1) / TILE;
    gemmNT<TILE, false, false, false><<<(unsigned)(t * t), GEMM_THREADS>>>(ctx.dA, n, ctx.dWork, ctx.dWork, n,
                                                                     (int)n, (int)n, (int)n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(A.data(), ctx.dA, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Add diagonal dominance to ensure positive definiteness
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n,
                      GpuContext& ctx) {
    // Validate by computing L * L^T and comparing with original matrix
    
    std::vector<double> reconstructed(n * n);
    
    // Compute L * L^T on the GPU
    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(ctx.dA, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice));
        const long long t = (n + TILE - 1) / TILE;
        gemmNT<TILE, false, false, false><<<(unsigned)(t * t), GEMM_THREADS>>>(ctx.dWork, n, ctx.dA, ctx.dA, n,
                                                                         (int)n, (int)n, (int)n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(reconstructed.data(), ctx.dWork, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    }
    
    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;
    
    for (size_t i = 0; i < n * n; ++i) {
        const double error = fabs(reconstructed[i] - A_orig[i]);
        maxError = std::max(maxError, error);
        
        const double rel = error / (fabs(A_orig[i]) + 1e-10);
        relError = std::max(relError, rel);
    }
    
    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);
    
    // Check if error is within tolerance
    if (relError > 1e-6) {
        printf("Validation failed: relative error too large\n");
        return false;
    }
    
    return true;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = atoi(argv[++i]);
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
    
    printf("Cholesky Decomposition Benchmark\n");
    printf("Matrix size: %zu x %zu\n", n, n);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Allocate matrix
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    
    // Set up GPU resources (device buffers, streams, pinned host memory)
    GpuContext ctx;
    gpuInit(ctx, A, n);
    
    // Generate positive definite matrix
    printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n, ctx);
    
    if (validate) {
        A_orig = A; // Save original for validation
    }
    
    // Perform Cholesky decomposition
    printf("Computing Cholesky decomposition...\n");
    auto start = std::chrono::high_resolution_clock::now();
    
    bool success = choleskyDecomposition(A, n, ctx);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!success) {
        gpuFree(ctx);
        printf("Cholesky decomposition failed\n");
        return 1;
    }
    
    printf("Computation time: %ld ms\n", duration.count());
    
    // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
    double ops = (double)n * n * n / 3.0;
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    printf("Performance: %.3f GFLOPS\n", gflops);
    
    // Print results for external validation
    if (printResults) {
        print_results(A, "CholeskyL");
    }
    
    // Validation
    if (validate) {
        printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, n, ctx);
        gpuFree(ctx);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    gpuFree(ctx);
    return 0;
}
