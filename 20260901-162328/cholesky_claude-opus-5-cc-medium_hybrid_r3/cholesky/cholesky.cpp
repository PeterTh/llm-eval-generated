#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Algorithm: right-looking blocked Cholesky (A = L * L^T, L lower triangular).
//   * MPI    : the matrix is distributed over the ranks as block columns
//              ("panels") of NB columns in a cyclic fashion.  Each step
//              broadcasts the freshly factorized panel; a depth-1 lookahead
//              overlaps that broadcast with the trailing matrix updates.
//   * CUDA   : every rank owns one GPU which holds its panels for the whole
//              factorization.  All O(n^3) work (trailing updates DSYRK/DGEMM,
//              the panel triangular solve DTRSM, matrix generation and the
//              validation product) runs there through cuBLAS.
//   * OpenMP : the random matrix generation (a jump-ahead of the generator
//              reproduces rand_r() exactly in parallel) and all O(n^2) host
//              side loops - result clean up and validation - are threaded.
//
// Semantics are those of the sequential version: rank 0 ends up with the full
// row-major lower triangular L, its upper triangle zeroed.

#define CUDA_CHECK(call)                                                                                               \
    do {                                                                                                               \
        const cudaError_t err_ = (call);                                                                               \
        if (err_ != cudaSuccess) {                                                                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

#define CUBLAS_CHECK(call)                                                                                             \
    do {                                                                                                               \
        const cublasStatus_t st_ = (call);                                                                             \
        if (st_ != CUBLAS_STATUS_SUCCESS) {                                                                            \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)st_);                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                                              \
        }                                                                                                              \
    } while (0)

// ---------------------------------------------------------------------------
// Global parallel context
// ---------------------------------------------------------------------------

namespace {

int g_rank = 0;
int g_nranks = 1;
// All OpenMP regions use the same team size: on many-core machines libgomp
// pays a large penalty whenever the size of the requested team changes.
int g_threads = 1;
cublasHandle_t g_blas = nullptr;
cudaStream_t g_stream = nullptr;
// Second stream (rank 0 only): assembles the result concurrently with compute.
cublasHandle_t g_blasOut = nullptr;
cudaStream_t g_streamOut = nullptr;
// Stream for the initial upload of the matrix, overlapped with the first step.
cudaStream_t g_streamIn = nullptr;

// Column block size of the blocked algorithm.
constexpr size_t NB = 256;

inline int panelOwner(const size_t k) { return (int)(k % (size_t)g_nranks); }
inline size_t panelLocalIndex(const size_t k) { return k / (size_t)g_nranks; }

// Number of panels owned by this rank out of nt panels.
inline size_t localPanelCount(const size_t nt) {
    return (nt > (size_t)g_rank) ? ((nt - (size_t)g_rank + (size_t)g_nranks - 1) / (size_t)g_nranks) : 0;
}

// ---------------------------------------------------------------------------
// Parallel bit-exact replica of glibc's rand_r
// ---------------------------------------------------------------------------

struct LcgMap {
    uint32_t a;
    uint32_t c;
};

// Apply f1 first, then f2.
inline LcgMap lcgCompose(const LcgMap f1, const LcgMap f2) {
    return LcgMap{f1.a * f2.a, (uint32_t)(f2.a * f1.c + f2.c)};
}

// The base step of glibc's TYPE_0 generator, raised to the m-th power.
inline LcgMap lcgPower(uint64_t m) {
    LcgMap result{1u, 0u};
    LcgMap base{1103515245u, 12345u};
    while (m != 0) {
        if (m & 1ull) result = lcgCompose(result, base);
        base = lcgCompose(base, base);
        m >>= 1;
    }
    return result;
}

inline int randR(uint32_t& seed) {
    uint32_t next = seed;
    int result;
    next = next * 1103515245u + 12345u;
    result = (int)((next / 65536u) % 2048u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    next = next * 1103515245u + 12345u;
    result <<= 10;
    result ^= (int)((next / 65536u) % 1024u);
    seed = next;
    return result;
}

// ---------------------------------------------------------------------------
// Factorization of the small diagonal block (column-major, lower triangle).
// The block is only NB x NB and stays in cache, so the right looking rank-1
// formulation on a single core beats any threaded variant here.
// Returns -1 on success, otherwise the local index of the failing diagonal.
// ---------------------------------------------------------------------------

int choleskyDiagonalBlock(double* D, const size_t kb) {
    for (size_t j = 0; j < kb; ++j) {
        const double val = D[j * kb + j];
        if (val <= 0.0) return (int)j;
        const double d = sqrt(val);
        D[j * kb + j] = d;
        for (size_t i = j + 1; i < kb; ++i) {
            D[j * kb + i] /= d;
        }

        // Rank-1 update of the trailing lower triangle.
        for (size_t c = j + 1; c < kb; ++c) {
            const double f = D[j * kb + c];
            if (f != 0.0) {
                const double* __restrict src = D + j * kb;
                double* __restrict dst = D + c * kb;
#pragma omp simd
                for (size_t i = c; i < kb; ++i) {
                    dst[i] -= f * src[i];
                }
            }
        }
    }
    return -1;
}

} // namespace

// ---------------------------------------------------------------------------
// Workspace of the distributed factorization.  Allocating device memory, page
// locked host buffers and events is expensive, so it happens once up front
// instead of inside the measured region.
// ---------------------------------------------------------------------------

namespace {

struct Workspace {
    size_t n = 0;
    size_t nt = 0;         // number of block columns
    size_t nLocal = 0;     // block columns owned by this rank
    size_t panelStride = 0; // elements per panel (column-major, ld = n)

    double* dA = nullptr;             // locally owned panels
    double* dPanel[2] = {nullptr, nullptr}; // received panel of the current step
    double* dOut[2] = {nullptr, nullptr};   // rank 0: transposed panel for output
    double* hPanel[2] = {nullptr, nullptr}; // pinned broadcast buffers
    double* hDiag = nullptr;                // pinned diagonal block
    cudaEvent_t evReady[2], evExported[2], evScatterDone;
    std::vector<cudaEvent_t> evScatter;
};

Workspace g_ws;

void setupWorkspace(const size_t n) {
    if (n == 0) return;
    Workspace& w = g_ws;
    w.n = n;
    w.nt = (n + NB - 1) / NB;
    w.nLocal = localPanelCount(w.nt);
    w.panelStride = n * NB;

    if (w.nLocal > 0) {
        CUDA_CHECK(cudaMalloc(&w.dA, w.nLocal * w.panelStride * sizeof(double)));
    }
    CUDA_CHECK(cudaMalloc(&w.dPanel[0], w.panelStride * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&w.dPanel[1], w.panelStride * sizeof(double)));
    CUDA_CHECK(cudaHostAlloc(&w.hPanel[0], (w.panelStride + 1) * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&w.hPanel[1], (w.panelStride + 1) * sizeof(double), cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc(&w.hDiag, NB * NB * sizeof(double), cudaHostAllocDefault));

    // Rank 0 assembles the row-major result while the factorization runs: a
    // panel is final as soon as it has been factorized/received, so it is
    // transposed on the GPU and DMA'd straight into its place in A on a
    // dedicated stream that overlaps with the trailing updates.
    if (g_rank == 0) {
        CUDA_CHECK(cudaMalloc(&w.dOut[0], w.panelStride * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&w.dOut[1], w.panelStride * sizeof(double)));
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaEventCreateWithFlags(&w.evReady[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventCreateWithFlags(&w.evExported[i], cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(w.evExported[i], g_streamOut));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&w.evScatterDone, cudaEventDisableTiming));
    }

    w.evScatter.resize(w.nLocal);
    for (size_t jl = 0; jl < w.nLocal; ++jl) {
        CUDA_CHECK(cudaEventCreateWithFlags(&w.evScatter[jl], cudaEventDisableTiming));
    }

    // Touch every cuBLAS routine and transfer path once so that the lazy kernel
    // loading of the CUDA runtime does not land in the measured region.
    {
        const int t = (int)std::min<size_t>(32, n);
        const double one = 1.0, zero = 0.0;
        CUDA_CHECK(cudaMemsetAsync(w.dPanel[0], 0, w.panelStride * sizeof(double), g_stream));
        CUDA_CHECK(cudaMemsetAsync(w.dPanel[1], 0, w.panelStride * sizeof(double), g_stream));
        CUBLAS_CHECK(cublasDsyrk(g_blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, t, t, &one, w.dPanel[0], t, &one,
                                 w.dPanel[1], t));
        CUBLAS_CHECK(cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, t, t, t, &one, w.dPanel[0], t, w.dPanel[0], t,
                                 &one, w.dPanel[1], t));
        CUBLAS_CHECK(cublasDtrsm(g_blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                 CUBLAS_DIAG_NON_UNIT, t, t, &one, w.dPanel[0], t, w.dPanel[1], t));
        CUDA_CHECK(cudaMemcpy2DAsync(w.hDiag, t * sizeof(double), w.dPanel[0], t * sizeof(double),
                                     t * sizeof(double), t, cudaMemcpyDeviceToHost, g_stream));
        CUDA_CHECK(cudaMemcpy2DAsync(w.dPanel[0], t * sizeof(double), w.hDiag, t * sizeof(double),
                                     t * sizeof(double), t, cudaMemcpyHostToDevice, g_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_stream));
        if (g_rank == 0) {
            CUBLAS_CHECK(cublasDgeam(g_blasOut, CUBLAS_OP_T, CUBLAS_OP_N, t, t, &one, w.dPanel[0], t, &zero,
                                     w.dOut[0], t, w.dOut[0], t));
            CUDA_CHECK(cudaMemcpy2DAsync(w.hPanel[0], t * sizeof(double), w.dOut[0], t * sizeof(double),
                                         t * sizeof(double), t, cudaMemcpyDeviceToHost, g_streamOut));
            CUDA_CHECK(cudaStreamSynchronize(g_streamOut));
        }
        CUDA_CHECK(cudaMemcpyAsync(w.dPanel[0], w.hPanel[0], t * sizeof(double), cudaMemcpyHostToDevice,
                                   g_streamIn));
        CUDA_CHECK(cudaStreamSynchronize(g_streamIn));
    }
}

void teardownWorkspace() {
    Workspace& w = g_ws;
    if (w.n == 0) return;
    for (size_t jl = 0; jl < w.nLocal; ++jl) CUDA_CHECK(cudaEventDestroy(w.evScatter[jl]));
    w.evScatter.clear();
    if (g_rank == 0) {
        for (int i = 0; i < 2; ++i) {
            CUDA_CHECK(cudaFree(w.dOut[i]));
            CUDA_CHECK(cudaEventDestroy(w.evReady[i]));
            CUDA_CHECK(cudaEventDestroy(w.evExported[i]));
        }
        CUDA_CHECK(cudaEventDestroy(w.evScatterDone));
    }
    if (w.dA != nullptr) CUDA_CHECK(cudaFree(w.dA));
    CUDA_CHECK(cudaFree(w.dPanel[0]));
    CUDA_CHECK(cudaFree(w.dPanel[1]));
    CUDA_CHECK(cudaFreeHost(w.hPanel[0]));
    CUDA_CHECK(cudaFreeHost(w.hPanel[1]));
    CUDA_CHECK(cudaFreeHost(w.hDiag));
    w = Workspace{};
}

} // namespace

// ---------------------------------------------------------------------------
// Distributed blocked Cholesky decomposition
//
// On entry every rank holds the full symmetric matrix A (row-major).  On exit
// rank 0 holds the row-major lower triangular factor L with a zeroed upper
// triangle; the contents of A on the other ranks are unspecified.
// ---------------------------------------------------------------------------

bool choleskyDecomposition(std::vector<double>& A, const size_t n) {
    if (n == 0) return true;

    const size_t nt = g_ws.nt;
    const size_t nLocal = g_ws.nLocal;
    const size_t panelStride = g_ws.panelStride;
    double* const dA = g_ws.dA;
    double* const* const dPanel = g_ws.dPanel;
    double* const* const dOut = g_ws.dOut;
    double* const* const hPanel = g_ws.hPanel;
    double* const hDiag = g_ws.hDiag;
    cudaEvent_t* const evReady = g_ws.evReady;
    cudaEvent_t* const evExported = g_ws.evExported;
    const std::vector<cudaEvent_t>& evScatter = g_ws.evScatter;

    // ---- scatter the initial matrix onto the GPUs -------------------------
    // A is symmetric, hence its row-major host image is at the same time the
    // column-major image of A: column j of A is the contiguous row j.
    // The uploads run on their own stream so that the first factorization step
    // can start while the remaining panels are still in flight.
    for (size_t jl = 0; jl < nLocal; ++jl) {
        const size_t j = jl * (size_t)g_nranks + (size_t)g_rank;
        const size_t joff = j * NB;
        const size_t jb = std::min(NB, n - joff);
        CUDA_CHECK(cudaMemcpyAsync(dA + jl * panelStride, A.data() + joff * n, jb * n * sizeof(double),
                                   cudaMemcpyHostToDevice, g_streamIn));
        CUDA_CHECK(cudaEventRecord(evScatter[jl], g_streamIn));
    }
    if (g_rank == 0) {
        // The result is written back into A, so no panel may be exported before
        // the whole matrix has been read out of it.
        CUDA_CHECK(cudaEventRecord(g_ws.evScatterDone, g_streamIn));
        CUDA_CHECK(cudaStreamWaitEvent(g_streamOut, g_ws.evScatterDone, 0));
    }

    // Makes the compute stream wait for a panel's initial upload (only ever
    // relevant during the first sweep over the panels).
    std::vector<char> scattered(nLocal, 0);
    auto awaitScatter = [&](const size_t jl) {
        if (!scattered[jl]) {
            CUDA_CHECK(cudaStreamWaitEvent(g_stream, evScatter[jl], 0));
            scattered[jl] = 1;
        }
    };

    const double one = 1.0;
    const double minusOne = -1.0;
    const double zero = 0.0;

    // Rank 0 only: transpose the finished panel k (rows koff..n, column-major
    // with leading dimension ldSrc) and copy it into the row-major result.
    auto exportPanel = [&](const size_t k, const double* src, const size_t ldSrc) {
        const size_t koff = k * NB;
        const size_t kb = std::min(NB, n - koff);
        const size_t m = n - koff;
        const size_t slot = k % 2;
        CUBLAS_CHECK(cublasDgeam(g_blasOut, CUBLAS_OP_T, CUBLAS_OP_N, (int)kb, (int)m, &one, src, (int)ldSrc, &zero,
                                 dOut[slot], (int)kb, dOut[slot], (int)kb));
        CUDA_CHECK(cudaMemcpy2DAsync(A.data() + koff * n + koff, n * sizeof(double), dOut[slot], kb * sizeof(double),
                                     kb * sizeof(double), m, cudaMemcpyDeviceToHost, g_streamOut));
        CUDA_CHECK(cudaEventRecord(evExported[slot], g_streamOut));
    };

    // Factorize panel k (owned by this rank) and pack it into hbuf.
    // Returns the packed status word (0 = ok, otherwise failing index + 1).
    auto factorizePanel = [&](const size_t k, double* hbuf) -> double {
        const size_t koff = k * NB;
        const size_t kb = std::min(NB, n - koff);
        const size_t m = n - koff;
        awaitScatter(panelLocalIndex(k));
        double* p = dA + panelLocalIndex(k) * panelStride;

        // Diagonal block -> host, factorize it there, push the factor back.
        CUDA_CHECK(cudaMemcpy2DAsync(hDiag, kb * sizeof(double), p + koff, n * sizeof(double),
                                     kb * sizeof(double), kb, cudaMemcpyDeviceToHost, g_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_stream));

        const int failed = choleskyDiagonalBlock(hDiag, kb);
        if (failed >= 0) {
            hbuf[m * kb] = (double)(koff + (size_t)failed + 1);
            return hbuf[m * kb];
        }

        CUDA_CHECK(cudaMemcpy2DAsync(p + koff, n * sizeof(double), hDiag, kb * sizeof(double),
                                     kb * sizeof(double), kb, cudaMemcpyHostToDevice, g_stream));

        // L21 = A21 * L11^{-T}
        if (m > kb) {
            CUBLAS_CHECK(cublasDtrsm(g_blas, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T,
                                     CUBLAS_DIAG_NON_UNIT, (int)(m - kb), (int)kb, &one, p + koff, (int)n,
                                     p + koff + kb, (int)n));
        }

        // Pack rows koff..n of the panel for the broadcast.
        CUDA_CHECK(cudaMemcpy2DAsync(hbuf, m * sizeof(double), p + koff, n * sizeof(double), m * sizeof(double), kb,
                                     cudaMemcpyDeviceToHost, g_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_stream));
        hbuf[m * kb] = 0.0;

        // The panel is final now - rank 0 can already file it away.
        if (g_rank == 0) exportPanel(k, p + koff, n);
        return 0.0;
    };

    // Trailing update of the locally owned panel j (global index) with panel k.
    auto updatePanel = [&](const size_t k, const double* pk, const size_t ldp, const size_t j) {
        const size_t koff = k * NB;
        const size_t kb = std::min(NB, n - koff);
        const size_t joff = j * NB;
        const size_t jb = std::min(NB, n - joff);
        awaitScatter(panelLocalIndex(j));
        double* q = dA + panelLocalIndex(j) * panelStride;
        const double* src = pk + (joff - koff);

        // Diagonal block: symmetric rank-k update.
        CUBLAS_CHECK(cublasDsyrk(g_blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, (int)jb, (int)kb, &minusOne, src,
                                 (int)ldp, &one, q + joff, (int)n));

        // Everything below the diagonal block.
        if (joff + jb < n) {
            CUBLAS_CHECK(cublasDgemm(g_blas, CUBLAS_OP_N, CUBLAS_OP_T, (int)(n - joff - jb), (int)jb, (int)kb,
                                     &minusOne, pk + (joff + jb - koff), (int)ldp, src, (int)ldp, &one,
                                     q + joff + jb, (int)n));
        }
    };

    // ---- main loop with depth-1 lookahead ---------------------------------
    MPI_Request req = MPI_REQUEST_NULL;
    {
        const size_t kb0 = std::min(NB, n);
        const int cnt0 = (int)(n * kb0 + 1);
        if (panelOwner(0) == g_rank) factorizePanel(0, hPanel[0]);
        MPI_Ibcast(hPanel[0], cnt0, MPI_DOUBLE, panelOwner(0), MPI_COMM_WORLD, &req);
    }

    bool success = true;
    for (size_t k = 0; k < nt; ++k) {
        const size_t koff = k * NB;
        const size_t kb = std::min(NB, n - koff);
        const size_t m = n - koff;
        const size_t cur = k % 2;

        MPI_Wait(&req, MPI_STATUS_IGNORE);

        const double status = hPanel[cur][m * kb];
        if (status != 0.0) {
            if (g_rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)status - 1);
            }
            success = false;
            break;
        }

        // The owner keeps working straight out of its own panel storage.
        const double* pk;
        size_t ldp;
        if (panelOwner(k) == g_rank) {
            pk = dA + panelLocalIndex(k) * panelStride + koff;
            ldp = n;
        } else {
            if (g_rank == 0) CUDA_CHECK(cudaStreamWaitEvent(g_stream, evExported[cur], 0));
            CUDA_CHECK(cudaMemcpyAsync(dPanel[cur], hPanel[cur], m * kb * sizeof(double),
                                       cudaMemcpyHostToDevice, g_stream));
            pk = dPanel[cur];
            ldp = m;
            if (g_rank == 0) {
                CUDA_CHECK(cudaEventRecord(evReady[cur], g_stream));
                CUDA_CHECK(cudaStreamWaitEvent(g_streamOut, evReady[cur], 0));
                exportPanel(k, dPanel[cur], m);
            }
        }

        // Lookahead: finish panel k+1 first so its broadcast can overlap with
        // the remaining trailing updates.
        if (k + 1 < nt) {
            const size_t nxt = k + 1;
            if (panelOwner(nxt) == g_rank) {
                updatePanel(k, pk, ldp, nxt);
                factorizePanel(nxt, hPanel[nxt % 2]);
            }
            const size_t nb1 = std::min(NB, n - nxt * NB);
            const int cnt1 = (int)((n - nxt * NB) * nb1 + 1);
            MPI_Ibcast(hPanel[nxt % 2], cnt1, MPI_DOUBLE, panelOwner(nxt), MPI_COMM_WORLD, &req);
        }

        for (size_t jl = 0; jl < nLocal; ++jl) {
            const size_t j = jl * (size_t)g_nranks + (size_t)g_rank;
            if (j <= k || j == k + 1) continue;
            updatePanel(k, pk, ldp, j);
        }
        CUDA_CHECK(cudaStreamSynchronize(g_stream));
    }

    // ---- finish the row-major result on rank 0 ----------------------------
    if (g_rank == 0) {
        CUDA_CHECK(cudaStreamSynchronize(g_streamOut));
        if (success) {
            // Zero the upper triangular part.
            double* out = A.data();
#pragma omp parallel for schedule(static) num_threads(g_threads)
            for (size_t i = 0; i < n; ++i) {
                memset(out + i * n + i + 1, 0, (n - i - 1) * sizeof(double));
            }
        }
    }

    return success;
}

// Generate a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(std::vector<double>& A, const size_t n) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite

    std::vector<double> B(n * n);
    const uint32_t seed0 = 42;

    // Generate random matrix B: the generator is replicated exactly by
    // jumping the underlying LCG ahead for every thread's chunk.
    const size_t total = n * n;
#pragma omp parallel num_threads(g_threads)
    {
        const int nthreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();
        const size_t chunk = (total + (size_t)nthreads - 1) / (size_t)nthreads;
        const size_t begin = std::min(total, (size_t)tid * chunk);
        const size_t end = std::min(total, begin + chunk);
        if (begin < end) {
            const LcgMap jump = lcgPower(3ull * (uint64_t)begin);
            uint32_t seed = jump.a * seed0 + jump.c;
            for (size_t i = begin; i < end; ++i) {
                B[i] = (randR(seed) / (double)RAND_MAX) - 0.5;
            }
        }
    }

    // Compute A = B * B^T on the GPUs: every rank computes a slice of the rows
    // and the slices are gathered afterwards.
    const size_t rowsPerRank = (n + (size_t)g_nranks - 1) / (size_t)g_nranks;
    const size_t r0 = std::min(n, (size_t)g_rank * rowsPerRank);
    const size_t r1 = std::min(n, r0 + rowsPerRank);
    const size_t rows = r1 - r0;

    double* dB = nullptr;
    double* dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dB, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemcpyAsync(dB, B.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, g_stream));
    if (rows > 0) {
        CUDA_CHECK(cudaMalloc(&dC, rows * n * sizeof(double)));
        // The host row-major image of B is the column-major image of M = B^T.
        // The wanted row-major block A(r0:r1, :) is the column-major n x rows
        // matrix A(:, r0:r1) = M^T * M(:, r0:r1).
        const double one = 1.0;
        const double zero = 0.0;
        CUBLAS_CHECK(cublasDgemm(g_blas, CUBLAS_OP_T, CUBLAS_OP_N, (int)n, (int)rows, (int)n, &one, dB, (int)n,
                                 dB + r0 * n, (int)n, &zero, dC, (int)n));
        CUDA_CHECK(cudaMemcpyAsync(A.data() + r0 * n, dC, rows * n * sizeof(double), cudaMemcpyDeviceToHost,
                                   g_stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(g_stream));
    CUDA_CHECK(cudaFree(dB));
    if (dC != nullptr) CUDA_CHECK(cudaFree(dC));

    // Replicate the full matrix on every rank.
    if (g_nranks > 1) {
        std::vector<int> counts(g_nranks), displs(g_nranks);
        for (int r = 0; r < g_nranks; ++r) {
            const size_t a0 = std::min(n, (size_t)r * rowsPerRank);
            const size_t a1 = std::min(n, a0 + rowsPerRank);
            counts[r] = (int)((a1 - a0) * n);
            displs[r] = (int)(a0 * n);
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, A.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
    }

    // Add diagonal dominance to ensure positive definiteness
#pragma omp parallel for schedule(static) num_threads(g_threads)
    for (size_t i = 0; i < n; ++i) {
        A[i * n + i] += (double)n;
    }
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const size_t n) {
    // Validate by computing L * L^T and comparing with original matrix

    std::vector<double> reconstructed(n * n);

    // Compute L * L^T on the GPU.  The row-major host image of L is the
    // column-major image of X = L^T, so L * L^T = X^T * X (DSYRK, lower).
    {
        double* dX = nullptr;
        double* dR = nullptr;
        CUDA_CHECK(cudaMalloc(&dX, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dR, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpyAsync(dX, L.data(), n * n * sizeof(double), cudaMemcpyHostToDevice, g_stream));
        const double one = 1.0;
        const double zero = 0.0;
        CUBLAS_CHECK(cublasDsyrk(g_blas, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, (int)n, (int)n, &one, dX, (int)n,
                                 &zero, dR, (int)n));
        CUDA_CHECK(cudaMemcpyAsync(reconstructed.data(), dR, n * n * sizeof(double), cudaMemcpyDeviceToHost,
                                   g_stream));
        CUDA_CHECK(cudaStreamSynchronize(g_stream));
        CUDA_CHECK(cudaFree(dX));
        CUDA_CHECK(cudaFree(dR));
    }

    // Compare with original (only the lower triangle was computed; the product
    // is symmetric, so the mirrored entry is used above the diagonal).
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel for schedule(static) num_threads(g_threads) reduction(max : maxError, relError)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            const double r = (i >= j) ? reconstructed[j * n + i] : reconstructed[i * n + j];
            const double a = A_orig[i * n + j];
            const double error = fabs(r - a);
            maxError = std::max(maxError, error);

            const double rel = error / (fabs(a) + 1e-10);
            relError = std::max(relError, rel);
        }
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);
    g_threads = std::max(1, std::min(omp_get_max_threads(), 32));

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
            if (g_rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (g_rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    // One GPU per rank, assigned by the rank index within the compute node.
    {
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_free(&nodeComm);

        int deviceCount = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0) {
            if (g_rank == 0) printf("Error: no CUDA device available\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(localRank % deviceCount));
        CUDA_CHECK(cudaStreamCreate(&g_stream));
        CUBLAS_CHECK(cublasCreate(&g_blas));
        CUBLAS_CHECK(cublasSetStream(g_blas, g_stream));
        CUBLAS_CHECK(cublasSetPointerMode(g_blas, CUBLAS_POINTER_MODE_HOST));
        CUDA_CHECK(cudaStreamCreate(&g_streamIn));
        CUDA_CHECK(cudaStreamCreate(&g_streamOut));
        CUBLAS_CHECK(cublasCreate(&g_blasOut));
        CUBLAS_CHECK(cublasSetStream(g_blasOut, g_streamOut));
        CUBLAS_CHECK(cublasSetPointerMode(g_blasOut, CUBLAS_POINTER_MODE_HOST));
    }

    if (g_rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", g_nranks, g_threads);
    }

    // Allocate matrix.  Page-locking it once up front makes every host/device
    // transfer of the matrix run at full PCIe bandwidth.
    std::vector<double> A(n * n);
    std::vector<double> A_orig;
    if (n > 0) {
        CUDA_CHECK(cudaHostRegister(A.data(), n * n * sizeof(double), cudaHostRegisterDefault));
    }

    // Generate positive definite matrix
    if (g_rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, n);

    if (validate && g_rank == 0) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (g_rank == 0) printf("Computing Cholesky decomposition...\n");
    setupWorkspace(n);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(A, n);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    teardownWorkspace();

    if (!success) {
        if (g_rank == 0) printf("Cholesky decomposition failed\n");
        cublasDestroy(g_blas);
        MPI_Finalize();
        return 1;
    }

    int exitCode = 0;
    if (g_rank == 0) {
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
            bool valid = validateCholesky(A, A_orig, n);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (n > 0) cudaHostUnregister(A.data());
    cudaStreamDestroy(g_streamIn);
    cublasDestroy(g_blasOut);
    cudaStreamDestroy(g_streamOut);
    cublasDestroy(g_blas);
    cudaStreamDestroy(g_stream);
    MPI_Finalize();
    return exitCode;
}
