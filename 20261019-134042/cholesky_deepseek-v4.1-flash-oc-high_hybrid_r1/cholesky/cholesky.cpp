#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// Hybrid MPI + OpenMP + CUDA Cholesky decomposition.
//
// Rows are distributed cyclically over MPI ranks (row i -> rank i % P).
// Each rank keeps its local rows on the GPU in column-major order and the
// factor is produced with a blocked, left-looking algorithm:
//
//   * A tiled CUDA kernel assembles A = B * B^T (+ n*I) for the local rows.
//   * For a panel of CHOL_B columns [K, K+B) every rank packs the panel rows'
//     already known prefix (columns < K) plus the panel diagonal block of A
//     and exchanges it with a single compact MPI_Allgatherv.
//   * A CUDA kernel computes the partial dot products over the prefix for all
//     panel rows (replicated) and all trailing local rows.  This is the bulk
//     of the work and runs with many warps in flight.
//   * Rank 0 finishes the small B x B panel block on the host (using the GPU
//     prefix partial sums), broadcasts it, and the remaining of each panel
//     column is propagated to the trailing rows by short CUDA kernels.
//   * The tail of the next panel's prefix is read back from each owner's
//     local factor rows.
//
// The accumulation order of every entry is kept identical to the reference
// sequential implementation (groups of four products with ordered additions
// followed by an FMA tail), so the produced factor is bit-for-bit the same.
// OpenMP parallelizes the panel scatter, result assembly and validation.

#define CHOL_B 64
static_assert(CHOL_B % 4 == 0, "panel width must be a multiple of 4");

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err__));                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

#define MPI_CHECK(call)                                                         \
    do {                                                                        \
        int ierr__ = (call);                                                    \
        if (ierr__ != MPI_SUCCESS) {                                            \
            fprintf(stderr, "MPI error at %s:%d (code %d)\n", __FILE__,         \
                    __LINE__, ierr__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

// Exact replication of the reference accumulation: groups of four independent
// products whose rounded values are added in order, with a fused tail.
__device__ __forceinline__ double cholDotFrom(double acc, const double* a,
                                              long long astep, const double* b,
                                              long long bstep, int k0, int k1) {
    a += static_cast<long long>(k0) * astep;
    b += static_cast<long long>(k0) * bstep;
    int k = k0;
    for (; k + 3 < k1; k += 4) {
        const double p0 = __dmul_rn(a[0], b[0]);
        const double p1 = __dmul_rn(a[astep], b[bstep]);
        const double p2 = __dmul_rn(a[2 * astep], b[2 * bstep]);
        const double p3 = __dmul_rn(a[3 * astep], b[3 * bstep]);
        acc = __dadd_rn(acc, p0);
        acc = __dadd_rn(acc, p1);
        acc = __dadd_rn(acc, p2);
        acc = __dadd_rn(acc, p3);
        a += 4 * astep;
        b += 4 * bstep;
    }
    for (; k < k1; ++k) {
        acc = __fma_rn(a[0], b[0], acc);
        a += astep;
        b += bstep;
    }
    return acc;
}

// Assemble A = B * B^T (+ n*I on the diagonal) for this rank's local rows.
// Each thread owns one output element and accumulates over k in the exact
// reference order, while tiles of B are staged through shared memory.
#define GEN_TI 32
#define GEN_TJ 32
#define GEN_TK 32

__global__ void cholGenerateKernel(double* __restrict__ A,
                                   const double* __restrict__ B, int n, int m,
                                   int rank, int nprocs) {
    __shared__ double si[GEN_TI][GEN_TK];
    __shared__ double sj[GEN_TJ][GEN_TK];

    const int l0 = static_cast<int>(blockIdx.y) * GEN_TI;
    const int j0 = static_cast<int>(blockIdx.x) * GEN_TJ;
    const int ti = static_cast<int>(threadIdx.y);
    const int tj = static_cast<int>(threadIdx.x);
    const int l = l0 + ti;
    const int j = j0 + tj;

    double acc = 0.0;
    for (int base = 0; base < n; base += GEN_TK) {
        const int len = min(GEN_TK, n - base);
        for (int idx = ti * GEN_TJ + tj; idx < (GEN_TI + GEN_TJ) * GEN_TK;
             idx += GEN_TI * GEN_TJ) {
            if (idx < GEN_TI * GEN_TK) {
                const int r = idx / GEN_TK;
                const int kk = idx - r * GEN_TK;
                const int li = l0 + r;
                if (li < m && kk < len) {
                    const long long gi =
                        static_cast<long long>(rank) +
                        static_cast<long long>(li) *
                            static_cast<long long>(nprocs);
                    si[r][kk] = B[static_cast<size_t>(gi) *
                                      static_cast<size_t>(n) + base + kk];
                } else {
                    si[r][kk] = 0.0;
                }
            } else {
                const int t = idx - GEN_TI * GEN_TK;
                const int c = t / GEN_TK;
                const int kk = t - c * GEN_TK;
                const int cj = j0 + c;
                if (cj < n && kk < len) {
                    sj[c][kk] = B[static_cast<size_t>(cj) *
                                      static_cast<size_t>(n) + base + kk];
                } else {
                    sj[c][kk] = 0.0;
                }
            }
        }
        __syncthreads();
        if (l < m && j < n) {
            acc = cholDotFrom(acc, &si[ti][0], 1, &sj[tj][0], 1, 0, len);
        }
        __syncthreads();
    }

    if (l < m && j < n) {
        const long long gi = static_cast<long long>(rank) +
                             static_cast<long long>(l) *
                                 static_cast<long long>(nprocs);
        if (gi == static_cast<long long>(j)) {
            acc += static_cast<double>(n);
        }
        A[static_cast<size_t>(j) * static_cast<size_t>(m) +
          static_cast<size_t>(l)] = acc;
    }
}

// Partial sums over the already finished prefix columns [0, K) for the
// trailing local rows.  Several rows share one block so the panel data is
// reused from the L1 cache.
#define CHOL_ROWS 4

__global__ void cholSKernel(const double* __restrict__ L,
                            const double* __restrict__ pT,
                            double* __restrict__ S, int m, int K, int Bact,
                            int localRows, int lStart) {
    const int tid = static_cast<int>(threadIdx.x);
    const int blk = static_cast<int>(blockIdx.x);
    if (blk < Bact) {
        const int b = tid;
        const int bp = blk;
        if (b >= Bact || bp < b) {
            return;
        }
        const double acc =
            cholDotFrom(0.0, pT + bp, CHOL_B, pT + b, CHOL_B, 0, K);
        S[static_cast<size_t>(bp) * CHOL_B + b] = acc;
        return;
    }
    const int b = tid % CHOL_B;
    if (b >= Bact) {
        return;
    }
    const int r = tid / CHOL_B;
    const int l = lStart + (blk - Bact) * CHOL_ROWS + r;
    if (l >= localRows) {
        return;
    }
    const double acc =
        cholDotFrom(0.0, L + static_cast<size_t>(l), m, pT + b, CHOL_B, 0, K);
    S[(static_cast<size_t>(CHOL_B) + static_cast<size_t>(l)) * CHOL_B + b] =
        acc;
}

// Pack the panel rows owned by this rank (prefix and diagonal block) into a
// compact, contiguous buffer (one host transfer per panel).
__global__ void cholPackKernel(const double* __restrict__ L,
                               double* __restrict__ pack, int m, int nprocs,
                               int K, int Bact, int firstRow, int countRows,
                               int rowStride) {
    const int total = countRows * rowStride;
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const int t = idx / rowStride;
    const int off = idx - t * rowStride;
    const int b = firstRow + nprocs * t;
    const int row = K + b;
    const int li = row / nprocs;
    if (off < K) {
        pack[idx] = L[static_cast<size_t>(off) * m + static_cast<size_t>(li)];
    } else {
        const int col = off - K;
        if (col > b) {
            pack[idx] = 0.0;
        } else {
            pack[idx] = L[static_cast<size_t>(K + col) * m +
                          static_cast<size_t>(li)];
        }
    }
}

// Write the replicated panel rows of the factor back to the owning rank's
// local device rows.
__global__ void cholPanelWriteKernel(double* __restrict__ L,
                                     const double* __restrict__ pT, int m,
                                     int nprocs, int rank, int K, int Bact) {
    const int total = Bact * Bact;
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                    static_cast<int>(threadIdx.x);
    if (idx >= total) {
        return;
    }
    const int bp = idx / Bact;
    const int b = idx - bp * Bact;
    if (b > bp) {
        return;
    }
    const long long gi = static_cast<long long>(K) + bp;
    if (static_cast<int>(gi % nprocs) != rank) {
        return;
    }
    L[static_cast<size_t>(K + b) * m + static_cast<size_t>(gi / nprocs)] =
        pT[static_cast<size_t>(K + b) * CHOL_B + bp];
}

// Update one panel column (global column j = K + b) for all trailing local
// rows.  Panel rows are handled on the host.
__global__ void cholColumnKernel(double* __restrict__ L,
                                 const double* __restrict__ pT,
                                 const double* __restrict__ S, int m, int K,
                                 int b, int lStart, int total, double diag) {
    const int t = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) +
                  static_cast<int>(threadIdx.x);
    if (t >= total) {
        return;
    }
    const int l = lStart + t;
    const int j = K + b;
    double acc = K > 0 ? S[(static_cast<size_t>(CHOL_B) + static_cast<size_t>(l)) *
                               CHOL_B + b]
                       : 0.0;
    acc = cholDotFrom(acc, L + static_cast<size_t>(l), m, pT + b, CHOL_B, K, j);
    const double val =
        (L[static_cast<size_t>(j) * m + static_cast<size_t>(l)] - acc) / diag;
    L[static_cast<size_t>(j) * m + static_cast<size_t>(l)] = val;
}

// Exact host-side arithmetic (separate rounded multiply/add plus a fused
// tail) used to reproduce the reference summation bit-for-bit.
static inline double cholMulRN(double a, double b) {
#if defined(__x86_64__) || defined(__i386__)
    double r = a;
    __asm__("mulsd %1, %0" : "+x"(r) : "x"(b));
    return r;
#else
    volatile double r = a * b;
    return r;
#endif
}

static inline double cholAddRN(double a, double b) {
#if defined(__x86_64__) || defined(__i386__)
    double r = a;
    __asm__("addsd %1, %0" : "+x"(r) : "x"(b));
    return r;
#else
    volatile double r = a + b;
    return r;
#endif
}

// Dot product of panel rows r1 and r2 over columns [K, j), continuing from the
// prefix partial sum computed on the GPU.
static double panelDot(const double* block, const double* Sp, int K, int j,
                       int r1, int r2) {
    double acc = K > 0 ? Sp[static_cast<size_t>(r1) * CHOL_B + r2] : 0.0;
    int k = K;
    const double* b1 = block + static_cast<size_t>(r1) * CHOL_B;
    const double* b2 = block + static_cast<size_t>(r2) * CHOL_B;
    for (; k + 3 < j; k += 4) {
        const double v0 = cholMulRN(b1[0], b2[0]);
        const double v1 = cholMulRN(b1[1], b2[1]);
        const double v2 = cholMulRN(b1[2], b2[2]);
        const double v3 = cholMulRN(b1[3], b2[3]);
        acc = cholAddRN(acc, v0);
        acc = cholAddRN(acc, v1);
        acc = cholAddRN(acc, v2);
        acc = cholAddRN(acc, v3);
        b1 += 4;
        b2 += 4;
    }
    for (; k < j; ++k) {
        acc = std::fma(*b1, *b2, acc);
        ++b1;
        ++b2;
    }
    return acc;
}

// Factor the Bact x Bact diagonal block of the current panel on the host.
// Columns are processed in order (each only touches the short within-panel
// range), so this stays cheap and avoids per-column thread-team overhead.
static void hostPanelFactor(double* Lblk, const double* Sp, const double* Asub,
                            int K, int Bact) {
    for (int b = 0; b < Bact; ++b) {
        const int j = K + b;
        const double acc = panelDot(Lblk, Sp, K, j, b, b);
        const double v = Asub[static_cast<size_t>(b) * CHOL_B + b] - acc;
        const double d = v > 0.0 ? std::sqrt(v) : -1.0;
        Lblk[static_cast<size_t>(b) * CHOL_B + b] = d;
        if (b + 1 >= Bact) {
            continue;
        }
        for (int bp = b + 1; bp < Bact; ++bp) {
            const double a = panelDot(Lblk, Sp, K, j, bp, b);
            Lblk[static_cast<size_t>(bp) * CHOL_B + b] =
                (Asub[static_cast<size_t>(bp) * CHOL_B + b] - a) / d;
        }
    }
}

static inline int localRowCount(size_t n, int r, int nprocs) {
    const size_t rn = static_cast<size_t>(r);
    if (rn >= n) {
        return 0;
    }
    return static_cast<int>((n - rn + static_cast<size_t>(nprocs) - 1) /
                            static_cast<size_t>(nprocs));
}

// Generate the random matrix B used to form A = B * B^T + n*I.  The seed and
// generation order match the reference implementation.
static void generateRandomMatrix(std::vector<double>& B, size_t n) {
    B.resize(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }
}

// Blocked, distributed, GPU-resident Cholesky.  The panel diagonal blocks are
// factored on the host (in exact reference arithmetic) and broadcast once per
// panel; all trailing updates run on the GPUs.
static bool choleskyDecomposition(double* d_L, double* d_pT, double* d_S,
                                  double* d_pack, double* h_reduce,
                                  double* h_pack, double* h_block,
                                  double* h_pt, double* h_Spanel,
                                  double* h_send, size_t n, int rank,
                                  int nprocs, int localRows,
                                  int scatterThreads) {
    if (n == 0) {
        return true;
    }

    const int ni = static_cast<int>(n);
    const int m = localRows;

    bool ok = true;
    int K = 0;
    while (K < ni) {
        const int Bact = std::min(CHOL_B, ni - K);

        // Gather panel rows' prefix [0,K) and the panel diagonal block of A.
        const int prefixCount = K * CHOL_B;
        const int rowStride = K + Bact;

        const int base = K % nprocs;
        int rem = (rank - base) % nprocs;
        if (rem < 0) {
            rem += nprocs;
        }
        int firstRow = 0;
        int countRows = 0;
        if (rem < Bact) {
            firstRow = rem;
            countRows = (Bact - 1 - rem) / nprocs + 1;
        }

        const int packCount = countRows * rowStride;
        const int packBlocks = (packCount + 255) / 256;
        if (packBlocks > 0) {
            cholPackKernel<<<packBlocks, 256>>>(d_L, d_pack, m, nprocs, K, Bact,
                                               firstRow, countRows, rowStride);
            CUDA_CHECK(cudaMemcpy(h_send, d_pack,
                                  static_cast<size_t>(packCount) *
                                      sizeof(double),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> gcounts(nprocs, 0);
        std::vector<int> gdispls(nprocs, 0);
        {
            int total = 0;
            for (int r = 0; r < nprocs; ++r) {
                int rr = (r - base) % nprocs;
                if (rr < 0) {
                    rr += nprocs;
                }
                int cnt = 0;
                if (rr < Bact) {
                    cnt = (Bact - 1 - rr) / nprocs + 1;
                }
                gcounts[r] = cnt * rowStride;
                gdispls[r] = total;
                total += gcounts[r];
            }
        }
        MPI_CHECK(MPI_Allgatherv(
            countRows > 0 ? h_send : nullptr, countRows * rowStride, MPI_DOUBLE,
            h_pack, gcounts.data(), gdispls.data(), MPI_DOUBLE, MPI_COMM_WORLD));

        // Scatter the gathered panel data into the layouts used on the device.
        // For panel row b, its owner is (K+b)%P and its local index is b/P.
#pragma omp parallel for schedule(static) num_threads(scatterThreads)
        for (int b = 0; b < Bact; ++b) {
            const int owner = (K + b) % nprocs;
            const int t = b / nprocs;
            const double* src = h_pack + static_cast<size_t>(gdispls[owner]) +
                                static_cast<size_t>(t) * rowStride;
            for (int k = 0; k < K; ++k) {
                h_reduce[static_cast<size_t>(k) * CHOL_B + b] = src[k];
            }
            double* rowA = h_reduce + static_cast<size_t>(prefixCount) +
                           static_cast<size_t>(b) * CHOL_B;
            for (int col = 0; col <= b; ++col) {
                rowA[col] = src[K + col];
            }
        }

        if (K > 0) {
            CUDA_CHECK(cudaMemcpy(d_pT, h_reduce,
                                  static_cast<size_t>(prefixCount) *
                                      sizeof(double),
                                  cudaMemcpyHostToDevice));
        }

        // Index of the first trailing row of this panel.
        const int need = K + Bact - rank;
        int lStart = need > 0 ? (need + nprocs - 1) / nprocs : 0;
        lStart = std::min(lStart, m);

        // Prefix partial sums for the trailing rows (runs in parallel with the
        // host panel factorization below).
        if (K > 0) {
            const int nrows = m - lStart;
            const int sblocks =
                Bact + (nrows + CHOL_ROWS - 1) / CHOL_ROWS;
            cholSKernel<<<sblocks, CHOL_B * CHOL_ROWS>>>(
                d_L, d_pT, d_S, m, K, Bact, m, lStart);
        }

        // Factor the panel's diagonal block on the host, on rank 0 only, then
        // replicate it (one collective per panel).
        if (rank == 0) {
            if (K > 0) {
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(
                    h_Spanel, d_S,
                    static_cast<size_t>(Bact) * CHOL_B * sizeof(double),
                    cudaMemcpyDeviceToHost));
            }
            hostPanelFactor(h_block, h_Spanel, h_reduce + prefixCount, K,
                            Bact);
            for (int b = 0; b < Bact; ++b) {
                for (int bp = 0; bp < Bact; ++bp) {
                    h_pt[static_cast<size_t>(b) * CHOL_B + bp] =
                        bp >= b ? h_block[static_cast<size_t>(bp) * CHOL_B + b]
                                : 0.0;
                }
            }
        }
        MPI_CHECK(MPI_Bcast(h_pt, Bact * CHOL_B, MPI_DOUBLE, 0,
                            MPI_COMM_WORLD));

        CUDA_CHECK(cudaMemcpy(
            d_pT + static_cast<size_t>(K) * CHOL_B, h_pt,
            static_cast<size_t>(Bact) * CHOL_B * sizeof(double),
            cudaMemcpyHostToDevice));

        int localBad = INT_MAX;
        for (int b = 0; b < Bact; ++b) {
            if (!(h_pt[static_cast<size_t>(b) * CHOL_B + b] > 0.0)) {
                localBad = K + b;
                break;
            }
        }
        int globalBad = INT_MAX;
        MPI_CHECK(MPI_Allreduce(&localBad, &globalBad, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD));
        if (globalBad != INT_MAX) {
            if (rank == 0) {
                printf("Error: Matrix is not positive definite at diagonal element %d\n",
                       globalBad);
            }
            ok = false;
            break;
        }

        // Publish the replicated panel rows into each owner's local rows.
        const int pwBlocks = (Bact * Bact + 255) / 256;
        cholPanelWriteKernel<<<pwBlocks, 256>>>(d_L, d_pT, m, nprocs, rank, K,
                                                Bact);

        // Complete the trailing part of every panel column.
        const int total = m - lStart;
        if (total > 0) {
            const int blocks = (total + 127) / 128;
            for (int b = 0; b < Bact; ++b) {
                cholColumnKernel<<<blocks, 128>>>(
                    d_L, d_pT, d_S, m, K, b, lStart, total,
                    h_pt[static_cast<size_t>(b) * CHOL_B + b]);
            }
        }

        K += Bact;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    return ok;
}

// Gather cyclic row blocks on the root and rebuild the full row-major matrix.
// For the factor the strict upper triangle is zeroed.
static void gatherRows(const std::vector<double>& localRowsData, size_t n,
                       int nprocs, int rank, const std::vector<int>& counts,
                       const std::vector<int>& displs, std::vector<double>& full,
                       std::vector<double>& scratch, bool zeroUpper) {
    const int localCount = static_cast<int>(localRowsData.size());
    double* send = localCount > 0 ? const_cast<double*>(localRowsData.data())
                                  : nullptr;

    if (rank == 0) {
        scratch.resize(static_cast<size_t>(n) * n);
    }
    double* recv = rank == 0 ? scratch.data() : nullptr;

    MPI_CHECK(MPI_Gatherv(send, localCount, MPI_DOUBLE, recv,
                          rank == 0 ? counts.data() : nullptr,
                          rank == 0 ? displs.data() : nullptr, MPI_DOUBLE, 0,
                          MPI_COMM_WORLD));

    if (rank != 0) {
        return;
    }

    full.assign(static_cast<size_t>(n) * n, 0.0);
#pragma omp parallel for schedule(static) num_threads(8)
    for (long long row = 0; row < static_cast<long long>(n); ++row) {
        const int r = static_cast<int>(row % nprocs);
        const size_t li = static_cast<size_t>(row) / static_cast<size_t>(nprocs);
        const double* src = scratch.data() + displs[r] + li * n;
        double* dst = full.data() + static_cast<size_t>(row) * n;
        if (!zeroUpper) {
            std::memcpy(dst, src, n * sizeof(double));
            continue;
        }
        const size_t lower = static_cast<size_t>(row) + 1;
        std::memcpy(dst, src, lower * sizeof(double));
        if (lower < n) {
            std::memset(dst + lower, 0, (n - lower) * sizeof(double));
        }
    }
}

static bool validateCholesky(const std::vector<double>& L,
                             const std::vector<double>& A_orig, size_t n,
                             int threads) {
    double maxError = 0.0;
    double relError = 0.0;

#pragma omp parallel num_threads(threads) reduction(max : maxError, relError)
    {
        std::vector<double> reconstructed(n);
#pragma omp for schedule(static)
        for (long long i = 0; i < static_cast<long long>(n); ++i) {
            const double* Li = &L[static_cast<size_t>(i) * n];
            for (size_t j = 0; j < n; ++j) {
                const double* Lj = &L[j * n];
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) {
                    sum += Li[k] * Lj[k];
                }
                reconstructed[j] = sum;
            }
            for (size_t j = 0; j < n; ++j) {
                const double ref = A_orig[static_cast<size_t>(i) * n + j];
                const double error = fabs(reconstructed[j] - ref);
                maxError = std::max(maxError, error);
                const double rel = error / (fabs(ref) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
    }

    printf("Max absolute error: %.10e\n", maxError);
    printf("Max relative error: %.10e\n", relError);

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
    MPI_Init(&argc, &argv);

    int rank = 0;
    int nprocs = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &nodeComm);
    int localRank = rank;
    int localSize = nprocs;
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &localSize);
    }

    int threads = omp_get_max_threads();
    if (getenv("OMP_NUM_THREADS") == nullptr) {
        threads = std::max(1, omp_get_num_procs() / std::max(1, localSize));
        threads = std::min(threads, 64);
        omp_set_num_threads(threads);
    }
    const int scatterThreads = std::max(1, std::min(threads, 8));

    size_t n = 512;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool badOption = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n = static_cast<size_t>(atoll(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            showHelp = true;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
            }
            badOption = true;
            break;
        }
    }

    if (showHelp || badOption) {
        if (rank == 0) {
            printUsage(argv[0]);
        }
        if (nodeComm != MPI_COMM_NULL) {
            MPI_Comm_free(&nodeComm);
        }
        MPI_Finalize();
        return badOption ? 1 : 0;
    }

    // CUDA is required: the factorization always runs on the GPU.
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA device available\n");
        }
        if (nodeComm != MPI_COMM_NULL) {
            MPI_Comm_free(&nodeComm);
        }
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int localRows = localRowCount(n, rank, nprocs);
    const size_t localElems = static_cast<size_t>(localRows) * n;

    double* d_L = nullptr;
    double* d_pT = nullptr;
    double* d_S = nullptr;
    double* d_pack = nullptr;
    double* h_reduce = nullptr;
    double* h_pack = nullptr;
    double* h_block = nullptr;
    double* h_pt = nullptr;
    double* h_Spanel = nullptr;
    double* h_send = nullptr;

    if (rank == 0) {
        printf("Generating positive definite matrix...\n");
    }
    std::vector<double> Bmat;
    generateRandomMatrix(Bmat, n);

    const size_t localBytes = static_cast<size_t>(localRows) * n * sizeof(double);
    if (localBytes > 0) {
        CUDA_CHECK(cudaMalloc(&d_L, localBytes));
    }
    if (n > 0) {
        double* d_B = nullptr;
        CUDA_CHECK(cudaMalloc(&d_B, n * n * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(d_B, Bmat.data(), n * n * sizeof(double),
                              cudaMemcpyHostToDevice));
        const int gx = static_cast<int>((n + GEN_TJ - 1) / GEN_TJ);
        const int gy = static_cast<int>((localRows + GEN_TI - 1) / GEN_TI);
        if (gx > 0 && gy > 0) {
            cholGenerateKernel<<<dim3(gx, gy), dim3(GEN_TJ, GEN_TI)>>>(
                d_L, d_B, static_cast<int>(n), localRows, rank, nprocs);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(d_B));
    }
    std::vector<double>().swap(Bmat);

    std::vector<double> A_orig;
    if (validate && localBytes > 0) {
        // Kept in column-major layout; converted while gathering.
        A_orig.resize(static_cast<size_t>(localRows) * n);
        CUDA_CHECK(cudaMemcpy(A_orig.data(), d_L, localBytes,
                              cudaMemcpyDeviceToHost));
    }

    if (n > 0) {
        CUDA_CHECK(cudaMalloc(&d_pT, n * CHOL_B * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_S,
                              (static_cast<size_t>(CHOL_B) +
                               static_cast<size_t>(localRows)) *
                                  CHOL_B * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_pack,
                              (static_cast<size_t>(CHOL_B) * (n + CHOL_B)) *
                                  sizeof(double)));
        h_reduce = static_cast<double*>(
            std::malloc((n * CHOL_B + CHOL_B * CHOL_B) * sizeof(double)));
        h_pack = static_cast<double*>(std::malloc(
            (static_cast<size_t>(CHOL_B) * (n + CHOL_B)) * sizeof(double)));
        h_block = static_cast<double*>(
            std::malloc(CHOL_B * CHOL_B * sizeof(double)));
        h_pt = static_cast<double*>(
            std::malloc(CHOL_B * CHOL_B * sizeof(double)));
        h_Spanel = static_cast<double*>(
            std::malloc(CHOL_B * CHOL_B * sizeof(double)));
        h_send = static_cast<double*>(std::malloc(
            (static_cast<size_t>(CHOL_B) * (n + CHOL_B)) * sizeof(double)));
        if (!h_reduce || !h_pack || !h_block || !h_pt || !h_Spanel || !h_send) {
            fprintf(stderr, "Host allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    if (rank == 0) {
        printf("Computing Cholesky decomposition...\n");
    }

    const auto start = std::chrono::steady_clock::now();
    const bool success = choleskyDecomposition(
        d_L, d_pT, d_S, d_pack, h_reduce, h_pack, h_block, h_pt, h_Spanel,
        h_send, n, rank, nprocs, localRows, scatterThreads);
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    const auto end = std::chrono::steady_clock::now();

    const double local_seconds = std::chrono::duration<double>(end - start).count();
    double seconds = 0.0;
    MPI_CHECK(MPI_Reduce(&local_seconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));

    if (rank == 0) {
        printf("Computation time: %ld ms\n",
               static_cast<long>(seconds * 1000.0));
        const double ops = static_cast<double>(n) * static_cast<double>(n) *
                           static_cast<double>(n) / 3.0;
        const double gflops = seconds > 0.0 ? ops / seconds / 1e9 : 0.0;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    if (!success) {
        if (rank == 0) {
            printf("Cholesky decomposition failed\n");
        }
        if (h_send) std::free(h_send);
        if (h_Spanel) std::free(h_Spanel);
        if (h_pt) std::free(h_pt);
        if (h_block) std::free(h_block);
        if (h_pack) std::free(h_pack);
        if (h_reduce) std::free(h_reduce);
        if (d_pack) cudaFree(d_pack);
        if (d_S) cudaFree(d_S);
        if (d_pT) cudaFree(d_pT);
        if (d_L) cudaFree(d_L);
        if (nodeComm != MPI_COMM_NULL) {
            MPI_Comm_free(&nodeComm);
        }
        MPI_Finalize();
        return 1;
    }

    // Read back the local factor (column-major on the device), transposing the
    // lower triangle into row-major form for gathering.  Only needed when a
    // result or validation is requested.
    std::vector<double> localL;
    std::vector<double> A_orig_rows;
    if (printResults || validate) {
        std::vector<double> coldata(localElems);
        if (localBytes > 0) {
            CUDA_CHECK(cudaMemcpy(coldata.data(), d_L, localBytes,
                                  cudaMemcpyDeviceToHost));
        }
        localL.assign(localElems, 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
        for (int l = 0; l < localRows; ++l) {
            const size_t gi = static_cast<size_t>(rank) +
                              static_cast<size_t>(nprocs) *
                                  static_cast<size_t>(l);
            for (size_t j = 0; j <= gi; ++j) {
                localL[static_cast<size_t>(l) * n + j] =
                    coldata[j * static_cast<size_t>(localRows) +
                            static_cast<size_t>(l)];
            }
        }
        std::vector<double>().swap(coldata);

        if (validate) {
            A_orig_rows.assign(localElems, 0.0);
#pragma omp parallel for schedule(static) num_threads(threads)
            for (int l = 0; l < localRows; ++l) {
                for (size_t j = 0; j < n; ++j) {
                    A_orig_rows[static_cast<size_t>(l) * n + j] =
                        A_orig[j * static_cast<size_t>(localRows) +
                               static_cast<size_t>(l)];
                }
            }
            std::vector<double>().swap(A_orig);
        }
    }

    // Free device buffers that are no longer needed.
    if (d_pack) cudaFree(d_pack);
    if (d_S) cudaFree(d_S);
    if (d_pT) cudaFree(d_pT);
    if (d_L) cudaFree(d_L);

    bool valid = true;
    if (printResults || validate) {
        std::vector<int> counts(nprocs, 0);
        std::vector<int> displs(nprocs, 0);
        int total = 0;
        for (int r = 0; r < nprocs; ++r) {
            counts[r] = localRowCount(n, r, nprocs) * static_cast<int>(n);
            displs[r] = total;
            total += counts[r];
        }

        std::vector<double> fullL;
        std::vector<double> scratch;
        gatherRows(localL, n, nprocs, rank, counts, displs, fullL, scratch, true);

        if (validate) {
            std::vector<double> fullA;
            std::vector<double> scratchA;
            gatherRows(A_orig_rows, n, nprocs, rank, counts, displs, fullA,
                       scratchA, false);
            if (rank == 0) {
                printf("Validating result...\n");
                valid = validateCholesky(fullL, fullA, n, threads);
                if (valid) {
                    printf("Validation: PASSED\n");
                } else {
                    printf("Validation: FAILED\n");
                }
            }
        }

        if (printResults && rank == 0) {
            print_results(fullL, "CholeskyL");
        }

    }

    if (h_send) std::free(h_send);
    if (h_Spanel) std::free(h_Spanel);
    if (h_pt) std::free(h_pt);
    if (h_block) std::free(h_block);
    if (h_pack) std::free(h_pack);
    if (h_reduce) std::free(h_reduce);
    if (nodeComm != MPI_COMM_NULL) {
        MPI_Comm_free(&nodeComm);
    }

    MPI_Finalize();
    return (rank == 0 && !valid) ? 1 : 0;
}
