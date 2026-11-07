#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed right-looking blocked Cholesky decomposition (MPI).
//
// Decomposes a symmetric positive definite matrix A into L * L^T where L is
// lower triangular.
//
// Data distribution: rows are grouped into chunks of RB consecutive rows and
// chunks are dealt out to the ranks in a reflected cyclic fashion
// (0,1,..,P-1,P-1,..,1,0,0,1,..). Every rank stores its rows (all n columns,
// row-major) contiguously in ascending global order. The fine granularity
// keeps the triangular workload balanced at every step; the algorithmic
// block size bs is chosen independently.
//
// For each block column k (columns [k0, k1)):
//   1. the rows of the diagonal block are collected and the small bs x bs
//      block is factored redundantly on every rank,
//   2. every rank solves the triangular system for its own rows below the
//      diagonal block (TRSM),
//   3. the resulting panel is exchanged (MPI shared-memory windows when all
//      ranks share a node, nonblocking all-gather otherwise, see Exchange),
//   4. every rank applies the rank-bs update to its own trailing rows (the
//      bulk of the n^3/3 flops) with a cache-blocked, register-blocked
//      kernel.
// The exchange of panel k+1 is overlapped with the trailing update of
// block k (lookahead).

namespace {

constexpr size_t RB = 4; // rows per distribution chunk (= kernel row block)

typedef double v4d __attribute__((vector_size(32)));

inline v4d loadv(const double* p) {
    v4d v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}
inline void storev(double* p, v4d v) { std::memcpy(p, &v, sizeof(v)); }

struct Dist {
    size_t n = 0;       // matrix size
    size_t bs = 0;      // algorithmic block size
    size_t nblocks = 0; // number of algorithmic blocks
    size_t nchunks = 0; // number of row chunks
    int P = 1;          // number of ranks
    int rank = 0;

    size_t blockRows(size_t b) const { return std::min(bs, n - b * bs); }
    size_t chunkRows(size_t c) const { return std::min(RB, n - c * RB); }

    // Reflected cyclic ownership of chunks.
    int ownerOf(size_t c) const {
        const size_t round = c / (size_t)P;
        const size_t pos = c % (size_t)P;
        return (int)((round & 1) ? ((size_t)P - 1 - pos) : pos);
    }
    // global chunk index of local chunk lc on rank r (may be >= nchunks)
    size_t globalChunk(int r, size_t lc) const {
        const size_t pos = (lc & 1) ? ((size_t)P - 1 - (size_t)r) : (size_t)r;
        return lc * (size_t)P + pos;
    }
    size_t localChunk(size_t c) const { return c / (size_t)P; }
    // local row index of the first row of local chunk lc
    size_t localRow(size_t lc) const { return lc * RB; }
    // first local chunk of rank r containing rows >= g
    size_t firstLocalChunkFrom(int r, size_t g) const {
        if (g >= n) return numLocalChunks(r);
        const size_t c = g / RB;
        size_t lc = c / (size_t)P; // all earlier rounds are < c
        while (globalChunk(r, lc) < c) ++lc;
        return lc;
    }
    size_t numLocalChunks(int r) const {
        size_t cnt = 0;
        while (globalChunk(r, cnt) < nchunks) ++cnt;
        return cnt;
    }
    // number of chunks of rank r with global chunk index >= c0
    size_t numLocalChunksFrom(int r, size_t c0) const {
        const size_t total = numLocalChunks(r);
        const size_t first = firstLocalChunkFrom(r, c0 * RB);
        return first < total ? total - first : 0;
    }
    size_t numLocalRows(int r) const {
        size_t rows = 0;
        for (size_t lc = 0, c; (c = globalChunk(r, lc)) < nchunks; ++lc) rows += chunkRows(c);
        return rows;
    }
    // number of rows of rank r with global index in [lo, hi)
    size_t numLocalRowsIn(int r, size_t lo, size_t hi) const {
        size_t rows = 0;
        for (size_t lc = firstLocalChunkFrom(r, lo), c; (c = globalChunk(r, lc)) < nchunks; ++lc) {
            const size_t g0 = c * RB;
            if (g0 >= hi) break;
            rows += std::min(hi, g0 + chunkRows(c)) - std::max(lo, g0);
        }
        return rows;
    }
};

size_t chooseBlockSize(size_t n) {
    size_t bs = 64;
    // Keep a few steps for the lookahead pipeline on small matrices.
    while (bs > 16 && (n + bs - 1) / bs < 4) bs /= 2;
    // bs must be a multiple of RB unless the matrix is a single block.
    return bs >= n ? n : bs;
}

// Panel layout: the panel of block k (columns [k0, k1) of the rows >= k1) is
// exchanged in tiles, one per row chunk: tile t covers rows k1 + RB*t ..
// k1 + RB*t + RB - 1 and stores bsk x RB values as tile[c * RB + r].
// tilePtr[t] points to tile t, wherever it lives (gathered buffer or another
// rank's shared-memory segment). The kernels read the panel in place.

inline v4d bcast(double x) {
    v4d v = {x, x, x, x};
    return v;
}
// Keep a vector value in a register so the compiler does not fold the load
// into every FMA as a memory operand (which makes the kernel load-port bound).
#define KEEP_IN_REGISTER(v) asm volatile("" : "+x"(v))

// Register-blocked rank-bsk update of four consecutive rows:
//   a_r[j] -= sum_c l_r[c] * L[k1 + j, k0 + c]   for j in [jb, je)
// Processes 4 rows x 12 columns (three tiles) with 12 vector accumulators,
// then 4 x 8, 4 x 4 and scalar remainders.
inline void updateRows4(double* __restrict a0, double* __restrict a1,
                        double* __restrict a2, double* __restrict a3,
                        const double* __restrict l0, const double* __restrict l1,
                        const double* __restrict l2, const double* __restrict l3,
                        const double* const* __restrict tilePtr,
                        size_t bsk, size_t jb, size_t je) {
    size_t j = jb;
    // leading columns up to a tile boundary
    for (; j < je && (j % RB) != 0; ++j) {
        const double* pt = tilePtr[j / RB] + (j % RB);
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (size_t c = 0; c < bsk; ++c, pt += RB) {
            const double p = *pt;
            s0 += l0[c] * p; s1 += l1[c] * p; s2 += l2[c] * p; s3 += l3[c] * p;
        }
        a0[j] -= s0; a1[j] -= s1; a2[j] -= s2; a3[j] -= s3;
    }
    for (; j + 3 * RB <= je; j += 3 * RB) {
        v4d c00 = {0, 0, 0, 0}, c01 = c00, c02 = c00, c10 = c00, c11 = c00, c12 = c00;
        v4d c20 = c00, c21 = c00, c22 = c00, c30 = c00, c31 = c00, c32 = c00;
        const double* pa = tilePtr[j / RB];
        const double* pb = tilePtr[j / RB + 1];
        const double* pc = tilePtr[j / RB + 2];
        for (size_t c = 0; c < bsk; ++c, pa += RB, pb += RB, pc += RB) {
            v4d p0 = loadv(pa), p1 = loadv(pb), p2 = loadv(pc);
            KEEP_IN_REGISTER(p0); KEEP_IN_REGISTER(p1); KEEP_IN_REGISTER(p2);
            v4d x;
            x = bcast(l0[c]); c00 += x * p0; c01 += x * p1; c02 += x * p2;
            x = bcast(l1[c]); c10 += x * p0; c11 += x * p1; c12 += x * p2;
            x = bcast(l2[c]); c20 += x * p0; c21 += x * p1; c22 += x * p2;
            x = bcast(l3[c]); c30 += x * p0; c31 += x * p1; c32 += x * p2;
        }
        storev(a0 + j, loadv(a0 + j) - c00); storev(a0 + j + 4, loadv(a0 + j + 4) - c01); storev(a0 + j + 8, loadv(a0 + j + 8) - c02);
        storev(a1 + j, loadv(a1 + j) - c10); storev(a1 + j + 4, loadv(a1 + j + 4) - c11); storev(a1 + j + 8, loadv(a1 + j + 8) - c12);
        storev(a2 + j, loadv(a2 + j) - c20); storev(a2 + j + 4, loadv(a2 + j + 4) - c21); storev(a2 + j + 8, loadv(a2 + j + 8) - c22);
        storev(a3 + j, loadv(a3 + j) - c30); storev(a3 + j + 4, loadv(a3 + j + 4) - c31); storev(a3 + j + 8, loadv(a3 + j + 8) - c32);
    }
    if (j + 2 * RB <= je) {
        v4d c00 = {0, 0, 0, 0}, c01 = c00, c10 = c00, c11 = c00;
        v4d c20 = c00, c21 = c00, c30 = c00, c31 = c00;
        const double* pa = tilePtr[j / RB];
        const double* pb = tilePtr[j / RB + 1];
        for (size_t c = 0; c < bsk; ++c, pa += RB, pb += RB) {
            v4d p0 = loadv(pa), p1 = loadv(pb);
            KEEP_IN_REGISTER(p0); KEEP_IN_REGISTER(p1);
            v4d x;
            x = bcast(l0[c]); c00 += x * p0; c01 += x * p1;
            x = bcast(l1[c]); c10 += x * p0; c11 += x * p1;
            x = bcast(l2[c]); c20 += x * p0; c21 += x * p1;
            x = bcast(l3[c]); c30 += x * p0; c31 += x * p1;
        }
        storev(a0 + j, loadv(a0 + j) - c00); storev(a0 + j + 4, loadv(a0 + j + 4) - c01);
        storev(a1 + j, loadv(a1 + j) - c10); storev(a1 + j + 4, loadv(a1 + j + 4) - c11);
        storev(a2 + j, loadv(a2 + j) - c20); storev(a2 + j + 4, loadv(a2 + j + 4) - c21);
        storev(a3 + j, loadv(a3 + j) - c30); storev(a3 + j + 4, loadv(a3 + j + 4) - c31);
        j += 2 * RB;
    }
    if (j + RB <= je) { // j is tile aligned here
        v4d c0 = {0, 0, 0, 0}, c1 = c0, c2 = c0, c3 = c0;
        const double* pa = tilePtr[j / RB];
        for (size_t c = 0; c < bsk; ++c, pa += RB) {
            v4d p0 = loadv(pa);
            KEEP_IN_REGISTER(p0);
            c0 += bcast(l0[c]) * p0; c1 += bcast(l1[c]) * p0;
            c2 += bcast(l2[c]) * p0; c3 += bcast(l3[c]) * p0;
        }
        storev(a0 + j, loadv(a0 + j) - c0);
        storev(a1 + j, loadv(a1 + j) - c1);
        storev(a2 + j, loadv(a2 + j) - c2);
        storev(a3 + j, loadv(a3 + j) - c3);
        j += RB;
    }
    for (; j < je; ++j) {
        const double* pt = tilePtr[j / RB] + (j % RB);
        double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (size_t c = 0; c < bsk; ++c, pt += RB) {
            const double p = *pt;
            s0 += l0[c] * p; s1 += l1[c] * p; s2 += l2[c] * p; s3 += l3[c] * p;
        }
        a0[j] -= s0; a1[j] -= s1; a2[j] -= s2; a3[j] -= s3;
    }
}

inline void updateRows1(double* __restrict a0, const double* __restrict l0,
                        const double* const* __restrict tilePtr,
                        size_t bsk, size_t jb, size_t je) {
    for (size_t j = jb; j < je; ++j) {
        const double* pt = tilePtr[j / RB] + (j % RB);
        double s0 = 0;
        for (size_t c = 0; c < bsk; ++c, pt += RB) s0 += l0[c] * (*pt);
        a0[j] -= s0;
    }
}

// Apply the rank-bsk update from panel kb (columns [k0, k1), tiles indexed
// by (j - k1) / RB) to the owned rows with global index in [rlo, rhi) (rlo and
// rhi are chunk aligned or equal to n), restricted to global columns
// [jb, min(je, last row of the chunk + 1)). Entries above the diagonal may
// receive garbage; the strictly upper triangle is zeroed at the end of the
// factorization. The columns are processed in blocks small enough for the
// panel slice to stay in L2 while all rows stream through it. `poll()` is
// called after each chunk.
template <typename Poll>
void updateTrailingRows(std::vector<double>& local, const Dist& d, const double* const* tilePtr,
                        size_t k0, size_t k1, size_t bsk, size_t rlo,
                        size_t rhi, size_t jb, size_t je, Poll poll) {
    const size_t n = d.n;
    constexpr size_t L2_BYTES = 192 * 1024;
    size_t CB = L2_BYTES / (bsk * sizeof(double));
    CB = std::max<size_t>(3 * RB, CB - CB % (3 * RB));
    for (size_t jcb = jb; jcb < je; jcb += CB) {
        const size_t jce = std::min(je, jcb + CB);
        // rows below rhi are needed only from the first row with g >= jcb
        const size_t rstart = std::max(rlo, jcb - jcb % RB);
        for (size_t lc = d.firstLocalChunkFrom(d.rank, rstart), c; (c = d.globalChunk(d.rank, lc)) < d.nchunks; ++lc) {
            const size_t g0 = c * RB;
            if (g0 >= rhi) break;
            const size_t cr = d.chunkRows(c);
            const size_t lrow = d.localRow(lc);
            const size_t cjb = jcb - k1;
            const size_t cje = std::min(jce, g0 + cr) - k1; // columns [jcb, min(jce, last row + 1))
            if (cje <= cjb) continue;
            if (cr == RB) {
                double* a0 = local.data() + lrow * n + k1;
                const double* l0 = local.data() + lrow * n + k0;
                updateRows4(a0, a0 + n, a0 + 2 * n, a0 + 3 * n, l0, l0 + n, l0 + 2 * n, l0 + 3 * n,
                            tilePtr, bsk, cjb, cje);
            } else {
                for (size_t r = 0; r < cr; ++r) {
                    double* a0 = local.data() + (lrow + r) * n + k1;
                    const double* l0 = local.data() + (lrow + r) * n + k0;
                    updateRows1(a0, l0, tilePtr, bsk, cjb, cje);
                }
            }
            poll();
        }
    }
}

// Unblocked Cholesky of the bsk x bsk block D (leading dimension bs), in place.
// Only the lower triangle is read and written. Returns false if not SPD.
bool factorDiagonalBlock(double* D, size_t bs, size_t bsk, size_t k0, bool report) {
    for (size_t j = 0; j < bsk; ++j) {
        double* rowj = D + j * bs;
        double sum = 0.0;
        for (size_t c = 0; c < j; ++c) sum += rowj[c] * rowj[c];
        const double val = rowj[j] - sum;
        if (val <= 0.0) {
            if (report)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", k0 + j);
            return false;
        }
        const double djj = sqrt(val);
        rowj[j] = djj;
        const double inv = 1.0 / djj;
        for (size_t i = j + 1; i < bsk; ++i) {
            double* rowi = D + i * bs;
            double s = 0.0;
            for (size_t c = 0; c < j; ++c) s += rowi[c] * rowj[c];
            rowi[j] = (rowi[j] - s) * inv;
        }
    }
    return true;
}

// Pack the column segment [k0, k0 + w) of the owned rows with global index in
// [lo, hi) into buf (row-major, w values per row). Returns rows packed.
size_t packRows(const std::vector<double>& local, const Dist& d, size_t lo, size_t hi, size_t k0,
                size_t w, double* buf) {
    const size_t n = d.n;
    size_t pr = 0;
    for (size_t lc = d.firstLocalChunkFrom(d.rank, lo), c; (c = d.globalChunk(d.rank, lc)) < d.nchunks; ++lc) {
        const size_t g0 = c * RB;
        if (g0 >= hi) break;
        const size_t rb = std::max(lo, g0), re = std::min(hi, g0 + d.chunkRows(c));
        for (size_t g = rb; g < re; ++g, ++pr)
            std::memcpy(buf + pr * w, local.data() + (d.localRow(lc) + g - g0) * n + k0, w * sizeof(double));
    }
    return pr;
}

// Allgatherv layout for rows in [lo, hi), w values per row.
void gatherLayout(const Dist& d, size_t lo, size_t hi, size_t w, std::vector<int>& counts,
                  std::vector<int>& displs) {
    int off = 0;
    for (int r = 0; r < d.P; ++r) {
        counts[r] = (int)(d.numLocalRowsIn(r, lo, hi) * w);
        displs[r] = off;
        off += counts[r];
    }
}

// Visit the gathered rows (rank-major, each rank's rows ascending) of global
// rows [lo, hi) in global order: f(g, srcRowPointer).
template <typename F>
void forEachGatheredRow(const Dist& d, size_t lo, size_t hi, size_t w, const double* recv,
                        const std::vector<int>& displs, F f) {
    for (int r = 0; r < d.P; ++r) {
        const double* src = recv + displs[r];
        size_t pr = 0;
        for (size_t lc = d.firstLocalChunkFrom(r, lo), c; (c = d.globalChunk(r, lc)) < d.nchunks; ++lc) {
            const size_t g0 = c * RB;
            if (g0 >= hi) break;
            const size_t rb = std::max(lo, g0), re = std::min(hi, g0 + d.chunkRows(c));
            for (size_t g = rb; g < re; ++g, ++pr) f(g, src + pr * w);
        }
    }
}

// Gather the diagonal block rows [k0, k1) (columns [k0, k1)) into D
// (leading dimension bs, global row order).
void unpackDiagonalBlock(const Dist& d, size_t k0, size_t k1, size_t bsk, const double* recv,
                         const std::vector<int>& displs, double* D) {
    forEachGatheredRow(d, k0, k1, bsk, recv, displs, [&](size_t g, const double* row) {
        std::memcpy(D + (g - k0) * d.bs, row, bsk * sizeof(double));
    });
}

// Exchange of the diagonal block rows and of the panel tiles between ranks.
//
// Generic path: the diagonal block rows are all-gathered (blocking, tiny);
// every rank packs the tiles of its rows >= k1 into a send buffer and the
// tiles are all-gathered nonblocking, polled during the trailing update.
//
// Single-node path: when all ranks share a node, the owners write the
// diagonal block rows into a small shared-memory window and every rank
// writes its panel tiles into its own segment of a shared window; the other
// ranks read them in place after a barrier, nothing is copied. Two panel
// windows alternate so that a rank writing panel k+1 never overwrites tiles
// of panel k that another rank is still reading.
class Exchange {
  public:
    Exchange(const Dist& d) : d_(d) {
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, d.rank, MPI_INFO_NULL, &nodeComm_);
        int nodeSize = 1;
        MPI_Comm_size(nodeComm_, &nodeSize);
        shared_ = (nodeSize == d.P);
        const size_t segElems = std::max<size_t>(1, d.numLocalChunks(d.rank) * d.bs * RB);
        if (shared_) {
            MPI_Info info;
            MPI_Info_create(&info);
            MPI_Info_set(info, "alloc_shared_noncontig", "true"); // segments local to each rank
            for (int w = 0; w < NWIN; ++w) {
                double* base = nullptr;
                MPI_Win_allocate_shared((MPI_Aint)(segElems * sizeof(double)), sizeof(double), info,
                                        nodeComm_, &base, &win_[w]);
                MPI_Win_lock_all(MPI_MODE_NOCHECK, win_[w]);
                segBase_[w].resize(d.P);
                for (int r = 0; r < d.P; ++r) {
                    MPI_Aint sz;
                    int disp;
                    MPI_Win_shared_query(win_[w], r, &sz, &disp, &segBase_[w][r]);
                }
            }
            MPI_Info_free(&info);
            // diagonal block window: one segment, allocated by rank 0
            double* base = nullptr;
            MPI_Win_allocate_shared((MPI_Aint)((d.rank == 0 ? d.bs * d.bs : 0) * sizeof(double)),
                                    sizeof(double), MPI_INFO_NULL, nodeComm_, &base, &diagWin_);
            MPI_Win_lock_all(MPI_MODE_NOCHECK, diagWin_);
            MPI_Aint sz;
            int disp;
            MPI_Win_shared_query(diagWin_, 0, &sz, &disp, &diagShared_);
        } else {
            sendBuf_.resize(segElems);
            for (int b = 0; b < 2; ++b) recvBuf_[b].resize(std::max<size_t>(1, d.nchunks * d.bs * RB));
            counts_.resize(d.P);
            displs_.resize(d.P);
            sendDiag_.resize(std::max<size_t>(1, d.bs * d.bs));
            recvDiag_.resize(std::max<size_t>(1, d.bs * d.bs));
        }
        for (int b = 0; b < 2; ++b) tilePtr_[b].resize(std::max<size_t>(1, d.nchunks));
    }
    ~Exchange() {
        if (shared_) {
            for (int w = 0; w < NWIN; ++w) {
                MPI_Win_unlock_all(win_[w]);
                MPI_Win_free(&win_[w]);
            }
            MPI_Win_unlock_all(diagWin_);
            MPI_Win_free(&diagWin_);
        }
        MPI_Comm_free(&nodeComm_);
    }
    Exchange(const Exchange&) = delete;
    Exchange& operator=(const Exchange&) = delete;

    // Publish the owned rows of the diagonal block [k0, k1) (columns [k0, k1)).
    void diagPut(const std::vector<double>& local, size_t k0, size_t k1, size_t bsk) {
        if (shared_) {
            const size_t n = d_.n, bs = d_.bs;
            for (size_t lc = d_.firstLocalChunkFrom(d_.rank, k0), c; (c = d_.globalChunk(d_.rank, lc)) < d_.nchunks; ++lc) {
                const size_t g0 = c * RB;
                if (g0 >= k1) break;
                const size_t rb = std::max(k0, g0), re = std::min(k1, g0 + d_.chunkRows(c));
                for (size_t g = rb; g < re; ++g)
                    std::memcpy(diagShared_ + (g - k0) * bs, local.data() + (d_.localRow(lc) + g - g0) * n + k0,
                                bsk * sizeof(double));
            }
            MPI_Win_sync(diagWin_);
        } else {
            gatherLayout(d_, k0, k1, bsk, counts_, displs_);
            packRows(local, d_, k0, k1, k0, bsk, sendDiag_.data());
        }
    }

    // Collect the diagonal block [k0, k1) into D (leading dimension bs).
    void diagGet(size_t k0, size_t k1, size_t bsk, double* D) {
        if (shared_) {
            MPI_Barrier(nodeComm_);
            MPI_Win_sync(diagWin_);
            std::memcpy(D, diagShared_, (k1 - k0) * d_.bs * sizeof(double));
        } else {
            MPI_Allgatherv(sendDiag_.data(), counts_[d_.rank], MPI_DOUBLE, recvDiag_.data(), counts_.data(),
                           displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            unpackDiagonalBlock(d_, k0, k1, bsk, recvDiag_.data(), displs_, D);
        }
    }

    // Destination of the tile of local chunk lc for the panel of step `step`
    // (rows >= k1, bsk columns).
    double* tileDst(size_t step, size_t lc, size_t k1, size_t bsk) {
        if (shared_) return segBase_[step % NWIN][d_.rank] + lc * bsk * RB;
        return sendBuf_.data() + (lc - d_.firstLocalChunkFrom(d_.rank, k1)) * bsk * RB;
    }

    // All tiles of this rank have been written: start the exchange.
    void panelStart(size_t step, size_t k1, size_t bsk) {
        if (shared_) {
            MPI_Win_sync(win_[step % NWIN]);
        } else {
            int off = 0;
            for (int r = 0; r < d_.P; ++r) {
                counts_[r] = (int)(d_.numLocalChunksFrom(r, k1 / RB) * bsk * RB);
                displs_[r] = off;
                off += counts_[r];
            }
            MPI_Iallgatherv(sendBuf_.data(), counts_[d_.rank], MPI_DOUBLE, recvBuf_[step % 2].data(),
                            counts_.data(), displs_.data(), MPI_DOUBLE, MPI_COMM_WORLD, &req_);
        }
    }

    void panelProgress() {
        if (shared_) return;
        int done = 0;
        MPI_Test(&req_, &done, MPI_STATUS_IGNORE);
    }

    // Complete the exchange; returns the tile pointer table (tile t = rows
    // k1 + RB*t ...).
    const double* const* panelFinish(size_t step, size_t k1, size_t bsk) {
        const size_t c0 = k1 / RB;
        std::vector<const double*>& tp = tilePtr_[step % 2];
        if (shared_) {
            MPI_Barrier(nodeComm_);
            MPI_Win_sync(win_[step % NWIN]);
            const std::vector<double*>& base = segBase_[step % NWIN];
            for (size_t c = c0; c < d_.nchunks; ++c)
                tp[c - c0] = base[d_.ownerOf(c)] + d_.localChunk(c) * bsk * RB;
        } else {
            MPI_Wait(&req_, MPI_STATUS_IGNORE);
            const double* recv = recvBuf_[step % 2].data();
            for (int r = 0; r < d_.P; ++r) {
                size_t off = (size_t)displs_[r];
                for (size_t lc = d_.firstLocalChunkFrom(r, k1), c; (c = d_.globalChunk(r, lc)) < d_.nchunks; ++lc) {
                    tp[c - c0] = recv + off;
                    off += bsk * RB;
                }
            }
        }
        return tp.data();
    }

  private:
    static constexpr int NWIN = 2;
    const Dist& d_;
    MPI_Comm nodeComm_ = MPI_COMM_NULL;
    bool shared_ = false;
    MPI_Win win_[NWIN] = {MPI_WIN_NULL, MPI_WIN_NULL};
    std::vector<double*> segBase_[NWIN];
    MPI_Win diagWin_ = MPI_WIN_NULL;
    double* diagShared_ = nullptr;
    std::vector<double> sendBuf_;
    std::vector<double> recvBuf_[2];
    std::vector<int> counts_, displs_;
    std::vector<double> sendDiag_, recvDiag_;
    std::vector<const double*> tilePtr_[2];
    MPI_Request req_ = MPI_REQUEST_NULL;
};

// Write the factored diagonal block back into the owned rows.
void storeDiagonalBlock(std::vector<double>& local, const Dist& d, size_t k0, size_t k1, size_t bsk,
                        const double* D) {
    const size_t n = d.n;
    for (size_t lc = d.firstLocalChunkFrom(d.rank, k0), c; (c = d.globalChunk(d.rank, lc)) < d.nchunks; ++lc) {
        const size_t g0 = c * RB;
        if (g0 >= k1) break;
        const size_t rb = std::max(k0, g0), re = std::min(k1, g0 + d.chunkRows(c));
        for (size_t g = rb; g < re; ++g)
            std::memcpy(local.data() + (d.localRow(lc) + g - g0) * n + k0, D + (g - k0) * d.bs,
                        bsk * sizeof(double));
    }
}

// Triangular solve of the owned rows >= k1 against the diagonal block D of
// columns [k0, k1), writing the results as tiles (one bsk x RB tile per
// chunk, missing rows of a partial chunk zero filled) to tileDst(lc).
// Column-oriented form, four rows at a time, so the inner loop vectorizes:
//   for m: x_r = row_r[m] / D[m][m];  row_r[cc] -= x_r * D[cc][m]  (cc > m)
template <typename TileDst>
void trsmAndPackPanel(std::vector<double>& local, const Dist& d, size_t k0, size_t k1, size_t bsk,
                      const double* D, std::vector<double>& DT, TileDst tileDst) {
    const size_t n = d.n, bs = d.bs;
    // DT[m * bs + cc] = D[cc][m] for cc > m, zero elsewhere; DT[m * bs + m] = 1 / D[m][m]
    DT.assign(bs * bsk, 0.0);
    for (size_t m = 0; m < bsk; ++m) {
        DT[m * bs + m] = 1.0 / D[m * bs + m];
        for (size_t cc = m + 1; cc < bsk; ++cc) DT[m * bs + cc] = D[cc * bs + m];
    }
    auto solveRows = [&](double* const* rows, size_t nr) {
        for (size_t m = 0; m < bsk; ++m) {
            const double* __restrict dt = DT.data() + m * bs;
            const double inv = dt[m];
            if (nr == RB) {
                double* __restrict r0 = rows[0];
                double* __restrict r1 = rows[1];
                double* __restrict r2 = rows[2];
                double* __restrict r3 = rows[3];
                const double x0 = r0[m] * inv, x1 = r1[m] * inv, x2 = r2[m] * inv, x3 = r3[m] * inv;
                r0[m] = x0; r1[m] = x1; r2[m] = x2; r3[m] = x3;
                for (size_t cc = m + 1; cc < bsk; ++cc) {
                    const double t = dt[cc];
                    r0[cc] -= x0 * t; r1[cc] -= x1 * t; r2[cc] -= x2 * t; r3[cc] -= x3 * t;
                }
            } else {
                for (size_t r = 0; r < nr; ++r) {
                    double* __restrict row = rows[r];
                    const double x = row[m] * inv;
                    row[m] = x;
                    for (size_t cc = m + 1; cc < bsk; ++cc) row[cc] -= x * dt[cc];
                }
            }
        }
    };
    for (size_t lc = d.firstLocalChunkFrom(d.rank, k1), c; (c = d.globalChunk(d.rank, lc)) < d.nchunks; ++lc) {
        const size_t cr = d.chunkRows(c);
        double* rows[RB];
        for (size_t r = 0; r < cr; ++r) rows[r] = local.data() + (d.localRow(lc) + r) * n + k0;
        solveRows(rows, cr);
        double* tile = tileDst(lc);
        if (cr == RB) {
            for (size_t cc = 0; cc < bsk; ++cc) {
                tile[cc * RB + 0] = rows[0][cc];
                tile[cc * RB + 1] = rows[1][cc];
                tile[cc * RB + 2] = rows[2][cc];
                tile[cc * RB + 3] = rows[3][cc];
            }
        } else {
            std::fill(tile, tile + bsk * RB, 0.0);
            for (size_t r = 0; r < cr; ++r)
                for (size_t cc = 0; cc < bsk; ++cc) tile[cc * RB + r] = rows[r][cc];
        }
    }
}

} // namespace

// Returns true on success (collective; the result is consistent on all ranks).
bool choleskyDecomposition(std::vector<double>& local, const Dist& d) {
    const size_t n = d.n, bs = d.bs;
    const int rank = d.rank;
    const bool report = (rank == 0);

    std::vector<double> D(bs * bs), Dnext(bs * bs);
    std::vector<double> DT;
    Exchange xchg(d);
    const double* const* tilePtr = nullptr;

    // Prologue: gather and factor diagonal block 0, solve and gather panel 0.
    {
        const size_t bsk = d.blockRows(0);
        xchg.diagPut(local, 0, bsk, bsk);
        xchg.diagGet(0, bsk, bsk, D.data());
        if (!factorDiagonalBlock(D.data(), bs, bsk, 0, report)) return false;
        storeDiagonalBlock(local, d, 0, bsk, bsk, D.data());
        if (d.nblocks > 1) {
            trsmAndPackPanel(local, d, 0, bsk, bsk, D.data(), DT,
                             [&](size_t lc) { return xchg.tileDst(0, lc, bsk, bsk); });
            xchg.panelStart(0, bsk, bsk);
            tilePtr = xchg.panelFinish(0, bsk, bsk);
        }
    }

    for (size_t kb = 0; kb + 1 < d.nblocks; ++kb) {
        const size_t k0 = kb * bs;
        const size_t bsk = d.blockRows(kb);
        const size_t k1 = k0 + bsk;
        const size_t bskn = d.blockRows(kb + 1);
        const size_t k2 = k1 + bskn;
        const auto noPoll = [] {};

        // (a) Update the rows of the next diagonal block (columns [k1, k2))
        //     with panel kb and publish them.
        updateTrailingRows(local, d, tilePtr, k0, k1, bsk, k1, k2, k1, k2, noPoll);
        xchg.diagPut(local, k1, k2, bskn);

        // (b) Narrow update: own rows >= k2, columns [k1, k2).
        updateTrailingRows(local, d, tilePtr, k0, k1, bsk, k2, n, k1, k2, noPoll);

        // (c) Factor the next diagonal block (redundantly on every rank), then
        //     solve panel kb+1 and start its exchange.
        xchg.diagGet(k1, k2, bskn, Dnext.data());
        if (!factorDiagonalBlock(Dnext.data(), bs, bskn, k1, report)) return false;
        storeDiagonalBlock(local, d, k1, k2, bskn, Dnext.data());
        if (k2 >= n) break; // that was the last block

        trsmAndPackPanel(local, d, k1, k2, bskn, Dnext.data(), DT,
                         [&](size_t lc) { return xchg.tileDst(kb + 1, lc, k2, bskn); });
        xchg.panelStart(kb + 1, k2, bskn);

        // (d) Wide update: own rows >= k2, columns [k2, g]. Poll the gather
        //     between chunks so that it progresses in the background.
        updateTrailingRows(local, d, tilePtr, k0, k1, bsk, k2, n, k2, n, [&] { xchg.panelProgress(); });

        // (e) Finish the exchange of panel kb+1.
        tilePtr = xchg.panelFinish(kb + 1, k2, bskn);
        D.swap(Dnext);
    }

    // Zero out the strictly upper triangular part of the owned rows.
    for (size_t lc = 0, c; (c = d.globalChunk(rank, lc)) < d.nchunks; ++lc) {
        const size_t cr = d.chunkRows(c);
        for (size_t r = 0; r < cr; ++r) {
            const size_t g = c * RB + r;
            double* row = local.data() + (d.localRow(lc) + r) * n;
            std::fill(row + g + 1, row + n, 0.0);
        }
    }
    return true;
}

// Generate the owned rows of a symmetric positive definite matrix
// A = B * B^T + n*I, with B a deterministic pseudo-random matrix.
void generatePositiveDefiniteMatrix(std::vector<double>& local, const Dist& d) {
    const size_t n = d.n;
    std::vector<double> B(n * n);
    unsigned int seed = 42;
    for (size_t i = 0; i < n * n; ++i) {
        B[i] = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
    }

    for (size_t lc = 0, c; (c = d.globalChunk(d.rank, lc)) < d.nchunks; ++lc) {
        const size_t cr = d.chunkRows(c);
        for (size_t r = 0; r < cr; ++r) {
            const size_t i = c * RB + r;
            const double* Bi = B.data() + i * n;
            double* Ai = local.data() + (d.localRow(lc) + r) * n;
            for (size_t j = 0; j < n; ++j) {
                const double* Bj = B.data() + j * n;
                double sum = 0.0;
                for (size_t k = 0; k < n; ++k) sum += Bi[k] * Bj[k];
                Ai[j] = sum;
            }
            Ai[i] += n;
        }
    }
}

// Gather the distributed matrix into a full row-major matrix on every rank
// (allRanks) or only on rank 0. Returns the full matrix (empty elsewhere).
std::vector<double> gatherMatrix(const std::vector<double>& local, const Dist& d, bool allRanks) {
    const size_t n = d.n;
    std::vector<int> counts(d.P), displs(d.P);
    gatherLayout(d, 0, n, n, counts, displs);
    const bool haveFull = allRanks || d.rank == 0;
    std::vector<double> packed(haveFull ? n * n : 0);
    if (allRanks) {
        MPI_Allgatherv(local.data(), counts[d.rank], MPI_DOUBLE, packed.data(),
                       counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    } else {
        MPI_Gatherv(local.data(), counts[d.rank], MPI_DOUBLE, packed.data(),
                    counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    std::vector<double> full;
    if (haveFull) {
        full.resize(n * n);
        forEachGatheredRow(d, 0, n, n, packed.data(), displs, [&](size_t g, const double* row) {
            std::memcpy(full.data() + g * n, row, n * sizeof(double));
        });
    }
    return full;
}

bool validateCholesky(const std::vector<double>& localL, const std::vector<double>& localA,
                      const Dist& d) {
    // Validate by computing L * L^T (own rows) and comparing with original.
    const size_t n = d.n;
    std::vector<double> L = gatherMatrix(localL, d, true);

    double maxError = 0.0;
    double relError = 0.0;
    for (size_t lc = 0, c; (c = d.globalChunk(d.rank, lc)) < d.nchunks; ++lc) {
        const size_t cr = d.chunkRows(c);
        for (size_t r = 0; r < cr; ++r) {
            const size_t i = c * RB + r;
            const double* Li = L.data() + i * n;
            const double* Ai = localA.data() + (d.localRow(lc) + r) * n;
            for (size_t j = 0; j < n; ++j) {
                const double* Lj = L.data() + j * n;
                const size_t kmax = std::min(i, j) + 1; // L is lower triangular
                double sum = 0.0;
                for (size_t k = 0; k < kmax; ++k) sum += Li[k] * Lj[k];
                const double error = fabs(sum - Ai[j]);
                maxError = std::max(maxError, error);
                const double rel = error / (fabs(Ai[j]) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
    }

    double errs[2] = {maxError, relError};
    double gerrs[2];
    MPI_Allreduce(errs, gerrs, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", gerrs[0]);
        printf("Max relative error: %.10e\n", gerrs[1]);
    }

    if (gerrs[1] > 1e-6) {
        if (d.rank == 0) printf("Validation failed: relative error too large\n");
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
    int rank = 0, P = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &P);

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
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", P);
    }

    Dist d;
    d.n = n;
    d.P = P;
    d.rank = rank;
    d.bs = chooseBlockSize(n);
    d.nblocks = (n + d.bs - 1) / d.bs;
    d.nchunks = (n + RB - 1) / RB;

    // Allocate owned rows
    std::vector<double> A(std::max<size_t>(1, d.numLocalRows(rank) * n));
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, d);

    if (validate) {
        A_orig = A; // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) { printf("Computing Cholesky decomposition...\n"); fflush(stdout); }
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    bool success = choleskyDecomposition(A, d);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const long durationMs = (long)((end - start) * 1000.0);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", durationMs);
        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full = gatherMatrix(A, d, false);
        if (rank == 0) print_results(full, "CholeskyL");
    }

    // Validation
    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, d);
        if (rank == 0) {
            if (valid) printf("Validation: PASSED\n");
            else printf("Validation: FAILED\n");
        }
        ret = valid ? 0 : 1;
    }

    MPI_Finalize();
    return ret;
}
