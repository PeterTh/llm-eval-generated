#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed Cholesky decomposition (MPI, blocked, right-looking with lookahead)
// Decomposes positive definite matrix A into L * L^T where L is lower triangular
//
// Rows are distributed block-cyclically (blocks of NB rows) across ranks. Each
// element L[i][j] is computed exactly as in the sequential algorithm:
//     sum = 0; for k < j: sum += L[i][k] * L[j][k]; L[i][j] = (A[i][j] - sum) / L[j][j]
// with the same summation order and rounding, so results are bit-identical to
// the sequential version. The partial sums over finished column panels are kept
// in a separate accumulator S (updated panel by panel in ascending k order), the
// remainder of each sum is finished when its column panel is computed.
//
// Communication: ranks on the same node access each other's rows directly
// through an MPI-3 shared memory window, synchronised by per-block / per-rank
// ready flags. Between nodes, every finished diagonal block and every column
// panel piece is sent once to a designated receiver rank on each other node,
// which stores it in a node-wide mirror of the remote rows.

static constexpr size_t NB = 32;  // panel width / row block size
static constexpr size_t JR = 6;   // columns j per micro-kernel (i is vectorized by 8)
static constexpr size_t DB = 8;   // panels per batched (far-field) update
static constexpr size_t KB = DB * NB;

typedef double v4 __attribute__((vector_size(32)));
typedef double v4u __attribute__((vector_size(32), aligned(8)));

// The reference (sequential) build evaluates  sum += x[k] * y[k]  for k < len as:
//   - k < 4*floor(len/4): rounded product, then in-order addition (auto-vectorized body)
//   - remaining k:        fused multiply-add (scalar tail)
// The helpers below reproduce exactly this rounding so results are bit-identical.
template <typename T>
static inline T noContract(T p) {
    asm("" : "+x"(p));  // keep the product rounded (no FMA contraction)
    return p;
}

static inline double dotContinue(double sum, const double* x, const double* y, size_t kfrom, size_t len) {
    const size_t kv = len & ~(size_t)3;
    size_t k = kfrom;
    for (; k < kv; ++k) sum += noContract(x[k] * y[k]);
    for (; k < len; ++k) sum = __builtin_fma(x[k], y[k], sum);
    return sum;
}

// Row blocks are distributed in a reflected ("snake") cyclic order over ranks,
// counted from the bottom of the matrix (where the work is): with b' = nblocks-1-b,
// cycle c = b' / size holds ranks 0..size-1 for even c and size-1..0 for odd c.
// This balances the triangular work per rank; a partial cycle lands on the
// cheap top rows.
static size_t posInCycle(size_t c, int r, int size) {
    return (c & 1) ? (size_t)(size - 1 - r) : (size_t)r;
}

// Number of blocks of rank r with reversed index b' < y
static size_t countBelowRev(size_t y, int r, int size) {
    const size_t c = y / (size_t)size;
    return c + (c * (size_t)size + posInCycle(c, r, size) < y ? 1 : 0);
}

static size_t countLocalRows(size_t n, int rank, int size) {
    const size_t nblocks = (n + NB - 1) / NB;
    size_t rows = countBelowRev(nblocks, rank, size) * NB;
    // The last block (b' = 0, always owned by rank 0) may be partial
    if (rank == 0 && n % NB != 0) rows -= NB - n % NB;
    return rows;
}

// out[q] = dotContinue(0, x, y[q], 0, len) for q < 4, interleaved for instruction-level parallelism
static inline void dot4(const double* x, const double* const* y, size_t len, double* out) {
    const size_t kv = len & ~(size_t)3;
    double s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0;
    size_t k = 0;
    for (; k < kv; ++k) {
        const double xk = x[k];
        s0 += noContract(xk * y[0][k]);
        s1 += noContract(xk * y[1][k]);
        s2 += noContract(xk * y[2][k]);
        s3 += noContract(xk * y[3][k]);
    }
    for (; k < len; ++k) {
        s0 = __builtin_fma(x[k], y[0][k], s0);
        s1 = __builtin_fma(x[k], y[1][k], s1);
        s2 = __builtin_fma(x[k], y[2][k], s2);
        s3 = __builtin_fma(x[k], y[3][k], s3);
    }
    out[0] = s0;
    out[1] = s1;
    out[2] = s2;
    out[3] = s3;
}

struct Dist {
    int rank = 0, size = 1;
    size_t n = 0, nblocks = 0, lrows = 0;
    int owner(size_t b) const {
        const size_t br = nblocks - 1 - b;
        return (int)posInCycle(br / (size_t)size, (int)(br % (size_t)size), size);
    }
    size_t blockRows(size_t b) const { return std::min(NB, n - b * NB); }
    size_t ownedBlocks(int r) const { return countBelowRev(nblocks, r, size); }
    // Local row index (on rank r) at which the first owned block with index > b starts.
    // Local blocks are stored in ascending global order; only the very last one can be partial.
    size_t localStartAfter(size_t b, int r, size_t rows) const {
        const size_t cnt = ownedBlocks(r) - countBelowRev(nblocks - 1 - b, r, size);  // owned blocks <= b
        return std::min(cnt * NB, rows);
    }
    size_t localStartAfter(size_t b) const { return localStartAfter(b, rank, lrows); }
    size_t localRowOfBlock(size_t b) const {
        const int r = owner(b);
        return (ownedBlocks(r) - countBelowRev(nblocks - b, r, size)) * NB;  // owned blocks < b
    }
    // Global row of local row l on rank r
    size_t globalRowOf(int r, size_t l) const {
        const size_t k = ownedBlocks(r) - 1 - l / NB;  // k-th smallest reversed index of rank r
        const size_t br = k * (size_t)size + posInCycle(k, r, size);
        return (nblocks - 1 - br) * NB + l % NB;
    }
    size_t globalRow(size_t l) const { return globalRowOf(rank, l); }
};

struct PendingRecv {
    MPI_Request req;
    size_t step;
    bool diag;
};

// Node-level shared storage and inter-node communication state
struct Ctx {
    Dist d;
    MPI_Comm nodeComm = MPI_COMM_NULL;
    MPI_Comm xferComm = MPI_COMM_NULL;         // private communicator for inter-node transfers
    int nodeRank = 0, nodeSize = 1, myNode = 0, nnodes = 1;
    std::vector<int> nodeOf;                   // node index of every global rank
    std::vector<std::vector<int>> nodeRanks;   // global ranks of every node (node-local order)
    std::vector<size_t> rowsOf;                // number of local rows of every global rank
    MPI_Win dataWin = MPI_WIN_NULL, flagWin = MPI_WIN_NULL;
    double* myRows = nullptr;                  // local rows (lrows x n), inside the shared window
    std::vector<double*> rankRows;             // rows of every global rank as seen on this node
    double* status = nullptr;                  // per block: -1 ok, else failing diagonal index
    int* diagReady = nullptr;                  // per block: diagonal block finished
    int* colDone = nullptr;                    // per global rank: last column panel finished
    std::vector<int> recvSrc;                  // remote ranks this rank receives for
    std::vector<std::vector<PendingRecv>> recvQ;
    std::vector<size_t> recvHead;
    std::vector<MPI_Request> sends;
    size_t ldS = 0;
    std::vector<double> St;                    // partial sums, St[j * ldS + li]  (j global, li local row)
    std::vector<double> Pt;                    // own rows of the current panel, transposed in tiles of 8 rows

    const double* row(size_t g) const {
        const size_t b = g / NB;
        return rankRows[d.owner(b)] + (d.localRowOfBlock(b) + g % NB) * d.n;
    }
};

static void publishFlag(Ctx& c, int* flag, int value) {
    MPI_Win_sync(c.dataWin);
    MPI_Win_sync(c.flagWin);
    std::atomic_ref<int>(*flag).store(value, std::memory_order_release);
}

// A receive from remote sender recvSrc[s] has completed: publish its data node-wide
static void publishRecv(Ctx& c, size_t s, const PendingRecv& pr) {
    if (pr.diag)
        publishFlag(c, &c.diagReady[pr.step], 1);
    else
        publishFlag(c, &c.colDone[c.recvSrc[s]], (int)pr.step);
}

// Drive inter-node communication: complete receives (in per-sender order) and publish them
static void progress(Ctx& c) {
    for (size_t s = 0; s < c.recvSrc.size(); ++s) {
        std::vector<PendingRecv>& q = c.recvQ[s];
        while (c.recvHead[s] < q.size()) {
            PendingRecv& pr = q[c.recvHead[s]];
            int flag = 0;
            MPI_Test(&pr.req, &flag, MPI_STATUS_IGNORE);
            if (!flag) break;
            publishRecv(c, s, pr);
            ++c.recvHead[s];
        }
    }
    if (!c.sends.empty()) {
        int outcount = 0;
        std::vector<int> idx(c.sends.size());
        MPI_Testsome((int)c.sends.size(), c.sends.data(), &outcount, idx.data(), MPI_STATUSES_IGNORE);
        if (outcount > 0)
            c.sends.erase(std::remove(c.sends.begin(), c.sends.end(), MPI_REQUEST_NULL), c.sends.end());
    }
}

template <typename Pred>
static void waitUntil(Ctx& c, Pred done) {
    while (!done()) {
        progress(c);
        __builtin_ia32_pause();
    }
    MPI_Win_sync(c.dataWin);
    MPI_Win_sync(c.flagWin);
}

static void waitDiag(Ctx& c, size_t b) {
    waitUntil(c, [&] { return std::atomic_ref<int>(c.diagReady[b]).load(std::memory_order_acquire) != 0; });
}

static void waitColumn(Ctx& c, int q, size_t b) {
    waitUntil(c, [&] { return std::atomic_ref<int>(c.colDone[q]).load(std::memory_order_acquire) >= (int)b; });
}

// Datatype for rows [lbeg, lend) of a rank's row array, columns [j0, j0+w)
static MPI_Datatype panelType(size_t lbeg, size_t lend, size_t w, size_t n) {
    MPI_Datatype t;
    MPI_Type_vector((int)(lend - lbeg), (int)w, (int)n, MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}

// Datatype for status[b] + diagonal block b (rows/columns of block b) at absolute addresses
static MPI_Datatype diagType(double* statusPtr, double* blockRows, size_t nbb, size_t j0, size_t n) {
    std::vector<int> lens(nbb + 1);
    std::vector<MPI_Aint> disp(nbb + 1);
    lens[0] = 1;
    MPI_Get_address(statusPtr, &disp[0]);
    for (size_t r = 0; r < nbb; ++r) {
        lens[r + 1] = (int)nbb;
        MPI_Get_address(blockRows + r * n + j0, &disp[r + 1]);
    }
    MPI_Datatype t;
    MPI_Type_create_hindexed((int)(nbb + 1), lens.data(), disp.data(), MPI_DOUBLE, &t);
    MPI_Type_commit(&t);
    return t;
}

static int tagPanel(size_t b) { return (int)(2 * (b % 16384)); }
static int tagDiag(size_t b) { return (int)(2 * (b % 16384) + 1); }

static void sendToOtherNodes(Ctx& c, void* buf, MPI_Datatype t, int tag) {
    for (int m = 0; m < c.nnodes; ++m) {
        if (m == c.myNode) continue;
        const std::vector<int>& nr = c.nodeRanks[m];
        const int dest = nr[(size_t)c.d.rank % nr.size()];
        MPI_Request req;
        MPI_Isend(buf, 1, t, dest, tag, c.xferComm, &req);
        c.sends.push_back(req);
    }
}

// Finish columns [j0, j0+nbb) of local rows [lbeg, lend) once diagonal block b is known:
//   L[i][j] = (A[i][j] - (S[i][j] + sum_{j0<=k<j} L[i][k] L[j][k])) / L[j][j]
template <size_t R>
static inline void panelRows(Ctx& c, size_t b, size_t l0) {
    const size_t n = c.d.n, j0 = b * NB, nbb = c.d.blockRows(b);
    double* Li[R];
    for (size_t r = 0; r < R; ++r) Li[r] = c.myRows + (l0 + r) * n;
    for (size_t jj = 0; jj < nbb; ++jj) {
        const size_t j = j0 + jj;
        const double* Lj = c.row(j);
        const size_t kv = std::max(j & ~(size_t)3, j0);
        double s[R];
        for (size_t r = 0; r < R; ++r) s[r] = c.St[j * c.ldS + l0 + r];
        size_t k = j0;
        for (; k < kv; ++k)
            for (size_t r = 0; r < R; ++r) s[r] += noContract(Li[r][k] * Lj[k]);
        for (; k < j; ++k)
            for (size_t r = 0; r < R; ++r) s[r] = __builtin_fma(Li[r][k], Lj[k], s[r]);
        for (size_t r = 0; r < R; ++r) Li[r][j] = (Li[r][j] - s[r]) / Lj[j];
    }
}

static void computePanel(Ctx& c, size_t b, size_t lbeg, size_t lend) {
    size_t l = lbeg;
    for (; l + 4 <= lend; l += 4) panelRows<4>(c, b, l);
    for (; l < lend; ++l) panelRows<1>(c, b, l);
}

// Transpose columns [k0, k0+klen) of local rows [lbeg, lend) into Pt (tiles of 8 rows, k-major)
static void buildPt(Ctx& c, size_t k0, size_t klen, size_t lbeg, size_t lend) {
    const size_t n = c.d.n, j0 = k0, nbb = klen;
    for (size_t t = lbeg / 8; t * 8 < lend; ++t) {
        double* P = &c.Pt[t * KB * 8];
        for (size_t r = 0; r < 8; ++r) {
            const size_t li = t * 8 + r;
            if (li < c.d.lrows) {
                const double* src = c.myRows + li * n + j0;
                for (size_t k = 0; k < nbb; ++k) P[k * 8 + r] = src[k];
            } else {
                for (size_t k = 0; k < nbb; ++k) P[k * 8 + r] = 0.0;
            }
        }
    }
}

// S[j][i..i+7] += sum_{k in panel} L[i][k] * L[j][k]   for R columns j (sequential in k per element)
template <size_t R>
static inline void updateKernel(const double* P, const double* const* Lj, size_t klen, double* S, size_t ldS) {
    v4 acc0[R], acc1[R];
    for (size_t r = 0; r < R; ++r) {
        acc0[r] = *(const v4u*)(S + r * ldS);
        acc1[r] = *(const v4u*)(S + r * ldS + 4);
    }
    for (size_t k = 0; k < klen; ++k) {
        const v4 p0 = *(const v4u*)(P + k * 8);
        const v4 p1 = *(const v4u*)(P + k * 8 + 4);
        for (size_t r = 0; r < R; ++r) {
            const double a = Lj[r][k];
            acc0[r] += noContract(p0 * a);
            acc1[r] += noContract(p1 * a);
        }
    }
    for (size_t r = 0; r < R; ++r) {
        *(v4u*)(S + r * ldS) = acc0[r];
        *(v4u*)(S + r * ldS + 4) = acc1[r];
    }
}

// Apply columns [k0, k0+klen) of L to the partial sums of own rows [lbeg, lend) for the columns of block J
static void updateBlock(Ctx& c, size_t k0, size_t klen, size_t lbeg, size_t lend, size_t J) {
    const size_t j0 = k0, nbb = klen;
    const size_t jb = J * NB, nbJ = c.d.blockRows(J);
    for (size_t t = lbeg / 8; t * 8 < lend; ++t) {
        const double* P = &c.Pt[t * KB * 8];
        size_t jj = 0;
        for (; jj + JR <= nbJ; jj += JR) {
            const double* Lj[JR];
            for (size_t r = 0; r < JR; ++r) Lj[r] = c.row(jb + jj + r) + j0;
            updateKernel<JR>(P, Lj, nbb, &c.St[(jb + jj) * c.ldS + t * 8], c.ldS);
        }
        for (; jj + 2 <= nbJ; jj += 2) {
            const double* Lj[2] = {c.row(jb + jj) + j0, c.row(jb + jj + 1) + j0};
            updateKernel<2>(P, Lj, nbb, &c.St[(jb + jj) * c.ldS + t * 8], c.ldS);
        }
        for (; jj < nbJ; ++jj) {
            const double* Lj[1] = {c.row(jb + jj) + j0};
            updateKernel<1>(P, Lj, nbb, &c.St[(jb + jj) * c.ldS + t * 8], c.ldS);
        }
    }
}

// Factor the locally owned diagonal block b, publish it and send it to other nodes
static void factorDiagonal(Ctx& c, size_t b) {
    const size_t n = c.d.n;
    const size_t j0 = b * NB, nbb = c.d.blockRows(b);
    const size_t lb = c.d.localRowOfBlock(b);
    double* rows = c.myRows + lb * n;
    long long fail = -1;
    for (size_t r = 0; r < nbb && fail < 0; ++r) {
        double* Li = rows + r * n;
        for (size_t jj = 0; jj <= r; ++jj) {
            const size_t j = j0 + jj;
            const double* Lj = rows + jj * n;
            const double sum = dotContinue(c.St[j * c.ldS + lb + r], Li, Lj, j0, j);
            if (jj == r) {
                const double val = Li[j] - sum;
                if (val <= 0.0) {
                    // Matrix is not positive definite
                    fail = (long long)j;
                    break;
                }
                Li[j] = sqrt(val);
            } else {
                Li[j] = (Li[j] - sum) / Lj[j];
            }
        }
    }
    // Zero out upper triangular part of the finished rows
    for (size_t r = 0; r < nbb; ++r) std::fill(rows + r * n + j0 + r + 1, rows + (r + 1) * n, 0.0);

    c.status[b] = (double)fail;
    publishFlag(c, &c.diagReady[b], 1);
    if (c.nnodes > 1) {
        MPI_Datatype t = diagType(&c.status[b], rows, nbb, j0, n);
        sendToOtherNodes(c, MPI_BOTTOM, t, tagDiag(b));
        MPI_Type_free(&t);
    }
}

// Column panel b of own rows [lbeg, lrows) is finished: publish and send to other nodes
static void finishColumn(Ctx& c, size_t b, size_t lbeg) {
    publishFlag(c, &c.colDone[c.d.rank], (int)b);
    if (c.nnodes > 1 && lbeg < c.d.lrows) {
        MPI_Datatype t = panelType(lbeg, c.d.lrows, c.d.blockRows(b), c.d.n);
        sendToOtherNodes(c, c.myRows + lbeg * c.d.n + b * NB, t, tagPanel(b));
        MPI_Type_free(&t);
    }
}

// Set up node communicator, shared window layout and pre-posted inter-node receives
static void setupContext(Ctx& c) {
    const Dist& d = c.d;
    const size_t n = d.n, nb = d.nblocks;

    MPI_Comm_dup(MPI_COMM_WORLD, &c.xferComm);
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, d.rank, MPI_INFO_NULL, &c.nodeComm);
    MPI_Comm_rank(c.nodeComm, &c.nodeRank);
    MPI_Comm_size(c.nodeComm, &c.nodeSize);

    // Identify nodes by the global rank of their node-local rank 0
    int leader = d.rank;
    MPI_Bcast(&leader, 1, MPI_INT, 0, c.nodeComm);
    std::vector<int> leaders(d.size);
    MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> uniq(leaders);
    std::sort(uniq.begin(), uniq.end());
    uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
    c.nnodes = (int)uniq.size();
    c.nodeOf.resize(d.size);
    c.nodeRanks.assign(c.nnodes, {});
    c.rowsOf.resize(d.size);
    for (int g = 0; g < d.size; ++g) {
        c.nodeOf[g] = (int)(std::lower_bound(uniq.begin(), uniq.end(), leaders[g]) - uniq.begin());
        c.nodeRanks[c.nodeOf[g]].push_back(g);  // ascending global rank == node-local order
        c.rowsOf[g] = countLocalRows(n, g, d.size);
    }
    c.myNode = c.nodeOf[d.rank];
    const std::vector<int>& local = c.nodeRanks[c.myNode];

    // Segment of node-local rank q: [own rows][mirrors of remote ranks g with g % nodeSize == q]
    std::vector<size_t> seg(c.nodeSize);
    std::vector<int> segOf(d.size);
    std::vector<size_t> offOf(d.size);
    for (int q = 0; q < c.nodeSize; ++q) {
        segOf[local[q]] = q;
        offOf[local[q]] = 0;
        seg[q] = c.rowsOf[local[q]] * n;
    }
    for (int g = 0; g < d.size; ++g) {
        if (c.nodeOf[g] == c.myNode) continue;
        const int q = g % c.nodeSize;
        segOf[g] = q;
        offOf[g] = seg[q];
        seg[q] += c.rowsOf[g] * n;
    }

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "alloc_shared_noncontig", "true");
    double* base = nullptr;
    MPI_Win_allocate_shared((MPI_Aint)(std::max<size_t>(seg[c.nodeRank], 1) * sizeof(double)), sizeof(double),
                            info, c.nodeComm, &base, &c.dataWin);
    MPI_Info_free(&info);
    std::vector<double*> bases(c.nodeSize);
    for (int q = 0; q < c.nodeSize; ++q) {
        MPI_Aint sz;
        int du;
        MPI_Win_shared_query(c.dataWin, q, &sz, &du, &bases[q]);
    }
    c.myRows = bases[c.nodeRank];
    c.rankRows.resize(d.size);
    for (int g = 0; g < d.size; ++g) c.rankRows[g] = bases[segOf[g]] + offOf[g];
    // Mirrors start zeroed (upper triangle of the factor stays zero)
    std::fill(c.myRows + c.rowsOf[d.rank] * n, c.myRows + seg[c.nodeRank], 0.0);

    // Flags owned by node-local rank 0: status[nb] (double), diagReady[nb], colDone[size] (int)
    const size_t flagBytes = c.nodeRank == 0 ? nb * sizeof(double) + (nb + (size_t)d.size) * sizeof(int) : 0;
    void* fbase = nullptr;
    MPI_Win_allocate_shared((MPI_Aint)std::max<size_t>(flagBytes, 1), 1, MPI_INFO_NULL, c.nodeComm, &fbase,
                            &c.flagWin);
    {
        MPI_Aint sz;
        int du;
        void* f0;
        MPI_Win_shared_query(c.flagWin, 0, &sz, &du, &f0);
        c.status = (double*)f0;
        c.diagReady = (int*)((char*)f0 + nb * sizeof(double));
        c.colDone = c.diagReady + nb;
    }
    if (c.nodeRank == 0) {
        for (size_t b = 0; b < nb; ++b) {
            c.status[b] = -1.0;
            c.diagReady[b] = 0;
        }
        for (int g = 0; g < d.size; ++g) c.colDone[g] = -1;
    }
    MPI_Win_lock_all(MPI_MODE_NOCHECK, c.dataWin);
    MPI_Win_lock_all(MPI_MODE_NOCHECK, c.flagWin);

    // Pre-post receives (in sender order) for the remote ranks mirrored by this rank
    for (int g = 0; g < d.size; ++g) {
        if (c.nodeOf[g] == c.myNode || segOf[g] != c.nodeRank) continue;
        c.recvSrc.push_back(g);
        std::vector<PendingRecv> q;
        double* mirror = c.rankRows[g];
        auto postDiag = [&](size_t b) {
            const size_t nbb = d.blockRows(b);
            MPI_Datatype t = diagType(&c.status[b], mirror + d.localRowOfBlock(b) * n, nbb, b * NB, n);
            PendingRecv pr{MPI_REQUEST_NULL, b, true};
            MPI_Irecv(MPI_BOTTOM, 1, t, g, tagDiag(b), c.xferComm, &pr.req);
            MPI_Type_free(&t);
            q.push_back(pr);
        };
        if (nb > 0 && d.owner(0) == g) postDiag(0);
        for (size_t b = 0; b < nb; ++b) {
            if (b + 1 < nb && d.owner(b + 1) == g) postDiag(b + 1);
            const size_t ls = d.localStartAfter(b, g, c.rowsOf[g]);
            if (ls < c.rowsOf[g]) {
                MPI_Datatype t = panelType(ls, c.rowsOf[g], d.blockRows(b), n);
                PendingRecv pr{MPI_REQUEST_NULL, b, false};
                MPI_Irecv(mirror + ls * n + b * NB, 1, t, g, tagPanel(b), c.xferComm, &pr.req);
                MPI_Type_free(&t);
                q.push_back(pr);
            }
        }
        c.recvQ.push_back(std::move(q));
        c.recvHead.push_back(0);
    }

    c.ldS = (d.lrows + 7) / 8 * 8;
    c.St.assign(std::max<size_t>(n * c.ldS, 1), 0.0);
    c.Pt.assign(std::max<size_t>(c.ldS * KB, 1), 0.0);
    MPI_Barrier(MPI_COMM_WORLD);
}

static void freeContext(Ctx& c) {
    MPI_Waitall((int)c.sends.size(), c.sends.data(), MPI_STATUSES_IGNORE);
    c.sends.clear();
    MPI_Win_unlock_all(c.flagWin);
    MPI_Win_unlock_all(c.dataWin);
    MPI_Win_free(&c.flagWin);
    MPI_Win_free(&c.dataWin);
    MPI_Comm_free(&c.nodeComm);
    MPI_Comm_free(&c.xferComm);
}

// Complete (or, after a failure at block bfail, cancel unsent) inter-node receives
static void drainReceives(Ctx& c, bool failed, size_t bfail) {
    for (size_t s = 0; s < c.recvSrc.size(); ++s) {
        std::vector<PendingRecv>& q = c.recvQ[s];
        for (; c.recvHead[s] < q.size(); ++c.recvHead[s]) {
            PendingRecv& pr = q[c.recvHead[s]];
            const bool sent = !failed || (pr.diag ? pr.step <= bfail : pr.step < bfail);
            if (sent) {
                MPI_Wait(&pr.req, MPI_STATUS_IGNORE);
                publishRecv(c, s, pr);  // other ranks on this node may still wait for it
            } else {
                MPI_Cancel(&pr.req);
                MPI_Wait(&pr.req, MPI_STATUS_IGNORE);
            }
        }
    }
    MPI_Waitall((int)c.sends.size(), c.sends.data(), MPI_STATUSES_IGNORE);
    c.sends.clear();
}

// The local rows (c.myRows) hold A on entry and L on exit; all rows of the
// factor are available node-wide through c.row().
bool choleskyDecomposition(Ctx& c) {
    const Dist& d = c.d;
    const size_t nb = d.nblocks;
    bool ok = true;
    size_t bfail = 0;

    if (nb > 0 && d.owner(0) == d.rank) factorDiagonal(c, 0);

    for (size_t b = 0; b < nb; ++b) {
        waitDiag(c, b);
        if (c.status[b] >= 0.0) {
            if (d.rank == 0)
                printf("Error: Matrix is not positive definite at diagonal element %zu\n", (size_t)c.status[b]);
            ok = false;
            bfail = b;
            break;
        }
        // Panels are grouped into batches of DB blocks. Columns inside the current
        // batch are updated panel by panel; columns beyond it are updated once per
        // batch with all its panels (deeper k, less memory traffic). Either way each
        // partial sum is accumulated in ascending k order.
        const size_t bs = b / DB * DB;              // first block of the current batch
        const size_t nextBatch = bs + DB;           // first block of the next batch
        const size_t j0 = b * NB, j1 = j0 + d.blockRows(b);
        const bool batchEnd = (b + 1 == nextBatch) && (b + 1 < nb);
        const size_t ls = d.localStartAfter(b);     // first own row below block b
        size_t lr = ls;
        bool didLookahead = false;
        if (b + 1 < nb && d.owner(b + 1) == d.rank) {
            // Lookahead: finish the next diagonal block first so others can proceed
            const size_t nb1 = d.blockRows(b + 1);
            computePanel(c, b, ls, ls + nb1);
            const size_t k0 = batchEnd ? bs * NB : j0;
            buildPt(c, k0, j1 - k0, ls, ls + nb1);
            updateBlock(c, k0, j1 - k0, ls, ls + nb1, b + 1);
            factorDiagonal(c, b + 1);
            lr = ls + nb1;
            didLookahead = true;
        }
        computePanel(c, b, lr, d.lrows);
        finishColumn(c, b, ls);

        auto trailing = [&](size_t k0, size_t Jbeg, size_t Jend) {
            buildPt(c, k0, j1 - k0, lr, d.lrows);
            for (size_t J = Jbeg; J < Jend; ++J) {
                const size_t lI = d.localStartAfter(J - 1);  // first own row of a block >= J
                if (lI >= d.lrows) break;
                const int q = d.owner(J);
                if (q != d.rank) waitColumn(c, q, b);
                size_t l0 = lI;
                if (didLookahead && J == b + 1) l0 = lI + d.blockRows(b + 1);  // already done
                if (l0 < d.lrows) updateBlock(c, k0, j1 - k0, l0, d.lrows, J);
                if ((J & 7) == 0) progress(c);
            }
        };
        if (batchEnd)
            trailing(bs * NB, b + 1, nb);                        // batched far-field update
        else
            trailing(j0, b + 1, std::min(nextBatch, nb));        // within-batch update with panel b
    }

    // Complete outstanding inter-node transfers
    drainReceives(c, !ok, bfail);
    MPI_Barrier(MPI_COMM_WORLD);
    return ok;
}

// Generate the local rows of a symmetric positive definite matrix
void generatePositiveDefiniteMatrix(double* A, const Dist& d) {
    // Method: Create A = B * B^T where B is random
    // This guarantees A is positive semi-definite
    // Then add identity to make it strictly positive definite
    const size_t n = d.n;
    size_t maxRows = 0;
    for (int q = 0; q < d.size; ++q) maxRows = std::max(maxRows, countLocalRows(n, q, d.size));

    // Generate the local rows of random matrix B (same global sequence as sequential)
    std::vector<double> Bmine(std::max<size_t>(maxRows * n, 1));
    unsigned int seed = 42;
    {
        size_t l = 0;
        for (size_t i = 0; i < n; ++i) {
            const bool mine = d.owner(i / NB) == d.rank;
            for (size_t k = 0; k < n; ++k) {
                const double v = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
                if (mine) Bmine[l * n + k] = v;
            }
            if (mine) ++l;
        }
    }

    // Compute A = B * B^T, rotating blocks of B rows around a ring of ranks
    std::vector<double> Bcur(Bmine);
    const int right = (d.rank + 1) % d.size, left = (d.rank - 1 + d.size) % d.size;
    for (int step = 0; step < d.size; ++step) {
        const int src = (d.rank - step + d.size) % d.size;
        const size_t srcRows = countLocalRows(n, src, d.size);
        for (size_t l = 0; l < d.lrows; ++l) {
            const double* Bi = &Bmine[l * n];
            size_t lj = 0;
            for (; lj + 4 <= srcRows; lj += 4) {
                const double* Bj[4] = {&Bcur[lj * n], &Bcur[(lj + 1) * n], &Bcur[(lj + 2) * n], &Bcur[(lj + 3) * n]};
                double sums[4];
                dot4(Bi, Bj, n, sums);
                for (size_t q = 0; q < 4; ++q) A[l * n + d.globalRowOf(src, lj + q)] = sums[q];
            }
            for (; lj < srcRows; ++lj) {
                const size_t j = d.globalRowOf(src, lj);
                A[l * n + j] = dotContinue(0.0, Bi, &Bcur[lj * n], 0, n);
            }
        }
        if (step + 1 < d.size) {
            MPI_Sendrecv_replace(Bcur.data(), (int)(maxRows * n), MPI_DOUBLE, right, 0, left, 0, MPI_COMM_WORLD,
                                 MPI_STATUS_IGNORE);
        }
    }

    // Add diagonal dominance to ensure positive definiteness
    for (size_t l = 0; l < d.lrows; ++l) {
        A[l * n + d.globalRow(l)] += n;
    }
}

// Assemble the full row-major factor from the node-shared blocks (rank 0's node holds all of them)
// Assemble the full row-major factor from the node-wide rows (rank 0's node holds all of them)
static void assembleFactor(const Ctx& c, std::vector<double>& L) {
    const size_t n = c.d.n;
    L.resize(n * n);
    for (size_t i = 0; i < n; ++i) std::memcpy(&L[i * n], c.row(i), n * sizeof(double));
}

bool validateCholesky(const std::vector<double>& L, const std::vector<double>& A_orig, const Dist& d) {
    // Validate by computing L * L^T and comparing with original matrix
    // L is the full factor (on every rank), A_orig holds the local rows
    const size_t n = d.n;

    // Compare with original
    double maxError = 0.0;
    double relError = 0.0;

    // Compute L * L^T for local rows
    for (size_t l = 0; l < d.lrows; ++l) {
        const size_t i = d.globalRow(l);
        for (size_t j0 = 0; j0 < n; j0 += 4) {
            double sums[4];
            const size_t m = std::min<size_t>(4, n - j0);
            if (m == 4) {
                const double* Lj[4] = {&L[j0 * n], &L[(j0 + 1) * n], &L[(j0 + 2) * n], &L[(j0 + 3) * n]};
                dot4(&L[i * n], Lj, n, sums);
            } else {
                for (size_t q = 0; q < m; ++q) sums[q] = dotContinue(0.0, &L[i * n], &L[(j0 + q) * n], 0, n);
            }
            for (size_t q = 0; q < m; ++q) {
                const size_t j = j0 + q;
                const double error = fabs(sums[q] - A_orig[l * n + j]);
                maxError = std::max(maxError, error);

                const double rel = error / (fabs(A_orig[l * n + j]) + 1e-10);
                relError = std::max(relError, rel);
            }
        }
    }
    double errs[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, errs, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = errs[0];
    relError = errs[1];

    if (d.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
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
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (root) {
        printf("Cholesky Decomposition Benchmark\n");
        printf("Matrix size: %zu x %zu\n", n, n);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    Ctx c;
    c.d.rank = rank;
    c.d.size = size;
    c.d.n = n;
    c.d.nblocks = (n + NB - 1) / NB;
    c.d.lrows = countLocalRows(n, rank, size);
    const Dist& d = c.d;

    // Allocate local rows of the matrix (node-shared memory)
    setupContext(c);
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (root) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(c.myRows, d);

    if (validate) {
        A_orig.assign(c.myRows, c.myRows + d.lrows * n); // Save original for validation
    }

    // Perform Cholesky decomposition
    if (root) printf("Computing Cholesky decomposition...\n");
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    bool success = choleskyDecomposition(c);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (!success) {
        if (root) printf("Cholesky decomposition failed\n");
        freeContext(c);
        MPI_Finalize();
        return 1;
    }

    std::vector<double> L;
    if (root) assembleFactor(c, L);
    freeContext(c);

    if (root) {
        printf("Computation time: %ld ms\n", duration.count());

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration.count() / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);

        // Print results for external validation
        if (printResults) {
            print_results(L, "CholeskyL");
        }
    }

    int ret = 0;
    // Validation
    if (validate) {
        if (root) printf("Validating result...\n");
        if (!root) L.resize(n * n);
        MPI_Datatype rowType;
        MPI_Type_contiguous((int)std::max<size_t>(n, 1), MPI_DOUBLE, &rowType);
        MPI_Type_commit(&rowType);
        if (n > 0) MPI_Bcast(L.data(), (int)n, rowType, 0, MPI_COMM_WORLD);
        MPI_Type_free(&rowType);
        bool valid = validateCholesky(L, A_orig, d);

        if (valid) {
            if (root) printf("Validation: PASSED\n");
            ret = 0;
        } else {
            if (root) printf("Validation: FAILED\n");
            ret = 1;
        }
    }

    MPI_Finalize();
    return ret;
}
