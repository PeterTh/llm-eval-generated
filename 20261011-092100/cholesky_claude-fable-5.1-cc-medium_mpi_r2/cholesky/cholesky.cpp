#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// Distributed-memory Cholesky decomposition (MPI).
//
// Decomposes a positive definite matrix A into L * L^T where L is lower triangular.
//
// Parallelization scheme
// ----------------------
// * Rows of the matrix are distributed over the MPI ranks in blocks of RB rows
//   (reflected block-cyclic: block b costs ~b^2, blocks are dealt from the
//   heaviest down to the least loaded rank). Every rank stores only its own
//   rows, in a SIMD friendly "row tile" layout: groups of MR consecutive local
//   rows are stored k-major (tile[k][0..MR)), so that for a fixed column k the
//   MR row entries are contiguous.
// * The factorization is a blocked left-looking algorithm over column panels of
//   NB columns (several row blocks, usually on different ranks). For each panel
//   J, every rank computes, for its own rows i and the NB columns j of the panel,
//       S[i][j] = sum_{k < j} L[i][k] * L[j][k]       (register tiled kernel)
//       L[i][j] = (A[i][j] - S[i][j]) / L[j][j]
//   The owners of the panel's row blocks compute the partial sums of the NB x NB
//   diagonal block (their own rows) and publish them; the tiny diagonal
//   factorization is then done redundantly on every rank. The rows of the next
//   panel are finished and published early in each step (look-ahead).
// * Communication: ranks that share memory (a node, or a socket when the MPI
//   library can tell) read the panels in place from an MPI-3 shared memory
//   window, synchronized with release/acquire flags. Between such groups the
//   panels are forwarded by pipelined binomial-tree broadcasts built from
//   point-to-point messages that are progressed from the compute loops. There
//   is no blocking collective in the factorization loop.
// * Every element's sum over k is accumulated strictly in increasing k order
//   and with exactly the same rounding as the compiled sequential loop of the
//   original (GCC vectorizes the sequential dot product in-order: the first
//   VF*floor(len/VF) products are rounded separately and added, the remaining
//   len%VF terms use fused multiply-add), so results are bitwise identical.
//   The file is compiled with -ffp-contract=off; FMAs are explicit.
// * Matrix generation (A = B * B^T + n*I) and validation (L * L^T vs. A) are
//   distributed too, using a ring exchange of row blocks between the ranks.

namespace {

// ---------------------------------------------------------------------------
// SIMD configuration (compiler vector extensions / AVX2 intrinsics, no external
// dependencies)
// ---------------------------------------------------------------------------
#if defined(__AVX2__)
#include <immintrin.h>
constexpr int VL = 4;  // doubles per vector register
constexpr int NR = 6;  // columns per register tile (12 accumulators + 2 + 1 registers)
#else
constexpr int VL = 2;
constexpr int NR = 4;
#endif
constexpr int MV = 2;        // vector registers per row tile
constexpr int MR = MV * VL;  // rows per row tile
constexpr int KC = 256;      // k blocking (A tile chunk stays in L1)
#if defined(__AVX__)
constexpr size_t VF = 4;  // vectorization factor of the original's reduction loops
#else
constexpr size_t VF = 2;
#endif

typedef double vd __attribute__((vector_size(VL * sizeof(double))));

inline vd vload(const double* p) {
    vd v;
    memcpy(&v, p, sizeof(v));
    return v;
}
inline void vstore(double* p, vd v) { memcpy(p, &v, sizeof(v)); }
inline vd vsplat(double x) {
    vd v;
    for (int i = 0; i < VL; ++i) v[i] = x;
    return v;
}

// Fused multiply-add on vectors (explicit; the file is compiled with -ffp-contract=off)
inline vd vfma(vd a, vd b, vd c) {
    vd r;
    for (int i = 0; i < VL; ++i) r[i] = __builtin_fma(a[i], b[i], c[i]);
    return r;
}

constexpr size_t gcd_(size_t a, size_t b) { return b == 0 ? a : gcd_(b, a % b); }
constexpr size_t lcm_(size_t a, size_t b) { return a / gcd_(a, b) * b; }
constexpr size_t RB = MR;                        // row distribution block (one row tile)
constexpr size_t PANEL_UNIT = lcm_(lcm_(RB, NR), VF);  // panel widths are multiples of this (24)
static_assert(RB % VF == 0, "row block must be a multiple of the vectorization factor");

// C[j][0..MR) (+)= sum_{k0 <= k < k0+klen} round(A[k][0..MR) * B[k][j])
//   (k in increasing order; product rounded, then added)
//   a      : row tile, element (k, r) at a[k*as + r]   (k relative to k0)
//   groups : groups[g] points to a packed column group, element (k, j) at
//            groups[g][k*NR + j] for column j of group g (absolute k)
//   c      : element (j, r) at c[j*cs + r]; accumulated into if accumulate
// ncols must be a multiple of NR.
void kernel(const double* __restrict a, size_t as, const double* const* groups, size_t k0,
            size_t ncols, size_t klen, double* __restrict c, size_t cs, bool accumulate) {
#if defined(__AVX2__)
    for (size_t jt = 0; jt < ncols; jt += NR) {
        __m256d acc[NR][MV];
        for (int j = 0; j < NR; ++j)
            for (int v = 0; v < MV; ++v)
                acc[j][v] = accumulate ? _mm256_loadu_pd(c + (jt + j) * cs + v * VL) : _mm256_setzero_pd();
        const double* bp = groups[jt / NR] + k0 * NR;
        const double* ap = a;
        for (size_t k = 0; k < klen; ++k, ap += as, bp += NR) {
            __m256d av[MV];
            for (int v = 0; v < MV; ++v) av[v] = _mm256_loadu_pd(ap + v * VL);
            for (int j = 0; j < NR; ++j) {
                const __m256d bv = _mm256_broadcast_sd(bp + j);
                for (int v = 0; v < MV; ++v) acc[j][v] = _mm256_add_pd(acc[j][v], _mm256_mul_pd(av[v], bv));
            }
        }
        for (int j = 0; j < NR; ++j)
            for (int v = 0; v < MV; ++v) _mm256_storeu_pd(c + (jt + j) * cs + v * VL, acc[j][v]);
    }
#else
    for (size_t jt = 0; jt < ncols; jt += NR) {
        vd acc[NR][MV];
        for (int j = 0; j < NR; ++j)
            for (int v = 0; v < MV; ++v) acc[j][v] = accumulate ? vload(c + (jt + j) * cs + v * VL) : vsplat(0.0);
        const double* bp = groups[jt / NR] + k0 * NR;
        const double* ap = a;
        for (size_t k = 0; k < klen; ++k, ap += as, bp += NR) {
            vd av[MV];
            for (int v = 0; v < MV; ++v) {
                av[v] = vload(ap + v * VL);
                __asm__("" : "+x"(av[v]));  // keep in a register (avoid re-loading per column)
            }
            for (int j = 0; j < NR; ++j) {
                const vd bv = vsplat(bp[j]);
                for (int v = 0; v < MV; ++v) acc[j][v] += av[v] * bv;
            }
        }
        for (int j = 0; j < NR; ++j)
            for (int v = 0; v < MV; ++v) vstore(c + (jt + j) * cs + v * VL, acc[j][v]);
    }
#endif
}

// Pack a (K x ncols) row-major panel (row stride ss) into consecutive column
// groups of NR columns: dst[g*K*NR + k*NR + j] = src[k*ss + g*NR + j].
void pack_groups(const double* src, size_t ss, size_t ncols, size_t K, double* dst) {
    for (size_t k = 0; k < K; ++k) {
        const double* s = src + k * ss;
        for (size_t g = 0; g < ncols / NR; ++g) {
            double* d = dst + g * K * NR + k * NR;
            for (int j = 0; j < NR; ++j) d[j] = s[g * NR + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Data distribution
// ---------------------------------------------------------------------------
struct Layout {
    int P = 1, rank = 0;
    size_t n = 0;        // matrix size
    size_t nb = 0;       // panel width (multiple of PANEL_UNIT)
    size_t nsb = 0;      // row blocks per panel (nb / RB)
    size_t nblocks = 0;  // number of row blocks (of RB rows)
    size_t npanels = 0;  // number of column panels
    size_t n_pad = 0;    // npanels * nb
    size_t m_blk = 0;    // my number of row blocks
    size_t m_loc = 0;    // my number of (real) rows
    size_t m_pad = 0;    // m_blk * RB (padded rows)
    size_t m_tiles = 0;  // m_pad / MR

    // Block assignment: row block gb lives on rank owner_of[gb] as its local
    // block lidx_of[gb]; blocks_of[r] lists the blocks of rank r in increasing order.
    std::vector<int> owner_of;
    std::vector<size_t> lidx_of;
    std::vector<std::vector<size_t>> blocks_of;
    std::vector<size_t> my_before;  // my_before[gb] = number of my blocks with index < gb

    int owner(size_t gb) const { return owner_of[gb]; }
    size_t nblk_of(int r) const { return blocks_of[r].size(); }
    size_t mpad_of(int r) const { return nblk_of(r) * RB; }
    size_t mloc_of(int r) const {
        const size_t nbr = nblk_of(r);
        if (nbr == 0) return 0;
        return (nbr - 1) * RB + std::min(RB, n - blocks_of[r].back() * RB);
    }
    // number of my row blocks with global block index < gb_end
    size_t nblk_before(size_t gb_end) const { return my_before[std::min(gb_end, nblocks)]; }
    // global row index of a local row of rank r (may be >= n for padding rows)
    size_t global_row_of(int r, size_t iloc) const { return blocks_of[r][iloc / RB] * RB + iloc % RB; }
    size_t global_row(size_t iloc) const { return global_row_of(rank, iloc); }
    size_t tile_stride() const { return n * MR; }
};

Layout makeLayout(size_t n, int P, int rank) {
    Layout L;
    L.P = P;
    L.rank = rank;
    L.n = n;
    size_t nb = 2 * PANEL_UNIT;  // 48 columns per panel
    // Use a narrower panel for small matrices (fewer idle ranks in the last steps).
    while (nb > PANEL_UNIT && (n + nb - 1) / nb < 4) nb -= PANEL_UNIT;
    L.nb = nb;
    L.nsb = nb / RB;
    L.nblocks = (n + RB - 1) / RB;
    L.npanels = (n + nb - 1) / nb;
    L.n_pad = L.npanels * nb;

    // Row block gb costs ~ (gb+1)^2 (it is updated by every panel before it).
    // Assign blocks from the heaviest down, each to the least loaded rank; this
    // yields a reflected block-cyclic distribution that balances both the total
    // work and the trailing rows of every step (within one block per rank).
    L.owner_of.assign(L.nblocks, 0);
    L.lidx_of.assign(L.nblocks, 0);
    L.blocks_of.assign(P, {});
    std::vector<double> load(P, 0.0);
    for (size_t gb = L.nblocks; gb-- > 0;) {
        int best = 0;
        for (int r = 1; r < P; ++r)
            if (load[r] < load[best]) best = r;
        const double w = (double)(gb + 1) * (double)(gb + 1);
        load[best] += w;
        L.owner_of[gb] = best;
        L.blocks_of[best].push_back(gb);
    }
    for (int r = 0; r < P; ++r) {
        std::reverse(L.blocks_of[r].begin(), L.blocks_of[r].end());
        for (size_t i = 0; i < L.blocks_of[r].size(); ++i) L.lidx_of[L.blocks_of[r][i]] = i;
    }
    L.my_before.assign(L.nblocks + 1, 0);
    for (size_t gb = 0; gb < L.nblocks; ++gb)
        L.my_before[gb + 1] = L.my_before[gb] + (L.owner_of[gb] == rank ? 1 : 0);

    L.m_blk = L.nblk_of(rank);
    L.m_loc = L.mloc_of(rank);
    L.m_pad = L.m_blk * RB;
    L.m_tiles = L.m_pad / MR;
    return L;
}

// Element (iloc, k) in the row-tile layout.
inline double tileAt(const std::vector<double>& T, const Layout& L, size_t iloc, size_t k) {
    return T[(iloc / MR) * L.tile_stride() + k * MR + iloc % MR];
}

// Column-major copy of my rows: element (k, iloc) at out[k*m_pad + iloc].
std::vector<double> tilesToColumns(const std::vector<double>& T, const Layout& L) {
    std::vector<double> cols(L.n * L.m_pad, 0.0);
    for (size_t t = 0; t < L.m_tiles; ++t) {
        const double* tp = T.data() + t * L.tile_stride();
        for (size_t k = 0; k < L.n; ++k)
            memcpy(cols.data() + k * L.m_pad + t * MR, tp + k * MR, MR * sizeof(double));
    }
    return cols;
}

// ---------------------------------------------------------------------------
// Distributed symmetric product: for my rows i and all columns j compute
//   X[i][:] . X[j][:]  (sum over k in increasing order)
// a_tiles : my rows of X in tile layout, my_cols : my rows of X column-major.
// store(t, j, s) receives the MR values for tile t, column j.
// ---------------------------------------------------------------------------
template <class Store>
void symmetricProductRing(const Layout& L, const std::vector<double>& a_tiles,
                          std::vector<double> my_cols, Store&& store) {
    const size_t n = L.n;
    const int P = L.P, rank = L.rank;
    constexpr size_t CB = 4 * PANEL_UNIT;  // columns per chunk (multiple of NR and RB)
    size_t maxpad = 0;
    for (int r = 0; r < P; ++r) maxpad = std::max(maxpad, L.mpad_of(r));

    // (+ CB: a partial chunk is processed as whole column groups, which may read
    //  past the end of the last row; those results are never used)
    std::vector<double> cur = std::move(my_cols);
    cur.resize(n * maxpad + CB);
    std::vector<double> nxt(n * maxpad + CB);
    std::vector<double> Bp(n * CB + 2 * VL);
    std::vector<double> S(L.m_tiles * CB * MR);
    const double* groups[CB / NR];
    for (size_t g = 0; g < CB / NR; ++g) groups[g] = Bp.data() + g * n * NR;

    const int right = (rank + 1) % P;
    const int left = (rank - 1 + P) % P;

    for (int s = 0; s < P; ++s) {
        const int src = (rank - s + P) % P;
        const size_t src_pad = L.mpad_of(src);
        const size_t src_loc = L.mloc_of(src);
        MPI_Request reqs[2] = {MPI_REQUEST_NULL, MPI_REQUEST_NULL};
        if (s + 1 < P) {
            const int nsrc = (rank - s - 1 + P) % P;
            MPI_Irecv(nxt.data(), (int)(n * L.mpad_of(nsrc)), MPI_DOUBLE, left, 100 + s,
                      MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(cur.data(), (int)(n * src_pad), MPI_DOUBLE, right, 100 + s, MPI_COMM_WORLD,
                      &reqs[1]);
        }
        const size_t kmain = n - n % VF;  // non-fused part
        for (size_t c0 = 0; c0 < src_pad; c0 += CB) {
            const size_t ncols_valid = c0 < src_loc ? std::min(CB, src_loc - c0) : 0;
            if (ncols_valid == 0) continue;
            const size_t ncols = (ncols_valid + NR - 1) / NR * NR;  // whole column groups
            pack_groups(cur.data() + c0, src_pad, ncols, n, Bp.data());
            for (size_t k0 = 0; k0 < kmain; k0 += KC) {
                const size_t klen = std::min<size_t>(KC, kmain - k0);
                for (size_t t = 0; t < L.m_tiles; ++t)
                    kernel(a_tiles.data() + t * L.tile_stride() + k0 * MR, MR, groups, k0, ncols,
                           klen, S.data() + t * CB * MR, MR, k0 > 0);
            }
            if (kmain == 0) std::fill(S.begin(), S.end(), 0.0);
            for (size_t t = 0; t < L.m_tiles; ++t) {
                const double* tp = a_tiles.data() + t * L.tile_stride();
                for (size_t jj = 0; jj < ncols_valid; ++jj) {
                    double* Sj = S.data() + t * CB * MR + jj * MR;
                    for (size_t k = kmain; k < n; ++k) {  // fused remainder
                        const vd bv = vsplat(cur[k * src_pad + c0 + jj]);
                        for (int v = 0; v < MV; ++v)
                            vstore(Sj + v * VL, vfma(vload(tp + k * MR + v * VL), bv, vload(Sj + v * VL)));
                    }
                    store(t, L.global_row_of(src, c0 + jj), Sj);
                }
            }
        }
        if (s + 1 < P) {
            MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
            std::swap(cur, nxt);
        }
    }
}

// ---------------------------------------------------------------------------
// Matrix generation: A = B * B^T + n * I, B random (same sequence as original)
// ---------------------------------------------------------------------------
void generatePositiveDefiniteMatrix(std::vector<double>& A, const Layout& L) {
    const size_t n = L.n;
    std::vector<double> Bt(L.m_tiles * L.tile_stride(), 0.0);
    std::vector<double> Bcols(n * L.m_pad, 0.0);
    unsigned int seed = 42;

    // Generate random matrix B (every rank walks the full sequence, keeps its rows)
    for (size_t i = 0; i < n; ++i) {
        const size_t gb = i / RB;
        if (L.owner(gb) == L.rank) {
            const size_t iloc = L.lidx_of[gb] * RB + i % RB;
            double* tp = Bt.data() + (iloc / MR) * L.tile_stride() + iloc % MR;
            double* cp = Bcols.data() + iloc;
            for (size_t k = 0; k < n; ++k) {
                const double v = (rand_r(&seed) / (double)RAND_MAX) - 0.5;
                tp[k * MR] = v;
                cp[k * L.m_pad] = v;
            }
        } else {
            for (size_t k = 0; k < n; ++k) (void)rand_r(&seed);
        }
    }

    // Compute A = B * B^T and add diagonal dominance
    A.assign(L.m_tiles * L.tile_stride(), 0.0);
    const double nd = (double)n;
    symmetricProductRing(L, Bt, std::move(Bcols), [&](size_t t, size_t j, const double* s) {
        double* dst = A.data() + t * L.tile_stride() + j * MR;
        for (int l = 0; l < MR; ++l) {
            const size_t iloc = t * MR + l;
            if (iloc >= L.m_loc) continue;
            double v = s[l];
            if (L.global_row(iloc) == j) v += nd;
            dst[l] = v;
        }
    });
}

// ---------------------------------------------------------------------------
// Validation: compare L * L^T with the original matrix
// ---------------------------------------------------------------------------
bool validateCholesky(const std::vector<double>& Lt, const std::vector<double>& A_orig,
                      const Layout& L) {
    double maxError = 0.0;
    double relError = 0.0;
    symmetricProductRing(L, Lt, tilesToColumns(Lt, L), [&](size_t t, size_t j, const double* s) {
        const double* ref = A_orig.data() + t * L.tile_stride() + j * MR;
        for (int l = 0; l < MR; ++l) {
            if (t * MR + l >= L.m_loc) continue;
            const double error = fabs(s[l] - ref[l]);
            maxError = std::max(maxError, error);
            const double rel = error / (fabs(ref[l]) + 1e-10);
            relError = std::max(relError, rel);
        }
    });
    double g[2] = {maxError, relError};
    MPI_Allreduce(MPI_IN_PLACE, g, 2, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    maxError = g[0];
    relError = g[1];

    if (L.rank == 0) {
        printf("Max absolute error: %.10e\n", maxError);
        printf("Max relative error: %.10e\n", relError);
    }

    // Check if error is within tolerance
    if (relError > 1e-6) {
        if (L.rank == 0) printf("Validation failed: relative error too large\n");
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Node topology and shared memory. Ranks of the same "node" share the panel
// buffers through an MPI-3 shared memory window (data is written once by the
// owner and read in place by everybody on the node); between nodes the data is
// forwarded by pipelined binomial-tree broadcasts built from point-to-point
// messages, which are progressed from the compute loops. A "node" here is a
// group of ranks that share memory: the ranks of one socket when the MPI
// library can tell (keeps the shared panel in socket-local memory), otherwise
// the ranks of the whole node.
// ---------------------------------------------------------------------------
struct Node {
    MPI_Comm comm = MPI_COMM_NULL;  // ranks of my node
    int size = 1, rank = 0;         // within the node
    int nnodes = 1, my_node = 0;
    std::vector<int> node_of;                 // world rank -> node index
    std::vector<int> leader_of;               // node index -> world rank of its first rank
    std::vector<std::vector<int>> members_of;  // node index -> world ranks (in node rank order)
};

Node makeNode(int P, int rank) {
    Node N;
#ifdef OMPI_COMM_TYPE_SOCKET
    MPI_Comm_split_type(MPI_COMM_WORLD, OMPI_COMM_TYPE_SOCKET, rank, MPI_INFO_NULL, &N.comm);
#else
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &N.comm);
#endif
    MPI_Comm_size(N.comm, &N.size);
    MPI_Comm_rank(N.comm, &N.rank);
    int leader = rank;
    MPI_Bcast(&leader, 1, MPI_INT, 0, N.comm);
    std::vector<int> leaders(P);
    MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, MPI_COMM_WORLD);
    N.leader_of = leaders;
    std::sort(N.leader_of.begin(), N.leader_of.end());
    N.leader_of.erase(std::unique(N.leader_of.begin(), N.leader_of.end()), N.leader_of.end());
    N.nnodes = (int)N.leader_of.size();
    N.node_of.resize(P);
    N.members_of.assign(N.nnodes, {});
    for (int r = 0; r < P; ++r) {
        N.node_of[r] = (int)(std::lower_bound(N.leader_of.begin(), N.leader_of.end(), leaders[r]) - N.leader_of.begin());
        N.members_of[N.node_of[r]].push_back(r);  // world rank order == node rank order (split key = rank)
    }
    N.my_node = N.node_of[rank];
    return N;
}

inline void flag_set(volatile uint64_t* f, uint64_t v) { __atomic_store_n(f, v, __ATOMIC_RELEASE); }
inline uint64_t flag_get(volatile uint64_t* f) { return __atomic_load_n(f, __ATOMIC_ACQUIRE); }
inline void flag_add(volatile uint64_t* f, uint64_t v) { __atomic_add_fetch(f, v, __ATOMIC_ACQ_REL); }

// Inter-node broadcasts into node-shared memory. The participants are one rank
// per node: the owner on its node, elsewhere a member of the node chosen per
// object (spreads the receive work over the node's ranks). Two kinds:
//  * direct: the buffer has the same meaning on every node; the owner's data
//    is already in place, the receiving rank receives into it and sets the flag.
//  * staged: the owner sends from src (contiguous), the receiving rank stores
//    it in a staging buffer, sets the stage flag, then post-processes it
//    (scatter into the panel) and sets the flag.
class BcastPool {
  public:
    using Post = void (*)(void*, int);
    BcastPool(const Node& N, int rank, int nobj, Post post, void* ctx)
        : N_(N), rank_(rank), objs_(nobj), post_(post), ctx_(post ? ctx : nullptr) {}

    // Start object o: count doubles; the owner sends from src, remote nodes
    // receive into dst (same address meaning on every node). stage == nullptr
    // means direct. Tags [tag, tag + MAXSEG) are used.
    void start(int o, const double* src, double* dst, size_t count, int owner, int tag, volatile uint64_t* stage,
               volatile uint64_t* flag, uint64_t flag_val) {
        Obj& ob = objs_[o];
        ob.src = src;
        ob.dst = dst;
        ob.count = count;
        ob.root_node = N_.node_of[owner];
        ob.owner = owner;
        ob.tag = tag;
        ob.stage = stage;
        ob.flag = flag;
        ob.flag_val = flag_val;
        ob.seg = std::max<size_t>(MINSEG, (count + MAXSEG - 1) / MAXSEG);
        ob.nseg = count == 0 ? 0 : (count + ob.seg - 1) / ob.seg;
        ob.recv_pending = ob.send_pending = 0;
        if (N_.my_node == ob.root_node) {
            if (rank_ == owner) {
                flag_set(flag, flag_val);  // data was written by the owner before start()
                for (size_t g = 0; g < ob.nseg; ++g) sendSegment(o, g);
            }
        } else if (rank_ == rep(o, (N_.my_node - ob.root_node + N_.nnodes) % N_.nnodes)) {
            const int rel = (N_.my_node - ob.root_node + N_.nnodes) % N_.nnodes;
            const int parent = rep(o, (rel & (rel - 1)));  // clear lowest set bit
            for (size_t g = 0; g < ob.nseg; ++g) {
                items_.push_back({MPI_REQUEST_NULL, o, (int)g});
                MPI_Irecv(ob.dst + g * ob.seg, segLen(ob, g), MPI_DOUBLE, parent, tag + (int)g, MPI_COMM_WORLD,
                          &items_.back().req);
                ++ob.recv_pending;
            }
            if (stage != nullptr) posts_.push_back(o);
            if (ob.nseg == 0) arrived(o);
        }
    }
    // Wait until the data of object o is available on this node (flag set).
    void wait(int o) {
        const Obj& ob = objs_[o];
        while (flag_get(ob.flag) < ob.flag_val) progress(true);
    }
    // Wait for all outstanding sends of this rank (buffers may be reused afterwards).
    void drain() {
        while (!sends_.empty()) progress(true);
    }
    // Test outstanding receives (every poll) and sends (occasionally); forward
    // received segments to the child nodes; post-process staged objects.
    // Throttled unless forced.
    void progress(bool force = false) {
        if (items_.empty() && sends_.empty() && posts_.empty()) return;
        const double now = MPI_Wtime();
        if (!force && now - last_ < 15e-6) return;
        last_ = now;
        if (!items_.empty()) {
            reqs_.resize(items_.size());
            for (size_t i = 0; i < items_.size(); ++i) reqs_[i] = items_[i].req;
            idx_.resize(items_.size());
            int outcount = 0;
            MPI_Testsome((int)reqs_.size(), reqs_.data(), &outcount, idx_.data(), MPI_STATUSES_IGNORE);
            if (outcount > 0) {
                std::sort(idx_.begin(), idx_.begin() + outcount);
                forward_.clear();
                for (int c = outcount - 1; c >= 0; --c) {  // remove completed items (descending)
                    forward_.push_back(items_[idx_[c]]);
                    items_[idx_[c]] = items_.back();
                    items_.pop_back();
                }
                for (const Item& it : forward_) {
                    sendSegment(it.obj, it.seg);
                    if (--objs_[it.obj].recv_pending == 0) arrived(it.obj);
                }
            }
        }
        for (size_t i = 0; i < posts_.size();) {
            const Obj& ob = objs_[posts_[i]];
            if (flag_get(ob.stage) >= ob.flag_val) {
                post_(ctx_, posts_[i]);
                flag_set(ob.flag, ob.flag_val);
                posts_[i] = posts_.back();
                posts_.pop_back();
            } else {
                ++i;
            }
        }
        if (!sends_.empty() && (force || ++spoll_ % 8 == 0)) {
            reqs_.resize(sends_.size());
            for (size_t i = 0; i < sends_.size(); ++i) reqs_[i] = sends_[i].req;
            idx_.resize(sends_.size());
            int outcount = 0;
            MPI_Testsome((int)reqs_.size(), reqs_.data(), &outcount, idx_.data(), MPI_STATUSES_IGNORE);
            if (outcount > 0) {
                std::sort(idx_.begin(), idx_.begin() + outcount);
                for (int c = outcount - 1; c >= 0; --c) {
                    --objs_[sends_[idx_[c]].obj].send_pending;
                    sends_[idx_[c]] = sends_.back();
                    sends_.pop_back();
                }
            }
        }
    }

  private:
    struct Obj {
        const double* src = nullptr;
        double* dst = nullptr;
        size_t count = 0, seg = 0, nseg = 0;
        int root_node = 0, owner = 0, tag = 0, recv_pending = 0, send_pending = 0;
        volatile uint64_t* stage = nullptr;
        volatile uint64_t* flag = nullptr;
        uint64_t flag_val = 0;
    };
    struct Item {
        MPI_Request req;
        int obj, seg;
    };
    static constexpr size_t MINSEG = 49152;  // doubles per segment (384 KB)
    static constexpr size_t MAXSEG = 8;      // max segments per object

    // data of object o has fully arrived on this (non-root) node
    void arrived(int o) {
        const Obj& ob = objs_[o];
        flag_set(ob.stage != nullptr ? ob.stage : ob.flag, ob.flag_val);
    }
    // participating rank for object o of the node with relative index rel
    int rep(int o, int rel) const {
        const Obj& ob = objs_[o];
        const int node = (rel + ob.root_node) % N_.nnodes;
        if (node == ob.root_node) return ob.owner;
        const std::vector<int>& m = N_.members_of[node];
        return m[(size_t)o % m.size()];
    }
    int segLen(const Obj& ob, size_t g) const { return (int)std::min(ob.seg, ob.count - g * ob.seg); }
    void sendSegment(int o, size_t g) {
        Obj& ob = objs_[o];
        const double* from = (rank_ == ob.owner) ? ob.src : ob.dst;
        const int rel = (N_.my_node - ob.root_node + N_.nnodes) % N_.nnodes;
        const int lowbit = rel == 0 ? N_.nnodes : (rel & -rel);
        for (int b = 1; b < lowbit && rel + b < N_.nnodes; b <<= 1) {
            sends_.push_back({MPI_REQUEST_NULL, o, (int)g});
            MPI_Isend(from + g * ob.seg, segLen(ob, g), MPI_DOUBLE, rep(o, rel + b), ob.tag + (int)g, MPI_COMM_WORLD,
                      &sends_.back().req);
            ++ob.send_pending;
        }
    }

    const Node& N_;
    int rank_;
    std::vector<Obj> objs_;
    Post post_;
    void* ctx_;
    std::vector<Item> items_;    // outstanding receives
    std::vector<Item> sends_;    // outstanding sends
    std::vector<Item> forward_;  // scratch
    std::vector<int> posts_;     // staged objects I post-process
    std::vector<MPI_Request> reqs_;
    std::vector<int> idx_;
    double last_ = 0.0;
    unsigned spoll_ = 0;
};

// ---------------------------------------------------------------------------
// Distributed blocked left-looking Cholesky decomposition
// ---------------------------------------------------------------------------
bool choleskyDecomposition(std::vector<double>& T, const Layout& L, const Node& N) {
    const size_t n = L.n, nb = L.nb, nsb = L.nsb, npanels = L.npanels, n_pad = L.n_pad;
    const int rank = L.rank;
    const size_t ts = L.tile_stride();
    const size_t ngroups = nb / NR;
    const size_t sub_tile = RB * nb;  // doubles of S per row block (one tile)
    constexpr size_t NBUF = 3;        // panel buffers in flight

    // Node-shared memory: NBUF panel buffers, NBUF diagonal-sum buffers, flags.
    // Panel J (in buffer J % NBUF) holds, for every column j of the panel,
    // L[j][k] for k < j0 and the stored A values for k in [j0, j0+nb), packed
    // in column groups of NR: group g is at buf + g * bts, element (k, j) at
    // [k*NR + j]. The owner of each RB-row block of the panel scatters its
    // columns into the groups.
    const size_t panel_sz = nb * n_pad + 2 * VL;
    const size_t sdt_sz = nsb * sub_tile;
    const size_t stage_sz = N.nnodes > 1 ? nb * n_pad : 0;  // staging for panel blocks from other nodes
    const size_t data_doubles = NBUF * (panel_sz + sdt_sz + stage_sz);
    const size_t nflags = NBUF * (3 * nsb + 1);
    MPI_Win win;
    double* base = nullptr;
    {
        const MPI_Aint bytes = N.rank == 0 ? (MPI_Aint)(data_doubles * sizeof(double) + nflags * sizeof(uint64_t)) : 0;
        MPI_Win_allocate_shared(bytes, sizeof(double), MPI_INFO_NULL, N.comm, &base, &win);
        MPI_Aint qsize;
        int qdisp;
        MPI_Win_shared_query(win, 0, &qsize, &qdisp, &base);
        if (N.rank == 0) memset(base, 0, (size_t)bytes);
        MPI_Barrier(N.comm);
    }
    double* panel[NBUF];
    double* sdt[NBUF];
    double* stage[NBUF];
    for (size_t b = 0; b < NBUF; ++b) {
        panel[b] = base + b * panel_sz;
        sdt[b] = base + NBUF * panel_sz + b * sdt_sz;
        stage[b] = base + NBUF * (panel_sz + sdt_sz) + b * stage_sz;
    }
    volatile uint64_t* flags = reinterpret_cast<volatile uint64_t*>(base + data_doubles);
    auto pflag = [&](size_t J, size_t s) { return flags + (J % NBUF) * nsb + s; };                   // panel block ready
    auto sflag = [&](size_t J, size_t s) { return flags + NBUF * nsb + (J % NBUF) * nsb + s; };      // diag sums ready
    auto gflag = [&](size_t J, size_t s) { return flags + 2 * NBUF * nsb + (J % NBUF) * nsb + s; };  // staged block arrived
    auto dflag = [&](size_t J) { return flags + 3 * NBUF * nsb + J % NBUF; };                         // steps finished

    std::vector<const double*> groups(ngroups);
    std::vector<double> S(L.m_tiles * nb * MR);  // partial sums of my trailing rows, per tile
    std::vector<double> Sd(nb * nb, 0.0);        // Sd[jj][ii]
    std::vector<double> Dt(nb * nb, 0.0);        // diagonal block, Dt[kk][ii] = L[ii][kk]

    // Scatter a row block (tile layout [k][RB], kend rows) of panel J into the column groups
    auto scatter_block = [&](const double* tp, size_t J, size_t s) {
        const size_t bts = (J + 1) * nb * NR;
        const size_t krows = (J + 1) * nb;
        const size_t kend = std::min(n, krows);
        double* dst[RB];
        for (size_t j = 0; j < RB; ++j) {
            const size_t jj = s * RB + j;
            dst[j] = panel[J % NBUF] + (jj / NR) * bts + jj % NR;
        }
        for (size_t k = 0; k < kend; ++k)
            for (size_t j = 0; j < RB; ++j) dst[j][k * NR] = tp[k * RB + j];
        for (size_t k = kend; k < krows; ++k)
            for (size_t j = 0; j < RB; ++j) dst[j][k * NR] = 0.0;
    };
    // Broadcast objects: [0, nsb) panel row blocks (staged), [nsb, 2 nsb)
    // diagonal partial sums (direct). Post-processing of a staged block on
    // another node: scatter it from the staging area into the panel.
    struct PostCtx {
        decltype(scatter_block)* scatter;
        double** stage;
        size_t nsb, n_pad, J;
    } post_ctx{&scatter_block, stage, nsb, n_pad, 0};
    auto post_fn = [](void* c, int o) {
        PostCtx* pc = static_cast<PostCtx*>(c);
        const size_t s = (size_t)o;
        (*pc->scatter)(pc->stage[pc->J % NBUF] + s * RB * pc->n_pad, pc->J, s);
    };
    BcastPool bc(N, rank, (int)(2 * nsb), post_fn, &post_ctx);
    auto progress = [&]() { bc.progress(); };
    // Before writing into the buffers of step J: everybody on the node must be
    // done with step J - NBUF, and my own forwarding sends must have completed.
    auto reuse_guard = [&](size_t J) {
        if (J < NBUF) return;
        const uint64_t want = (uint64_t)(J / NBUF) * (uint64_t)N.size;
        while (flag_get(dflag(J)) < want) bc.progress(true);
        bc.drain();
    };

    auto nsb_of = [&](size_t J) { return std::min(nsb, L.nblocks - J * nsb); };
    auto panel_groups = [&](size_t J) {
        const size_t bts = (J + 1) * nb * NR;
        const double* buf = panel[J % NBUF];
        for (size_t g = 0; g < ngroups; ++g) groups[g] = buf + g * bts;
    };
    auto start_panel_bcasts = [&](size_t J) {
        const size_t nsb_cur = nsb_of(J);
        const size_t kend = std::min(n, (J + 1) * nb);
        reuse_guard(J);
        post_ctx.J = J;
        for (size_t s = 0; s < nsb_cur; ++s) {
            const int owner = L.owner(J * nsb + s);
            double* st = stage[J % NBUF] + s * RB * n_pad;
            const double* src = nullptr;
            if (owner == rank) {
                const double* tp = T.data() + L.lidx_of[J * nsb + s] * ts;
                scatter_block(tp, J, s);
                if (N.nnodes > 1) {  // send from a stable copy (the tile's diagonal rows change later)
                    memcpy(st, tp, kend * RB * sizeof(double));
                    src = st;
                }
            }
            bc.start((int)s, src, st, kend * RB, owner, (int)(1000 + ((J % 4) * nsb + s) * 32), gflag(J, s),
                     pflag(J, s), J + 1);
        }
    };

    // Partial sums over k < j0 for tiles [ta, tb)
    auto gemm = [&](size_t ta, size_t tb, size_t j0) {
        for (size_t k0 = 0; k0 < j0; k0 += KC) {
            const size_t klen = std::min<size_t>(KC, j0 - k0);
            for (size_t t = ta; t < tb; ++t) {
                kernel(T.data() + t * ts + k0 * MR, MR, groups.data(), k0, nb, klen, S.data() + t * nb * MR,
                       MR, k0 > 0);
                progress();
            }
        }
        if (j0 == 0) std::fill(S.begin() + ta * nb * MR, S.begin() + tb * nb * MR, 0.0);
    };
    // Triangular tail of the panel and final division for trailing tiles [ta, tb)
    auto tail = [&](size_t ta, size_t tb, size_t j0, size_t nb_cur) {
        for (size_t t = ta; t < tb; ++t) {
            double* tp = T.data() + t * ts;
            const double* St = S.data() + t * nb * MR;
            for (size_t jj = 0; jj < nb_cur; ++jj) {
                vd s[MV];
                for (int v = 0; v < MV; ++v) s[v] = vload(St + jj * MR + v * VL);
                const size_t kmain = jj - jj % VF;
                for (size_t kk = 0; kk < kmain; ++kk) {
                    const vd ljk = vsplat(Dt[kk * nb + jj]);
                    const double* ap = tp + (j0 + kk) * MR;
                    for (int v = 0; v < MV; ++v) s[v] += vload(ap + v * VL) * ljk;
                }
                for (size_t kk = kmain; kk < jj; ++kk) {  // fused remainder
                    const vd ljk = vsplat(Dt[kk * nb + jj]);
                    const double* ap = tp + (j0 + kk) * MR;
                    for (int v = 0; v < MV; ++v) s[v] = vfma(vload(ap + v * VL), ljk, s[v]);
                }
                const vd ljj = vsplat(Dt[jj * nb + jj]);
                double* ap = tp + (j0 + jj) * MR;
                for (int v = 0; v < MV; ++v) vstore(ap + v * VL, (vload(ap + v * VL) - s[v]) / ljj);
            }
            progress();
        }
    };

    bool ok = true;
    if (npanels > 0) start_panel_bcasts(0);

    for (size_t J = 0; J < npanels && ok; ++J) {
        const size_t j0 = J * nb;
        const size_t nb_cur = std::min(nb, n - j0);
        const size_t nsb_cur = nsb_of(J);
        const size_t gb0 = J * nsb;
        const bool has_next = J + 1 < npanels;

        for (size_t s = 0; s < nsb_cur; ++s) bc.wait((int)s);
        panel_groups(J);

        // My tiles: [t_panel, t_first) rows of this panel, [t_first, t_next) rows
        // of the next panel, [t_next, m_tiles) the remaining trailing rows.
        const size_t t_panel = L.nblk_before(gb0);
        const size_t t_first = L.nblk_before(gb0 + nsb_cur);
        const size_t t_next = has_next ? L.nblk_before(gb0 + 2 * nsb) : t_first;
        const size_t t_mid = t_next + (L.m_tiles - t_next) / 2;

        // 1. Partial sums of the diagonal block: the owner of each row block of
        //    the panel computes them for its rows first and publishes them.
        if (j0 > 0) {
            size_t idx = 0;
            for (size_t s = 0; s < nsb_cur; ++s) {
                const int owner = L.owner(gb0 + s);
                if (owner == rank) {
                    const size_t tb = t_panel + idx;
                    ++idx;
                    gemm(tb, tb + 1, j0);
                    memcpy(sdt[J % NBUF] + s * sub_tile, S.data() + tb * nb * MR, sub_tile * sizeof(double));
                }
                bc.start((int)(nsb + s), sdt[J % NBUF] + s * sub_tile, sdt[J % NBUF] + s * sub_tile, sub_tile, owner,
                         (int)(10000 + ((J % 4) * nsb + s) * 32), nullptr, sflag(J, s), J + 1);
            }
        }

        // 2. Partial sums for my rows of the next panel (look-ahead) and the
        //    first half of the remaining trailing rows
        gemm(t_first, t_next, j0);
        gemm(t_next, t_mid, j0);

        // 3. Diagonal block: A values from the panel, partial sums from the owners,
        //    then factor it (redundantly on every rank, identical arithmetic)
        if (j0 > 0) {
            for (size_t s = 0; s < nsb_cur; ++s) bc.wait((int)(nsb + s));
            const double* Sdt = sdt[J % NBUF];
            for (size_t jj = 0; jj < nb; ++jj)
                for (size_t ii = 0; ii < nb; ++ii) Sd[jj * nb + ii] = Sdt[(ii / MR) * nb * MR + jj * MR + ii % MR];
        } else {
            std::fill(Sd.begin(), Sd.end(), 0.0);
        }
        for (size_t kk = 0; kk < nb; ++kk)
            for (size_t ii = 0; ii < nb; ++ii) Dt[kk * nb + ii] = groups[ii / NR][(j0 + kk) * NR + ii % NR];
        for (size_t jj = 0; jj < nb_cur && ok; ++jj) {
            double* Sj = Sd.data() + jj * nb;
            const size_t kmain = jj - jj % VF;
            for (size_t kk = 0; kk < kmain; ++kk) {
                const double* Lk = Dt.data() + kk * nb;  // Lk[ii] = L[ii][kk]
                const double ljk = Lk[jj];
                for (size_t ii = jj; ii < nb_cur; ++ii) Sj[ii] += Lk[ii] * ljk;
            }
            for (size_t kk = kmain; kk < jj; ++kk) {  // fused remainder
                const double* Lk = Dt.data() + kk * nb;
                const double ljk = Lk[jj];
                for (size_t ii = jj; ii < nb_cur; ++ii) Sj[ii] = __builtin_fma(Lk[ii], ljk, Sj[ii]);
            }
            double* Aj = Dt.data() + jj * nb;  // Aj[ii] = A[ii][jj] -> L[ii][jj]
            const double val = Aj[jj] - Sj[jj];
            if (val <= 0.0) {
                // Matrix is not positive definite
                if (rank == 0) printf("Error: Matrix is not positive definite at diagonal element %zu\n", j0 + jj);
                ok = false;
                break;
            }
            const double ljj = sqrt(val);
            Aj[jj] = ljj;
            for (size_t ii = jj + 1; ii < nb_cur; ++ii) Aj[ii] = (Aj[ii] - Sj[ii]) / ljj;
            for (size_t ii = 0; ii < jj; ++ii) Aj[ii] = 0.0;  // upper part
        }
        if (!ok) break;

        // 4. Owners store the factored diagonal block rows
        {
            size_t idx = 0;
            for (size_t s = 0; s < nsb_cur; ++s) {
                if (L.owner(gb0 + s) != rank) continue;
                const size_t t0 = t_panel + idx;
                ++idx;
                for (size_t kk = 0; kk < nb_cur; ++kk)
                    memcpy(T.data() + t0 * ts + (j0 + kk) * MR, Dt.data() + kk * nb + s * RB, MR * sizeof(double));
            }
        }

        // 5. Finish my rows of the next panel and publish them (look-ahead:
        //    overlaps with the bulk of this step)
        if (has_next) {
            tail(t_first, t_next, j0, nb_cur);
            start_panel_bcasts(J + 1);
        }

        // 6. Second half of the trailing rows, then the tail of all of them
        gemm(t_mid, L.m_tiles, j0);
        tail(t_next, L.m_tiles, j0, nb_cur);
        flag_add(dflag(J), 1);  // done with the buffers of step J
    }

    // The failure decision is identical on all ranks (same arithmetic), so no
    // rank is left waiting; just make sure no messages are in flight.
    bc.drain();
    MPI_Barrier(N.comm);
    MPI_Win_free(&win);
    if (!ok) return false;

    // Zero out upper triangular part (and padding rows)
    for (size_t t = 0; t < L.m_tiles; ++t) {
        double* tp = T.data() + t * ts;
        for (int l = 0; l < MR; ++l) {
            const size_t iloc = t * MR + l;
            const size_t kstart = (iloc < L.m_loc) ? L.global_row(iloc) + 1 : 0;
            for (size_t k = kstart; k < n; ++k) tp[k * MR + l] = 0.0;
        }
    }
    return true;
}

// Gather the full matrix (row-major) on rank 0.
void gatherMatrix(const std::vector<double>& T, const Layout& L, std::vector<double>& A) {
    const size_t n = L.n;
    std::vector<double> rows(L.m_loc * n);
    for (size_t iloc = 0; iloc < L.m_loc; ++iloc)
        for (size_t k = 0; k < n; ++k) rows[iloc * n + k] = tileAt(T, L, iloc, k);

    if (L.rank == 0) {
        A.assign(n * n, 0.0);
        std::vector<double> tmp;
        for (int r = 0; r < L.P; ++r) {
            const double* src = rows.data();
            if (r != 0) {
                tmp.resize(L.mloc_of(r) * n);
                MPI_Recv(tmp.data(), (int)tmp.size(), MPI_DOUBLE, r, 7, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                src = tmp.data();
            }
            for (size_t lb = 0; lb < L.nblk_of(r); ++lb) {
                const size_t gb = L.blocks_of[r][lb];
                const size_t nrows = std::min(RB, n - gb * RB);
                memcpy(A.data() + gb * RB * n, src + lb * RB * n, nrows * n * sizeof(double));
            }
        }
    } else {
        MPI_Send(rows.data(), (int)rows.size(), MPI_DOUBLE, 0, 7, MPI_COMM_WORLD);
    }
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Matrix size (default: 512)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int P = 1, rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &P);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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

    const Layout L = makeLayout(n, P, rank);
    Node N = makeNode(P, rank);

    // Allocate (distributed) matrix
    std::vector<double> A;
    std::vector<double> A_orig;

    // Generate positive definite matrix
    if (rank == 0) printf("Generating positive definite matrix...\n");
    generatePositiveDefiniteMatrix(A, L);

    if (validate) {
        A_orig = A;  // Save original for validation
    }

    // Perform Cholesky decomposition
    if (rank == 0) printf("Computing Cholesky decomposition...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    bool success = choleskyDecomposition(A, L, N);

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const long duration_ms = (long)((end - start) * 1000.0);

    if (!success) {
        if (rank == 0) printf("Cholesky decomposition failed\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("Computation time: %ld ms\n", duration_ms);

        // Calculate GFLOPS (approximately n³/3 operations for Cholesky)
        double ops = (double)n * n * n / 3.0;
        double gflops = ops / (duration_ms / 1000.0) / 1e9;
        printf("Performance: %.3f GFLOPS\n", gflops);
    }

    // Print results for external validation
    if (printResults) {
        std::vector<double> full;
        gatherMatrix(A, L, full);
        if (rank == 0) print_results(full, "CholeskyL");
    }

    // Validation
    int ret = 0;
    if (validate) {
        if (rank == 0) printf("Validating result...\n");
        bool valid = validateCholesky(A, A_orig, L);

        if (rank == 0) {
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        ret = valid ? 0 : 1;
    }

    MPI_Comm_free(&N.comm);
    MPI_Finalize();
    return ret;
}
