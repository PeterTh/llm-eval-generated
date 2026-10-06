#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr unsigned int INF = 1000000000;
constexpr unsigned int MAX_DISTANCE = 200;

// Maximum number of intermediate nodes (k values) processed per communication step
constexpr size_t KBLOCK = 32;
// Column tile width (in elements) for cache reuse of pivot row panels
constexpr size_t JTILE = 1024;
// Number of element updates between progress calls for pending broadcasts
constexpr size_t TEST_INTERVAL = size_t(1) << 18;

// Index calculation for flattened 2D array
inline constexpr size_t idx2(const size_t i, const size_t j, const size_t n) noexcept {
    return j * n + i;
}

// ---------------------------------------------------------------------------
// 2D block distribution over a Pr x Pc process grid.
// Rank (pr, pc) owns rows [rowBegin, rowEnd) and columns [colBegin, colEnd).
// Local storage is row-major: local[(i - rowBegin) * nc + (j - colBegin)].
// ---------------------------------------------------------------------------
static size_t partBegin(const size_t n, const int parts, const int idx) {
    const size_t base = n / parts;
    const size_t rem = n % parts;
    const size_t r = static_cast<size_t>(idx);
    return r * base + std::min(r, rem);
}

static int partOwner(const size_t n, const int parts, const size_t i) {
    const size_t base = n / parts;
    const size_t rem = n % parts;
    // First 'rem' parts have base+1 elements
    if (i < rem * (base + 1)) return static_cast<int>(i / (base + 1));
    return static_cast<int>(rem + (i - rem * (base + 1)) / base);
}

struct Grid {
    size_t n = 0;
    int Pr = 1, Pc = 1;
    int pr = 0, pc = 0;
    size_t rowBegin = 0, rowEnd = 0, colBegin = 0, colEnd = 0;
    size_t nr() const { return rowEnd - rowBegin; }
    size_t nc() const { return colEnd - colBegin; }
    MPI_Comm rowComm = MPI_COMM_NULL;  // ranks in the same process row (rank = pc)
    MPI_Comm colComm = MPI_COMM_NULL;  // ranks in the same process column (rank = pr)
};

static Grid makeGrid(const size_t n, const int rank, const int size) {
    Grid g;
    g.n = n;
    int dims[2] = {0, 0};
    MPI_Dims_create(size, 2, dims);
    g.Pr = dims[0];
    g.Pc = dims[1];
    g.pr = rank / g.Pc;
    g.pc = rank % g.Pc;
    g.rowBegin = partBegin(n, g.Pr, g.pr);
    g.rowEnd = partBegin(n, g.Pr, g.pr + 1);
    g.colBegin = partBegin(n, g.Pc, g.pc);
    g.colEnd = partBegin(n, g.Pc, g.pc + 1);
    MPI_Comm_split(MPI_COMM_WORLD, g.pr, g.pc, &g.rowComm);
    MPI_Comm_split(MPI_COMM_WORLD, g.pc, g.pr, &g.colComm);
    return g;
}

void initializeDistanceMatrix(std::vector<unsigned int>& dist, const size_t numNodes,
                              const unsigned int rangeMin, const unsigned int rangeMax,
                              const Grid& g) {
    unsigned int seed = 42;
    const double range = static_cast<double>(rangeMax - rangeMin) + 1.0;
    const size_t nc = g.nc();

    // Replay the sequential generator, keeping only locally owned elements
    const size_t first = g.rowBegin * numNodes;
    for (size_t i = 0; i < first; ++i) {
        rand_r(&seed);
    }
    for (size_t i = g.rowBegin; i < g.rowEnd; ++i) {
        for (size_t j = 0; j < numNodes; ++j) {
            const int r = rand_r(&seed);
            if (j >= g.colBegin && j < g.colEnd) {
                dist[(i - g.rowBegin) * nc + (j - g.colBegin)] =
                    rangeMin + (unsigned int)(range * r / (double)RAND_MAX);
            }
        }
    }

    // Set diagonal to zero (distance from node to itself is 0)
    for (size_t i = std::max(g.rowBegin, g.colBegin); i < std::min(g.rowEnd, g.colEnd); ++i) {
        dist[(i - g.rowBegin) * nc + (i - g.colBegin)] = 0;
    }
}

void initializePathMatrix(std::vector<unsigned int>& path, const size_t numNodes,
                          const Grid& g) {
    (void)numNodes;
    // Equivalent to the original initialization: path[i][j] = i for all i, j
    const size_t nc = g.nc();
    for (size_t i = g.rowBegin; i < g.rowEnd; ++i) {
        unsigned int* row = path.data() + (i - g.rowBegin) * nc;
        std::fill(row, row + nc, static_cast<unsigned int>(i));
    }
}

// ---------------------------------------------------------------------------
// Relaxation kernels
// ---------------------------------------------------------------------------

// Relax d[j] against dk + pk[j] for j in [jBegin, jEnd) with intermediate node k
static inline void relaxRange(unsigned int* __restrict d, unsigned int* __restrict p,
                              const unsigned int* __restrict pk, const unsigned int dk,
                              const unsigned int k, const size_t jBegin, const size_t jEnd) {
#pragma GCC ivdep
    for (size_t j = jBegin; j < jEnd; ++j) {
        const unsigned int newDist = dk + pk[j];
        const unsigned int cur = d[j];
        const bool better = newDist < cur;
        d[j] = better ? newDist : cur;
        p[j] = better ? k : p[j];
    }
}

// Relax d[j] sequentially against four consecutive intermediate nodes k..k+3
// (same per-element update order as four separate relaxRange calls)
static inline void relaxRange4(unsigned int* __restrict d, unsigned int* __restrict p,
                               const unsigned int* __restrict pk0,
                               const unsigned int* __restrict pk1,
                               const unsigned int* __restrict pk2,
                               const unsigned int* __restrict pk3,
                               const unsigned int* dk, const unsigned int k,
                               const size_t jBegin, const size_t jEnd) {
    const unsigned int dk0 = dk[0], dk1 = dk[1], dk2 = dk[2], dk3 = dk[3];
#pragma GCC ivdep
    for (size_t j = jBegin; j < jEnd; ++j) {
        unsigned int cur = d[j];
        unsigned int pc = p[j];
        unsigned int nd;
        nd = dk0 + pk0[j]; pc = nd < cur ? k : pc;     cur = nd < cur ? nd : cur;
        nd = dk1 + pk1[j]; pc = nd < cur ? k + 1 : pc; cur = nd < cur ? nd : cur;
        nd = dk2 + pk2[j]; pc = nd < cur ? k + 2 : pc; cur = nd < cur ? nd : cur;
        nd = dk3 + pk3[j]; pc = nd < cur ? k + 3 : pc; cur = nd < cur ? nd : cur;
        d[j] = cur;
        p[j] = pc;
    }
}

// Apply iterations k = kb + t, t in [t0, t1), to d[jBegin, jEnd) of one row.
// rowPanel row t has stride 'stride'; dik[t] is dist[i][kb+t] at time kb+t.
static inline void relaxRowBlock(unsigned int* d, unsigned int* p,
                                 const unsigned int* rowPanel, const size_t stride,
                                 const unsigned int* dik, const size_t kb,
                                 const size_t t0, const size_t t1,
                                 const size_t jBegin, const size_t jEnd) {
    size_t t = t0;
    for (; t + 4 <= t1; t += 4) {
        const unsigned int* pk = rowPanel + t * stride;
        relaxRange4(d, p, pk, pk + stride, pk + 2 * stride, pk + 3 * stride, dik + t,
                    static_cast<unsigned int>(kb + t), jBegin, jEnd);
    }
    for (; t < t1; ++t) {
        relaxRange(d, p, rowPanel + t * stride, dik[t], static_cast<unsigned int>(kb + t),
                   jBegin, jEnd);
    }
}

// Half-open interval of local indices
struct Range {
    size_t lo, hi;
};

// Local index interval [0, total) minus up to two excluded local intervals
static int subtractRanges(const size_t total, Range ex0, Range ex1, Range out[3]) {
    if (ex0.lo > ex1.lo) std::swap(ex0, ex1);
    int cnt = 0;
    size_t pos = 0;
    for (const Range& e : {ex0, ex1}) {
        if (e.lo >= e.hi) continue;
        if (e.lo > pos) out[cnt++] = {pos, e.lo};
        pos = std::max(pos, e.hi);
    }
    if (pos < total) out[cnt++] = {pos, total};
    return cnt;
}

// One block of intermediate nodes k = kb + t, t in [0, bsz).
// The sequential algorithm reads, in iteration k, dist[i][k] and dist[k][j], which are
// not modified during iteration k. The "panels" store exactly these values:
//   colPanel[il * bsz + t] = dist[i][kb+t] at the start of iteration kb+t
//   rowPanel[t * nc + jl]  = dist[kb+t][j] at the start of iteration kb+t
// Each element is updated with increasing k, so dist and path are bit-identical
// to the sequential algorithm.
struct KBlockDesc {
    size_t kb;
    size_t bsz;
    int pr;  // process row owning rows kb..kb+bsz
    int pc;  // process column owning columns kb..kb+bsz
};

// Update local rows 'rows' x local columns 'cols' with all iterations of a block
static void updateRegion(unsigned int* D, unsigned int* P, const size_t nc,
                         const Range* rows, const int nRows,
                         const Range* cols, const int nCols,
                         const unsigned int* colPanel, const unsigned int* rowPanel,
                         const size_t kb, const size_t bsz,
                         MPI_Request* reqs, const int nReqs) {
    int done = (nReqs == 0);
    size_t work = 0;
    for (int c = 0; c < nCols; ++c) {
        for (size_t jb = cols[c].lo; jb < cols[c].hi; jb += JTILE) {
            const size_t je = std::min(jb + JTILE, cols[c].hi);
            for (int r = 0; r < nRows; ++r) {
                for (size_t il = rows[r].lo; il < rows[r].hi; ++il) {
                    // Drive progress of pending non-blocking broadcasts
                    if (!done) {
                        work += (je - jb) * bsz;
                        if (work >= TEST_INTERVAL) {
                            work = 0;
                            MPI_Testall(nReqs, reqs, &done, MPI_STATUSES_IGNORE);
                        }
                    }
                    relaxRowBlock(D + il * nc, P + il * nc, rowPanel, nc, colPanel + il * bsz,
                                  kb, 0, bsz, jb, je);
                }
            }
        }
    }
}

class FloydWarshallSolver {
public:
    FloydWarshallSolver(const Grid& g, unsigned int* D, unsigned int* P)
        : g_(g), D_(D), P_(P), nr_(g.nr()), nc_(g.nc()) {
        rowPanel_[0].resize(KBLOCK * std::max<size_t>(nc_, 1));
        rowPanel_[1].resize(KBLOCK * std::max<size_t>(nc_, 1));
        colPanel_[0].resize(std::max<size_t>(nr_, 1) * KBLOCK);
        colPanel_[1].resize(std::max<size_t>(nr_, 1) * KBLOCK);
        diagRow_.resize(KBLOCK * KBLOCK);
        diagCol_.resize(KBLOCK * KBLOCK);
    }

    void run() {
        const size_t n = g_.n;
        if (n == 0) return;
        buildBlocks();

        int cur = 0;
        preparePanels(blocks_[0], cur);
        MPI_Request reqs[2];
        startPanelBroadcast(blocks_[0], cur, reqs);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

        for (size_t b = 0; b < blocks_.size(); ++b) {
            const KBlockDesc& blk = blocks_[b];
            const bool hasNext = b + 1 < blocks_.size();
            const int nxt = cur ^ 1;
            const unsigned int* cp = colPanel_[cur].data();
            const unsigned int* rp = rowPanel_[cur].data();

            // Local exclusions: panels of the current block were fully processed
            // during their preparation; panels of the next block are processed first.
            Range exRowCur{0, 0}, exColCur{0, 0}, exRowNxt{0, 0}, exColNxt{0, 0};
            if (g_.pr == blk.pr) exRowCur = localRows(blk);
            if (g_.pc == blk.pc) exColCur = localCols(blk);

            int nReqs = 0;
            if (hasNext) {
                const KBlockDesc& nb = blocks_[b + 1];
                const bool inRow = (g_.pr == nb.pr);
                const bool inCol = (g_.pc == nb.pc);
                if (inRow) exRowNxt = localRows(nb);
                if (inCol) exColNxt = localCols(nb);
                if (inRow || inCol) {
                    // Lookahead: bring the next block's panels up to date, prepare them
                    // and start broadcasting before doing the bulk of the update.
                    Range rows[3], cols[3];
                    if (inRow) {
                        const int nCols = subtractRanges(nc_, exColCur, {0, 0}, cols);
                        updateRegion(D_, P_, nc_, &exRowNxt, 1, cols, nCols, cp, rp,
                                     blk.kb, blk.bsz, nullptr, 0);
                    }
                    if (inCol) {
                        const int nRows = subtractRanges(nr_, exRowCur, exRowNxt, rows);
                        updateRegion(D_, P_, nc_, rows, nRows, &exColNxt, 1, cp, rp,
                                     blk.kb, blk.bsz, nullptr, 0);
                    }
                    preparePanels(nb, nxt);
                }
                startPanelBroadcast(nb, nxt, reqs);
                nReqs = 2;
            }

            // Bulk update of the remaining local elements
            Range rows[3], cols[3];
            const int nRows = subtractRanges(nr_, exRowCur, exRowNxt, rows);
            const int nCols = subtractRanges(nc_, exColCur, exColNxt, cols);
            updateRegion(D_, P_, nc_, rows, nRows, cols, nCols, cp, rp, blk.kb, blk.bsz,
                         reqs, nReqs);

            if (hasNext) {
                MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
                cur = nxt;
            }
        }
    }

private:
    void buildBlocks() {
        const size_t n = g_.n;
        // Block boundaries: union of row and column partition boundaries, so that each
        // block lies within one process row and one process column.
        std::vector<size_t> bounds;
        for (int r = 0; r <= g_.Pr; ++r) bounds.push_back(partBegin(n, g_.Pr, r));
        for (int c = 0; c <= g_.Pc; ++c) bounds.push_back(partBegin(n, g_.Pc, c));
        std::sort(bounds.begin(), bounds.end());
        bounds.erase(std::unique(bounds.begin(), bounds.end()), bounds.end());

        const size_t minLocal = std::max<size_t>(1, n / std::max(g_.Pr, g_.Pc));
        const size_t kBlock = std::clamp<size_t>((minLocal / 4 + 3) / 4 * 4, 8, KBLOCK);
        for (size_t s = 0; s + 1 < bounds.size(); ++s) {
            for (size_t k = bounds[s]; k < bounds[s + 1]; k += kBlock) {
                const size_t bsz = std::min(kBlock, bounds[s + 1] - k);
                blocks_.push_back({k, bsz, partOwner(n, g_.Pr, k), partOwner(n, g_.Pc, k)});
            }
        }
    }

    Range localRows(const KBlockDesc& b) const {
        return {b.kb - g_.rowBegin, b.kb + b.bsz - g_.rowBegin};
    }
    Range localCols(const KBlockDesc& b) const {
        return {b.kb - g_.colBegin, b.kb + b.bsz - g_.colBegin};
    }

    // Compute the panels of block 'blk' into buffer set 'buf' and apply the block's
    // iterations to the panel elements themselves. Requires all earlier blocks to be
    // fully applied to the panel elements. Called by all ranks in process row blk.pr
    // and process column blk.pc.
    void preparePanels(const KBlockDesc& blk, const int buf) {
        const bool inRow = (g_.pr == blk.pr);
        const bool inCol = (g_.pc == blk.pc);
        if (!inRow && !inCol) return;
        const size_t B = blk.bsz;
        unsigned int* rp = rowPanel_[buf].data();
        unsigned int* cp = colPanel_[buf].data();
        unsigned int* dR = diagRow_.data();  // dR[t * B + c] = dist[kb+t][kb+c] at time kb+t
        unsigned int* dC = diagCol_.data();  // dC[r * B + t] = dist[kb+r][kb+t] at time kb+t

        // Step 1: diagonal block (plain Floyd-Warshall on the B x B block)
        if (inRow && inCol) {
            const size_t r0 = blk.kb - g_.rowBegin;
            const size_t c0 = blk.kb - g_.colBegin;
            for (size_t t = 0; t < B; ++t) {
                const unsigned int k = static_cast<unsigned int>(blk.kb + t);
                std::memcpy(dR + t * B, D_ + (r0 + t) * nc_ + c0, B * sizeof(unsigned int));
                for (size_t r = 0; r < B; ++r) {
                    unsigned int* d = D_ + (r0 + r) * nc_ + c0;
                    unsigned int* p = P_ + (r0 + r) * nc_ + c0;
                    dC[r * B + t] = d[t];
                    relaxRange(d, p, dR + t * B, d[t], k, 0, B);
                }
            }
        }

        // Step 2: share diagonal panels
        if (inCol) MPI_Bcast(dR, static_cast<int>(B * B), MPI_UNSIGNED, blk.pr, g_.colComm);
        if (inRow) MPI_Bcast(dC, static_cast<int>(B * B), MPI_UNSIGNED, blk.pc, g_.rowComm);

        // Step 3a: row panel (pivot rows x local columns). Pivot row kb+r first gets
        // iterations kb..kb+r-1 (giving its snapshot), then iterations kb+r..kb+B-1.
        if (inRow) {
            const size_t r0 = blk.kb - g_.rowBegin;
            Range cols[3];
            const int nCols = subtractRanges(nc_, inCol ? localCols(blk) : Range{0, 0},
                                             {0, 0}, cols);
            for (int c = 0; c < nCols; ++c) {
                for (size_t jb = cols[c].lo; jb < cols[c].hi; jb += JTILE) {
                    const size_t je = std::min(jb + JTILE, cols[c].hi);
                    for (size_t r = 0; r < B; ++r) {
                        unsigned int* d = D_ + (r0 + r) * nc_;
                        relaxRowBlock(d, P_ + (r0 + r) * nc_, rp, nc_, dC + r * B, blk.kb,
                                      0, r, jb, je);
                        std::memcpy(rp + r * nc_ + jb, d + jb, (je - jb) * sizeof(unsigned int));
                    }
                    for (size_t r = 0; r < B; ++r) {
                        relaxRowBlock(D_ + (r0 + r) * nc_, P_ + (r0 + r) * nc_, rp, nc_,
                                      dC + r * B, blk.kb, r, B, jb, je);
                    }
                }
            }
            if (inCol) {
                const size_t c0 = blk.kb - g_.colBegin;
                for (size_t t = 0; t < B; ++t) {
                    std::memcpy(rp + t * nc_ + c0, dR + t * B, B * sizeof(unsigned int));
                }
            }
        }

        // Step 3b: column panel (local rows x pivot columns), processed in transposed
        // form so that the inner loops run (vectorized) over rows.
        if (inCol) {
            const size_t c0 = blk.kb - g_.colBegin;
            Range rows[3];
            const int nRows = subtractRanges(nr_, inRow ? localRows(blk) : Range{0, 0},
                                             {0, 0}, rows);
            size_t M = 0;
            for (int r = 0; r < nRows; ++r) M += rows[r].hi - rows[r].lo;
            if (Dt_.size() < B * M) {
                Dt_.resize(B * M);
                Pt_.resize(B * M);
                St_.resize(B * M);
            }
            unsigned int* Dt = Dt_.data();
            unsigned int* Pt = Pt_.data();
            unsigned int* St = St_.data();
            // Transpose in
            size_t m = 0;
            for (int r = 0; r < nRows; ++r) {
                for (size_t il = rows[r].lo; il < rows[r].hi; ++il, ++m) {
                    const unsigned int* d = D_ + il * nc_ + c0;
                    const unsigned int* p = P_ + il * nc_ + c0;
                    for (size_t c = 0; c < B; ++c) {
                        Dt[c * M + m] = d[c];
                        Pt[c * M + m] = p[c];
                    }
                }
            }
            // Iterations: snapshot column t, then relax all B columns against it
            for (size_t t = 0; t < B; ++t) {
                const unsigned int k = static_cast<unsigned int>(blk.kb + t);
                unsigned int* snap = St + t * M;
                std::memcpy(snap, Dt + t * M, M * sizeof(unsigned int));
                for (size_t c = 0; c < B; ++c) {
                    const unsigned int pkj = dR[t * B + c];
                    unsigned int* __restrict dc = Dt + c * M;
                    unsigned int* __restrict pc = Pt + c * M;
                    const unsigned int* __restrict sn = snap;
#pragma GCC ivdep
                    for (size_t q = 0; q < M; ++q) {
                        const unsigned int newDist = sn[q] + pkj;
                        const unsigned int cur = dc[q];
                        const bool better = newDist < cur;
                        dc[q] = better ? newDist : cur;
                        pc[q] = better ? k : pc[q];
                    }
                }
            }
            // Transpose out
            m = 0;
            for (int r = 0; r < nRows; ++r) {
                for (size_t il = rows[r].lo; il < rows[r].hi; ++il, ++m) {
                    unsigned int* d = D_ + il * nc_ + c0;
                    unsigned int* p = P_ + il * nc_ + c0;
                    unsigned int* cpi = cp + il * B;
                    for (size_t c = 0; c < B; ++c) {
                        d[c] = Dt[c * M + m];
                        p[c] = Pt[c * M + m];
                        cpi[c] = St[c * M + m];
                    }
                }
            }
            if (inRow) {
                const size_t r0 = blk.kb - g_.rowBegin;
                std::memcpy(cp + r0 * B, dC, B * B * sizeof(unsigned int));
            }
        }
    }

    void startPanelBroadcast(const KBlockDesc& blk, const int buf, MPI_Request reqs[2]) {
        MPI_Ibcast(rowPanel_[buf].data(), static_cast<int>(blk.bsz * nc_), MPI_UNSIGNED,
                   blk.pr, g_.colComm, &reqs[0]);
        MPI_Ibcast(colPanel_[buf].data(), static_cast<int>(nr_ * blk.bsz), MPI_UNSIGNED,
                   blk.pc, g_.rowComm, &reqs[1]);
    }

    const Grid& g_;
    unsigned int* D_;
    unsigned int* P_;
    const size_t nr_, nc_;
    std::vector<KBlockDesc> blocks_;
    std::vector<unsigned int> rowPanel_[2], colPanel_[2];
    std::vector<unsigned int> diagRow_, diagCol_;
    std::vector<unsigned int> Dt_, Pt_, St_;  // transposed column panel work buffers
};

void floydWarshall(std::vector<unsigned int>& dist,
                   std::vector<unsigned int>& path,
                   const Grid& g) {
    FloydWarshallSolver solver(g, dist.data(), path.data());
    solver.run();
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

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    const bool root = (rank == 0);

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
        printf("Floyd-Warshall All-Pairs Shortest Path Benchmark\n");
        printf("Number of nodes: %zu\n", numNodes);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Allocate the locally owned blocks of the matrices (2D block distribution)
    Grid g = makeGrid(numNodes, rank, size);
    std::vector<unsigned int> dist(g.nr() * g.nc());
    std::vector<unsigned int> path(g.nr() * g.nc());
    
    // Initialize
    if (root) printf("Initializing graph...\n");
    initializeDistanceMatrix(dist, numNodes, 1, MAX_DISTANCE, g);
    initializePathMatrix(path, numNodes, g);
    
    // Run Floyd-Warshall
    if (root) printf("Computing shortest paths...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    floydWarshall(dist, path, g);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();
    const long durationMs = static_cast<long>((end - start) * 1000.0);
    
    if (root) {
        printf("Computation time: %ld ms\n", durationMs);

        // Calculate operations per second
        // Floyd-Warshall has O(n³) complexity
        double ops = (double)numNodes * numNodes * numNodes;
        double gflops = ops / (durationMs / 1000.0) / 1e9;
        printf("Performance: %.3f GOPS\n", gflops);
    }
    
    int exitCode = 0;
    if (printResults || validate) {
        // Gather the full distance matrix on rank 0
        std::vector<int> counts, displs;
        std::vector<unsigned int> packed;
        if (root) {
            counts.resize(size);
            displs.resize(size);
            size_t off = 0;
            for (int r = 0; r < size; ++r) {
                const int rr = r / g.Pc, rc = r % g.Pc;
                const size_t cnt = (partBegin(numNodes, g.Pr, rr + 1) - partBegin(numNodes, g.Pr, rr)) *
                                   (partBegin(numNodes, g.Pc, rc + 1) - partBegin(numNodes, g.Pc, rc));
                counts[r] = static_cast<int>(cnt);
                displs[r] = static_cast<int>(off);
                off += cnt;
            }
            packed.resize(off);
        }
        MPI_Gatherv(dist.data(), static_cast<int>(dist.size()), MPI_UNSIGNED,
                    packed.data(), counts.data(), displs.data(), MPI_UNSIGNED,
                    0, MPI_COMM_WORLD);

        if (root) {
            std::vector<unsigned int> fullDist(numNodes * numNodes);
            for (int r = 0; r < size; ++r) {
                const int rr = r / g.Pc, rc = r % g.Pc;
                const size_t r0 = partBegin(numNodes, g.Pr, rr), r1 = partBegin(numNodes, g.Pr, rr + 1);
                const size_t c0 = partBegin(numNodes, g.Pc, rc), c1 = partBegin(numNodes, g.Pc, rc + 1);
                const unsigned int* src = packed.data() + displs[r];
                for (size_t i = r0; i < r1; ++i) {
                    std::memcpy(fullDist.data() + i * numNodes + c0, src + (i - r0) * (c1 - c0),
                                (c1 - c0) * sizeof(unsigned int));
                }
            }

            // Print results for external validation (integer hash-based)
            if (printResults) {
                print_results_int(fullDist, "DistanceMatrix");
            }

            // Validation
            if (validate) {
                printf("Validating result...\n");
                bool valid = validateResult(fullDist, numNodes);

                if (valid) {
                    printf("Validation: PASSED\n");
                    exitCode = 0;
                } else {
                    printf("Validation: FAILED\n");
                    exitCode = 1;
                }
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }
    
    MPI_Comm_free(&g.rowComm);
    MPI_Comm_free(&g.colComm);
    MPI_Finalize();
    return exitCode;
}
