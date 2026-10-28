// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA implementation
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Algorithm semantics (unchanged from the sequential reference):
//   - For every unclustered seed, a candidate cluster is grown greedily: the
//     unclustered point with the smallest "max distance to current members"
//     that stays below the threshold is added (lowest index wins ties).
//   - The seed whose candidate cluster is largest (lowest index wins ties) is
//     committed as a cluster; the process repeats until all points are
//     clustered.
//
// Parallelization:
//   - MPI:    seeds are distributed cyclically over ranks. Each iteration the
//             globally best seed is selected with MPI_Allreduce(MPI_MAXLOC) and
//             its member list is broadcast from the owning rank.
//   - CUDA:   candidate-cluster growth runs on the GPU, one thread block per
//             seed, with block-cooperative argmin reductions and pruning.
//   - OpenMP: host-side reductions and dirty-flag propagation across threads.
//
// Exact (semantics-preserving) reuse:
//   - The growth of a seed depends only on the unclustered points within the
//     threshold of that seed. Cached cardinalities are therefore recomputed
//     only for seeds that lost such a point in the last committed cluster.
//   - Distances use one explicit expression on host and device (fma of dx*dx
//     with the rounded dy*dy, which is the arithmetic of the original build),
//     and the code is compiled without implicit contraction, so every path
//     yields bit-identical values.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#ifdef __CUDACC__
#define QT_HD __host__ __device__
#else
#define QT_HD
#endif

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Candidate buffer budget (elements per parity buffer) for one GPU launch.
static const long long CAND_BUDGET = 1LL << 25;
// Threads per block for candidate-cluster growth (one block per seed).
static const int GROW_THREADS = 128;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
        // Make sure we don't make more points than we need
        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            const double dy = std::sqrt(std::fma(r, r, -(dx * dx))) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }
            
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Euclidean distance between two points (shared by host and device code).
// The sum of squares is evaluated as a fused multiply-add of dx*dx with the
// rounded dy*dy, which is the arithmetic of the original build; host and device
// use the same explicit form so every path yields bit-identical distances.
QT_HD inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(std::fma(dx, dx, dy * dy));
}

// ---------------------------------------------------------------------------
// Uniform spatial grid. Cell edge >= threshold, so every point within the
// threshold of a point lies in that point's 3x3 cell neighbourhood.
// ---------------------------------------------------------------------------
struct GridInfo {
    double h;
    int ncx;
    int ncy;
};

QT_HD inline int cellCoord(double v, double h, int n) {
    const int c = static_cast<int>(v / h);
    return c < 0 ? 0 : (c >= n ? n - 1 : c);
}

QT_HD inline int cellIndexOf(const GridInfo& g, const Point& p) {
    return cellCoord(p.y, g.h, g.ncy) * g.ncx + cellCoord(p.x, g.h, g.ncx);
}

struct HostGrid {
    GridInfo g{};
    std::vector<int> cellStart;  // size ncells + 1
    std::vector<int> cellPts;    // point indices grouped by cell (ascending)
    std::vector<int> cellOf;     // cell index of each point
};

HostGrid buildGrid(const std::vector<Point>& pts, double threshold) {
    HostGrid hg;
    const double minH = MAX_WIDTH / 2048.0;
    double h = std::isfinite(threshold)
                   ? std::max(threshold * (1.0 + 1e-6), minH)
                   : MAX_WIDTH;
    hg.g.h = h;
    hg.g.ncx = static_cast<int>(MAX_WIDTH / h) + 1;
    hg.g.ncy = static_cast<int>(MAX_HEIGHT / h) + 1;

    const int ncells = hg.g.ncx * hg.g.ncy;
    const int N = static_cast<int>(pts.size());
    hg.cellOf.resize(N);
    hg.cellStart.assign(ncells + 1, 0);
    for (int i = 0; i < N; ++i) {
        hg.cellOf[i] = cellIndexOf(hg.g, pts[i]);
        hg.cellStart[hg.cellOf[i] + 1]++;
    }
    for (int c = 0; c < ncells; ++c) {
        hg.cellStart[c + 1] += hg.cellStart[c];
    }
    hg.cellPts.resize(N);
    std::vector<int> fill(hg.cellStart.begin(), hg.cellStart.begin() + ncells);
    for (int i = 0; i < N; ++i) {
        hg.cellPts[fill[hg.cellOf[i]]++] = i;
    }
    return hg;
}

// Number of points in the 3x3 cell neighbourhood of p (upper bound on the
// number of candidates a seed at p can ever have).
static long long neighbourhoodPop(const HostGrid& hg, const Point& p) {
    const int cx = cellCoord(p.x, hg.g.h, hg.g.ncx);
    const int cy = cellCoord(p.y, hg.g.h, hg.g.ncy);
    long long total = 0;
    for (int dy = -1; dy <= 1; ++dy) {
        const int y = cy + dy;
        if (y < 0 || y >= hg.g.ncy) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int x = cx + dx;
            if (x < 0 || x >= hg.g.ncx) continue;
            const int c = y * hg.g.ncx + x;
            total += hg.cellStart[c + 1] - hg.cellStart[c];
        }
    }
    return total;
}

// ---------------------------------------------------------------------------
// GPU candidate-cluster growth
// ---------------------------------------------------------------------------
struct DevGrid {
    const Point* pts;            // coordinates indexed by point id
    const int* cellStart;        // CSR offsets of the cells
    const int* cellPts;          // point ids grouped by cell
    const unsigned char* clustered;
    GridInfo g;
    double threshold;
};

// Candidate ordering by (maxDist, index): smaller distance first, lowest index on ties.
QT_HD inline bool candLess(double v1, int i1, double v2, int i2) {
    return v1 < v2 || (v1 == v2 && i1 < i2);
}

// Warp-aggregated append: lanes flagged in `msk` obtain consecutive slots of
// the shared counter. One atomic per warp instead of one per element avoids
// serialising the whole block on a single counter. Returns this lane's slot
// offset (valid only for flagged lanes). All lanes of the warp must call it.
__device__ __forceinline__ int warpAppend(int* counter, unsigned msk, int lane) {
    int base = 0;
    if (lane == 0 && msk) base = atomicAdd(counter, __popc(msk));
    base = __shfl_sync(0xffffffffu, base, 0);
    return base + __popc(msk & ((1u << lane) - 1u));
}

// One thread block grows the candidate cluster of one seed.
//
// Invariant: the candidate list holds every unclustered point c != seed with
// maxDist(c) < threshold, where maxDist(c) is the maximum distance from c to
// the current members. Each step commits the candidate with the smallest
// (maxDist, index) pair, then updates and prunes the remaining candidates.
// This reproduces the sequential findClosestPoint() exactly: maxDist only
// grows as members are added, so pruned candidates can never qualify again.
//
// Each thread keeps the best (maxDist, index) of the survivors it wrote, so the
// next step's argmin needs no extra pass over the candidates: warp shuffles
// reduce it, and one shared exchange plus one barrier combines the warps.
// Candidate state is kept minimal (id and maxDist); coordinates are gathered
// from the point table, which stays resident in L2.
//
// If `members` is non-null, the member list is written to it (one seed only).
__global__ void growKernel(DevGrid D, const int* seeds, const int* offs, int stride,
                           int* candId, double* candMd, int* cards, int* members)
{
    constexpr int NW = GROW_THREADS / 32;
    __shared__ double s_wbv[NW];
    __shared__ int s_wbi[NW];
    __shared__ int s_cnt[2];  // live candidate counts of the two ping-pong buffers

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const unsigned full = 0xffffffffu;
    const int seed = seeds[blockIdx.x];
    const int base = offs[blockIdx.x];
    const Point sp = D.pts[seed];
    const int cx = cellCoord(sp.x, D.g.h, D.g.ncx);
    const int cy = cellCoord(sp.y, D.g.h, D.g.ncy);

    if (tid == 0) {
        s_cnt[0] = 0;
        s_cnt[1] = 0;
    }
    __syncthreads();

    // Thread-local best (maxDist, index) over the live candidates it wrote.
    double bv = INFINITY;
    int bi = 0x7fffffff;

    // Gather unclustered neighbours within the threshold into buffer 0.
    const int base0 = base;
    for (int dy = -1; dy <= 1; ++dy) {
        const int y = cy + dy;
        if (y < 0 || y >= D.g.ncy) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            const int x = cx + dx;
            if (x < 0 || x >= D.g.ncx) continue;
            const int c = y * D.g.ncx + x;
            const int end = D.cellStart[c + 1];
            for (int j0 = D.cellStart[c]; j0 < end; j0 += GROW_THREADS) {
                const int j = j0 + tid;
                int p = -1;
                double d = 0.0;
                bool keep = false;
                if (j < end) {
                    p = D.cellPts[j];
                    if (p != seed && !D.clustered[p]) {
                        d = distance(D.pts[p], sp);
                        keep = d < D.threshold;
                    }
                }
                const unsigned msk = __ballot_sync(full, keep);
                const int slot = warpAppend(&s_cnt[0], msk, lane);
                if (keep) {
                    candId[base0 + slot] = p;
                    candMd[base0 + slot] = d;
                    if (candLess(d, p, bv, bi)) {
                        bv = d;
                        bi = p;
                    }
                }
            }
        }
    }
    __syncthreads();

    int cur = 0;
    int mcount = 1;  // cluster cardinality so far
    if (tid == 0 && members) members[0] = seed;

    while (true) {
        // Block-wide argmin over (maxDist, index) from the thread-local bests.
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_xor_sync(full, bv, off);
            const int oi = __shfl_xor_sync(full, bi, off);
            if (candLess(ov, oi, bv, bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            s_wbv[warp] = bv;
            s_wbi[warp] = bi;
        }
        __syncthreads();  // S1: warp bests and the live count of buffer `cur` are visible

        const int n = s_cnt[cur];  // uniform across the block
        if (n == 0) break;

        int best = s_wbi[0];
        double bestV = s_wbv[0];
        #pragma unroll
        for (int w = 1; w < NW; ++w) {
            if (candLess(s_wbv[w], s_wbi[w], bestV, best)) {
                bestV = s_wbv[w];
                best = s_wbi[w];
            }
        }
        if (tid == 0) {
            s_cnt[cur ^ 1] = 0;  // destination counter, not read in this step
            if (members) members[mcount] = best;
            mcount++;
        }
        __syncthreads();  // S2: destination counter reset before the atomics below

        // Update maxDist of the remaining candidates, prune the ones that can no
        // longer stay below the threshold, and compact survivors into the other
        // buffer (order within a buffer is irrelevant: ties are resolved by id).
        const Point bpt = D.pts[best];
        const int srcOff = cur * stride + base;
        const int dstOff = (cur ^ 1) * stride + base;
        bv = INFINITY;
        bi = 0x7fffffff;
        for (int i0 = 0; i0 < n; i0 += GROW_THREADS) {
            const int i = i0 + tid;
            int id = -1;
            double m = 0.0;
            bool keep = false;
            if (i < n) {
                id = candId[srcOff + i];
                if (id != best) {
                    m = candMd[srcOff + i];
                    const double d = distance(D.pts[id], bpt);
                    m = (m < d) ? d : m;  // max_dist = std::max(max_dist, dist)
                    keep = m < D.threshold;
                }
            }
            const unsigned msk = __ballot_sync(full, keep);
            const int slot = warpAppend(&s_cnt[cur ^ 1], msk, lane);
            if (keep) {
                candId[dstOff + slot] = id;
                candMd[dstOff + slot] = m;
                if (candLess(m, id, bv, bi)) {
                    bv = m;
                    bi = id;
                }
            }
        }
        cur ^= 1;
    }

    if (tid == 0) cards[blockIdx.x] = mcount;
}

// Device state for one MPI rank (bound to one GPU).
struct Gpu {
    const std::vector<Point>* pts = nullptr;
    const HostGrid* hg = nullptr;
    long long budget = 0;
    DevGrid dg{};
    Point* d_pts = nullptr;
    int* d_cellStart = nullptr;
    int* d_cellPts = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_offs = nullptr;
    int* d_cards = nullptr;
    int* d_members = nullptr;
    // Candidate scratch (id, maxDist), two ping-pong parities of `budget` slots
    int* d_candId = nullptr;
    double* d_candMd = nullptr;
};

static void gpuInit(Gpu& G, const std::vector<Point>& pts, const HostGrid& hg, double threshold) {
    G.pts = &pts;
    G.hg = &hg;
    const size_t N = pts.size();
    const size_t ncells = hg.cellStart.size() - 1;

    long long maxPop = 1;
    for (const Point& p : pts) maxPop = std::max(maxPop, neighbourhoodPop(hg, p));
    G.budget = std::max(CAND_BUDGET, maxPop);

    CUDA_CHECK(cudaMalloc(&G.d_pts, N * sizeof(Point)));
    CUDA_CHECK(cudaMemcpy(G.d_pts, pts.data(), N * sizeof(Point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&G.d_cellStart, (ncells + 1) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(G.d_cellStart, hg.cellStart.data(), (ncells + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&G.d_cellPts, N * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(G.d_cellPts, hg.cellPts.data(), N * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&G.d_clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&G.d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&G.d_offs, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&G.d_cards, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&G.d_members, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&G.d_candId, 2 * G.budget * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&G.d_candMd, 2 * G.budget * sizeof(double)));

    G.dg = DevGrid{G.d_pts, G.d_cellStart, G.d_cellPts, G.d_clustered, hg.g,
                   threshold};
}

static void gpuFree(Gpu& G) {
    cudaFree(G.d_pts);
    cudaFree(G.d_cellStart);
    cudaFree(G.d_cellPts);
    cudaFree(G.d_clustered);
    cudaFree(G.d_seeds);
    cudaFree(G.d_offs);
    cudaFree(G.d_cards);
    cudaFree(G.d_members);
    cudaFree(G.d_candId);
    cudaFree(G.d_candMd);
}

static void gpuSetClustered(Gpu& G, const std::vector<unsigned char>& clustered) {
    CUDA_CHECK(cudaMemcpy(G.d_clustered, clustered.data(), clustered.size(),
                          cudaMemcpyHostToDevice));
}

// Cardinality of the candidate cluster of every seed in `seeds`.
static void gpuGrow(Gpu& G, const std::vector<int>& seeds, std::vector<int>& cardsOut) {
    cardsOut.resize(seeds.size());
    const std::vector<Point>& pts = *G.pts;
    std::vector<int> offs;
    size_t i = 0;
    while (i < seeds.size()) {
        // Pack seeds into one launch while their scratch regions fit the budget.
        offs.clear();
        long long total = 0;
        size_t j = i;
        while (j < seeds.size()) {
            const long long cap = neighbourhoodPop(*G.hg, pts[seeds[j]]);
            if (j > i && total + cap > G.budget) break;
            offs.push_back(static_cast<int>(total));
            total += cap;
            ++j;
        }
        const int nb = static_cast<int>(j - i);
        CUDA_CHECK(cudaMemcpy(G.d_seeds, seeds.data() + i, nb * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(G.d_offs, offs.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
        growKernel<<<nb, GROW_THREADS>>>(
            G.dg, G.d_seeds, G.d_offs, static_cast<int>(G.budget),
            G.d_candId, G.d_candMd, G.d_cards, nullptr);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cardsOut.data() + i, G.d_cards, nb * sizeof(int),
                              cudaMemcpyDeviceToHost));
        i = j;
    }
}

// Member list (in addition order) of the candidate cluster of one seed.
static std::vector<int> gpuMembers(Gpu& G, int seed, int expectedCard) {
    const int one_seed = seed;
    const int zero_off = 0;
    CUDA_CHECK(cudaMemcpy(G.d_seeds, &one_seed, sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(G.d_offs, &zero_off, sizeof(int), cudaMemcpyHostToDevice));
    growKernel<<<1, GROW_THREADS>>>(G.dg, G.d_seeds, G.d_offs, static_cast<int>(G.budget),
                                    G.d_candId, G.d_candMd, G.d_cards, G.d_members);
    CUDA_CHECK(cudaGetLastError());
    int card = 0;
    CUDA_CHECK(cudaMemcpy(&card, G.d_cards, sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> members(card);
    CUDA_CHECK(cudaMemcpy(members.data(), G.d_members, card * sizeof(int), cudaMemcpyDeviceToHost));
    if (card != expectedCard) {
        fprintf(stderr, "internal error: cardinality mismatch for seed %d (%d vs %d)\n",
                seed, card, expectedCard);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return members;
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (distributed over MPI ranks, GPU-accelerated)
// ---------------------------------------------------------------------------
static std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                         const HostGrid& hg,
                                         const double threshold,
                                         Gpu& gpu,
                                         const int rank,
                                         const int nprocs) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<unsigned char> dirty(N, 0);  // cached cardinality is stale
    std::vector<int> card(N, 0);             // cached cardinality of owned seeds

    // Cyclic distribution of seeds over ranks balances dense and sparse regions.
    std::vector<int> owned;
    for (int i = rank; i < N; i += nprocs) {
        owned.push_back(i);
        dirty[i] = 1;
    }

    std::vector<Cluster> clusters;
    int numClustered = 0;

    while (numClustered < N) {
        gpuSetClustered(gpu, clustered);

        // 1. Recompute cardinalities of owned seeds whose neighbourhood changed.
        std::vector<int> todo;
        for (int s : owned) {
            if (dirty[s] && !clustered[s]) todo.push_back(s);
        }
        if (!todo.empty()) {
            std::vector<int> res;
            gpuGrow(gpu, todo, res);
            for (size_t k = 0; k < todo.size(); ++k) {
                card[todo[k]] = res[k];
                dirty[todo[k]] = 0;
            }
        }

        // 2. Best owned seed: largest cardinality, lowest index on ties.
        int lc = 0;
        int ls = INT_MAX;
        #pragma omp parallel
        {
            int tc = 0;
            int ts = INT_MAX;
            #pragma omp for nowait
            for (size_t k = 0; k < owned.size(); ++k) {
                const int s = owned[k];
                if (clustered[s]) continue;
                const int c = card[s];
                if (c > tc || (c == tc && s < ts)) {
                    tc = c;
                    ts = s;
                }
            }
            #pragma omp critical
            {
                if (tc > lc || (tc == lc && ts < ls)) {
                    lc = tc;
                    ls = ts;
                }
            }
        }

        // 3. Global best. MPI_MAXLOC returns the maximum value and, on ties,
        //    the minimum index, matching the sequential first-maximum rule.
        int in[2] = {lc, ls};
        int out[2] = {0, 0};
        MPI_Allreduce(in, out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int bestCard = out[0];
        const int bestSeed = out[1];
        if (bestCard <= 0) break;  // no cluster can be formed

        // 4. Member list from the owning rank, shared with everyone.
        const int owner = bestSeed % nprocs;
        std::vector<int> members;
        if (owner == rank) {
            members = gpuMembers(gpu, bestSeed, bestCard);
        } else {
            members.resize(bestCard);
        }
        MPI_Bcast(members.data(), bestCard, MPI_INT, owner, MPI_COMM_WORLD);

        // 5. Commit the cluster.
        for (int q : members) clustered[q] = 1;
        numClustered += bestCard;
        clusters.push_back(Cluster{members, bestSeed});

        // 6. Owned unclustered seeds within the threshold of a newly clustered
        //    point have changed neighbourhoods: mark them dirty.
        const int ncx = hg.g.ncx;
        #pragma omp parallel for schedule(dynamic, 32)
        for (size_t k = 0; k < members.size(); ++k) {
            const int q = members[k];
            const int cq = hg.cellOf[q];
            const int cx = cq % ncx;
            const int cy = cq / ncx;
            for (int dy = -1; dy <= 1; ++dy) {
                const int y = cy + dy;
                if (y < 0 || y >= hg.g.ncy) continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    const int x = cx + dx;
                    if (x < 0 || x >= ncx) continue;
                    const int c = y * ncx + x;
                    for (int j = hg.cellStart[c]; j < hg.cellStart[c + 1]; ++j) {
                        const int p = hg.cellPts[j];
                        if (p % nprocs != rank || clustered[p]) continue;
                        if (distance(points[q], points[p]) < threshold) {
                            #pragma omp atomic write
                            dirty[p] = 1;
                        }
                    }
                }
            }
        }
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                      const std::vector<Point>& points,
                      const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");

    // Diameters (max pairwise distance) are independent per cluster
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }
    
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
        
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    
    // Count clustered points
    int clustered_count = 0;
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
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
    const bool root = (rank == 0);

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
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
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank to a GPU and divide the host cores among node-local ranks.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "rank %d: no CUDA devices available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
    
    // Generate synthetic data (identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    const HostGrid grid = buildGrid(points, threshold);
    Gpu gpu;
    gpuInit(gpu, points, grid, threshold);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters =
        qtClustering(points, grid, threshold, gpu, rank, nprocs);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
    int exitCode = 0;
    if (root) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time.count()));
        printf("Clusters found: %zu\n", clusters.size());
        
        // Calculate statistics and performance metrics
        int total_clustered = 0;
        int max_cluster_size = 0;
        
        for (size_t i = 0; i < clusters.size(); ++i) {
            const int size = static_cast<int>(clusters[i].members.size());
            total_clustered += size;
            max_cluster_size = std::max(max_cluster_size, size);
        }
        
        const double avg_cluster_size = clusters.empty() ? 0.0 :
            static_cast<double>(total_clustered) / clusters.size();
        
        printf("Points clustered: %d / %d (%.1f%%)\n",
               total_clustered, num_points,
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        
        // Performance metrics
        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n",
               clusters_per_sec, points_per_sec);
        
        // Print results for external validation
        if (printResults) {
            // Serialize cluster membership for hashing
            std::vector<double> membershipData;
            membershipData.reserve(num_points);
            std::vector<int> membership(num_points, -1);
            for (size_t c = 0; c < clusters.size(); ++c) {
                for (size_t i = 0; i < clusters[c].members.size(); ++i) {
                    membership[clusters[c].members[i]] = static_cast<int>(c);
                }
            }
            for (int m : membership) {
                membershipData.push_back(static_cast<double>(m));
            }
            print_results(membershipData, "ClusterMembership");
        }
        
        // Validation
        if (validate) {
            const bool valid = validateClusters(clusters, points, threshold);
            
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }

    if (validate) {
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    gpuFree(gpu);
    MPI_Finalize();
    return exitCode;
}
