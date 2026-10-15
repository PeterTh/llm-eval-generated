// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA version
//
// Parallelization strategy:
//  * Points are sorted into a uniform grid (cell >= threshold), so a cluster
//    grown from a seed only ever touches the 3x3 cells around that seed.
//  * Candidate seeds are distributed cyclically over MPI ranks (one GPU each).
//    Every GPU grows one candidate cluster per CUDA thread block, keeping an
//    incremental "max distance to cluster" per candidate (O(size*K) instead of
//    O(size^2*N)).
//  * A seed's candidate cluster only changes when a point within the threshold
//    of the seed gets removed, so cardinalities are cached and only
//    invalidated seeds are recomputed in subsequent rounds.
//  * The global winner (max cardinality, lowest index) is selected with an
//    MPI_Allreduce; each rank recomputes the (single) winning cluster itself.
//  * OpenMP is used for host-side preprocessing, statistics and validation.
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

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
            const double dy = std::sqrt(r * r - dx * dx) * sign;
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                       \
                    cudaGetErrorString(err_), __FILE__, __LINE__);            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                     \
        }                                                                     \
    } while (0)

static const int BLOCK = 256;
static const int NWARPS = BLOCK / 32;

// Candidate record kept in per-block scratch memory
struct Cand {
    double x, y;
    double md2;  // squared max distance to the current cluster members
    float fx, fy;  // single precision copies used for cheap conservative tests
    float mdf;     // (float)md2; > 1e30 marks a candidate that must be dropped
    int orig;
};

// Device-side view of the (grid-sorted) problem
struct DevData {
    const double2* pts;        // points sorted by grid cell
    const int* orig;           // sorted position -> original index
    const int* origToPos;      // original index -> sorted position
    const int* cellStart;      // size ncell+1
    unsigned char* clustered;  // by sorted position
    int N, nx, ny;
    double inv_cell;
    double thr2;               // largest squared distance whose sqrt is < threshold
};

// Squared distance (no FMA contraction, matches host evaluation order). Since
// sqrt is monotone, max/min/threshold decisions on d^2 equal those on d.
__device__ __forceinline__ double dist2(const double ax, const double ay,
                                        const double bx, const double by) {
    const double dx = __dsub_rn(ax, bx);
    const double dy = __dsub_rn(ay, by);
    return __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
}

__device__ __forceinline__ bool lexLess(double d1, int o1, double d2, int o2) {
    return d1 < d2 || (d1 == d2 && o1 < o2);
}

// Grow the QT candidate cluster of a seed with one thread block.
// Each candidate tracks the max distance to the current members; dead candidates
// (>= threshold) are compacted away on every pass.
// Returns the cluster size; if out != nullptr, writes member original indices.
__device__ int growCluster(const DevData& D, const int seedPos, Cand* bufA, Cand* bufB,
                           int* out) {
    __shared__ int s_n, s_cnt;
    __shared__ double s_wd[NWARPS];
    __shared__ int s_wo[NWARPS], s_wk[NWARPS];
    __shared__ int s_sel;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const double INF = 1e300;

    const double2 sp = D.pts[seedPos];
    if (tid == 0) { s_n = 0; s_cnt = 0; }
    __syncthreads();

    Cand* cur = bufA;
    Cand* nxt = bufB;

    const int cx = min(static_cast<int>(sp.x * D.inv_cell), D.nx - 1);
    const int cy = min(static_cast<int>(sp.y * D.inv_cell), D.ny - 1);
    const int x0 = max(cx - 1, 0), x1 = min(cx + 1, D.nx - 1);
    for (int y = max(cy - 1, 0); y <= min(cy + 1, D.ny - 1); ++y) {
        const int a = D.cellStart[y * D.nx + x0];
        const int b = D.cellStart[y * D.nx + x1 + 1];
        for (int p0 = a; p0 < b; p0 += BLOCK) {
            const int p = p0 + tid;
            bool ok = false;
            double2 q = make_double2(0, 0);
            double d2 = 0;
            if (p < b && p != seedPos && !D.clustered[p]) {
                q = D.pts[p];
                d2 = dist2(q.x, q.y, sp.x, sp.y);
                ok = d2 <= D.thr2;
            }
            const unsigned m = __ballot_sync(0xffffffffu, ok);
            int base = 0;
            if (lane == 0 && m) base = atomicAdd(&s_cnt, __popc(m));
            base = __shfl_sync(0xffffffffu, base, 0);
            if (ok) {
                Cand c; c.x = q.x; c.y = q.y; c.md2 = d2; c.orig = D.orig[p];
                c.fx = static_cast<float>(q.x); c.fy = static_cast<float>(q.y);
                c.mdf = static_cast<float>(d2);
                cur[base + __popc(m & ((1u << lane) - 1u))] = c;
            }
        }
    }
    __syncthreads();
    if (tid == 0) { s_n = s_cnt; s_cnt = 0; }
    __syncthreads();

    int members = 1;
    if (out && tid == 0) out[0] = D.orig[seedPos];
    double lx = sp.x, ly = sp.y;
    float lfx = static_cast<float>(sp.x), lfy = static_cast<float>(sp.y);
    bool first = true;

    while (true) {
        const int n = s_n;
        double bd = INF;
        float bdf = 3e38f;
        int bo = 0x7fffffff, bk = -1;
        for (int k0 = 0; k0 < n; k0 += BLOCK) {
            const int k = k0 + tid;
            bool alive = false;
            Cand c;
            if (k < n) {
                c = cur[k];
                if (c.mdf <= 1e30f) {
                    alive = true;
                    if (!first) {
                        // Cheap float test: skip the exact update when the new member
                        // is certainly not farther than the current max distance.
                        const float dx = c.fx - lfx, dy = c.fy - lfy;
                        const float d2f = dx * dx + dy * dy;
                        if (!(d2f + 1e-3f + 1e-5f * d2f < c.mdf * (1.0f - 1e-6f))) {
                            const double d2 = dist2(c.x, c.y, lx, ly);
                            if (d2 > c.md2) { c.md2 = d2; c.mdf = static_cast<float>(d2); }
                            alive = c.md2 <= D.thr2;
                        }
                    }
                }
            }
            const unsigned m = __ballot_sync(0xffffffffu, alive);
            if (m) {
                int base = 0;
                if (lane == 0) base = atomicAdd(&s_cnt, __popc(m));
                base = __shfl_sync(0xffffffffu, base, 0);
                if (alive) {
                    const int idx = base + __popc(m & ((1u << lane) - 1u));
                    nxt[idx] = c;
                    if (c.mdf <= bdf && lexLess(c.md2, c.orig, bd, bo)) {
                        bd = c.md2; bdf = static_cast<float>(bd); bo = c.orig; bk = idx;
                    }
                }
            }
        }
        first = false;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const double od = __shfl_down_sync(0xffffffffu, bd, off);
            const int oo = __shfl_down_sync(0xffffffffu, bo, off);
            const int ok = __shfl_down_sync(0xffffffffu, bk, off);
            if (lexLess(od, oo, bd, bo)) { bd = od; bo = oo; bk = ok; }
        }
        if (lane == 0) { s_wd[tid >> 5] = bd; s_wo[tid >> 5] = bo; s_wk[tid >> 5] = bk; }
        __syncthreads();
        if (tid == 0) {
            double cd = s_wd[0]; int co = s_wo[0], ck = s_wk[0];
            for (int w = 1; w < NWARPS; ++w)
                if (lexLess(s_wd[w], s_wo[w], cd, co)) { cd = s_wd[w]; co = s_wo[w]; ck = s_wk[w]; }
            s_sel = ck;
            if (ck >= 0) {
                nxt[ck].mdf = 3e38f;  // becomes a member; dropped on the next pass
                if (out) out[members] = nxt[ck].orig;
            }
            s_n = s_cnt;
            s_cnt = 0;
        }
        __syncthreads();
        const int sel = s_sel;
        if (sel < 0) break;
        lx = nxt[sel].x;
        ly = nxt[sel].y;
        lfx = nxt[sel].fx;
        lfy = nxt[sel].fy;
        ++members;
        Cand* t = cur; cur = nxt; nxt = t;
    }
    __syncthreads();
    return members;
}

// Seed kernel: persistent blocks fetch dirty seeds through an atomic ticket.
__global__ void seedKernel(DevData D, const int* ownPos, int* card, const int* dirty,
                           const int* dirtyCount, int* ticket, Cand* scratch,
                           const size_t stride) {
    __shared__ int s_t;
    Cand* bufA = scratch + 2 * blockIdx.x * stride;
    Cand* bufB = bufA + stride;
    const int total = *dirtyCount;
    while (true) {
        if (threadIdx.x == 0) s_t = atomicAdd(ticket, 1);
        __syncthreads();
        const int t = s_t;
        if (t >= total) break;
        const int i = dirty[t];
        const int c = growCluster(D, ownPos[i], bufA, bufB, nullptr);
        if (threadIdx.x == 0) card[i] = c;
        __syncthreads();
    }
}

// Mark seeds whose cached candidate cluster may have been affected by the last
// removed cluster (bbox in wbox, members in wmem) as dirty (-2).
__global__ void dirtyKernel(DevData D, const int* ownPos, int nOwn, int* card,
                            const double2* wpts, const int* wcount, const double* wbox,
                            int* dirty, int* dirtyCount) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nOwn) return;
    const int pos = ownPos[i];
    int c = card[i];
    if (c == -1) return;
    if (D.clustered[pos]) { card[i] = -1; return; }
    if (c != -2) {
        const int wc = *wcount;
        const double2 p = D.pts[pos];
        const double m = D.thr2 * 1.00001;
        const double mb = sqrt(m);
        if (wc > 0 && p.x >= wbox[0] - mb && p.x <= wbox[1] + mb &&
            p.y >= wbox[2] - mb && p.y <= wbox[3] + mb) {
            for (int j = 0; j < wc; ++j) {
                const double2 q = wpts[j];
                if (dist2(q.x, q.y, p.x, p.y) <= m) { c = -2; break; }
            }
        }
        if (c == -2) card[i] = -2;
    }
    if (c == -2) dirty[atomicAdd(dirtyCount, 1)] = i;
}

__global__ void argmaxKernel(DevData D, const int* ownPos, int nOwn, const int* card,
                             unsigned long long* key) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nOwn) return;
    const int c = card[i];
    if (c < 1) return;
    const unsigned long long k = static_cast<unsigned long long>(c) * (D.N + 1) +
                                 static_cast<unsigned long long>(D.N - 1 - D.orig[ownPos[i]]);
    atomicMax(key, k);
}

// Recompute the winning cluster, mark it clustered, export members and bbox.
__global__ void winnerKernel(DevData D, const int seedPos, Cand* bufA, Cand* bufB,
                             int* wmem, double2* wpts, int* wcount, double* wbox) {
    __shared__ double s_box[4 * NWARPS];
    const int c = growCluster(D, seedPos, bufA, bufB, wmem);
    double mnx = 1e300, mxx = -1e300, mny = 1e300, mxy = -1e300;
    for (int j = threadIdx.x; j < c; j += BLOCK) {
        const int p = D.origToPos[wmem[j]];
        D.clustered[p] = 1;
        const double2 q = D.pts[p];
        wpts[j] = q;
        mnx = fmin(mnx, q.x); mxx = fmax(mxx, q.x);
        mny = fmin(mny, q.y); mxy = fmax(mxy, q.y);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mnx = fmin(mnx, __shfl_down_sync(0xffffffffu, mnx, off));
        mxx = fmax(mxx, __shfl_down_sync(0xffffffffu, mxx, off));
        mny = fmin(mny, __shfl_down_sync(0xffffffffu, mny, off));
        mxy = fmax(mxy, __shfl_down_sync(0xffffffffu, mxy, off));
    }
    const int w = threadIdx.x >> 5;
    if ((threadIdx.x & 31) == 0) {
        s_box[w * 4 + 0] = mnx; s_box[w * 4 + 1] = mxx;
        s_box[w * 4 + 2] = mny; s_box[w * 4 + 3] = mxy;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int k = 1; k < NWARPS; ++k) {
            s_box[0] = fmin(s_box[0], s_box[k * 4 + 0]);
            s_box[1] = fmax(s_box[1], s_box[k * 4 + 1]);
            s_box[2] = fmin(s_box[2], s_box[k * 4 + 2]);
            s_box[3] = fmax(s_box[3], s_box[k * 4 + 3]);
        }
        wbox[0] = s_box[0]; wbox[1] = s_box[1]; wbox[2] = s_box[2]; wbox[3] = s_box[3];
        *wcount = c;
    }
}

__global__ void resetKernel(int* ticket, int* dirtyCount, unsigned long long* key) {
    *ticket = 0;
    *dirtyCount = 0;
    *key = 0;
}

// Bind one GPU per rank (round robin over the GPUs of a node) and create the context.
void initGpu(const int rank) {
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int lrank = 0;
    MPI_Comm_rank(local, &lrank);
    MPI_Comm_free(&local);
    CUDA_CHECK(cudaSetDevice(lrank % ndev));
    CUDA_CHECK(cudaFree(0));
}

// Main QT clustering algorithm (distributed over MPI ranks, one GPU per rank).
// All ranks return the full cluster list.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const int N = static_cast<int>(points.size());

    int dev = 0, nsm = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));

    // ---- grid sort (cell size >= threshold, bounded cell count) ----
    const double cell = std::max(threshold, std::min(MAX_WIDTH, MAX_HEIGHT) / 2048.0);
    const int nx = std::max(1, static_cast<int>(MAX_WIDTH / cell) + 1);
    const int ny = std::max(1, static_cast<int>(MAX_HEIGHT / cell) + 1);
    const double inv_cell = 1.0 / cell;
    const int ncell = nx * ny;
    std::vector<int> cellOf(N);
    std::vector<int> cellStart(ncell + 1, 0);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const int cx = std::min(static_cast<int>(points[i].x * inv_cell), nx - 1);
        const int cy = std::min(static_cast<int>(points[i].y * inv_cell), ny - 1);
        cellOf[i] = cy * nx + cx;
    }
    for (int i = 0; i < N; ++i) cellStart[cellOf[i] + 1]++;
    for (int c = 0; c < ncell; ++c) cellStart[c + 1] += cellStart[c];
    std::vector<double2> spts(N);
    std::vector<int> sorig(N), origToPos(N);
    {
        std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
        for (int i = 0; i < N; ++i) {
            const int p = fill[cellOf[i]]++;
            spts[p] = make_double2(points[i].x, points[i].y);
            sorig[p] = i;
            origToPos[i] = p;
        }
    }
    // Upper bound on neighbor-list length
    int Kmax = 1;
    #pragma omp parallel for reduction(max : Kmax) collapse(2)
    for (int y = 0; y < ny; ++y) {
        for (int x = 0; x < nx; ++x) {
            const int x0 = std::max(x - 1, 0), x1 = std::min(x + 1, nx - 1);
            int s = 0;
            for (int yy = std::max(y - 1, 0); yy <= std::min(y + 1, ny - 1); ++yy)
                s += cellStart[yy * nx + x1 + 1] - cellStart[yy * nx + x0];
            Kmax = std::max(Kmax, s);
        }
    }

    // ---- seeds owned by this rank (cyclic by original index) ----
    std::vector<int> ownPos;
    for (int i = rank; i < N; i += nranks) ownPos.push_back(origToPos[i]);
    const int nOwn = static_cast<int>(ownPos.size());

    // ---- device allocations ----
    double2 *d_pts, *d_wpts; int *d_orig, *d_origToPos, *d_cellStart, *d_ownPos, *d_card, *d_dirty;
    int *d_dirtyCount, *d_ticket, *d_wmem, *d_wcount;
    Cand* d_scratch;
    unsigned char* d_clustered; double* d_wbox; unsigned long long* d_key;
    const size_t stride = static_cast<size_t>(Kmax);
    size_t freeB = 0, totB = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeB, &totB));
    size_t maxBlocks = static_cast<size_t>(nsm) * 4;
    const size_t perBlock = 2 * stride * sizeof(Cand);
    const size_t budget = freeB / 2;
    maxBlocks = std::max<size_t>(1, std::min(maxBlocks, budget / perBlock));
    // +1 scratch slot for the winner kernel
    const size_t nSlots = maxBlocks + 1;

    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_orig, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_cellStart, sizeof(int) * (ncell + 1)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_ownPos, sizeof(int) * std::max(nOwn, 1)));
    CUDA_CHECK(cudaMalloc(&d_card, sizeof(int) * std::max(nOwn, 1)));
    CUDA_CHECK(cudaMalloc(&d_dirty, sizeof(int) * std::max(nOwn, 1)));
    CUDA_CHECK(cudaMalloc(&d_dirtyCount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_ticket, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wmem, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_wcount, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_wbox, sizeof(double) * 4));
    CUDA_CHECK(cudaMalloc(&d_key, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_scratch, perBlock * nSlots));
    CUDA_CHECK(cudaMalloc(&d_wpts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&d_origToPos, sizeof(int) * N));
    CUDA_CHECK(cudaMemcpy(d_origToPos, origToPos.data(), sizeof(int) * N, cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_pts, spts.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_orig, sorig.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cellStart, cellStart.data(), sizeof(int) * (ncell + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, N));
    if (nOwn > 0) CUDA_CHECK(cudaMemcpy(d_ownPos, ownPos.data(), sizeof(int) * nOwn, cudaMemcpyHostToDevice));
    {
        std::vector<int> init(std::max(nOwn, 1), -2);
        CUDA_CHECK(cudaMemcpy(d_card, init.data(), sizeof(int) * init.size(), cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(d_wcount, 0, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_wbox, 0, sizeof(double) * 4));

    DevData D;
    // largest squared distance whose (correctly rounded) sqrt is still < threshold
    double t2 = threshold * threshold;
    while (std::sqrt(t2) >= threshold) t2 = std::nextafter(t2, 0.0);
    while (std::sqrt(std::nextafter(t2, 1e300)) < threshold) t2 = std::nextafter(t2, 1e300);
    D.thr2 = t2;
    D.origToPos = d_origToPos;
    D.pts = d_pts; D.orig = d_orig; D.cellStart = d_cellStart; D.clustered = d_clustered;
    D.N = N; D.nx = nx; D.ny = ny; D.inv_cell = inv_cell;

    std::vector<Cluster> clusters;
    std::vector<int> wmem;
    const int gridOwn = (nOwn + BLOCK - 1) / BLOCK;

    while (true) {
        resetKernel<<<1, 1>>>(d_ticket, d_dirtyCount, d_key);
        if (nOwn > 0) {
            dirtyKernel<<<gridOwn, BLOCK>>>(D, d_ownPos, nOwn, d_card, d_wpts, d_wcount,
                                            d_wbox, d_dirty, d_dirtyCount);
            seedKernel<<<static_cast<unsigned>(maxBlocks), BLOCK>>>(
                D, d_ownPos, d_card, d_dirty, d_dirtyCount, d_ticket, d_scratch, stride);
            argmaxKernel<<<gridOwn, BLOCK>>>(D, d_ownPos, nOwn, d_card, d_key);
        }
        unsigned long long key = 0;
        CUDA_CHECK(cudaMemcpy(&key, d_key, sizeof(key), cudaMemcpyDeviceToHost));
        unsigned long long gkey = 0;
        MPI_Allreduce(&key, &gkey, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
        if (gkey == 0) break;

        const int winnerOrig = N - 1 - static_cast<int>(gkey % static_cast<unsigned long long>(N + 1));
        winnerKernel<<<1, BLOCK>>>(D, origToPos[winnerOrig],
                                   d_scratch + 2 * maxBlocks * stride,
                                   d_scratch + (2 * maxBlocks + 1) * stride,
                                   d_wmem, d_wpts, d_wcount, d_wbox);
        int wc = 0;
        CUDA_CHECK(cudaMemcpy(&wc, d_wcount, sizeof(int), cudaMemcpyDeviceToHost));
        wmem.resize(wc);
        CUDA_CHECK(cudaMemcpy(wmem.data(), d_wmem, sizeof(int) * wc, cudaMemcpyDeviceToHost));

        Cluster cl;
        cl.seed_point = winnerOrig;
        cl.members.resize(wc);
        for (int j = 0; j < wc; ++j) cl.members[j] = wmem[j];
        clusters.push_back(std::move(cl));
    }

    cudaFree(d_pts); cudaFree(d_orig); cudaFree(d_cellStart); cudaFree(d_clustered);
    cudaFree(d_ownPos); cudaFree(d_card); cudaFree(d_dirty); cudaFree(d_dirtyCount);
    cudaFree(d_ticket); cudaFree(d_wmem); cudaFree(d_wcount); cudaFree(d_wbox);
    cudaFree(d_key); cudaFree(d_scratch); cudaFree(d_wpts); cudaFree(d_origToPos);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute all cluster diameters in parallel
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (long c = 0; c < static_cast<long>(clusters.size()); ++c) {
        const auto& members = clusters[c].members;
        double max_diameter = 0.0;
        for (size_t i = 0; i < members.size(); ++i) {
            for (size_t j = i + 1; j < members.size(); ++j) {
                max_diameter = std::max(max_diameter, distance(points[members[i]], points[members[j]]));
            }
        }
        diameters[c] = max_diameter;
    }

    // Check each cluster
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

static int realMain(int argc, char** argv, const int rank, const int nranks) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    const bool root = (rank == 0);
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
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (root) printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                         num_points, threshold);
        return 1;
    }
    
    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    initGpu(rank);
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    long max_cluster_time = cluster_time.count();
    MPI_Allreduce(MPI_IN_PLACE, &max_cluster_time, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    cluster_time = std::chrono::milliseconds(max_cluster_time);
    
    if (!root) return 0;
    (void)nranks;
    printf("Clustering time: %ld ms\n", cluster_time.count());
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
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const int rc = realMain(argc, argv, rank, nranks);
    MPI_Finalize();
    return rc;
}
