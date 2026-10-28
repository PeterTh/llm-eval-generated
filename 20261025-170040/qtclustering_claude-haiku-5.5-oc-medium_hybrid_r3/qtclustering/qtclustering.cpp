// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//  - MPI:    the seeds of each greedy growth round are dealt cyclically across
//            ranks. Each rank reports its best (cardinality, seed) and an
//            allgather selects the global winner.
//  - OpenMP: each rank runs one host thread per GPU it drives. Threads pull
//            batches of seeds with dynamic scheduling. OpenMP also parallelizes
//            the host-side validation.
//  - CUDA:   one thread block grows the candidate cluster of one seed. A uniform
//            grid (cell size >= threshold) limits the initial candidates to the
//            seed's 3x3 cell neighborhood. Each growth step is a block-wide argmin
//            over the active candidate list, and the list is compacted every step.
//            A candidate whose max-distance to the cluster reaches the threshold
//            can never become eligible again, so dropping it is exact.
//
// Semantics match the sequential algorithm: the candidate chosen at each step
// is the one with the smallest max-distance (ties: smallest index), the seed
// with the largest cluster wins (ties: smallest index), and distances use the
// same IEEE arithmetic on host and device (FMA contraction is disabled).

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

static constexpr int kBlockSize = 256;                              // threads per CUDA block
static constexpr size_t kWorkspaceBudgetBytes = size_t(1) << 30;    // per GPU context

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

[[noreturn]] static void fatal(const char* msg) {
    fprintf(stderr, "Fatal: %s\n", msg);
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
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

// Calculate Euclidean distance between two points (host)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Same arithmetic as distance(), for use on the device
__device__ __forceinline__ double distanceDev(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Lexicographic order on (distance, index): smaller distance wins, ties go to lower index
__device__ __forceinline__ bool pairLess(double av, int ai, double bv, int bi) {
    return av < bv || (av == bv && ai < bi);
}

// Uniform grid over the points. Cell size is at least the threshold, so every point
// within distance < threshold of a seed lies in the seed's 3x3 cell neighborhood.
// Points are sorted by cell (CSR layout), cellPts[cellStart[k]..cellStart[k+1]).
struct Grid {
    int gx = 1;
    int gy = 1;
    std::vector<int> cellOf;
    std::vector<int> cellStart;
    std::vector<int> cellPts;
};

Grid buildGrid(const std::vector<Point>& pts, const double threshold) {
    Grid g;
    const int N = static_cast<int>(pts.size());
    double minX = pts[0].x, maxX = pts[0].x, minY = pts[0].y, maxY = pts[0].y;
    for (const Point& p : pts) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }
    const double extent = std::max(maxX - minX, maxY - minY);
    // Cell size >= threshold (with margin); also bound the cell count to about N
    double cs = threshold * (1.0 + 1e-9);
    cs = std::max(cs, extent / std::max(1.0, std::sqrt(static_cast<double>(N))));
    if (!(cs > 0.0)) cs = 1.0;

    g.gx = static_cast<int>(std::floor((maxX - minX) / cs)) + 1;
    g.gy = static_cast<int>(std::floor((maxY - minY) / cs)) + 1;
    const int numCells = g.gx * g.gy;

    g.cellOf.resize(N);
    for (int i = 0; i < N; ++i) {
        int cx = static_cast<int>(std::floor((pts[i].x - minX) / cs));
        int cy = static_cast<int>(std::floor((pts[i].y - minY) / cs));
        cx = std::min(std::max(cx, 0), g.gx - 1);
        cy = std::min(std::max(cy, 0), g.gy - 1);
        g.cellOf[i] = cy * g.gx + cx;
    }

    g.cellStart.assign(numCells + 1, 0);
    for (int i = 0; i < N; ++i) g.cellStart[g.cellOf[i] + 1]++;
    for (int k = 0; k < numCells; ++k) g.cellStart[k + 1] += g.cellStart[k];
    g.cellPts.resize(N);
    std::vector<int> fill(g.cellStart.begin(), g.cellStart.end() - 1);
    for (int i = 0; i < N; ++i) g.cellPts[fill[g.cellOf[i]]++] = i;
    return g;
}

// Per-OpenMP-thread GPU context: one device, persistent buffers
struct GpuCtx {
    int dev = 0;
    int maxBatch = 0;      // max seeds grown concurrently (one block each)
    int gx = 1;
    int gy = 1;
    Point* dPts = nullptr;
    int* dCellOf = nullptr;
    int* dCellStart = nullptr;
    int* dCellPts = nullptr;
    unsigned char* dClustered = nullptr;
    int* dSeeds = nullptr;
    int* dCards = nullptr;
    int* dMembers = nullptr;
    int* wsIdx = nullptr;
    double* wsDist = nullptr;

    void init(int device, const std::vector<Point>& pts, const Grid& grid) {
        const int N = static_cast<int>(pts.size());
        const int numCells = grid.gx * grid.gy;
        dev = device;
        gx = grid.gx;
        gy = grid.gy;
        CUDA_CHECK(cudaSetDevice(dev));
        const size_t perBlock = 24ull * static_cast<size_t>(N);  // 2 x (int + double) x N
        size_t mb = kWorkspaceBudgetBytes / perBlock;
        mb = std::max<size_t>(1, std::min<size_t>(mb, static_cast<size_t>(N)));
        maxBatch = static_cast<int>(mb);

        CUDA_CHECK(cudaMalloc(&dPts, sizeof(Point) * N));
        CUDA_CHECK(cudaMalloc(&dCellOf, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&dCellStart, sizeof(int) * (numCells + 1)));
        CUDA_CHECK(cudaMalloc(&dCellPts, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&dClustered, sizeof(unsigned char) * N));
        CUDA_CHECK(cudaMalloc(&dSeeds, sizeof(int) * maxBatch));
        CUDA_CHECK(cudaMalloc(&dCards, sizeof(int) * maxBatch));
        CUDA_CHECK(cudaMalloc(&dMembers, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&wsIdx, sizeof(int) * 2 * static_cast<size_t>(N) * maxBatch));
        CUDA_CHECK(cudaMalloc(&wsDist, sizeof(double) * 2 * static_cast<size_t>(N) * maxBatch));

        CUDA_CHECK(cudaMemcpy(dPts, pts.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCellOf, grid.cellOf.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCellStart, grid.cellStart.data(), sizeof(int) * (numCells + 1),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dCellPts, grid.cellPts.data(), sizeof(int) * N,
                              cudaMemcpyHostToDevice));
    }

    void destroy() {
        CUDA_CHECK(cudaSetDevice(dev));
        cudaFree(dPts);
        cudaFree(dCellOf);
        cudaFree(dCellStart);
        cudaFree(dCellPts);
        cudaFree(dClustered);
        cudaFree(dSeeds);
        cudaFree(dCards);
        cudaFree(dMembers);
        cudaFree(wsIdx);
        cudaFree(wsDist);
    }
};

// Append candidate (c, m) to the output list if keep is set. Warp-aggregated atomics
// on the shared counter give each kept entry a unique slot; slot order is irrelevant
// because selection is by the (distance, index) minimum. Also tracks the thread-local
// minimum (bv, bi).
__device__ __forceinline__ void consumeCandidate(const bool keep, const int c, const double m,
                                                 int* outIdx, double* outDist, int* s_cnt,
                                                 double& bv, int& bi) {
    const int lane = threadIdx.x & 31;
    const unsigned mask = __ballot_sync(0xffffffffu, keep);
    const int wcnt = __popc(mask);
    const int leader = mask ? (__ffs(mask) - 1) : 0;
    int base = 0;
    if (lane == leader && wcnt > 0) base = atomicAdd(s_cnt, wcnt);
    base = __shfl_sync(0xffffffffu, base, leader);
    if (keep) {
        const int pos = base + __popc(mask & ((1u << lane) - 1u));
        outIdx[pos] = c;
        outDist[pos] = m;
        if (pairLess(m, c, bv, bi)) {
            bv = m;
            bi = c;
        }
    }
}

// Block-wide argmin over (distance, index). Result (-1 if none) is returned to all
// threads. Contains barriers; the caller must not write s_res before its next barrier.
__device__ __forceinline__ int blockArgmin(double bv, int bi, double* s_wv, int* s_wi, int* s_res) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, bv, off);
        const int oi = __shfl_down_sync(0xffffffffu, bi, off);
        if (pairLess(ov, oi, bv, bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        s_wv[warp] = bv;
        s_wi[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        constexpr int nw = kBlockSize / 32;
        double v = (lane < nw) ? s_wv[lane] : CUDART_INF;
        int ii = (lane < nw) ? s_wi[lane] : INT_MAX;
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, v, off);
            const int oi = __shfl_down_sync(0xffffffffu, ii, off);
            if (pairLess(ov, oi, v, ii)) {
                v = ov;
                ii = oi;
            }
        }
        if (lane == 0) *s_res = (ii == INT_MAX) ? -1 : ii;
    }
    __syncthreads();
    return *s_res;
}

// Grow the candidate cluster for each seed in seeds[0..gridDim.x).
// One thread block per seed; writes the cluster cardinality to cards[blockIdx.x].
// If members_out is non-null (single-block launch only), the members are written in
// the order they were added.
//
// Algorithm for one seed (identical to the sequential greedy growth):
//   active list = unclustered points (other than seed) within threshold of seed,
//                 with running max-distance to the members (initially dist to seed)
//   repeat: pick the active point with minimum max-distance (ties: lowest index),
//           add it to the cluster, update the running max-distances of the remaining
//           active points, and drop points whose max-distance reached the threshold.
// Dropped points can never become eligible again because max-distance only grows.
__global__ void __launch_bounds__(kBlockSize)
growClusterKernel(const Point* __restrict__ pts,
                  const int* __restrict__ cellOf,
                  const int* __restrict__ cellStart,
                  const int* __restrict__ cellPts,
                  const int gx, const int gy,
                  const unsigned char* __restrict__ clustered,
                  const int* __restrict__ seeds,
                  const int N, const double threshold,
                  int* __restrict__ wsIdx, double* __restrict__ wsDist,
                  int* __restrict__ cards, int* __restrict__ members_out)
{
    __shared__ int s_cnt;
    __shared__ int s_res;
    __shared__ double s_wv[kBlockSize / 32];
    __shared__ int s_wi[kBlockSize / 32];

    const int tid = threadIdx.x;
    const int blk = blockIdx.x;
    const int seed = seeds[blk];
    const Point pseed = pts[seed];
    const size_t NN = static_cast<size_t>(N);

    int* idxBuf = wsIdx + static_cast<size_t>(blk) * 2 * NN;
    double* distBuf = wsDist + static_cast<size_t>(blk) * 2 * NN;

    // Seed's 3x3 cell neighborhood: rows of contiguous cells -> up to 3 point ranges
    const int cell = cellOf[seed];
    const int cx = cell % gx;
    const int cy = cell / gx;
    const int x0 = max(cx - 1, 0);
    const int x1 = min(cx + 1, gx - 1);
    const int y0 = max(cy - 1, 0);
    const int y1 = min(cy + 1, gy - 1);
    int rs[3];
    int cum[3];
    int nr = 0;
    int K = 0;
    for (int y = y0; y <= y1; ++y) {
        const int rowBase = y * gx;
        rs[nr] = cellStart[rowBase + x0];
        cum[nr] = K;
        K += cellStart[rowBase + x1 + 1] - rs[nr];
        ++nr;
    }

    if (members_out != nullptr && blk == 0 && tid == 0) members_out[0] = seed;
    int members = 1;   // the seed

    // Step 0: active list from the neighborhood (seed excluded, clustered excluded)
    if (tid == 0) s_cnt = 0;
    __syncthreads();
    double bv = CUDART_INF;
    int bi = INT_MAX;
    for (int t0 = 0; t0 < K; t0 += kBlockSize) {
        const int k = t0 + tid;
        bool keep = false;
        int c = 0;
        double m = 0.0;
        if (k < K) {
            int r = 0;
            if (nr > 1 && k >= cum[1]) r = 1;
            if (nr > 2 && k >= cum[2]) r = 2;
            c = cellPts[rs[r] + (k - cum[r])];
            if (!clustered[c] && c != seed) {
                m = distanceDev(pts[c], pseed);   // max-distance to {seed}
                keep = m < threshold;
            }
        }
        consumeCandidate(keep, c, m, idxBuf, distBuf, &s_cnt, bv, bi);
    }
    int next = blockArgmin(bv, bi, s_wv, s_wi, &s_res);
    int nin = s_cnt;
    __syncthreads();   // all threads have read s_cnt before it is reset

    // Greedy growth: each iteration adds `next` to the cluster
    int cur = 0;
    while (next >= 0) {
        const int best = next;
        if (members_out != nullptr && blk == 0 && tid == 0) members_out[members] = best;
        ++members;

        const int* inIdx = idxBuf + (cur ? NN : 0);
        const double* inDist = distBuf + (cur ? NN : 0);
        int* outIdx = idxBuf + (cur ? 0 : NN);
        double* outDist = distBuf + (cur ? 0 : NN);

        if (tid == 0) s_cnt = 0;
        __syncthreads();

        const Point pb = pts[best];
        bv = CUDART_INF;
        bi = INT_MAX;
        for (int t0 = 0; t0 < nin; t0 += kBlockSize) {
            const int i = t0 + tid;
            bool keep = false;
            int c = 0;
            double m = 0.0;
            if (i < nin) {
                c = inIdx[i];
                if (c != best) {
                    m = fmax(inDist[i], distanceDev(pts[c], pb));
                    keep = m < threshold;
                }
            }
            consumeCandidate(keep, c, m, outIdx, outDist, &s_cnt, bv, bi);
        }
        next = blockArgmin(bv, bi, s_wv, s_wi, &s_res);
        const int nout = s_cnt;
        __syncthreads();   // all threads have read s_cnt before it is reset

        cur ^= 1;
        nin = nout;
    }

    if (tid == 0) cards[blk] = members;
}

// Main QT clustering algorithm
// Each round grows a candidate cluster from every unclustered seed and keeps the
// largest one (ties: lowest seed index). Every rank holds an identical copy of the
// unclustered list and clustered flags, and therefore computes the same clusters.
std::vector<Cluster> qtClustering(std::vector<GpuCtx>& ctx,
                                  const int N,
                                  const double threshold,
                                  const int rank,
                                  const int nranks) {
    const int nthr = static_cast<int>(ctx.size());
    const int maxBatch = ctx[0].maxBatch;

    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<unsigned char> clustered(N, 0);

    std::vector<Cluster> clusters;
    std::vector<int> localSeeds;
    std::vector<int> cards;
    std::vector<int> pairs(2 * static_cast<size_t>(nranks));

    while (!unclustered.empty()) {
        const int nU = static_cast<int>(unclustered.size());

        // Cyclic distribution of the seeds across ranks
        localSeeds.clear();
        for (int i = rank; i < nU; i += nranks) {
            localSeeds.push_back(unclustered[i]);
        }
        const int nl = static_cast<int>(localSeeds.size());
        cards.assign(nl, 0);

        const int B = std::min(maxBatch, std::max(64, (nl + 4 * nthr - 1) / (4 * nthr)));
        const int nb = (nl + B - 1) / B;

        // Grow the candidate clusters of all local seeds, one device per OpenMP thread
#pragma omp parallel num_threads(nthr)
        {
            GpuCtx& g = ctx[omp_get_thread_num()];
            CUDA_CHECK(cudaSetDevice(g.dev));
            CUDA_CHECK(cudaMemcpy(g.dClustered, clustered.data(), sizeof(unsigned char) * N,
                                  cudaMemcpyHostToDevice));

#pragma omp for schedule(dynamic, 1)
            for (int b = 0; b < nb; ++b) {
                const int s0 = b * B;
                const int cnt = std::min(B, nl - s0);
                CUDA_CHECK(cudaMemcpy(g.dSeeds, localSeeds.data() + s0, sizeof(int) * cnt,
                                      cudaMemcpyHostToDevice));
                growClusterKernel<<<cnt, kBlockSize>>>(g.dPts, g.dCellOf, g.dCellStart, g.dCellPts,
                                                       g.gx, g.gy, g.dClustered, g.dSeeds,
                                                       N, threshold, g.wsIdx, g.wsDist,
                                                       g.dCards, nullptr);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(cards.data() + s0, g.dCards, sizeof(int) * cnt,
                                      cudaMemcpyDeviceToHost));
            }
        }

        // Local best: largest cardinality, ties to the lowest seed index
        int localBestCard = -1;
        int localBestSeed = -1;
        for (int i = 0; i < nl; ++i) {
            if (cards[i] > localBestCard) {
                localBestCard = cards[i];
                localBestSeed = localSeeds[i];
            }
        }

        // Global best across ranks
        const int mine[2] = {localBestCard, localBestSeed};
        MPI_Allgather(mine, 2, MPI_INT, pairs.data(), 2, MPI_INT, MPI_COMM_WORLD);
        int bestCard = -1;
        int bestSeed = -1;
        for (int r = 0; r < nranks; ++r) {
            const int c = pairs[2 * r];
            const int s = pairs[2 * r + 1];
            if (c > bestCard || (c == bestCard && s >= 0 && (bestSeed < 0 || s < bestSeed))) {
                bestCard = c;
                bestSeed = s;
            }
        }

        if (bestSeed < 0 || bestCard <= 0) break;   // no more clusters can be formed

        // Recompute the winning cluster's members (identical result on every rank)
        GpuCtx& g = ctx[0];
        CUDA_CHECK(cudaSetDevice(g.dev));
        CUDA_CHECK(cudaMemcpy(g.dSeeds, &bestSeed, sizeof(int), cudaMemcpyHostToDevice));
        growClusterKernel<<<1, kBlockSize>>>(g.dPts, g.dCellOf, g.dCellStart, g.dCellPts,
                                             g.gx, g.gy, g.dClustered, g.dSeeds,
                                             N, threshold, g.wsIdx, g.wsDist,
                                             g.dCards, g.dMembers);
        CUDA_CHECK(cudaGetLastError());
        int len = 0;
        CUDA_CHECK(cudaMemcpy(&len, g.dCards, sizeof(int), cudaMemcpyDeviceToHost));
        if (len != bestCard) fatal("cluster recomputation mismatch");

        Cluster cluster;
        cluster.seed_point = bestSeed;
        cluster.members.resize(len);
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), g.dMembers, sizeof(int) * len,
                              cudaMemcpyDeviceToHost));
        for (int m : cluster.members) clustered[m] = 1;
        clusters.push_back(std::move(cluster));

        // Remove clustered points from the unclustered list (order preserved)
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
    }

    return clusters;
}

bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Diameters are computed in parallel (max is order independent)
    std::vector<double> diameters(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic)
    for (long long c = 0; c < static_cast<long long>(clusters.size()); ++c) {
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
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }
    
    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Map ranks to GPUs: each rank drives the devices assigned to it on its node
    MPI_Comm localComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
    int localRank = 0;
    int localSize = 1;
    MPI_Comm_rank(localComm, &localRank);
    MPI_Comm_size(localComm, &localSize);
    MPI_Comm_free(&localComm);

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev < 1) fatal("no CUDA devices available");
    const int nthr = std::max(1, ndev / localSize);

    // Spatial grid for neighborhood gathering (cell size >= threshold)
    const Grid grid = buildGrid(points, threshold);

    omp_set_dynamic(0);
    std::vector<GpuCtx> ctx(nthr);
#pragma omp parallel num_threads(nthr)
    {
        const int t = omp_get_thread_num();
        const int dev = (localRank * nthr + t) % ndev;
        ctx[t].init(dev, points, grid);
    }
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_start = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(ctx, num_points, threshold, rank, nranks);
    
    MPI_Barrier(MPI_COMM_WORLD);
    const double cluster_end = MPI_Wtime();
    const long cluster_ms = static_cast<long>((cluster_end - cluster_start) * 1000.0);

    for (auto& g : ctx) g.destroy();

    int rc = 0;
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", cluster_ms);
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
        const double time_sec = (cluster_end - cluster_start);
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
                rc = 1;
            }
        }
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return rc;
}
