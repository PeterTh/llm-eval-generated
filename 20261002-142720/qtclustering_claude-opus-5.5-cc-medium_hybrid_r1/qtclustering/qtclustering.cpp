// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (results are identical to the sequential version):
//  * Only points closer than the threshold to a seed can ever join its
//    candidate cluster, so a fixed-radius neighbor list (CSR, sorted by point
//    index) is built once on the host with OpenMP using a uniform grid.
//  * The diameter of "cluster + candidate" is maintained incrementally per
//    candidate (max is exact), which turns each growth step into O(degree).
//  * A seed's candidate cluster depends only on the clustered state of its
//    neighbors, so cardinalities are cached and only seeds adjacent to newly
//    clustered points are recomputed in later iterations.
//  * Seeds are distributed cyclically over MPI ranks (one GPU per rank); each
//    GPU grows its candidate clusters in parallel (one warp per seed for small
//    neighborhoods, one thread block per seed for large ones).
//  * The global best seed (max cardinality, lowest index on ties) is found
//    with MPI_Allreduce(MPI_MAXLOC); every rank then rebuilds the winning
//    cluster on its own GPU, so no member lists need to be communicated.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <math_constants.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

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

// Calculate Euclidean distance between two points.
// The rounding sequence (dy*dy rounded, then fused dx*dx + that) is spelled
// out explicitly so host and device produce bit-identical distances.
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(std::fma(dx, dx, dy * dy));
}

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

static constexpr int BLOCK_THREADS = 128;
static constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / 32;
static constexpr int CAP_WARP = 256;    // max neighbors handled by one warp (shared memory)
static constexpr int CAP_BLOCK = 1024;  // max neighbors kept in shared memory by one block

// Squared distance with the same rounding sequence as the host distance();
// distance == sqrt(squared distance) bit for bit.
__device__ __forceinline__ double deviceDistance2(double ax, double ay, double bx, double by) {
    const double dx = __dsub_rn(ax, bx);
    const double dy = __dsub_rn(ay, by);
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

// Lexicographic (diameter, position) minimum, i.e. smallest diameter and the
// lowest point index among ties, exactly like the sequential scan.
__device__ __forceinline__ void argminCombine(double& d, int& p, double od, int op) {
    if (od < d || (od == d && op < p)) {
        d = od;
        p = op;
    }
}

__device__ __forceinline__ void warpArgmin(double& d, int& p) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const double od = __shfl_xor_sync(0xffffffffu, d, o);
        const int op = __shfl_xor_sync(0xffffffffu, p, o);
        argminCombine(d, p, od, op);
    }
}

// Grow the candidate cluster of seed s with a group of threads (a warp, or a
// whole block if BLOCK is true) and write its members in insertion order.
// sx/sy hold candidate coordinates, sd the (bits of the) squared diameter the cluster would
// have if the candidate were added (INF = not eligible). Since sqrt is
// monotonic and correctly rounded, sqrt(max d^2) == max sqrt(d^2) and
// sqrt(x) < threshold <=> x < threshold2_bound, so only selection needs the
// rounded (unsquared) diameter, which is evaluated lazily.
template <bool BLOCK>
__device__ int growCluster(const int s,
                           const double2* __restrict__ pts,
                           const long long* __restrict__ off,
                           const int* __restrict__ nb,
                           const unsigned char* __restrict__ clustered,
                           const double threshold2_bound,
                           double* sx, double* sy, long long* sd,
                           const int lane, const int nthreads,
                           double* red_d, int* red_p,
                           int* members_out) {
    // Squared diameters are non-negative, so their IEEE bit patterns order
    // exactly like the values: compare/max them on the integer pipeline.
    const double INF = CUDART_INF;
    const long long INF_BITS = __double_as_longlong(CUDART_INF);
    const long long bound_bits = __double_as_longlong(threshold2_bound);
    const long long base = off[s];
    const int deg = static_cast<int>(off[s + 1] - base);
    const double2 ps = pts[s];

    // Running per-thread best: squared value bm, its sqrt bd, position bp.
    // Positions are visited in increasing order, so a later element only wins
    // with a strictly smaller (rounded) diameter.
    long long bm = INF_BITS;
    double bd = INF;
    int bp = INT_MAX;
    for (int p = lane; p < deg; p += nthreads) {
        const int j = nb[base + p];
        const double2 q = pts[j];
        sx[p] = q.x;
        sy[p] = q.y;
        long long m = INF_BITS;
        if (!clustered[j]) {
            const long long t = __double_as_longlong(deviceDistance2(q.x, q.y, ps.x, ps.y));
            if (t < bound_bits) m = t;
        }
        sd[p] = m;
        if (m < bm) {
            const double r = __dsqrt_rn(__longlong_as_double(m));
            bm = m;
            if (r < bd) { bd = r; bp = p; }
        }
    }

    if (lane == 0) members_out[0] = s;
    int card = 1;
    int buf = 0;

    while (true) {
        // Group-wide argmin on (rounded diameter, position)
        warpArgmin(bd, bp);
        if constexpr (BLOCK) {
            const int w = threadIdx.x >> 5;
            if ((threadIdx.x & 31) == 0) {
                red_d[buf * WARPS_PER_BLOCK + w] = bd;
                red_p[buf * WARPS_PER_BLOCK + w] = bp;
            }
            __syncthreads();
            bd = red_d[buf * WARPS_PER_BLOCK];
            bp = red_p[buf * WARPS_PER_BLOCK];
#pragma unroll
            for (int k = 1; k < WARPS_PER_BLOCK; ++k) {
                argminCombine(bd, bp, red_d[buf * WARPS_PER_BLOCK + k],
                              red_p[buf * WARPS_PER_BLOCK + k]);
            }
            buf ^= 1;
        } else {
            __syncwarp();
        }

        if (!(bd < INF)) break;  // no point keeps the diameter below threshold

        const int sel = bp;
        if (lane == 0) members_out[card] = nb[base + sel];
        ++card;

        const double cx = sx[sel];
        const double cy = sy[sel];
        bm = INF_BITS;
        bd = INF;
        bp = INT_MAX;
        for (int p = lane; p < deg; p += nthreads) {
            long long m = sd[p];
            if (m < INF_BITS) {
                if (p == sel) {
                    m = INF_BITS;
                } else {
                    const long long t =
                        __double_as_longlong(deviceDistance2(sx[p], sy[p], cx, cy));
                    m = max(m, t);
                    if (m >= bound_bits) m = INF_BITS;  // diameter only grows: drop for good
                }
                sd[p] = m;
                if (m < bm) {
                    const double r = __dsqrt_rn(__longlong_as_double(m));
                    bm = m;
                    if (r < bd) { bd = r; bp = p; }
                }
            }
        }
    }
    return card;
}

// Members of seed s are stored at memb[off[s] + s], room for deg(s) + 1 entries.
//
// Single launch per iteration: seeds[0, nl) have large neighborhoods and are
// processed one per block by the first large_blocks blocks (grid-stride; global
// scratch if they do not fit into shared memory); seeds[nl, nl + ns) have
// deg <= CAP_WARP and are processed one per warp by the remaining blocks.
static constexpr int SMEM_BYTES = 3 * 8 * CAP_BLOCK;
static_assert(3 * 8 * CAP_WARP * WARPS_PER_BLOCK <= SMEM_BYTES, "warp buffers exceed smem");

__global__ void __launch_bounds__(BLOCK_THREADS)
cardinalityKernel(const int* __restrict__ seeds, const int nl, const int ns,
                  const int large_blocks, int* __restrict__ card,
                  const double2* __restrict__ pts, const long long* __restrict__ off,
                  const int* __restrict__ nb, const unsigned char* __restrict__ clustered,
                  const double threshold2_bound, double* __restrict__ scratch,
                  const long long scratch_stride, int* __restrict__ memb) {
    __shared__ __align__(16) unsigned char smem[SMEM_BYTES];
    __shared__ double red_d[2 * WARPS_PER_BLOCK];
    __shared__ int red_p[2 * WARPS_PER_BLOCK];

    if (static_cast<int>(blockIdx.x) < large_blocks) {
        double* sx = reinterpret_cast<double*>(smem);
        double* sy = sx + CAP_BLOCK;
        long long* sd = reinterpret_cast<long long*>(sy + CAP_BLOCK);
        for (int i = blockIdx.x; i < nl; i += large_blocks) {
            __syncthreads();
            const int s = seeds[i];
            const int deg = static_cast<int>(off[s + 1] - off[s]);
            double *px = sx, *py = sy;
            long long* pd = sd;
            if (deg > CAP_BLOCK) {
                px = scratch + blockIdx.x * 3 * scratch_stride;
                py = px + scratch_stride;
                pd = reinterpret_cast<long long*>(py + scratch_stride);
            }
            const int c = growCluster<true>(s, pts, off, nb, clustered, threshold2_bound,
                                            px, py, pd, threadIdx.x, blockDim.x, red_d, red_p,
                                            memb + off[s] + s);
            if (threadIdx.x == 0) card[i] = c;
        }
    } else {
        const int w = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int g = (blockIdx.x - large_blocks) * WARPS_PER_BLOCK + w;
        if (g >= ns) return;
        double* sx = reinterpret_cast<double*>(smem) + w * CAP_WARP;
        double* sy = reinterpret_cast<double*>(smem) + (WARPS_PER_BLOCK + w) * CAP_WARP;
        long long* sd = reinterpret_cast<long long*>(smem) + (2 * WARPS_PER_BLOCK + w) * CAP_WARP;
        const int s = seeds[nl + g];
        const int c = growCluster<false>(s, pts, off, nb, clustered, threshold2_bound,
                                         sx, sy, sd, lane, 32, nullptr, nullptr,
                                         memb + off[s] + s);
        if (lane == 0) card[nl + g] = c;
    }
}

__global__ void markClusteredKernel(const int* __restrict__ members, const int k,
                                    unsigned char* __restrict__ clustered) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < k) clustered[members[i]] = 1;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Build the fixed-radius neighbor lists (dist < threshold, excluding self),
// sorted by point index, using a uniform grid and OpenMP.
static void buildNeighborLists(const std::vector<Point>& points, const double threshold,
                               std::vector<long long>& off, std::vector<int>& nb) {
    const int N = static_cast<int>(points.size());
    const double extent = std::max(MAX_WIDTH, MAX_HEIGHT);
    int ncell = static_cast<int>(std::floor(extent / threshold));
    ncell = std::max(1, std::min(ncell, 1024));
    const double cs = extent / ncell;  // cell size >= threshold

    auto cellOf = [&](double v) {
        int c = static_cast<int>(v / cs);
        return std::min(std::max(c, 0), ncell - 1);
    };

    const int ncells = ncell * ncell;
    std::vector<int> cellCount(ncells + 1, 0), cellStart(ncells + 1, 0), cellPts(N);
    std::vector<int> pcell(N);
    for (int i = 0; i < N; ++i) {
        pcell[i] = cellOf(points[i].y) * ncell + cellOf(points[i].x);
        cellCount[pcell[i]]++;
    }
    for (int c = 0; c < ncells; ++c) cellStart[c + 1] = cellStart[c] + cellCount[c];
    {
        std::vector<int> fill(cellStart.begin(), cellStart.end() - 1);
        for (int i = 0; i < N; ++i) cellPts[fill[pcell[i]]++] = i;
    }

    auto forEachNeighbor = [&](int i, auto&& fn) {
        const int cx = pcell[i] % ncell, cy = pcell[i] / ncell;
        for (int yy = std::max(cy - 1, 0); yy <= std::min(cy + 1, ncell - 1); ++yy) {
            for (int xx = std::max(cx - 1, 0); xx <= std::min(cx + 1, ncell - 1); ++xx) {
                const int c = yy * ncell + xx;
                for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                    const int j = cellPts[k];
                    if (j != i && distance(points[i], points[j]) < threshold) fn(j);
                }
            }
        }
    };

    // Avoid spinning up a large thread team for tiny inputs
    const int nthreads = std::max(1, std::min(omp_get_max_threads(), N / 512 + 1));

    off.assign(N + 1, 0);
    #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
    for (int i = 0; i < N; ++i) {
        long long cnt = 0;
        forEachNeighbor(i, [&](int) { ++cnt; });
        off[i + 1] = cnt;
    }
    for (int i = 0; i < N; ++i) off[i + 1] += off[i];

    nb.resize(std::max<long long>(off[N], 1));
    #pragma omp parallel for schedule(dynamic, 64) num_threads(nthreads)
    for (int i = 0; i < N; ++i) {
        long long pos = off[i];
        forEachNeighbor(i, [&](int j) { nb[pos++] = j; });
        std::sort(nb.begin() + off[i], nb.begin() + off[i + 1]);
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;

    // Neighbor lists (identical on every rank)
    std::vector<long long> off;
    std::vector<int> nb;
    buildNeighborLists(points, threshold, off, nb);
    int maxdeg = 0;
    for (int i = 0; i < N; ++i) maxdeg = std::max(maxdeg, static_cast<int>(off[i + 1] - off[i]));

    // Smallest double T with sqrt(T) >= threshold, so that for squared
    // distances x: sqrt(x) < threshold  <=>  x < T (sqrt is monotonic).
    double threshold2_bound = threshold * threshold;
    while (threshold2_bound > 0.0 && std::sqrt(threshold2_bound) >= threshold)
        threshold2_bound = std::nextafter(threshold2_bound, 0.0);
    while (std::sqrt(threshold2_bound) < threshold)
        threshold2_bound = std::nextafter(threshold2_bound, std::numeric_limits<double>::infinity());

    // Device data
    int dev = 0, numSMs = 1;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
    const int blockGridMax = numSMs * 8;

    double2* d_pts = nullptr;
    long long* d_off = nullptr;
    int* d_nb = nullptr;
    unsigned char* d_clustered = nullptr;
    int *d_members = nullptr, *d_memb = nullptr;
    double* d_scratch = nullptr;
    const long long scratch_stride = (maxdeg > CAP_BLOCK) ? maxdeg : 0;

    CUDA_CHECK(cudaMalloc(&d_pts, sizeof(double2) * std::max(N, 1)));
    CUDA_CHECK(cudaMalloc(&d_off, sizeof(long long) * (N + 1)));
    CUDA_CHECK(cudaMalloc(&d_nb, sizeof(int) * nb.size()));
    CUDA_CHECK(cudaMalloc(&d_clustered, std::max(N, 1)));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * std::max(N, 1)));
    CUDA_CHECK(cudaMalloc(&d_memb, sizeof(int) * (off[N] + N)));  // per-seed candidate members
    if (scratch_stride > 0) {
        CUDA_CHECK(cudaMalloc(&d_scratch, sizeof(double) * 3 * scratch_stride * blockGridMax));
    }

    {
        std::vector<double2> hp(N);
        for (int i = 0; i < N; ++i) hp[i] = make_double2(points[i].x, points[i].y);
        CUDA_CHECK(cudaMemcpy(d_pts, hp.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemcpy(d_off, off.data(), sizeof(long long) * (N + 1), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nb, nb.data(), sizeof(int) * nb.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, std::max(N, 1)));

    // Seed lists and cardinalities live in mapped pinned memory, so each
    // iteration costs a single kernel launch and synchronization.
    int *h_seeds = nullptr, *h_card = nullptr, *h_members = nullptr;
    int *d_seeds = nullptr, *d_card = nullptr;
    CUDA_CHECK(cudaHostAlloc(&h_seeds, sizeof(int) * std::max(N, 1), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_card, sizeof(int) * std::max(N, 1), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_seeds, h_seeds, 0));
    CUDA_CHECK(cudaHostGetDevicePointer(&d_card, h_card, 0));
    CUDA_CHECK(cudaMallocHost(&h_members, sizeof(int) * std::max(N, 1)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // Host state
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> cardCache(N, 0);
    std::vector<unsigned char> dirtyMark(N, 0);
    std::vector<int> ownedUnclustered;  // ascending point indices owned by this rank
    std::vector<int> dirty;
    for (int i = rank; i < N; i += nranks) {
        ownedUnclustered.push_back(i);
        dirty.push_back(i);
    }

    int remaining = N;
    while (remaining > 0) {
        // 1. Recompute cardinalities of seeds whose neighborhood changed
        if (!dirty.empty()) {
            int nl = 0;
            for (int s : dirty) {
                if (off[s + 1] - off[s] > CAP_WARP) h_seeds[nl++] = s;
            }
            int ns = 0;
            for (int s : dirty) {
                if (off[s + 1] - off[s] <= CAP_WARP) h_seeds[nl + ns++] = s;
            }
            const int large_blocks = std::min(nl, blockGridMax);
            const int grid = large_blocks + (ns + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            cardinalityKernel<<<grid, BLOCK_THREADS, 0, stream>>>(
                d_seeds, nl, ns, large_blocks, d_card, d_pts, d_off, d_nb, d_clustered,
                threshold2_bound, d_scratch, scratch_stride, d_memb);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaStreamSynchronize(stream));
            for (int i = 0; i < ns + nl; ++i) cardCache[h_seeds[i]] = h_card[i];
        }

        // 2. Local best seed: max cardinality, lowest index on ties
        const long long nown = static_cast<long long>(ownedUnclustered.size());
        long long bestKey = -1;
        #pragma omp parallel for reduction(max : bestKey) schedule(static) if (nown > 32768)
        for (long long i = 0; i < nown; ++i) {
            const int s = ownedUnclustered[i];
            const long long key = (static_cast<long long>(cardCache[s]) << 32) |
                                  static_cast<long long>(INT_MAX - s);
            bestKey = std::max(bestKey, key);
        }
        int local[2] = {-1, INT_MAX};
        if (bestKey >= 0) {
            local[0] = static_cast<int>(bestKey >> 32);
            local[1] = INT_MAX - static_cast<int>(bestKey & 0xffffffffLL);
        }

        // 3. Global best seed (MAXLOC resolves ties to the lowest index)
        int global[2];
        MPI_Allreduce(local, global, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_card = global[0];
        const int best_seed = global[1];
        if (best_card <= 0 || best_seed < 0 || best_seed >= N) break;

        // 4. The owner of the winning seed already holds its members on the GPU
        //    (computed together with its cached cardinality): broadcast them
        //    and mark them clustered everywhere.
        const int owner = best_seed % nranks;
        const int* d_win = d_members;
        if (rank == owner) {
            d_win = d_memb + off[best_seed] + best_seed;
            CUDA_CHECK(cudaMemcpyAsync(h_members, d_win, sizeof(int) * best_card,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        if (nranks > 1) {
            CUDA_CHECK(cudaStreamSynchronize(stream));  // h_members may still be in flight
            MPI_Bcast(h_members, best_card, MPI_INT, owner, MPI_COMM_WORLD);
        }
        if (rank != owner) {
            CUDA_CHECK(cudaMemcpyAsync(d_members, h_members, sizeof(int) * best_card,
                                       cudaMemcpyHostToDevice, stream));
        }
        markClusteredKernel<<<(best_card + 255) / 256, 256, 0, stream>>>(
            d_win, best_card, d_clustered);
        CUDA_CHECK(cudaGetLastError());

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(h_members, h_members + best_card);
        for (int m : cluster.members) clustered[m] = 1;
        remaining -= best_card;

        // 5. Seeds owned by this rank whose neighborhood lost points become dirty
        dirty.clear();
        for (int m : cluster.members) {
            for (long long k = off[m]; k < off[m + 1]; ++k) {
                const int c = nb[k];
                if (!clustered[c] && (c % nranks) == rank && !dirtyMark[c]) {
                    dirtyMark[c] = 1;
                    dirty.push_back(c);
                }
            }
        }
        for (int c : dirty) dirtyMark[c] = 0;

        ownedUnclustered.erase(
            std::remove_if(ownedUnclustered.begin(), ownedUnclustered.end(),
                           [&clustered](int idx) { return clustered[idx]; }),
            ownedUnclustered.end());

        clusters.push_back(std::move(cluster));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFreeHost(h_seeds));
    CUDA_CHECK(cudaFreeHost(h_card));
    CUDA_CHECK(cudaFreeHost(h_members));
    CUDA_CHECK(cudaFree(d_pts));
    CUDA_CHECK(cudaFree(d_off));
    CUDA_CHECK(cudaFree(d_nb));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_memb));
    if (d_scratch) CUDA_CHECK(cudaFree(d_scratch));

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Compute cluster diameters (max distance between any two points) in parallel
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
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

static int runBenchmark(int argc, char** argv) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
               num_points, threshold);
        return 1;
    }
    
    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    long cluster_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    MPI_Allreduce(MPI_IN_PLACE, &cluster_time_ms, 1, MPI_LONG, MPI_MAX, MPI_COMM_WORLD);
    auto cluster_time = std::chrono::milliseconds(cluster_time_ms);
    
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
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Bind each rank to a GPU of its node (round-robin over node-local ranks)
    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    int num_devices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    if (num_devices <= 0) {
        fprintf(stderr, "Error: no CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(local_rank % num_devices));
    CUDA_CHECK(cudaFree(nullptr));  // create the CUDA context up front

    // Only rank 0 reports
    if (rank != 0) {
        if (!freopen("/dev/null", "w", stdout)) fclose(stdout);
    }

    int rc = runBenchmark(argc, argv);
    // All ranks hold identical clusters, but use rank 0's verdict
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    fflush(stdout);
    MPI_Finalize();
    return rc;
}
