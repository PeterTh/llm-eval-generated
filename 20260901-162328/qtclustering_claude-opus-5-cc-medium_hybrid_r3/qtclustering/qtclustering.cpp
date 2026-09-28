// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy
// ------------------------
// The dominant cost is the "candidate cluster" search: in every round of the
// outer loop, every still-unclustered point is tried as a seed and grown
// greedily.  Those seed evaluations are completely independent of each other,
// which gives three nested levels of parallelism that are exploited here:
//
//   * MPI    - the seed set of a round is distributed over the ranks of the
//              cluster; the winning (seed, cardinality) pair is combined with
//              a single MAXLOC all-reduce per round.
//   * OpenMP - each rank drives all of the GPUs assigned to it with one host
//              thread per device, and performs the (much cheaper) host-side
//              phases (winner re-grow, membership bookkeeping, validation)
//              multi-threaded.
//   * CUDA   - one thread block grows one candidate cluster; the block-wide
//              "closest point" search of every growth step is a fused
//              update + argmin reduction over the seed's candidate list.
//
// Algorithmic (semantics preserving) improvements that make the above pay off:
//
//   * The maximum distance of a candidate to the current cluster is maintained
//     incrementally (max is associative), turning the O(N*k) inner scan of the
//     original into O(N).
//   * Because that running maximum only grows, a candidate that once exceeded
//     the threshold can never come back; each seed therefore only ever needs to
//     look at the points that are within the threshold of the seed itself.
//     That candidate list is built once per seed.
//   * A first wave of the most promising seeds (largest candidate lists)
//     establishes a best cardinality that provably prunes every remaining seed
//     whose candidate list is too small to beat (or tie-and-win against) it.
//     Those candidate list sizes are computed once and then maintained
//     incrementally as points get clustered.
//   * The pairwise distances are precomputed into a device side matrix while it
//     fits into GPU memory, which reduces the growth steps to a coalesced
//     streaming loop.  For larger inputs the kernel falls back to computing
//     distances from the (gathered, hence still coalesced) candidate
//     coordinates, skipping the square root whenever the squared distance
//     cannot raise the running maximum.
//
// All comparisons keep the tie-breaking of the original code (lowest index
// wins), and all distances are computed with the exact same non-contracted
// IEEE double operations on host and device, so the produced clustering is
// bit-for-bit identical to the sequential reference.

#include <algorithm>
#include <chrono>
#include <cmath>
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

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t _err = (call);                                        \
        if (_err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(_err), __FILE__, __LINE__);              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                       \
        }                                                                       \
    } while (0)

static int g_rank = 0;
static int g_size = 1;
static std::vector<int> g_myDevices;   // GPUs this rank drives (one thread each)

// The host-side loops of the clustering run are entered very often but are
// individually small, so the team size has to be matched to the work item
// count - spawning hundreds of threads for a few thousand iterations costs far
// more than it saves.
static inline int ompThreadsFor(long long work, long long grain) {
    long long t = work / grain;
    if (t < 1) return 1;
    const long long maxT = omp_get_max_threads();
    return static_cast<int>(t < maxT ? t : maxT);
}

// Work out which GPUs belong to this rank and create their contexts.  Doing
// this up front keeps the one-off CUDA driver initialisation out of the
// measured clustering phase.
static void initDevices() {
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount <= 0) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_size(shmComm, &localSize);
    MPI_Comm_free(&shmComm);

    // Ranks share the node's GPUs; if there are at least as many GPUs as ranks
    // on the node, a rank drives several of them with one OpenMP thread each.
    g_myDevices.clear();
    if (localSize <= deviceCount) {
        const int first = localRank * deviceCount / localSize;
        const int last = (localRank + 1) * deviceCount / localSize;
        for (int d = first; d < last; ++d) g_myDevices.push_back(d);
    } else {
        g_myDevices.push_back(localRank % deviceCount);
    }

    const int nGpu = static_cast<int>(g_myDevices.size());
    #pragma omp parallel for num_threads(nGpu) schedule(static, 1)
    for (int g = 0; g < nGpu; ++g) {
        cudaSetDevice(g_myDevices[g]);
        cudaFree(nullptr);   // force context creation
    }
}

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
// The host build uses -ffp-contract=off and the device build -fmad=false so
// that both evaluate exactly the same sequence of IEEE double operations.
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Device side
// ---------------------------------------------------------------------------

__device__ __forceinline__ double devDistance(double ax, double ay, double bx, double by) {
    const double dx = __dsub_rn(ax, bx);
    const double dy = __dsub_rn(ay, by);
    return __dsqrt_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
}

// Fetch d(a, b): either from the precomputed distance matrix or on the fly.
__device__ __forceinline__ double devDist(const double* __restrict__ mat,
                                          const double2* __restrict__ pts,
                                          int N, int a, int b) {
    if (mat != nullptr) return mat[static_cast<size_t>(a) * N + b];
    const double2 pa = pts[a];
    const double2 pb = pts[b];
    return devDistance(pa.x, pa.y, pb.x, pb.y);
}

// Fill the (symmetric) distance matrix.
__global__ void buildDistanceMatrix(double* __restrict__ mat,
                                    const double2* __restrict__ pts, int N) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y;
    if (j >= N) return;
    const double2 pa = pts[i];
    const double2 pb = pts[j];
    mat[static_cast<size_t>(i) * N + j] = devDistance(pa.x, pa.y, pb.x, pb.y);
}

__device__ __forceinline__ int blockSum(int* scratch, int v) {
    const int tid = threadIdx.x;
    scratch[tid] = v;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) scratch[tid] += scratch[tid + off];
        __syncthreads();
    }
    const int r = scratch[0];
    __syncthreads();
    return r;
}

// ub[p] = 1 + |{c != p : d(p,c) < threshold}| for the initial (fully
// unclustered) state; see the discussion at the call site.
__global__ void initBounds(const double* __restrict__ mat,
                           const double2* __restrict__ pts, int N,
                           double threshold, int* __restrict__ ub) {
    extern __shared__ int scratch[];
    const int tid = threadIdx.x;
    for (int s = blockIdx.x; s < N; s += gridDim.x) {
        int cnt = 0;
        for (int c = tid; c < N; c += blockDim.x) {
            if (c != s && devDist(mat, pts, N, s, c) < threshold) cnt++;
        }
        const int total = blockSum(scratch, cnt);
        if (tid == 0) ub[s] = total + 1;
    }
}

// Subtract the points that just became clustered from the bounds of the seeds
// that are still available.
__global__ void updateBounds(const double* __restrict__ mat,
                             const double2* __restrict__ pts, int N,
                             double threshold,
                             const int* __restrict__ seeds, int nseeds,
                             const int* __restrict__ members, int nmembers,
                             int* __restrict__ ub) {
    extern __shared__ int scratch[];
    const int tid = threadIdx.x;
    for (int i = blockIdx.x; i < nseeds; i += gridDim.x) {
        const int s = seeds[i];
        int cnt = 0;
        for (int j = tid; j < nmembers; j += blockDim.x) {
            if (devDist(mat, pts, N, s, members[j]) < threshold) cnt++;
        }
        const int total = blockSum(scratch, cnt);
        if (tid == 0) ub[s] -= total;
    }
}

// Block-wide argmin over (value, index) with lowest-index tie breaking.
// Returns the winning slot position through *posOut.
__device__ __forceinline__ void blockArgMin(double* sval, int* sidx, int* spos,
                                            double& bval, int& bidx, int& bpos) {
    const int tid = threadIdx.x;
    sval[tid] = bval;
    sidx[tid] = bidx;
    spos[tid] = bpos;
    __syncthreads();
    for (int off = blockDim.x >> 1; off > 0; off >>= 1) {
        if (tid < off) {
            const int o = tid + off;
            if (sidx[o] >= 0 && (sidx[tid] < 0 || sval[o] < sval[tid] ||
                                 (sval[o] == sval[tid] && sidx[o] < sidx[tid]))) {
                sval[tid] = sval[o];
                sidx[tid] = sidx[o];
                spos[tid] = spos[o];
            }
        }
        __syncthreads();
    }
    bval = sval[0];
    bidx = sidx[0];
    bpos = spos[0];
    __syncthreads();
}

// Grow one candidate cluster per thread block and report its cardinality.
__global__ void growClusters(const double* __restrict__ mat,
                             const double2* __restrict__ pts, int N,
                             double threshold,
                             const unsigned char* __restrict__ clustered,
                             const int* __restrict__ seeds, int nseeds,
                             int* __restrict__ listIdx, double* __restrict__ listVal,
                             double* __restrict__ listSq, double2* __restrict__ listPos,
                             int* __restrict__ cardOut) {
    extern __shared__ char smem[];
    double* sval = reinterpret_cast<double*>(smem);
    int* sidx = reinterpret_cast<int*>(sval + blockDim.x);
    int* spos = sidx + blockDim.x;
    __shared__ int sCount;
    __shared__ double2 sChosenPos;

    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.x) * N;
    int* myIdx = listIdx + base;
    double* myVal = listVal + base;
    // Only used when there is no distance matrix: the candidate coordinates
    // are gathered once so that the growth steps read them coalesced, and the
    // squared running maximum lets us skip the (expensive) square root
    // whenever the new distance cannot raise the maximum - which is the common
    // case.  Comparing squares is exact here because sqrt is monotone.
    const bool onTheFly = (mat == nullptr);
    double* mySq = onTheFly ? listSq + base : nullptr;
    double2* myPos = onTheFly ? listPos + base : nullptr;

    for (int s = blockIdx.x; s < nseeds; s += gridDim.x) {
        const int seed = seeds[s];

        // Phase A: build the candidate list = unclustered points within the
        // threshold of the seed, together with their distance to the seed
        // (which is their running max-distance to the one-element cluster).
        const double2 seedPos = pts[seed];
        if (tid == 0) {
            sCount = 0;
            sChosenPos = seedPos;
        }
        __syncthreads();
        for (int c = tid; c < N; c += blockDim.x) {
            if (clustered[c] || c == seed) continue;
            const double2 pc = pts[c];
            double d, d2 = 0.0;
            if (onTheFly) {
                const double dx = __dsub_rn(seedPos.x, pc.x);
                const double dy = __dsub_rn(seedPos.y, pc.y);
                d2 = __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
                d = __dsqrt_rn(d2);
            } else {
                d = mat[static_cast<size_t>(seed) * N + c];
            }
            if (d < threshold) {
                const int p = atomicAdd(&sCount, 1);
                myIdx[p] = c;
                myVal[p] = d;
                if (onTheFly) {
                    mySq[p] = d2;
                    myPos[p] = pc;
                }
            }
        }
        __syncthreads();
        const int m = sCount;

        // Phase B: greedily add the candidate with the smallest running max
        // distance until nothing fits below the threshold any more.
        int card = 1;
        int chosen = -1;   // last added member; -1 => list values are current
        while (true) {
            double bval = CUDART_INF;
            int bidx = -1, bpos = -1;
            const double2 cp = sChosenPos;
            for (int j = tid; j < m; j += blockDim.x) {
                double v = myVal[j];
                if (v == CUDART_INF) continue;
                const int c = myIdx[j];
                if (chosen >= 0) {
                    if (onTheFly) {
                        const double2 pc = myPos[j];
                        const double dx = __dsub_rn(cp.x, pc.x);
                        const double dy = __dsub_rn(cp.y, pc.y);
                        const double d2 = __dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy));
                        if (d2 > mySq[j]) {
                            mySq[j] = d2;
                            v = __dsqrt_rn(d2);
                            if (!(v < threshold)) v = CUDART_INF;
                            myVal[j] = v;
                            if (v == CUDART_INF) continue;
                        }
                    } else {
                        const double d = mat[static_cast<size_t>(chosen) * N + c];
                        if (d > v) {
                            v = d;
                            if (!(v < threshold)) v = CUDART_INF;
                            myVal[j] = v;
                            if (v == CUDART_INF) continue;
                        }
                    }
                }
                if (bidx < 0 || v < bval || (v == bval && c < bidx)) {
                    bval = v;
                    bidx = c;
                    bpos = j;
                }
            }
            __syncthreads();
            blockArgMin(sval, sidx, spos, bval, bidx, bpos);
            if (bidx < 0) break;
            if (tid == 0) {
                myVal[bpos] = CUDART_INF;
                sChosenPos = pts[bidx];
            }
            chosen = bidx;
            card++;
            __syncthreads();
        }

        if (tid == 0) cardOut[s] = card;
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// Host side: per-GPU context
// ---------------------------------------------------------------------------

struct GpuContext {
    int device = 0;
    int blocks = 0;
    int threads = 0;
    double* d_mat = nullptr;      // may be null -> distances computed on the fly
    double2* d_pts = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_ub = nullptr;
    int* d_members = nullptr;
    int* d_listIdx = nullptr;
    double* d_listVal = nullptr;
    double* d_listSq = nullptr;    // only allocated for the matrix-free path
    double2* d_listPos = nullptr;  // only allocated for the matrix-free path
    cudaStream_t stream = nullptr;
    std::vector<int> h_seeds;
    std::vector<int> h_out;
};

static int roundPow2AtMost(int v, int lo, int hi) {
    int p = lo;
    while (p * 2 <= v && p * 2 <= hi) p *= 2;
    return p;
}

// ---------------------------------------------------------------------------
// Host re-grow of the winning cluster (OpenMP parallel).
// Mirrors the device kernel exactly, so it yields identical members.
// ---------------------------------------------------------------------------
static void growClusterHost(int seed, const std::vector<unsigned char>& clustered,
                            const std::vector<Point>& points, double threshold,
                            int N, std::vector<int>& members,
                            std::vector<int>& idxBuf, std::vector<double>& valBuf) {
    members.clear();
    members.push_back(seed);

    idxBuf.clear();
    valBuf.clear();
    const Point ps = points[seed];
    for (int c = 0; c < N; ++c) {
        if (clustered[c] || c == seed) continue;
        const double dx = ps.x - points[c].x;
        const double dy = ps.y - points[c].y;
        const double d2 = dx * dx + dy * dy;
        const double d = std::sqrt(d2);
        if (d < threshold) {
            idxBuf.push_back(c);
            valBuf.push_back(d);
            valBuf.push_back(d2);   // interleaved: [distance, squared distance]
        }
    }

    const int m = static_cast<int>(idxBuf.size());
    const int* idx = idxBuf.data();
    double* val = valBuf.data();
    const double INF = std::numeric_limits<double>::infinity();
    int chosen = -1;

    while (true) {
        double bestVal = INF;
        int bestIdx = -1, bestPos = -1;

        const Point pchosen = chosen >= 0 ? points[chosen] : ps;
        const int nt = ompThreadsFor(m, 8192);   // tiny lists stay serial
        (void)nt;                                // (used by the OpenMP clause below)
        #pragma omp parallel num_threads(nt) if(nt > 1)
        {
            double lval = INF;
            int lidx = -1, lpos = -1;
            #pragma omp for schedule(static) nowait
            for (int j = 0; j < m; ++j) {
                double v = val[2 * j];
                if (v == INF) continue;
                const int c = idx[j];
                if (chosen >= 0) {
                    // Squares only decide whether the running maximum grows;
                    // sqrt is monotone, so this is exact - and the square root
                    // is skipped in the common case.
                    const double dx = pchosen.x - points[c].x;
                    const double dy = pchosen.y - points[c].y;
                    const double d2 = dx * dx + dy * dy;
                    if (d2 > val[2 * j + 1]) {
                        val[2 * j + 1] = d2;
                        v = std::sqrt(d2);
                        if (!(v < threshold)) v = INF;
                        val[2 * j] = v;
                        if (v == INF) continue;
                    }
                }
                if (lidx < 0 || v < lval || (v == lval && c < lidx)) {
                    lval = v; lidx = c; lpos = j;
                }
            }
            #pragma omp critical
            {
                if (lidx >= 0 && (bestIdx < 0 || lval < bestVal ||
                                  (lval == bestVal && lidx < bestIdx))) {
                    bestVal = lval; bestIdx = lidx; bestPos = lpos;
                }
            }
        }

        if (bestIdx < 0) break;
        val[2 * bestPos] = INF;
        chosen = bestIdx;
        members.push_back(bestIdx);
    }
}

// ---------------------------------------------------------------------------
// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
// ---------------------------------------------------------------------------
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N <= 0) return clusters;

    const std::vector<int>& myDevices = g_myDevices;
    const int nGpu = static_cast<int>(myDevices.size());

    // --- per-GPU setup -----------------------------------------------------
    std::vector<GpuContext> ctx(nGpu);
    std::vector<double2> hpts(N);
    for (int i = 0; i < N; ++i) hpts[i] = make_double2(points[i].x, points[i].y);

    const int threadsPerBlock = roundPow2AtMost(N / 4, 32, 256);

    #pragma omp parallel for num_threads(nGpu) schedule(static, 1)
    for (int g = 0; g < nGpu; ++g) {
        GpuContext& c = ctx[g];
        c.device = myDevices[g];
        CUDA_CHECK(cudaSetDevice(c.device));
        CUDA_CHECK(cudaStreamCreate(&c.stream));
        c.threads = threadsPerBlock;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, c.device));

        size_t freeMem = 0, totalMem = 0;
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));

        CUDA_CHECK(cudaMalloc(&c.d_pts, sizeof(double2) * N));
        CUDA_CHECK(cudaMemcpy(c.d_pts, hpts.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&c.d_clustered, N));
        CUDA_CHECK(cudaMalloc(&c.d_seeds, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&c.d_card, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&c.d_ub, sizeof(int) * N));
        CUDA_CHECK(cudaMalloc(&c.d_members, sizeof(int) * N));

        // Precompute the full distance matrix when it comfortably fits; it
        // turns the growth kernel into a purely memory bound loop.
        const size_t matBytes = sizeof(double) * static_cast<size_t>(N) * N;
        const size_t budget = static_cast<size_t>(0.75 * static_cast<double>(freeMem));
        if (matBytes <= budget && cudaMalloc(&c.d_mat, matBytes) == cudaSuccess) {
            dim3 blk(256);
            dim3 grd((N + 255) / 256, N);
            buildDistanceMatrix<<<grd, blk>>>(c.d_mat, c.d_pts, N);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            c.d_mat = nullptr;
        }

        // As many resident blocks as the device can hold, bounded by the
        // scratch memory each block needs (one candidate list of size N).
        CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
        const bool onTheFly = (c.d_mat == nullptr);
        size_t perBlock = static_cast<size_t>(N) * (sizeof(int) + sizeof(double));
        if (onTheFly) perBlock += static_cast<size_t>(N) * (sizeof(double) + sizeof(double2));
        size_t maxBlocks = static_cast<size_t>(0.6 * static_cast<double>(freeMem)) / perBlock;
        int blocks = prop.multiProcessorCount * 8;
        if (blocks > N) blocks = N;
        if (static_cast<size_t>(blocks) > maxBlocks) blocks = static_cast<int>(maxBlocks);
        if (blocks < 1) blocks = 1;
        c.blocks = blocks;

        const size_t slots = static_cast<size_t>(blocks) * N;
        CUDA_CHECK(cudaMalloc(&c.d_listIdx, sizeof(int) * slots));
        CUDA_CHECK(cudaMalloc(&c.d_listVal, sizeof(double) * slots));
        if (onTheFly) {
            CUDA_CHECK(cudaMalloc(&c.d_listSq, sizeof(double) * slots));
            CUDA_CHECK(cudaMalloc(&c.d_listPos, sizeof(double2) * slots));
        }
        c.h_seeds.reserve(N);
        c.h_out.resize(N);
    }

    // Total number of candidate clusters that can be grown concurrently by the
    // whole job - used as the size of the pruning wave.
    int localWidth = 0;
    for (int g = 0; g < nGpu; ++g) localWidth += ctx[g].blocks;
    int globalWidth = 0;
    MPI_Allreduce(&localWidth, &globalWidth, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    if (globalWidth < 1) globalWidth = 1;

    // --- host state (replicated on every rank) -----------------------------
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered;
    unclustered.reserve(N);
    for (int i = 0; i < N; ++i) unclustered.push_back(i);

    std::vector<int> allSeeds, waveSeeds, mySeeds;
    std::vector<int> memberBuf, idxBuf;
    std::vector<double> valBuf;
    idxBuf.reserve(N);
    valBuf.reserve(2 * static_cast<size_t>(N));

    // ub[p] = 1 + number of still unclustered points (other than p) within the
    // threshold of p.  Every member of a cluster has to be within the threshold
    // of every other member, so this is an upper bound on the cardinality that
    // seed p can reach.  It is computed once and then maintained incrementally
    // (identically and redundantly on every rank, so it needs no communication).
    std::vector<int> ub(N, 1);
    {
        GpuContext& c = ctx[0];
        CUDA_CHECK(cudaSetDevice(c.device));
        initBounds<<<std::min(N, c.blocks), c.threads, c.threads * sizeof(int)>>>(
            c.d_mat, c.d_pts, N, threshold, c.d_ub);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(ub.data(), c.d_ub, sizeof(int) * N, cudaMemcpyDeviceToHost));
    }

    // Runs one wave of seeds through the local GPUs and returns the best
    // (cardinality, seed) found locally.
    auto runWave = [&](const std::vector<int>& seeds, int& bestCard, int& bestSeed) {
        bestCard = -1;
        bestSeed = -1;
        const int n = static_cast<int>(seeds.size());
        if (n == 0) return;
        // Round-robin over the GPUs; the wave is ordered by decreasing work so
        // this balances the load well.
        for (int g = 0; g < nGpu; ++g) ctx[g].h_seeds.clear();
        for (int i = 0; i < n; ++i) ctx[i % nGpu].h_seeds.push_back(seeds[i]);

        std::vector<int> gCard(nGpu, -1), gSeed(nGpu, -1);
        #pragma omp parallel for num_threads(nGpu) schedule(static, 1)
        for (int g = 0; g < nGpu; ++g) {
            GpuContext& c = ctx[g];
            const int ns = static_cast<int>(c.h_seeds.size());
            if (ns == 0) continue;
            CUDA_CHECK(cudaSetDevice(c.device));
            CUDA_CHECK(cudaMemcpyAsync(c.d_seeds, c.h_seeds.data(), sizeof(int) * ns,
                                       cudaMemcpyHostToDevice, c.stream));
            const int grid = std::min(ns, c.blocks);
            const size_t shmem = c.threads * (sizeof(double) + 2 * sizeof(int));
            growClusters<<<grid, c.threads, shmem, c.stream>>>(
                c.d_mat, c.d_pts, N, threshold, c.d_clustered, c.d_seeds, ns,
                c.d_listIdx, c.d_listVal, c.d_listSq, c.d_listPos, c.d_card);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(c.h_out.data(), c.d_card, sizeof(int) * ns,
                                       cudaMemcpyDeviceToHost, c.stream));
            CUDA_CHECK(cudaStreamSynchronize(c.stream));
            int bc = -1, bs = -1;
            for (int i = 0; i < ns; ++i) {
                const int card = c.h_out[i];
                const int sd = c.h_seeds[i];
                if (card > bc || (card == bc && sd < bs)) { bc = card; bs = sd; }
            }
            gCard[g] = bc;
            gSeed[g] = bs;
        }
        for (int g = 0; g < nGpu; ++g) {
            if (gSeed[g] < 0) continue;
            if (gCard[g] > bestCard || (gCard[g] == bestCard && gSeed[g] < bestSeed)) {
                bestCard = gCard[g];
                bestSeed = gSeed[g];
            }
        }
    };

    // --- main clustering loop ---------------------------------------------
    while (!unclustered.empty()) {
        const int S = static_cast<int>(unclustered.size());

        // Upload the current clustered mask.
        #pragma omp parallel for num_threads(nGpu) schedule(static, 1)
        for (int g = 0; g < nGpu; ++g) {
            CUDA_CHECK(cudaSetDevice(ctx[g].device));
            CUDA_CHECK(cudaMemcpy(ctx[g].d_clustered, clustered.data(), N,
                                  cudaMemcpyHostToDevice));
        }

        // Wave 1 consists of the seeds with the largest upper bounds - enough
        // of them to fill the whole machine.  Only that split has to be exact
        // (it is a strict total order, hence deterministic on every rank); the
        // ordering inside the two parts is irrelevant for the result.
        const int wave1 = std::min(S, globalWidth);
        allSeeds.assign(unclustered.begin(), unclustered.end());
        const auto byBound = [&ub](int a, int b) {
            if (ub[a] != ub[b]) return ub[a] > ub[b];
            return a < b;
        };
        if (wave1 < S) {
            std::nth_element(allSeeds.begin(), allSeeds.begin() + wave1, allSeeds.end(), byBound);
        }
        // Descending order within the wave: combined with the round-robin
        // distribution below this spreads the expensive seeds evenly.
        std::sort(allSeeds.begin(), allSeeds.begin() + wave1, byBound);

        int globalCard = -1, globalSeed = -1;
        for (int pass = 0; pass < 2; ++pass) {
            waveSeeds.clear();
            if (pass == 0) {
                for (int i = 0; i < wave1; ++i) waveSeeds.push_back(allSeeds[i]);
            } else {
                // Wave 2: only seeds that can still beat - or tie and win
                // against - the current best are worth evaluating.
                for (int i = wave1; i < S; ++i) {
                    const int sd = allSeeds[i];
                    const int bound = ub[sd];
                    if (bound > globalCard || (bound == globalCard && sd < globalSeed)) {
                        waveSeeds.push_back(sd);
                    }
                }
            }
            const int nw = static_cast<int>(waveSeeds.size());
            if (nw == 0) continue;

            // Strided split over the ranks keeps the (sorted) work balanced.
            mySeeds.clear();
            for (int i = g_rank; i < nw; i += g_size) mySeeds.push_back(waveSeeds[i]);

            int bc = -1, bs = -1;
            runWave(mySeeds, bc, bs);

            int in[2] = {bc, bs < 0 ? std::numeric_limits<int>::max() : bs};
            int out[2];
            MPI_Allreduce(in, out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
            // MAXLOC keeps the smallest "location" among equal values, which is
            // exactly the lowest-seed-index tie break of the original code.
            if (out[0] > globalCard || (out[0] == globalCard && out[1] < globalSeed)) {
                globalCard = out[0];
                globalSeed = out[1];
            }
        }

        if (globalSeed < 0 || globalCard <= 0) break;

        // Re-grow the winner on the host (cheap, and identical bit for bit).
        growClusterHost(globalSeed, clustered, points, threshold, N, memberBuf,
                        idxBuf, valBuf);

        Cluster cluster;
        cluster.seed_point = globalSeed;
        cluster.members = memberBuf;
        clusters.push_back(std::move(cluster));

        for (int mIdx : memberBuf) clustered[mIdx] = 1;
        unclustered.erase(std::remove_if(unclustered.begin(), unclustered.end(),
                                         [&clustered](int idx) { return clustered[idx] != 0; }),
                          unclustered.end());

        // Maintain the upper bounds: the points that just became clustered are
        // no longer available to any remaining seed.  Summed over all rounds
        // this costs O(N^2) in total instead of O(N^2) per round.
        {
            const int nu = static_cast<int>(unclustered.size());
            const int nm = static_cast<int>(memberBuf.size());
            if (nu > 0) {
                GpuContext& c = ctx[0];
                CUDA_CHECK(cudaSetDevice(c.device));
                CUDA_CHECK(cudaMemcpy(c.d_seeds, unclustered.data(), sizeof(int) * nu,
                                      cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(c.d_members, memberBuf.data(), sizeof(int) * nm,
                                      cudaMemcpyHostToDevice));
                updateBounds<<<std::min(nu, c.blocks), c.threads, c.threads * sizeof(int)>>>(
                    c.d_mat, c.d_pts, N, threshold, c.d_seeds, nu, c.d_members, nm, c.d_ub);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(ub.data(), c.d_ub, sizeof(int) * N,
                                      cudaMemcpyDeviceToHost));
            }
        }
    }

    // --- cleanup -----------------------------------------------------------
    for (int g = 0; g < nGpu; ++g) {
        GpuContext& c = ctx[g];
        cudaSetDevice(c.device);
        cudaFree(c.d_mat);
        cudaFree(c.d_pts);
        cudaFree(c.d_clustered);
        cudaFree(c.d_seeds);
        cudaFree(c.d_card);
        cudaFree(c.d_ub);
        cudaFree(c.d_members);
        cudaFree(c.d_listIdx);
        cudaFree(c.d_listVal);
        cudaFree(c.d_listSq);
        cudaFree(c.d_listPos);
        cudaStreamDestroy(c.stream);
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;

    printf("Validating clusters:\n");

    const int nc = static_cast<int>(clusters.size());
    std::vector<double> diameters(nc, 0.0);

    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < nc; ++c) {
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

    for (int c = 0; c < nc; ++c) {
        const double max_diameter = diameters[c];
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %d: size=%zu, seed=%d, diameter=%.4f\n",
                   c, clusters[c].members.size(), clusters[c].seed_point, max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %d has diameter %.4f > threshold %.4f\n",
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
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

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

    if (num_points <= 0 || threshold <= 0.0) {
        if (g_rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }

    if (g_rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bring up the CUDA contexts before the measured region.
    initDevices();

    // Generate synthetic data (replicated - identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &cluster_time, 1, MPI_LONG, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (g_rank != 0) {
        MPI_Finalize();
        return 0;
    }

    printf("Clustering time: %ld ms\n", cluster_time);
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
    const double time_sec = cluster_time / 1000.0;
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
    int rc = 0;
    if (validate) {
        const bool valid = validateClusters(clusters, points, threshold);

        if (valid) {
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            rc = 1;
        }
    }

    MPI_Finalize();
    return rc;
}
