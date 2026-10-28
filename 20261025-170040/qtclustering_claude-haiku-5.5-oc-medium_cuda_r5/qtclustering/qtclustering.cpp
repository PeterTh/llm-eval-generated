// QT Clustering Benchmark - CUDA GPU Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Each outer iteration grows a greedy candidate cluster from every unclustered
// point (one CUDA block processes a seed at a time) and keeps the seed that
// produces the largest cluster (ties go to the lowest index). The winning
// cluster is then re-grown once to recover its members. A seed's result only
// depends on its neighborhood, so after each cluster is formed only the seeds
// neighboring its members are recomputed; the others keep their cached size.
//
// Within a block, the greedy growth is data-parallel: each step updates the
// max-distance of the live candidates using only the newest member, performs a
// block-wide argmin, and compacts the candidate list. A candidate whose
// max-distance reaches the threshold can never be selected again (max-distances
// only grow), so it is dropped. The initial candidates of a seed are its
// neighbors within the threshold, precomputed once on the GPU. Results are
// identical to the sequential reference algorithm.

#include <cuda_runtime.h>

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

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err_), #call);                 \
            exit(EXIT_FAILURE);                                                 \
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
// (Sequential by nature: the RNG stream and rejection sampling define the data.)
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

// Calculate Euclidean distance between two points (host; used for validation)
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Device code
// ---------------------------------------------------------------------------

constexpr int kBlockThreads = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxWarps = kBlockThreads / kWarpSize;
constexpr int kNoIndex = INT_MAX;

// Squared distance, evaluated as fma(dx, dx, dy*dy). This is the exact
// operation order the reference build uses, so the squared distances (and
// therefore every threshold and tie decision) match bit for bit.
__device__ __forceinline__ double sqDistance(const Point& a, const Point& b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return __fma_rn(dx, dx, __dmul_rn(dy, dy));
}

// Squared distances are stored as their IEEE-754 bit patterns. For non-negative
// doubles the bit pattern orders exactly like the value, so max and comparisons
// run on 64-bit integers.
typedef unsigned long long Bits;
constexpr Bits kNoKey = ~0ull;

__device__ __forceinline__ Bits toBits(double v) { return __double_as_longlong(v); }
__device__ __forceinline__ double fromBits(Bits b) { return __longlong_as_double(b); }

// Strict ordering of candidates by (sqrt(sq), index), i.e. by the distance the
// reference algorithm compares, with the lowest index winning ties.
// Exact: if two squared values differ by more than 2^12 ulps (relative gap
// >= 2^-42), their correctly rounded square roots keep the same strict order,
// so square roots are computed only for (near-)ties.
__device__ __forceinline__ bool keyLess(Bits a, int ia, Bits b, int ib) {
    if (ia == kNoIndex) return false;
    if (ib == kNoIndex) return true;
    const Bits diff = (a > b) ? (a - b) : (b - a);
    if (diff > 4096ull) return a < b;
    const double ra = sqrt(fromBits(a));
    const double rb = sqrt(fromBits(b));
    if (ra != rb) return ra < rb;
    return ia < ib;
}

// Warp-wide argmin; lane 0 ends with the minimum.
__device__ __forceinline__ void warpArgMin(Bits& key, int& idx) {
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        const Bits ok = __shfl_down_sync(0xffffffffu, key, offset);
        const int oi = __shfl_down_sync(0xffffffffu, idx, offset);
        if (keyLess(ok, oi, key, idx)) {
            key = ok;
            idx = oi;
        }
    }
}

// Read-only inputs and per-launch outputs shared by all blocks.
struct SweepParams {
    const Point* pts;
    int N;                          // number of points (= list capacity)
    const unsigned char* clustered; // 1 if the point is already clustered
    const int* cand;                // unclustered point indices
    int numCand;
    const int* nbrOff;              // CSR neighbor offsets (N + 1), or nullptr
    const int* nbrIdx;              // CSR neighbor indices (within thr2 of point)
    const Bits* nbrSq;              // CSR squared distances (bit patterns)
    double threshold;
    double thr2;                    // conservative squared threshold for pruning
    Bits thr2Bits;                  // bit pattern of thr2 (thr2 >= 0)
    int* cardinality;               // output: cluster cardinality per launched seed
    int* listIdx;                   // scratch: 2 * N per block
    Bits* listSq;                   // scratch: 2 * N per block (bit patterns)
    int* members;                   // scratch: N per block
};

struct BlockShared {
    Bits warpKey[kMaxWarps];
    int warpIdx[kMaxWarps];
    Bits bestKey;
    int bestIdx;
    int last;       // most recently added member
    int newCount;   // survivors written to the destination list this step
};

// Grow the greedy cluster from `seed` (block-cooperative).
// Equivalent to the reference generateCandidateCluster: repeatedly add the
// unclustered, non-member candidate with the smallest max distance to the
// current members, as long as that max distance is below `threshold`.
//
// Step 0 candidates are the seed's unclustered neighbors (CSR, with exact
// squared distances) or, without the CSR, all unclustered points. Later steps
// use the double-buffered live list in listIdx/listSq (max distances so far).
// Returns the cluster cardinality; members are written to `members` in
// insertion order.
__device__ int growCluster(const int seed, const SweepParams& p,
                           int* listIdx, Bits* listSq, int* members, BlockShared& sh) {
    const int tid = threadIdx.x;
    const int nThreads = blockDim.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid / kWarpSize;
    const int numWarps = (nThreads + kWarpSize - 1) / kWarpSize;
    const int capacity = p.N;

    if (tid == 0) {
        sh.newCount = 0;
        sh.last = seed;
        members[0] = seed;
    }
    __syncthreads();

    int count = 1;
    const int* srcIdx = p.cand;
    const Bits* srcSq = nullptr;     // nullptr: max distance is 0 before the first step
    int srcCount = p.numCand;
    bool exactFirst = false;         // first step: srcSq already holds distance to seed
    if (p.nbrOff != nullptr) {
        const int begin = p.nbrOff[seed];
        srcIdx = p.nbrIdx + begin;
        srcSq = p.nbrSq + begin;
        srcCount = p.nbrOff[seed + 1] - begin;
        exactFirst = true;
    }
    int parity = 0;

    while (true) {
        const int last = sh.last;
        const Point lastPt = p.pts[last];
        int* dstIdx = listIdx + parity * capacity;
        Bits* dstSq = listSq + parity * capacity;

        // Update the max distance of each live candidate with the newest member,
        // keep candidates that can still be selected, and find this thread's best.
        Bits localKey = kNoKey;
        int localIdx = kNoIndex;
        // Warp-uniform round loop so every lane reaches the ballot/shuffle below.
        for (int round = 0; round < srcCount; round += nThreads) {
            const int j = round + tid;
            int c = -1;
            Bits m = 0;
            bool keep = false;
            if (j < srcCount) {
                c = srcIdx[j];
                if (c != last && p.clustered[c] == 0) {
                    if (srcSq == nullptr) {
                        m = toBits(sqDistance(p.pts[c], lastPt));
                    } else if (exactFirst) {
                        m = srcSq[j];
                    } else {
                        m = max(srcSq[j], toBits(sqDistance(p.pts[c], lastPt)));
                    }
                    keep = m < p.thr2Bits;
                }
            }

            // Warp-aggregated compaction: one shared atomic per warp
            const unsigned int keepMask = __ballot_sync(0xffffffffu, keep);
            if (keepMask != 0) {
                const int leader = __ffs(keepMask) - 1;
                int warpBase = 0;
                if (lane == leader) {
                    warpBase = atomicAdd(&sh.newCount, __popc(keepMask));
                }
                warpBase = __shfl_sync(0xffffffffu, warpBase, leader);
                if (keep) {
                    const int pos = warpBase + __popc(keepMask & ((1u << lane) - 1u));
                    dstIdx[pos] = c;
                    dstSq[pos] = m;
                }
            }

            if (keep && keyLess(m, c, localKey, localIdx)) {
                localKey = m;
                localIdx = c;
            }
        }

        // Block-wide argmin
        warpArgMin(localKey, localIdx);
        if (lane == 0) {
            sh.warpKey[warp] = localKey;
            sh.warpIdx[warp] = localIdx;
        }
        __syncthreads();
        const int newCount = sh.newCount;  // final survivor count for this step
        if (warp == 0) {
            Bits k = kNoKey;
            int i = kNoIndex;
            if (lane < numWarps) {
                k = sh.warpKey[lane];
                i = sh.warpIdx[lane];
            }
            warpArgMin(k, i);
            if (lane == 0) {
                sh.bestKey = k;
                sh.bestIdx = i;
            }
        }
        __syncthreads();

        // Uniform decision across the block: no candidate, or diameter would
        // reach the threshold.
        const int best = sh.bestIdx;
        const double bestDist = sqrt(fromBits(sh.bestKey));
        if (best == kNoIndex || !(bestDist < p.threshold)) break;

        if (tid == 0) {
            members[count] = best;
            sh.last = best;
            sh.newCount = 0;  // safe: all threads already read newCount
        }
        ++count;
        srcIdx = dstIdx;
        srcSq = dstSq;
        srcCount = newCount;
        exactFirst = false;
        parity ^= 1;
        __syncthreads();
    }
    return count;
}

// Each block processes seeds seeds[blockIdx.x], seeds[blockIdx.x + gridDim.x], ...
// and writes the greedy cluster cardinality of each seed to cardinality[s].
// Per-block scratch: 2 * N list entries (index + sq) and N member entries.
__global__ void __launch_bounds__(kBlockThreads)
qtSeedSweepKernel(const SweepParams p, const int* __restrict__ seeds, const int numSeeds) {
    __shared__ BlockShared sh;

    int* listIdx = p.listIdx + static_cast<size_t>(blockIdx.x) * 2 * p.N;
    Bits* listSq = p.listSq + static_cast<size_t>(blockIdx.x) * 2 * p.N;
    int* members = p.members + static_cast<size_t>(blockIdx.x) * p.N;

    for (int s = blockIdx.x; s < numSeeds; s += gridDim.x) {
        const int card = growCluster(seeds[s], p, listIdx, listSq, members, sh);
        if (threadIdx.x == 0) {
            p.cardinality[s] = card;
        }
    }
}

// Neighbor lists: for each point, the other points with squared distance < thr2
// (same arithmetic and operand order as the growth step, so values are identical).
__global__ void countNeighborsKernel(const Point* __restrict__ pts, const int N,
                                     const double thr2, int* __restrict__ deg) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    const Point ps = pts[s];
    int cnt = 0;
    for (int c = 0; c < N; ++c) {
        if (c != s && sqDistance(pts[c], ps) < thr2) ++cnt;
    }
    deg[s] = cnt;
}

__global__ void fillNeighborsKernel(const Point* __restrict__ pts, const int N,
                                    const double thr2, const int* __restrict__ off,
                                    int* __restrict__ nbrIdx, Bits* __restrict__ nbrSq) {
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= N) return;
    const Point ps = pts[s];
    int pos = off[s];
    for (int c = 0; c < N; ++c) {
        if (c == s) continue;
        const double d = sqDistance(pts[c], ps);
        if (d < thr2) {
            nbrIdx[pos] = c;
            nbrSq[pos] = toBits(d);
            ++pos;
        }
    }
}

// Marks every point whose neighbor list contains a newly clustered member. The
// neighbor relation is symmetric (squared distances are bitwise symmetric), so
// the rows of the members identify exactly the seeds whose growth may change.
__global__ void markDirtyKernel(const int* __restrict__ members, const int numMembers,
                                const int* __restrict__ off, const int* __restrict__ nbrIdx,
                                unsigned char* __restrict__ dirty) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= numMembers) return;
    const int m = members[t];
    for (int k = off[m]; k < off[m + 1]; ++k) {
        dirty[nbrIdx[k]] = 1;
    }
}

// ---------------------------------------------------------------------------
// Host orchestration
// ---------------------------------------------------------------------------

template <typename T>
class DeviceBuffer {
  public:
    explicit DeviceBuffer(size_t count) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), std::max<size_t>(count, 1) * sizeof(T)));
    }
    ~DeviceBuffer() {
        if (ptr_) cudaFree(ptr_);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }

  private:
    T* ptr_ = nullptr;
};

// Main QT clustering algorithm (GPU)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    // Conservative squared threshold for pruning: any candidate with
    // sqrt(max_sq) < threshold also satisfies max_sq < thr2 (margin >> rounding).
    const double thr2 = threshold * threshold * (1.0 + 1e-9);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    int numSMs = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, device));

    DeviceBuffer<Point> d_pts(N);
    CUDA_CHECK(cudaMemcpy(d_pts.get(), points.data(), N * sizeof(Point),
                          cudaMemcpyHostToDevice));

    // Neighbor lists (CSR), built on the GPU if they fit in a quarter of free
    // device memory; otherwise each growth scans the unclustered points instead.
    const int nbrBlocks = (N + kBlockThreads - 1) / kBlockThreads;
    DeviceBuffer<int> d_deg(N);
    countNeighborsKernel<<<nbrBlocks, kBlockThreads>>>(d_pts.get(), N, thr2, d_deg.get());
    CUDA_CHECK(cudaGetLastError());
    std::vector<int> deg(N);
    CUDA_CHECK(cudaMemcpy(deg.data(), d_deg.get(), N * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<int> nbrOffHost(N + 1, 0);
    for (int i = 0; i < N; ++i) {
        nbrOffHost[i + 1] = nbrOffHost[i] + deg[i];
    }
    const size_t totalNbr = static_cast<size_t>(nbrOffHost[N]);
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const bool useNbr = totalNbr * (sizeof(int) + sizeof(Bits)) <= freeBytes / 4;

    DeviceBuffer<int> d_nbrOff(useNbr ? N + 1 : 0);
    DeviceBuffer<int> d_nbrIdx(useNbr ? totalNbr : 0);
    DeviceBuffer<Bits> d_nbrSq(useNbr ? totalNbr : 0);
    if (useNbr) {
        CUDA_CHECK(cudaMemcpy(d_nbrOff.get(), nbrOffHost.data(), (N + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));
        fillNeighborsKernel<<<nbrBlocks, kBlockThreads>>>(d_pts.get(), N, thr2,
                                                          d_nbrOff.get(), d_nbrIdx.get(),
                                                          d_nbrSq.get());
        CUDA_CHECK(cudaGetLastError());
    }
    std::vector<int>().swap(deg);
    std::vector<int>().swap(nbrOffHost);

    // Number of concurrently resident seed blocks. Scratch per block is
    // 2N * (sizeof(int) + sizeof(double)) + N * sizeof(int) bytes.
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t bytesPerBlock =
        static_cast<size_t>(N) * (2 * (sizeof(int) + sizeof(double)) + sizeof(int));
    const size_t memBlocks = std::max<size_t>(1, (freeBytes / 2) / bytesPerBlock);
    const int maxBlocks = static_cast<int>(std::min<size_t>(
        {static_cast<size_t>(N), static_cast<size_t>(numSMs) * 4, memBlocks}));

    DeviceBuffer<unsigned char> d_clustered(N);
    DeviceBuffer<int> d_cand(N);
    DeviceBuffer<int> d_seeds(N);
    DeviceBuffer<int> d_seedOne(1);
    DeviceBuffer<int> d_card(N);
    DeviceBuffer<int> d_listIdx(static_cast<size_t>(maxBlocks) * 2 * N);
    DeviceBuffer<Bits> d_listSq(static_cast<size_t>(maxBlocks) * 2 * N);
    DeviceBuffer<int> d_scratchMembers(static_cast<size_t>(maxBlocks) * N);
    DeviceBuffer<int> d_clusterMembers(N);
    DeviceBuffer<unsigned char> d_dirty(N);

    SweepParams params{};
    params.pts = d_pts.get();
    params.N = N;
    params.clustered = d_clustered.get();
    params.cand = d_cand.get();
    params.nbrOff = useNbr ? d_nbrOff.get() : nullptr;
    params.nbrIdx = useNbr ? d_nbrIdx.get() : nullptr;
    params.nbrSq = useNbr ? d_nbrSq.get() : nullptr;
    params.threshold = threshold;
    params.thr2 = thr2;
    std::memcpy(&params.thr2Bits, &thr2, sizeof(Bits));  // thr2 >= 0: bit order == value order
    params.cardinality = d_card.get();
    params.listIdx = d_listIdx.get();
    params.listSq = d_listSq.get();
    params.members = d_scratchMembers.get();

    // Cached greedy cardinality per point. With neighbor lists, the growth from a
    // seed depends only on its neighbors, so the cached value stays exact until a
    // neighbor is clustered (then the seed is marked dirty and recomputed).
    // Without neighbor lists every seed is recomputed on every iteration.
    std::vector<int> cardOf(N, 0);
    std::vector<unsigned char> dirty(N, 1);
    std::vector<int> seedsToRun;
    seedsToRun.reserve(N);
    std::vector<int> cardHost(N);

    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_clustered.get(), clustered.data(), N,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_cand.get(), unclustered_indices.data(),
                              U * sizeof(int), cudaMemcpyHostToDevice));
        params.numCand = U;

        // Grow a candidate cluster from every dirty unclustered seed
        seedsToRun.clear();
        for (int s : unclustered_indices) {
            if (dirty[s]) seedsToRun.push_back(s);
        }
        const int R = static_cast<int>(seedsToRun.size());
        if (R > 0) {
            CUDA_CHECK(cudaMemcpy(d_seeds.get(), seedsToRun.data(), R * sizeof(int),
                                  cudaMemcpyHostToDevice));
            const int grid = std::min(R, maxBlocks);
            qtSeedSweepKernel<<<grid, kBlockThreads>>>(params, d_seeds.get(), R);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(cardHost.data(), d_card.get(), R * sizeof(int),
                                  cudaMemcpyDeviceToHost));
            for (int i = 0; i < R; ++i) {
                cardOf[seedsToRun[i]] = cardHost[i];
                dirty[seedsToRun[i]] = 0;
            }
        }

        // Largest cardinality wins; ties go to the earliest (lowest) seed
        int best_pos = -1;
        int max_cardinality = -1;
        for (int i = 0; i < U; ++i) {
            const int card_i = cardOf[unclustered_indices[i]];
            if (card_i > max_cardinality) {
                max_cardinality = card_i;
                best_pos = i;
            }
        }

        if (best_pos < 0 || max_cardinality <= 0) break;  // No more clusters can be formed

        // Re-grow the winning seed (single block) to recover its members
        const int best_seed = unclustered_indices[best_pos];
        CUDA_CHECK(cudaMemcpy(d_seedOne.get(), &best_seed, sizeof(int),
                              cudaMemcpyHostToDevice));
        qtSeedSweepKernel<<<1, kBlockThreads>>>(params, d_seedOne.get(), 1);
        CUDA_CHECK(cudaGetLastError());

        int card = 0;
        CUDA_CHECK(cudaMemcpy(&card, d_card.get(), sizeof(int), cudaMemcpyDeviceToHost));
        if (card != max_cardinality) {
            fprintf(stderr, "Internal error: cluster re-computation mismatch\n");
            exit(EXIT_FAILURE);
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(card);
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), d_scratchMembers.get(),
                              card * sizeof(int), cudaMemcpyDeviceToHost));

        // Mark all members as clustered
        for (int m : cluster.members) {
            clustered[m] = 1;
        }

        // Seeds whose neighborhood contains a new member must be recomputed
        if (useNbr) {
            CUDA_CHECK(cudaMemcpy(d_clusterMembers.get(), cluster.members.data(),
                                  card * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_dirty.get(), dirty.data(), N, cudaMemcpyHostToDevice));
            const int markBlocks = (card + kBlockThreads - 1) / kBlockThreads;
            markDirtyKernel<<<markBlocks, kBlockThreads>>>(
                d_clusterMembers.get(), card, d_nbrOff.get(), d_nbrIdx.get(), d_dirty.get());
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(dirty.data(), d_dirty.get(), N, cudaMemcpyDeviceToHost));
        } else {
            std::fill(dirty.begin(), dirty.end(), static_cast<unsigned char>(1));
        }
        clusters.push_back(std::move(cluster));

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        
        // Check diameter (max distance between any two points)
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
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
    
    // Initialize the CUDA runtime/context outside the timed region
    CUDA_CHECK(cudaFree(nullptr));

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);
    
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
