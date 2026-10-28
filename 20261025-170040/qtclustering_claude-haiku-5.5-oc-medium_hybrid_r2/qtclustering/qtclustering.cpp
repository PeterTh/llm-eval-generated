// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy:
//   * MPI:    the candidate seeds of each clustering round are distributed
//             cyclically over ranks. Every round the ranks exchange their local
//             best (cardinality, seed) with MPI_Allgather and apply the same
//             selection rule, so all ranks hold identical clustering state.
//   * CUDA:   the greedy candidate-cluster growth of a batch of seeds runs on
//             the GPU, one thread block per seed (fp64, block-wide argmin).
//   * OpenMP: host-side per-round work (spatial-grid neighbourhood enumeration
//             and packing of candidate lists) and cluster validation run on
//             all cores of the rank.
//
// Semantics are identical to the sequential algorithm: the seed with the
// largest candidate cluster wins (lowest seed index on ties); the greedy growth
// adds the candidate with the smallest maximum distance to the members that is
// still below the threshold (lowest point index on ties). Floating-point
// contraction (FMA) is disabled on both host and device so that the distance
// values, and therefore all comparisons, are bit-identical on CPU and GPU.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

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

// Best seed of a round: largest cardinality, lowest seed index on ties.
// Plain-old-data (two ints) so it can be exchanged directly with MPI.
struct SeedChoice {
    int card;
    int seed;
};

static int g_rank = 0;
static int g_nranks = 1;

// Output helper: only rank 0 writes to stdout.
static void rootPrintf(const char* fmt, ...) {
    if (g_rank != 0) return;
    va_list ap;
    va_start(ap, fmt);
    vprintf(fmt, ap);
    va_end(ap);
}

#define CUDA_CHECK(expr)                                                              \
    do {                                                                              \
        cudaError_t err_ = (expr);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #expr, __FILE__,        \
                    __LINE__, cudaGetErrorString(err_));                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
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

// Calculate Euclidean distance between two points
inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

// ---------------------------------------------------------------------------
// Spatial grid over the currently unclustered points. Cell size >= threshold,
// so every point within the threshold of a seed lies in the 3x3 cell block
// around the seed. The grid is a superset filter; exact distances are checked
// on the device.
// ---------------------------------------------------------------------------
struct Grid {
    double minx_ = 0.0, miny_ = 0.0, cs_ = 1.0;
    int gx_ = 1, gy_ = 1;
    std::vector<int> start_;  // CSR offsets per cell
    std::vector<int> items_;  // point indices grouped by cell

    int cellX(double x) const {
        return std::clamp(static_cast<int>((x - minx_) / cs_), 0, gx_ - 1);
    }
    int cellY(double y) const {
        return std::clamp(static_cast<int>((y - miny_) / cs_), 0, gy_ - 1);
    }

    void build(const std::vector<int>& U, const std::vector<Point>& pts, double threshold) {
        double minx = std::numeric_limits<double>::max(), maxx = -minx;
        double miny = minx, maxy = -minx;
        for (int idx : U) {
            const Point& p = pts[idx];
            minx = std::min(minx, p.x);
            maxx = std::max(maxx, p.x);
            miny = std::min(miny, p.y);
            maxy = std::max(maxy, p.y);
        }
        const double ex = std::max(maxx - minx, maxy - miny);
        const int G = std::min(1024, std::max(1, static_cast<int>(std::sqrt(4.0 * U.size()))));
        // Slightly enlarged cells keep the "3x3 superset" property robust to rounding.
        cs_ = std::max(threshold, ex / G) * (1.0 + 1e-6);
        minx_ = minx;
        miny_ = miny;
        gx_ = static_cast<int>((maxx - minx) / cs_) + 1;
        gy_ = static_cast<int>((maxy - miny) / cs_) + 1;

        const size_t nc = static_cast<size_t>(gx_) * gy_;
        start_.assign(nc + 1, 0);
        std::vector<int> cell(U.size());
        for (size_t i = 0; i < U.size(); ++i) {
            const Point& p = pts[U[i]];
            cell[i] = cellY(p.y) * gx_ + cellX(p.x);
            start_[cell[i] + 1]++;
        }
        for (size_t c = 0; c < nc; ++c) start_[c + 1] += start_[c];

        items_.resize(U.size());
        std::vector<int> cursor(start_.begin(), start_.end() - 1);
        for (size_t i = 0; i < U.size(); ++i) {
            items_[cursor[cell[i]]++] = U[i];
        }
    }

    // Visits the CSR ranges of the (up to 9) cells around point p.
    template <class F>
    void forEachNearCell(const Point& p, F&& f) const {
        const int cx = cellX(p.x), cy = cellY(p.y);
        for (int y = std::max(cy - 1, 0); y <= std::min(cy + 1, gy_ - 1); ++y) {
            for (int x = std::max(cx - 1, 0); x <= std::min(cx + 1, gx_ - 1); ++x) {
                const int id = y * gx_ + x;
                f(start_[id], start_[id + 1]);
            }
        }
    }

    // Number of points in the neighbourhood of p (including p itself if indexed).
    int neighborhoodSize(const Point& p) const {
        int n = 0;
        forEachNearCell(p, [&](int b, int e) { n += e - b; });
        return n;
    }

    // Writes all indexed points of the neighbourhood except `seed` to out.
    void gather(const Point& p, int seed, int* out) const {
        int n = 0;
        forEachNearCell(p, [&](int b, int e) {
            for (int k = b; k < e; ++k) {
                const int it = items_[k];
                if (it != seed) out[n++] = it;
            }
        });
    }
};

// ---------------------------------------------------------------------------
// Host greedy growth for a single seed. Used to materialize the members of the
// winning cluster (identical algorithm to the device kernel).
// Returns the cardinality; members are appended in insertion order.
// ---------------------------------------------------------------------------
static int hostGreedyCluster(int seed, const std::vector<int>& U, const std::vector<Point>& pts,
                             double threshold, std::vector<int>& members) {
    std::vector<int> cand;
    std::vector<double> maxd;  // max distance from candidate to current members
    for (int c : U) {
        if (c == seed) continue;
        const double d = distance(pts[c], pts[seed]);
        if (d < threshold) {
            cand.push_back(c);
            maxd.push_back(d);
        }
    }

    members.assign(1, seed);
    while (!cand.empty()) {
        size_t bi = 0;
        for (size_t i = 1; i < cand.size(); ++i) {
            if (maxd[i] < maxd[bi] || (maxd[i] == maxd[bi] && cand[i] < cand[bi])) bi = i;
        }
        const int p = cand[bi];
        members.push_back(p);

        size_t w = 0;
        for (size_t i = 0; i < cand.size(); ++i) {
            if (i == bi) continue;
            const double d = distance(pts[cand[i]], pts[p]);
            const double m = std::max(maxd[i], d);
            if (m < threshold) {
                cand[w] = cand[i];
                maxd[w] = m;
                ++w;
            }
        }
        cand.resize(w);
        maxd.resize(w);
    }
    return static_cast<int>(members.size());
}

// ---------------------------------------------------------------------------
// CUDA kernel: one block per seed. Each thread owns a strided slice of the
// candidate list of its seed. Per growth step, the candidates' running max
// distance is updated with the newly added member and the block computes the
// argmin (value, then point index) over eligible candidates.
// ---------------------------------------------------------------------------
namespace {

constexpr int kBlockThreads = 256;
constexpr int kNoIndex = INT_MAX;
constexpr double kInf = std::numeric_limits<double>::infinity();

__device__ __forceinline__ bool argLess(double v1, int c1, double v2, int c2) {
    return v1 < v2 || (v1 == v2 && c1 < c2);
}

// Block-wide argmin over (value, index). Result is written to *outVal/*outIdx
// and visible to all threads of the block on return.
__device__ __forceinline__ void blockArgMin(double v, int c, double* sVal, int* sIdx,
                                            double* outVal, int* outIdx) {
    const int tid = threadIdx.x;
    sVal[tid] = v;
    sIdx[tid] = c;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s && argLess(sVal[tid + s], sIdx[tid + s], sVal[tid], sIdx[tid])) {
            sVal[tid] = sVal[tid + s];
            sIdx[tid] = sIdx[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        *outVal = sVal[0];
        *outIdx = sIdx[0];
    }
    __syncthreads();
}

__global__ void __launch_bounds__(kBlockThreads)
seedGreedyKernel(const Point* __restrict__ pts, const int* __restrict__ seeds,
                 const int* __restrict__ offs, const int* __restrict__ cand,
                 double* __restrict__ maxd, unsigned char* __restrict__ alive,
                 int* __restrict__ cards, double threshold) {
    __shared__ double sVal[kBlockThreads];
    __shared__ int sIdx[kBlockThreads];
    __shared__ double bestVal;
    __shared__ int bestIdx;

    const int b = blockIdx.x;
    const int tid = threadIdx.x;
    const int seed = seeds[b];
    const int o = offs[b];
    const int e = offs[b + 1];
    const Point ps = pts[seed];

    // Initial state: cluster = {seed}; maxd = distance to seed; eligible if < threshold.
    double lv = kInf;
    int lc = kNoIndex;
    for (int k = o + tid; k < e; k += blockDim.x) {
        const int c = cand[k];
        const Point pc = pts[c];
        const double dx = pc.x - ps.x;
        const double dy = pc.y - ps.y;
        const double d = sqrt(dx * dx + dy * dy);
        maxd[k] = d;
        const bool a = d < threshold;
        alive[k] = a ? 1 : 0;
        if (a && argLess(d, c, lv, lc)) {
            lv = d;
            lc = c;
        }
    }
    blockArgMin(lv, lc, sVal, sIdx, &bestVal, &bestIdx);

    int card = 1;
    while (bestIdx != kNoIndex) {
        const int chosen = bestIdx;
        ++card;
        const Point pp = pts[chosen];

        // Add `chosen` to the cluster: update running max distances, prune
        // candidates that can no longer be below the threshold, and compute
        // the next argmin in the same pass.
        lv = kInf;
        lc = kNoIndex;
        for (int k = o + tid; k < e; k += blockDim.x) {
            if (!alive[k]) continue;
            const int c = cand[k];
            if (c == chosen) {
                alive[k] = 0;
                continue;
            }
            const Point pc = pts[c];
            const double dx = pc.x - pp.x;
            const double dy = pc.y - pp.y;
            const double d = sqrt(dx * dx + dy * dy);
            double m = maxd[k];
            if (d > m) m = d;
            maxd[k] = m;
            if (m < threshold) {
                if (argLess(m, c, lv, lc)) {
                    lv = m;
                    lc = c;
                }
            } else {
                alive[k] = 0;  // max distance only grows: never eligible again
            }
        }
        blockArgMin(lv, lc, sVal, sIdx, &bestVal, &bestIdx);
    }

    if (tid == 0) cards[b] = card;
}

}  // namespace

// Owns the device-side data for the seed-evaluation kernel.
class SeedGreedyGpu {
  public:
    static constexpr size_t kMaxBatchEntries = size_t(1) << 24;

    SeedGreedyGpu(const std::vector<Point>& pts, double threshold)
        : threshold_(threshold),
          seedCap_(pts.size()),
          entryCap_(std::max(kMaxBatchEntries, pts.size())) {
        CUDA_CHECK(cudaMalloc(&d_pts_, sizeof(Point) * pts.size()));
        CUDA_CHECK(cudaMemcpy(d_pts_, pts.data(), sizeof(Point) * pts.size(),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_seeds_, sizeof(int) * seedCap_));
        CUDA_CHECK(cudaMalloc(&d_offs_, sizeof(int) * (seedCap_ + 1)));
        CUDA_CHECK(cudaMalloc(&d_cards_, sizeof(int) * seedCap_));
        CUDA_CHECK(cudaMalloc(&d_cand_, sizeof(int) * entryCap_));
        CUDA_CHECK(cudaMalloc(&d_maxd_, sizeof(double) * entryCap_));
        CUDA_CHECK(cudaMalloc(&d_alive_, entryCap_));
    }

    ~SeedGreedyGpu() {
        cudaFree(d_pts_);
        cudaFree(d_seeds_);
        cudaFree(d_offs_);
        cudaFree(d_cards_);
        cudaFree(d_cand_);
        cudaFree(d_maxd_);
        cudaFree(d_alive_);
    }

    SeedGreedyGpu(const SeedGreedyGpu&) = delete;
    SeedGreedyGpu& operator=(const SeedGreedyGpu&) = delete;

    size_t entryCapacity() const { return entryCap_; }

    // Evaluates numSeeds seeds; seed b has candidates cand[offs[b] .. offs[b+1]).
    void evaluate(const int* seeds, const int* offs, int numSeeds, const int* cand,
                  int numEntries, int* cards) {
        if (numSeeds <= 0) return;
        CUDA_CHECK(cudaMemcpy(d_seeds_, seeds, sizeof(int) * numSeeds, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_offs_, offs, sizeof(int) * (numSeeds + 1), cudaMemcpyHostToDevice));
        if (numEntries > 0) {
            CUDA_CHECK(cudaMemcpy(d_cand_, cand, sizeof(int) * numEntries, cudaMemcpyHostToDevice));
        }
        seedGreedyKernel<<<numSeeds, kBlockThreads>>>(d_pts_, d_seeds_, d_offs_, d_cand_, d_maxd_,
                                                      d_alive_, d_cards_, threshold_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cards, d_cards_, sizeof(int) * numSeeds, cudaMemcpyDeviceToHost));
    }

  private:
    double threshold_;
    size_t seedCap_;
    size_t entryCap_;
    Point* d_pts_ = nullptr;
    int* d_seeds_ = nullptr;
    int* d_offs_ = nullptr;
    int* d_cards_ = nullptr;
    int* d_cand_ = nullptr;
    double* d_maxd_ = nullptr;
    unsigned char* d_alive_ = nullptr;
};

// One clustering round: evaluates the locally owned unclustered seeds
// (cyclic distribution over ranks) and returns the best local choice.
static SeedChoice evaluateRound(const std::vector<int>& U, const std::vector<Point>& pts,
                                double threshold, SeedGreedyGpu& gpu) {
    Grid grid;
    grid.build(U, pts, threshold);

    const int nU = static_cast<int>(U.size());
    const int nLocal = (g_rank < nU) ? (nU - g_rank + g_nranks - 1) / g_nranks : 0;
    auto seedAt = [&](int li) { return U[g_rank + li * g_nranks]; };

    // Candidate count per local seed (neighbourhood superset minus the seed).
    std::vector<int> counts(nLocal);
#pragma omp parallel for schedule(dynamic, 64)
    for (int li = 0; li < nLocal; ++li) {
        counts[li] = grid.neighborhoodSize(pts[seedAt(li)]) - 1;
    }

    SeedChoice best{0, kNoIndex};
    const size_t cap = gpu.entryCapacity();
    std::vector<int> hSeeds, hOffs, hCand, hCards;

    int lo = 0;
    while (lo < nLocal) {
        // Form a batch that fits the device buffers (always at least one seed).
        int hi = lo;
        size_t total = 0;
        while (hi < nLocal && (hi == lo || total + static_cast<size_t>(counts[hi]) <= cap)) {
            total += static_cast<size_t>(counts[hi]);
            ++hi;
        }
        const int B = hi - lo;

        hSeeds.resize(B);
        hOffs.assign(B + 1, 0);
        for (int b = 0; b < B; ++b) {
            hSeeds[b] = seedAt(lo + b);
            hOffs[b + 1] = hOffs[b] + counts[lo + b];
        }
        hCand.resize(total);
#pragma omp parallel for schedule(dynamic, 16)
        for (int b = 0; b < B; ++b) {
            grid.gather(pts[hSeeds[b]], hSeeds[b], hCand.data() + hOffs[b]);
        }

        hCards.resize(B);
        gpu.evaluate(hSeeds.data(), hOffs.data(), B, hCand.data(), static_cast<int>(total),
                     hCards.data());

        for (int b = 0; b < B; ++b) {
            const int card = hCards[b], seed = hSeeds[b];
            if (card > best.card || (card == best.card && seed < best.seed)) {
                best = {card, seed};
            }
        }
        lo = hi;
    }
    return best;
}

// Main QT clustering algorithm. Every rank executes the same sequence of
// rounds and returns an identical cluster list.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold,
                                  SeedGreedyGpu& gpu) {
    const int N = static_cast<int>(points.size());
    std::vector<char> clustered(N, 0);
    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<Cluster> clusters;

    while (!unclustered.empty()) {
        const SeedChoice local = evaluateRound(unclustered, points, threshold, gpu);

        std::vector<SeedChoice> all(g_nranks);
        MPI_Allgather(&local, 2, MPI_INT, all.data(), 2, MPI_INT, MPI_COMM_WORLD);

        SeedChoice best{0, kNoIndex};
        for (const SeedChoice& c : all) {
            if (c.card > best.card || (c.card == best.card && c.seed < best.seed)) best = c;
        }

        // No more clusters can be formed
        if (best.card <= 0) break;

        std::vector<int> members;
        const int card = hostGreedyCluster(best.seed, unclustered, points, threshold, members);
        if (card != best.card) {
            fprintf(stderr, "Internal error: cluster cardinality mismatch (seed %d: %d vs %d)\n",
                    best.seed, card, best.card);
            MPI_Abort(MPI_COMM_WORLD, 3);
        }

        for (int m : members) clustered[m] = 1;
        clusters.push_back(Cluster{std::move(members), best.seed});

        // Remove clustered points from unclustered list (order preserved)
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered.end());
    }

    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    rootPrintf("Validating clusters:\n");

    // Diameters are independent per cluster: compute in parallel.
    std::vector<double> diameter(clusters.size(), 0.0);
#pragma omp parallel for schedule(dynamic)
    for (long c = 0; c < static_cast<long>(clusters.size()); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameter[c] = max_diameter;
    }
    
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameter[c];
        
        if (c < 10) { // Print first 10 clusters
            rootPrintf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",
                       c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            rootPrintf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n",
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
                rootPrintf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
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
    
    rootPrintf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
               points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    rootPrintf("Usage: %s [options]\n", progName);
    rootPrintf("Options:\n");
    rootPrintf("  -n <num>     Number of points (default: 1000)\n");
    rootPrintf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    rootPrintf("  -v           Enable validation\n");
    rootPrintf("  -r           Print results for external validation\n");
    rootPrintf("  -h           Show this help message\n");
}

static int runBenchmark(int argc, char** argv) {
    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments (identical on all ranks)
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
            rootPrintf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        rootPrintf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        return 1;
    }
    
    rootPrintf("QT Clustering Benchmark\n");
    rootPrintf("Number of points: %d\n", num_points);
    rootPrintf("Distance threshold: %.2f\n", threshold);
    rootPrintf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Generate synthetic data on rank 0 and distribute it
    std::vector<Point> points(num_points);
    if (g_rank == 0) generateSyntheticData(points, num_points);
    MPI_Bcast(points.data(), 2 * num_points, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Bind each rank to a GPU on its node
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "Error: no CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    CUDA_CHECK(cudaFree(nullptr));  // create the context outside the timed region

    SeedGreedyGpu gpu(points, threshold);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, gpu);
    
    const double elapsed_local = MPI_Wtime() - t0;
    double elapsed = 0.0;
    MPI_Reduce(&elapsed_local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const long cluster_ms = static_cast<long>(elapsed * 1000.0);
    
    rootPrintf("Clustering time: %ld ms\n", cluster_ms);
    rootPrintf("Clusters found: %zu\n", clusters.size());
    
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
    
    rootPrintf("Points clustered: %d / %d (%.1f%%)\n", 
               total_clustered, num_points, 
               100.0 * total_clustered / num_points);
    rootPrintf("Average cluster size: %.2f\n", avg_cluster_size);
    rootPrintf("Maximum cluster size: %d\n", max_cluster_size);
    
    // Performance metrics
    const double time_sec = cluster_ms / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    rootPrintf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    
    // Print results for external validation
    if (printResults && g_rank == 0) {
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
        const bool valid = (g_rank == 0) ? validateClusters(clusters, points, threshold) : false;
        if (valid) {
            rootPrintf("Validation: PASSED\n");
        } else {
            rootPrintf("Validation: FAILED\n");
            rc = 1;
        }
    }
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return rc;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    const int rc = runBenchmark(argc, argv);

    MPI_Finalize();
    return rc;
}
