// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization (every layer is always used):
//   MPI    Each clustering round the unclustered seeds are dealt out to the
//          ranks. Every rank reports its best (cardinality, seed) pair and all
//          ranks agree on the global winner with a single allgather.
//   CUDA   Each rank drives one GPU (rank % device count). One thread block
//          grows the candidate cluster of one seed; the nearest-candidate
//          search of every growth step is a parallel reduction.
//   OpenMP The threshold neighbour graph, which is the bulk of the host-side
//          work, is built by a team of threads.
//
// Algorithmic changes that keep the output identical to the sequential search:
//   - Only points within the threshold of the seed can ever join its cluster,
//     so candidates are taken from a precomputed neighbour graph.
//   - The maximum distance of each candidate to the growing cluster is updated
//     incrementally (max is exact), and a candidate whose maximum reaches the
//     threshold is dropped for good (the maximum never decreases).
//   - Seeds are visited in order of an upper bound on their cardinality (one
//     plus their unclustered neighbours), and the search stops as soon as no
//     remaining seed can beat the best cardinality found.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Seeds evaluated per GPU launch; also bounds the device workspace
static const int kBatchSeeds = 512;
// Threads per block of the seed-growth kernel (must be a multiple of 32, at most 1024)
static const int kBlockThreads = 128;
static const int kWarpsPerBlock = kBlockThreads / 32;

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

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

// Calculate Euclidean distance between two points.
// The fused multiply-add matches the expression the device kernel evaluates,
// so both sides produce bit-identical distances.
inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(std::fma(dx, dx, dy * dy));
}

// Threshold neighbour graph in CSR form: the neighbours of point i are
// index[offset[i]] .. index[offset[i+1]-1], i.e. every j != i with
// distance(i, j) < threshold, in ascending order.
struct NeighborGraph {
    std::vector<int64_t> offset;
    std::vector<int> index;
    int maxDegree = 0;
};

// Build the neighbour graph with a uniform grid (OpenMP over the points)
NeighborGraph buildNeighborGraph(const std::vector<Point>& points, const double threshold) {
    const int N = static_cast<int>(points.size());

    double minX = points[0].x, maxX = minX;
    double minY = points[0].y, maxY = minY;
    for (const Point& p : points) {
        minX = std::min(minX, p.x);
        maxX = std::max(maxX, p.x);
        minY = std::min(minY, p.y);
        maxY = std::max(maxY, p.y);
    }
    const double width = std::max(maxX - minX, maxY - minY);

    // Cells are at least as wide as the threshold (slightly enlarged so rounding
    // cannot push a close pair two cells apart) and the grid is capped at 1024 per axis
    double cell = threshold * (1.0 + 1e-9);
    if (!(cell >= width / 1024.0)) cell = width / 1024.0;  // also catches a NaN threshold
    if (!(cell > 0.0)) cell = 1.0;                         // all points coincide
    const int nx = static_cast<int>((maxX - minX) / cell) + 1;
    const int ny = static_cast<int>((maxY - minY) / cell) + 1;

    std::vector<int> pointCell(N);
    std::vector<int> cellStart(static_cast<size_t>(nx) * ny + 1, 0);
    for (int i = 0; i < N; ++i) {
        const int cx = std::min(nx - 1, static_cast<int>((points[i].x - minX) / cell));
        const int cy = std::min(ny - 1, static_cast<int>((points[i].y - minY) / cell));
        pointCell[i] = cy * nx + cx;
        cellStart[pointCell[i] + 1]++;
    }
    for (size_t c = 0; c + 1 < cellStart.size(); ++c) {
        cellStart[c + 1] += cellStart[c];
    }
    std::vector<int> cellPoints(N);
    {
        std::vector<int> next(cellStart.begin(), cellStart.end() - 1);
        for (int i = 0; i < N; ++i) {
            cellPoints[next[pointCell[i]]++] = i;
        }
    }

    // Calls visit(j) for every neighbour j of point i (in grid order)
    auto forEachNeighbor = [&](const int i, auto&& visit) {
        const int cx = pointCell[i] % nx;
        const int cy = pointCell[i] / nx;
        for (int y = std::max(cy - 1, 0); y <= std::min(cy + 1, ny - 1); ++y) {
            for (int x = std::max(cx - 1, 0); x <= std::min(cx + 1, nx - 1); ++x) {
                const int c = y * nx + x;
                for (int k = cellStart[c]; k < cellStart[c + 1]; ++k) {
                    const int j = cellPoints[k];
                    if (j != i && distance(points[i], points[j]) < threshold) {
                        visit(j);
                    }
                }
            }
        }
    };

    // Two passes over the points (count, then fill) so no per-point allocation is needed
    NeighborGraph graph;
    graph.offset.assign(N + 1, 0);
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        int64_t degree = 0;
        forEachNeighbor(i, [&degree](int) { ++degree; });
        graph.offset[i + 1] = degree;
    }
    for (int i = 0; i < N; ++i) {
        graph.offset[i + 1] += graph.offset[i];
        graph.maxDegree = std::max(graph.maxDegree,
                                   static_cast<int>(graph.offset[i + 1] - graph.offset[i]));
    }

    graph.index.resize(static_cast<size_t>(graph.offset[N]));
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < N; ++i) {
        int* out = graph.index.data() + graph.offset[i];
        forEachNeighbor(i, [&out](int j) { *out++ = j; });
        std::sort(graph.index.data() + graph.offset[i], out);
    }
    return graph;
}

// Lexicographic (value, position) comparison: smaller value wins, ties go to the smaller position
__device__ __forceinline__ bool lessValuePos(double v1, int p1, double v2, int p2) {
    return v1 < v2 || (v1 == v2 && p1 < p2);
}

// Grow the candidate cluster of one seed per block. Block b handles seeds[b].
// Each step selects the candidate with the smallest maximum distance to the
// cluster (ties to the smallest index; neighbour lists are sorted) and then
// refreshes the maxima of the remaining candidates. Candidates whose maximum
// reaches the threshold are dropped. cards[b] receives the cluster cardinality.
__global__ void evaluateSeeds(const double* __restrict__ px, const double* __restrict__ py,
                              const int64_t* __restrict__ nbrOffset,
                              const int* __restrict__ nbrIndex,
                              const uint8_t* __restrict__ clustered,
                              const int* __restrict__ seeds,
                              const double threshold, const int stride,
                              double* __restrict__ wsMaxDist,
                              uint8_t* __restrict__ wsAlive,
                              int* __restrict__ cards) {
    __shared__ double warpVal[kWarpsPerBlock];
    __shared__ int warpPos[kWarpsPerBlock];
    __shared__ int chosen;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int seed = seeds[blockIdx.x];
    const int64_t begin = nbrOffset[seed];
    const int deg = static_cast<int>(nbrOffset[seed + 1] - begin);
    const int* nbr = nbrIndex + begin;
    double* maxDist = wsMaxDist + static_cast<size_t>(blockIdx.x) * stride;
    uint8_t* alive = wsAlive + static_cast<size_t>(blockIdx.x) * stride;

    // Maximum distance of each candidate to the cluster, which is just the seed so far
    const double sx = px[seed];
    const double sy = py[seed];
    for (int j = tid; j < deg; j += kBlockThreads) {
        const int p = nbr[j];
        const double dx = px[p] - sx;
        const double dy = py[p] - sy;
        const double d = __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
        maxDist[j] = d;
        alive[j] = (clustered[p] == 0 && d < threshold) ? 1 : 0;
    }
    __syncthreads();

    int card = 1;  // the seed itself
    while (true) {
        // Nearest eligible candidate: per-thread scan, then warp and block reductions
        double bv = INFINITY;
        int bp = INT_MAX;
        for (int j = tid; j < deg; j += kBlockThreads) {
            if (alive[j] && maxDist[j] < bv) {
                bv = maxDist[j];
                bp = j;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, bv, off);
            const int op = __shfl_down_sync(0xffffffffu, bp, off);
            if (lessValuePos(ov, op, bv, bp)) {
                bv = ov;
                bp = op;
            }
        }
        if (lane == 0) {
            warpVal[warp] = bv;
            warpPos[warp] = bp;
        }
        __syncthreads();
        if (warp == 0) {
            const bool has = lane < kWarpsPerBlock;
            bv = has ? warpVal[lane] : INFINITY;
            bp = has ? warpPos[lane] : INT_MAX;
            for (int off = 16; off > 0; off >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, bv, off);
                const int op = __shfl_down_sync(0xffffffffu, bp, off);
                if (lessValuePos(ov, op, bv, bp)) {
                    bv = ov;
                    bp = op;
                }
            }
            if (lane == 0) chosen = bp;
        }
        __syncthreads();

        const int q = chosen;
        if (q == INT_MAX) break;  // no eligible candidate left

        // Add the chosen point and refresh the maxima of the remaining candidates
        ++card;
        const int cp = nbr[q];
        const double cx = px[cp];
        const double cy = py[cp];
        for (int j = tid; j < deg; j += kBlockThreads) {
            if (!alive[j]) continue;
            if (j == q) {
                alive[j] = 0;
                continue;
            }
            const int p = nbr[j];
            const double dx = px[p] - cx;
            const double dy = py[p] - cy;
            const double d = __dsqrt_rn(__fma_rn(dx, dx, __dmul_rn(dy, dy)));
            const double m = fmax(maxDist[j], d);
            maxDist[j] = m;
            if (m >= threshold) alive[j] = 0;
        }
        __syncthreads();
    }

    if (tid == 0) cards[blockIdx.x] = card;
}

// Selects this rank's GPU and creates its context. Runtime startup, like MPI_Init,
// so it is done before the clustering timer starts.
void initCudaDevice(const int rank) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "rank %d: no CUDA device available\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));
    CUDA_CHECK(cudaFree(nullptr));  // forces context creation
}

// Owns the device copies of the problem and evaluates batches of seeds on this rank's GPU
class SeedEvaluator {
  public:
    SeedEvaluator(const std::vector<Point>& points, const NeighborGraph& graph,
                  const double threshold)
        : n_(static_cast<int>(points.size())),
          threshold_(threshold),
          stride_(std::max(graph.maxDegree, 1)) {
        std::vector<double> xs(n_), ys(n_);
        for (int i = 0; i < n_; ++i) {
            xs[i] = points[i].x;
            ys[i] = points[i].y;
        }
        CUDA_CHECK(cudaMalloc(&px_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&py_, n_ * sizeof(double)));
        CUDA_CHECK(cudaMemcpy(px_, xs.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(py_, ys.data(), n_ * sizeof(double), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&nbrOffset_, graph.offset.size() * sizeof(int64_t)));
        CUDA_CHECK(cudaMemcpy(nbrOffset_, graph.offset.data(),
                              graph.offset.size() * sizeof(int64_t), cudaMemcpyHostToDevice));
        const size_t indexCount = std::max<size_t>(graph.index.size(), 1);
        CUDA_CHECK(cudaMalloc(&nbrIndex_, indexCount * sizeof(int)));
        CUDA_CHECK(cudaMemcpy(nbrIndex_, graph.index.data(),
                              graph.index.size() * sizeof(int), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc(&clustered_, std::max(n_, 1)));
        CUDA_CHECK(cudaMalloc(&seeds_, kBatchSeeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&cards_, kBatchSeeds * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&wsMaxDist_, static_cast<size_t>(kBatchSeeds) * stride_ * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&wsAlive_, static_cast<size_t>(kBatchSeeds) * stride_));
    }

    ~SeedEvaluator() {
        cudaFree(px_);
        cudaFree(py_);
        cudaFree(nbrOffset_);
        cudaFree(nbrIndex_);
        cudaFree(clustered_);
        cudaFree(seeds_);
        cudaFree(cards_);
        cudaFree(wsMaxDist_);
        cudaFree(wsAlive_);
    }

    SeedEvaluator(const SeedEvaluator&) = delete;
    SeedEvaluator& operator=(const SeedEvaluator&) = delete;

    // Cluster cardinality of each of the count seeds (count <= kBatchSeeds)
    void evaluate(const int* seeds, const int count, const std::vector<uint8_t>& clustered,
                  int* cards) {
        CUDA_CHECK(cudaMemcpy(clustered_, clustered.data(), n_, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(seeds_, seeds, count * sizeof(int), cudaMemcpyHostToDevice));
        evaluateSeeds<<<count, kBlockThreads>>>(px_, py_, nbrOffset_, nbrIndex_, clustered_,
                                                seeds_, threshold_, stride_,
                                                wsMaxDist_, wsAlive_, cards_);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(cards, cards_, count * sizeof(int), cudaMemcpyDeviceToHost));
    }

  private:
    int n_;
    double threshold_;
    int stride_;
    double* px_ = nullptr;
    double* py_ = nullptr;
    int64_t* nbrOffset_ = nullptr;
    int* nbrIndex_ = nullptr;
    uint8_t* clustered_ = nullptr;
    int* seeds_ = nullptr;
    int* cards_ = nullptr;
    double* wsMaxDist_ = nullptr;
    uint8_t* wsAlive_ = nullptr;
};

// Host reference of the greedy growth from one seed (same rules as the device
// kernel). Returns the members in the order they were added, seed first.
std::vector<int> growCluster(const int seed, const std::vector<Point>& points,
                             const NeighborGraph& graph, const std::vector<uint8_t>& clustered,
                             const double threshold) {
    const int64_t begin = graph.offset[seed];
    const int64_t end = graph.offset[seed + 1];
    std::vector<int> cand;
    std::vector<double> maxDist;
    for (int64_t k = begin; k < end; ++k) {
        const int p = graph.index[k];
        if (!clustered[p]) {
            cand.push_back(p);
            maxDist.push_back(distance(points[p], points[seed]));
        }
    }
    std::vector<uint8_t> alive(cand.size(), 1);

    std::vector<int> members{seed};
    while (true) {
        int best = -1;
        double bestDist = std::numeric_limits<double>::max();
        for (size_t k = 0; k < cand.size(); ++k) {
            if (alive[k] && maxDist[k] < bestDist) {
                bestDist = maxDist[k];
                best = static_cast<int>(k);
            }
        }
        if (best < 0) break;

        const int chosen = cand[best];
        alive[best] = 0;
        members.push_back(chosen);
        for (size_t k = 0; k < cand.size(); ++k) {
            if (!alive[k]) continue;
            maxDist[k] = std::max(maxDist[k], distance(points[cand[k]], points[chosen]));
            if (maxDist[k] >= threshold) alive[k] = 0;
        }
    }
    return members;
}

// Main QT clustering algorithm. Every rank runs the same loop and holds the
// same clustering state; only the seed search is divided among the ranks.
std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold,
                                  const int rank, const int size) {
    const int N = static_cast<int>(points.size());
    const NeighborGraph graph = buildNeighborGraph(points, threshold);
    SeedEvaluator evaluator(points, graph, threshold);

    std::vector<uint8_t> clustered(N, 0);
    // Number of unclustered neighbours of each point (bounds its cluster cardinality)
    std::vector<int> unclusteredNeighbors(N);
    for (int i = 0; i < N; ++i) {
        unclusteredNeighbors[i] = static_cast<int>(graph.offset[i + 1] - graph.offset[i]);
    }

    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);
    std::vector<Cluster> clusters;

    std::vector<int> local;
    std::vector<int> cards(kBatchSeeds);
    while (!unclustered.empty()) {
        // This rank's share of the unclustered seeds (cyclic over the list)
        local.clear();
        for (size_t i = rank; i < unclustered.size(); i += size) {
            local.push_back(unclustered[i]);
        }
        // Most promising seeds first; the cardinality bound is non-increasing along the list
        std::sort(local.begin(), local.end(), [&](int a, int b) {
            if (unclusteredNeighbors[a] != unclusteredNeighbors[b]) {
                return unclusteredNeighbors[a] > unclusteredNeighbors[b];
            }
            return a < b;
        });

        // Best (cardinality, seed) on this rank: largest cardinality, then smallest seed
        int bestCard = 0;
        int bestSeed = INT_MAX;
        for (size_t start = 0; start < local.size(); start += kBatchSeeds) {
            if (1 + unclusteredNeighbors[local[start]] < bestCard) break;

            const int count = static_cast<int>(std::min<size_t>(kBatchSeeds, local.size() - start));
            evaluator.evaluate(&local[start], count, clustered, cards.data());
            for (int i = 0; i < count; ++i) {
                const int seed = local[start + i];
                if (cards[i] > bestCard || (cards[i] == bestCard && seed < bestSeed)) {
                    bestCard = cards[i];
                    bestSeed = seed;
                }
            }
        }

        // Combine the per-rank winners. Ranks without seeds report (0, INT_MAX).
        const int localBest[2] = {bestCard, bestSeed};
        std::vector<int> allBest(2 * size);
        MPI_Allgather(localBest, 2, MPI_INT, allBest.data(), 2, MPI_INT, MPI_COMM_WORLD);
        int globalCard = 0;
        int globalSeed = INT_MAX;
        for (int r = 0; r < size; ++r) {
            const int c = allBest[2 * r];
            const int s = allBest[2 * r + 1];
            if (c > globalCard || (c == globalCard && s < globalSeed)) {
                globalCard = c;
                globalSeed = s;
            }
        }
        if (globalCard <= 0) break;  // No more clusters can be formed

        // Every rank rebuilds the winning cluster so the member order matches the sequential run
        std::vector<int> members = growCluster(globalSeed, points, graph, clustered, threshold);
        if (static_cast<int>(members.size()) != globalCard) {
            fprintf(stderr, "Internal error: GPU cardinality %d != host cardinality %zu for seed %d\n",
                    globalCard, members.size(), globalSeed);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        Cluster cluster;
        cluster.seed_point = globalSeed;
        cluster.members = members;
        clusters.push_back(cluster);

        // Mark all members as clustered and update the neighbour counts
        for (int m : members) {
            clustered[m] = 1;
            for (int64_t k = graph.offset[m]; k < graph.offset[m + 1]; ++k) {
                unclusteredNeighbors[graph.index[k]]--;
            }
        }

        // Remove clustered points from unclustered list
        unclustered.erase(
            std::remove_if(unclustered.begin(), unclustered.end(),
                          [&clustered](int idx) { return clustered[idx]; }),
            unclustered.end()
        );
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

// Runs the benchmark on this rank. Only rank 0 writes to stdout.
int run(int argc, char** argv, const int rank, const int numRanks) {
    const bool root = rank == 0;
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
        if (root) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n",
                   num_points, threshold);
        }
        return 1;
    }

    if (root) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Generate synthetic data (deterministic, so every rank holds the same points)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);

    // Runtime startup is excluded from the timing, like MPI_Init
    initCudaDevice(rank);

    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, numRanks);

    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start);

    // Every rank holds the same result; the rest of the benchmark is reporting
    if (!root) return 0;

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
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = run(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
