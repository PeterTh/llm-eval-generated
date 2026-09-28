// QT Clustering Benchmark - CUDA Parallel Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy (CUDA):
//   * The dominant cost of the algorithm is growing one candidate cluster for
//     *every* still unclustered seed point in every outer iteration.  Those
//     candidate clusters are completely independent of each other, so one CUDA
//     block grows the candidate cluster of one seed (blocks loop over seeds
//     with a grid stride).
//   * Inside a block, all threads cooperate on the O(N) "find the unclustered
//     point whose maximum distance to the current members is minimal" scan and
//     on the following block-wide argmin reduction.
//   * The per-candidate maximum distance to the cluster members is carried
//     incrementally (max over members == max of previous max and the distance
//     to the newly added member), which is bit-identical to recomputing the
//     full maximum, and points whose maximum distance has reached the
//     threshold are dropped from the candidate list (the maximum is
//     monotonically non-decreasing, so they can never become admissible
//     again).
//   * After a cluster has been committed, only seeds that have at least one of
//     the newly clustered points within `threshold` can change their result;
//     all other candidate clusters are provably identical to the previous
//     iteration, so their cardinality is reused.
//
// All tie-breaking rules of the sequential code (smallest index wins) are
// reproduced exactly and all distance arithmetic is performed in double
// precision, so the produced clustering is identical to the sequential one.
// Distances are compared in squared form (monotone, and the threshold test uses
// an exactly derived cutoff), which avoids a double precision square root in
// the innermost loop.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        const cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                    __LINE__, cudaGetErrorString(err__));                      \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
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
// Device code
// ---------------------------------------------------------------------------

// Sentinel for "no admissible candidate found"; larger than any squared
// distance that can occur.
constexpr double kNoCandidate = std::numeric_limits<double>::infinity();

// Squared Euclidean distance.  All comparisons the algorithm performs on
// distances (ordering of candidates and the threshold test) are monotone, so
// they can be carried out on squared distances instead, which avoids a very
// expensive double precision square root per candidate.  The threshold test
// uses the exact cutoff computed by squaredThresholdCutoff() below, so it is
// bit-exactly the same predicate as `sqrt(d2) < threshold`.
__device__ __forceinline__ double devDistanceSq(const double2 a, const double2 b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return dx * dx + dy * dy;
}

// Smallest squared distance whose (correctly rounded, monotone) square root
// reaches `threshold`, i.e. `d2 < cutoff` <=> `sqrt(d2) < threshold`.
static double squaredThresholdCutoff(const double threshold) {
    double c = threshold * threshold;
    if (std::sqrt(c) < threshold) {
        while (std::sqrt(std::nextafter(c, HUGE_VAL)) < threshold) {
            c = std::nextafter(c, HUGE_VAL);
        }
        c = std::nextafter(c, HUGE_VAL);
    } else {
        while (c > 0.0 && std::sqrt(std::nextafter(c, 0.0)) >= threshold) {
            c = std::nextafter(c, 0.0);
        }
    }
    return c;
}

// Block wide argmin over (value, index) with smallest-index tie-breaking.
// Also publishes the number of surviving candidates of the pass and resets the
// pass counter for the next pass.  Every thread of the block must call this.
template <int BS>
__device__ __forceinline__ void finishPass(double val, int idx, double* s_warpVal,
                                           int* s_warpIdx, int* s_cnt, int* s_len,
                                           double* s_bestVal, int* s_bestIdx) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    constexpr int NWARPS = BS / 32;

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, val, off);
        const int oi = __shfl_down_sync(0xffffffffu, idx, off);
        if (ov < val || (ov == val && oi < idx)) {
            val = ov;
            idx = oi;
        }
    }
    if (lane == 0) {
        s_warpVal[warp] = val;
        s_warpIdx[warp] = idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        double bv = s_warpVal[0];
        int bi = s_warpIdx[0];
#pragma unroll
        for (int w = 1; w < NWARPS; ++w) {
            const double ov = s_warpVal[w];
            const int oi = s_warpIdx[w];
            if (ov < bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        *s_bestVal = bv;
        *s_bestIdx = bi;
        *s_len = *s_cnt;
        *s_cnt = 0;
    }
    __syncthreads();
}

// Grow one candidate cluster per seed.  `cardOut` is indexed by point id.
// If `membersOut` is non-null (single seed / single block launch) the member
// list of the grown cluster is written to it in insertion order.
template <int BS>
__global__ void qtGrowClustersKernel(const double2* __restrict__ pts, const int N,
                                     const unsigned char* __restrict__ clustered,
                                     const int* __restrict__ seeds, const int numSeeds,
                                     const double thresholdSq,
                                     int* __restrict__ cardOut,
                                     int* __restrict__ idxScratch,
                                     double* __restrict__ distScratch,
                                     int* __restrict__ membersOut) {
    __shared__ double s_warpVal[BS / 32];
    __shared__ int s_warpIdx[BS / 32];
    __shared__ int s_cnt;
    __shared__ int s_len;
    __shared__ double s_bestVal;
    __shared__ int s_bestIdx;

    const int tid = threadIdx.x;
    const size_t base = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(N) * 2;
    int* const idxA = idxScratch + base;
    int* const idxB = idxA + N;
    double* const dstA = distScratch + base;
    double* const dstB = dstA + N;

    if (tid == 0) {
        s_cnt = 0;
    }
    __syncthreads();

    for (int si = blockIdx.x; si < numSeeds; si += gridDim.x) {
        const int seed = seeds[si];
        const double2 seedP = pts[seed];

        int* curI = idxA;
        int* nxtI = idxB;
        double* curD = dstA;
        double* nxtD = dstB;

        // First pass: all unclustered points within `threshold` of the seed are
        // the only points that can ever join this candidate cluster.
        double bv = kNoCandidate;
        int bi = INT_MAX;
        for (int i = tid; i < N; i += BS) {
            if (clustered[i] || i == seed) continue;
            const double d = devDistanceSq(pts[i], seedP);
            if (d < thresholdSq) {
                const int p = atomicAdd(&s_cnt, 1);
                nxtI[p] = i;
                nxtD[p] = d;
                if (d < bv || (d == bv && i < bi)) {
                    bv = d;
                    bi = i;
                }
            }
        }
        finishPass<BS>(bv, bi, s_warpVal, s_warpIdx, &s_cnt, &s_len, &s_bestVal, &s_bestIdx);

        int listLen = s_len;
        int chosen = s_bestIdx;
        int card = 1;
        {
            int* t = curI; curI = nxtI; nxtI = t;
            double* u = curD; curD = nxtD; nxtD = u;
        }

        while (chosen != INT_MAX) {
            if (membersOut != nullptr && tid == 0) {
                membersOut[card] = chosen;
            }
            ++card;

            const double2 newP = pts[chosen];
            double pv = kNoCandidate;
            int pi = INT_MAX;
            for (int j = tid; j < listLen; j += BS) {
                const int idx = curI[j];
                if (idx == chosen) continue;  // just became a member
                double d = curD[j];
                const double dn = devDistanceSq(pts[idx], newP);
                if (dn > d) d = dn;
                if (d < thresholdSq) {
                    const int p = atomicAdd(&s_cnt, 1);
                    nxtI[p] = idx;
                    nxtD[p] = d;
                    if (d < pv || (d == pv && idx < pi)) {
                        pv = d;
                        pi = idx;
                    }
                }
            }
            finishPass<BS>(pv, pi, s_warpVal, s_warpIdx, &s_cnt, &s_len, &s_bestVal, &s_bestIdx);

            listLen = s_len;
            chosen = s_bestIdx;
            int* t = curI; curI = nxtI; nxtI = t;
            double* u = curD; curD = nxtD; nxtD = u;
        }

        if (tid == 0) {
            cardOut[seed] = card;
            if (membersOut != nullptr) {
                membersOut[0] = seed;
            }
        }
    }
}

// Mark all still unclustered points that have at least one of the freshly
// clustered points within `threshold`; those are the seeds whose candidate
// cluster has to be recomputed.  The surviving seeds are compacted into
// `seedsOut`.
__global__ void qtCollectDirtySeedsKernel(const double2* __restrict__ pts, const int N,
                                          const unsigned char* __restrict__ clustered,
                                          const int* __restrict__ newMembers,
                                          const int newMemberCount,
                                          const double thresholdSq,
                                          int* __restrict__ seedsOut,
                                          int* __restrict__ seedCount) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
         i += blockDim.x * gridDim.x) {
        if (clustered[i]) continue;
        const double2 p = pts[i];
        bool dirty = false;
        for (int k = 0; k < newMemberCount; ++k) {
            if (devDistanceSq(p, pts[newMembers[k]]) < thresholdSq) {
                dirty = true;
                break;
            }
        }
        if (dirty) {
            seedsOut[atomicAdd(seedCount, 1)] = i;
        }
    }
}

// ---------------------------------------------------------------------------
// Host driver
// ---------------------------------------------------------------------------

namespace {

// Growing one candidate cluster is a long chain of short, dependent passes, so
// the parallelism that matters is the number of resident blocks (= seeds in
// flight).  With enough seeds, small blocks win; when only a handful of seeds
// is left (in particular the single-block re-run that extracts the member list
// of the winning cluster) wide blocks shorten the critical path instead.
int chooseBlockSize(const int numSeeds) {
    return (numSeeds >= 128) ? 64 : 256;
}

void launchGrow(const int blocks, const double2* pts, const int N,
                const unsigned char* clustered, const int* seeds, const int numSeeds,
                const double thresholdSq, int* cardOut, int* idxScratch,
                double* distScratch, int* membersOut) {
    if (chooseBlockSize(numSeeds) == 64) {
        qtGrowClustersKernel<64><<<blocks, 64>>>(pts, N, clustered, seeds, numSeeds,
                                                 thresholdSq, cardOut, idxScratch,
                                                 distScratch, membersOut);
    } else {
        qtGrowClustersKernel<256><<<blocks, 256>>>(pts, N, clustered, seeds, numSeeds,
                                                   thresholdSq, cardOut, idxScratch,
                                                   distScratch, membersOut);
    }
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace

// Main QT clustering algorithm (CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Exact squared-distance equivalent of the `dist < threshold` predicate.
    const double thresholdSq = squaredThresholdCutoff(threshold);

    // How many candidate clusters can we grow concurrently?  Every block needs
    // two candidate lists (index + running maximum distance) of up to N entries.
    const size_t perBlockBytes = static_cast<size_t>(N) * 2 * (sizeof(int) + sizeof(double));
    size_t freeBytes = 0, totalBytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeBytes, &totalBytes));
    const size_t pointBytes = static_cast<size_t>(N) * sizeof(double2);
    const size_t fixedBytes = pointBytes + static_cast<size_t>(N) * (2 * sizeof(int) + 1) + 4096;
    size_t budget = (freeBytes > fixedBytes) ? (freeBytes - fixedBytes) : 0;
    budget = budget - budget / 5;  // leave some headroom
    int maxBlocks = static_cast<int>(std::min<size_t>(budget / perBlockBytes, 262144));
    if (maxBlocks < 1) maxBlocks = 1;
    if (maxBlocks > N) maxBlocks = N;

    double2* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_card = nullptr;
    int* d_members = nullptr;
    int* d_seedCount = nullptr;
    int* d_idxScratch = nullptr;
    double* d_distScratch = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, pointBytes));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_seeds, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_members, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_seedCount, sizeof(int)));
    // Grab as much candidate-list scratch space as the device is willing to
    // give us; every block needs two lists of up to N entries.
    while (true) {
        const size_t entries = static_cast<size_t>(maxBlocks) * 2 * static_cast<size_t>(N);
        const cudaError_t e1 = cudaMalloc(&d_idxScratch, entries * sizeof(int));
        const cudaError_t e2 = (e1 == cudaSuccess)
                                   ? cudaMalloc(&d_distScratch, entries * sizeof(double))
                                   : cudaErrorMemoryAllocation;
        if (e1 == cudaSuccess && e2 == cudaSuccess) break;
        if (e1 == cudaSuccess) cudaFree(d_idxScratch);
        d_idxScratch = nullptr;
        d_distScratch = nullptr;
        cudaGetLastError();
        if (maxBlocks == 1) {
            fprintf(stderr, "CUDA error: out of memory allocating clustering scratch\n");
            exit(EXIT_FAILURE);
        }
        maxBlocks = std::max(1, maxBlocks / 2);
    }

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), pointBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_clustered, 0, static_cast<size_t>(N)));

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    std::vector<int> cardinality(N, 0);
    std::vector<int> best_members;
    best_members.resize(N);

    // First iteration: every point is a seed that needs to be evaluated.
    int numSeeds = N;
    CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(),
                          static_cast<size_t>(N) * sizeof(int), cudaMemcpyHostToDevice));

    while (!unclustered_indices.empty()) {
        if (numSeeds > 0) {
            const int blocks = std::min(maxBlocks, numSeeds);
            launchGrow(blocks, d_points, N, d_clustered, d_seeds, numSeeds,
                       thresholdSq, d_card, d_idxScratch, d_distScratch, nullptr);
            CUDA_CHECK(cudaMemcpy(cardinality.data(), d_card,
                                  static_cast<size_t>(N) * sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }

        // Pick the seed with the largest cardinality; ties go to the smallest
        // index, matching the sequential scan order.
        int max_cardinality = -1;
        int best_seed = -1;
        for (const int seed : unclustered_indices) {
            if (cardinality[seed] > max_cardinality) {
                max_cardinality = cardinality[seed];
                best_seed = seed;
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        // Re-grow the winning cluster to obtain its member list.
        CUDA_CHECK(cudaMemcpy(d_seeds, &best_seed, sizeof(int), cudaMemcpyHostToDevice));
        launchGrow(1, d_points, N, d_clustered, d_seeds, 1, thresholdSq, d_card,
                   d_idxScratch, d_distScratch, d_members);
        CUDA_CHECK(cudaMemcpy(best_members.data(), d_members,
                              static_cast<size_t>(max_cardinality) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.assign(best_members.begin(), best_members.begin() + max_cardinality);
        clusters.push_back(cluster);

        for (const int m : cluster.members) clustered[m] = 1;
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), static_cast<size_t>(N),
                              cudaMemcpyHostToDevice));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end());

        if (unclustered_indices.empty()) break;

        // Only seeds close to the removed cluster can change their result.
        CUDA_CHECK(cudaMemset(d_seedCount, 0, sizeof(int)));
        const int dirtyBlocks = std::min((N + 255) / 256, 4096);
        qtCollectDirtySeedsKernel<<<dirtyBlocks, 256>>>(d_points, N, d_clustered, d_members,
                                                        max_cardinality, thresholdSq, d_seeds,
                                                        d_seedCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(&numSeeds, d_seedCount, sizeof(int), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_seedCount));
    CUDA_CHECK(cudaFree(d_idxScratch));
    CUDA_CHECK(cudaFree(d_distScratch));

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

    // Initialize the CUDA context up front so that the measured region only
    // contains the actual clustering work.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));

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
