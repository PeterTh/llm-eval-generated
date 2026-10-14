// QT Clustering Benchmark - CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// The clustering runs on the GPU: every candidate seed grows its cluster in
// its own warp, and the best candidate cluster is chosen with the
// same tie-breaking rules as the sequential algorithm, so the results are
// identical to the sequential version.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

static constexpr int kWarpsPerBlock = 4;                  // One seed per warp in the kernels
static constexpr int kFirstBatch = 64;                    // Seeds in the first GPU batch
static constexpr size_t kScratchBudget = size_t(1) << 30; // Bytes for neighbor-list scratch

// Per-neighbor state inside a seed's growth
static constexpr int kCandidateState = -1;  // Still available to this cluster
static constexpr int kNoPick = INT_MAX;     // Reduction sentinel: no candidate

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err_));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

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

// Same arithmetic as distance(), with explicit round-to-nearest operations so
// that no fused multiply-add is introduced. Host code is built with
// -ffp-contract=off, so both sides produce bit-identical results.
__device__ __forceinline__ double deviceDistance(const Point& p1, const Point& p2) {
    const double dx = __dsub_rn(p1.x, p2.x);
    const double dy = __dsub_rn(p1.y, p2.y);
    return __dsqrt_rn(__dadd_rn(__dmul_rn(dx, dx), __dmul_rn(dy, dy)));
}

// A point can ever join a seed's cluster only if it is unclustered and within
// the threshold of the seed: a cluster's diameter is at least the distance from
// the seed to any of its members.
__device__ __forceinline__ bool isNeighbor(const Point* points, const unsigned char* clustered,
                                           const int c, const int seed, const Point& seed_point,
                                           const double threshold) {
    return c != seed && !clustered[c] && deviceDistance(points[c], seed_point) < threshold;
}

// Keep the lexicographically smaller (value, index) pair
__device__ __forceinline__ void keepSmaller(double& v, int& i, double ov, int oi) {
    if (ov < v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

// Upper bound on each seed's cluster size: the seed plus its neighbors.
// Its neighbor count is exact, so the neighbor list is sized from it.
// One warp per seed, scanning the points 32 at a time.
__global__ void boundKernel(const Point* points, const unsigned char* clustered, const int N,
                            const double threshold, const int* seeds, const int num_seeds,
                            int* bound_out) {
    constexpr unsigned kFullMask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int slot = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (slot >= num_seeds) return;

    const int seed = seeds[slot];
    const Point seed_point = points[seed];
    int count = 0;
    for (int base = 0; base < N; base += 32) {
        const int c = base + lane;
        const bool hit = c < N && isNeighbor(points, clustered, c, seed, seed_point, threshold);
        count += __popc(__ballot_sync(kFullMask, hit));
    }
    if (lane == 0) bound_out[slot] = 1 + count;
}

// Greedy growth of one candidate cluster per warp, starting from seeds[slot].
// The warp first collects the seed's neighbors in ascending point order. Each step
// then adds the neighbor whose maximum distance to the members is smallest (lowest
// point index on ties) among those below the threshold. The per-neighbor maximum is
// maintained incrementally: max over members is exact, so this equals recomputing it
// from scratch. state[] records each member's insertion position (the seed is 0),
// which gives the cluster's member list in the same order as the sequential code.
// Each warp works independently, so only warp-level synchronization is needed.
__global__ void growClusterKernel(const Point* points, const unsigned char* clustered, const int N,
                                  const double threshold, const int* seeds, const int num_seeds,
                                  const int* offsets, int* nbr_all, double* dmax_all,
                                  int* state_all, int* active_all, int* card_out) {
    constexpr unsigned kFullMask = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int slot = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (slot >= num_seeds) return;

    const int seed = seeds[slot];
    const Point seed_point = points[seed];
    int* nbr = nbr_all + offsets[slot];
    double* dmax = dmax_all + offsets[slot];
    int* state = state_all + offsets[slot];
    int* active = active_all + offsets[slot];

    // Collect neighbors in ascending point order: a ballot gives each hit its
    // rank within the chunk of 32 points
    int len = 0;
    for (int base = 0; base < N; base += 32) {
        const int c = base + lane;
        const bool hit = c < N && isNeighbor(points, clustered, c, seed, seed_point, threshold);
        const unsigned mask = __ballot_sync(kFullMask, hit);
        if (hit) {
            const int pos = len + __popc(mask & ((1u << lane) - 1u));
            nbr[pos] = c;
            dmax[pos] = 0.0;
            state[pos] = kCandidateState;
            active[pos] = pos;
        }
        len += __popc(mask);
    }
    __syncwarp(kFullMask);

    // active[] lists the neighbors that can still be chosen, in ascending order.
    // A neighbor leaves it once it joins the cluster or its maximum distance reaches
    // the threshold; that is permanent because maximum distances only grow.
    int n_active = len;
    int count = 1;
    int last = seed;
    while (true) {
        const Point last_point = points[last];
        double best_v = INFINITY;
        int best_j = kNoPick;

        // Neighbor index j orders the same way as point index, so the
        // lowest-j tie-break matches the lowest-point-index rule
        for (int i = lane; i < n_active; i += 32) {
            const int j = active[i];
            const double d = deviceDistance(points[nbr[j]], last_point);
            const double m = dmax[j] < d ? d : dmax[j];
            dmax[j] = m;
            if (m < threshold && m < best_v) {
                best_v = m;
                best_j = j;
            }
        }

        for (int off = 16; off > 0; off >>= 1) {
            keepSmaller(best_v, best_j, __shfl_down_sync(kFullMask, best_v, off),
                        __shfl_down_sync(kFullMask, best_j, off));
        }
        const int pick = __shfl_sync(kFullMask, best_j, 0);
        if (pick == kNoPick) break;

        if (lane == 0) state[pick] = count;
        ++count;
        last = nbr[pick];
        __syncwarp(kFullMask);

        // Stable in-place compaction. Each chunk is read before it is written, and
        // writes never pass the chunk being read, so no unread entry is overwritten.
        int write = 0;
        for (int base = 0; base < n_active; base += 32) {
            const int i = base + lane;
            const bool in_range = i < n_active;
            const int j = in_range ? active[i] : 0;
            const bool keep = in_range && j != pick && dmax[j] < threshold;
            const unsigned mask = __ballot_sync(kFullMask, keep);
            if (keep) active[write + __popc(mask & ((1u << lane) - 1u))] = j;
            write += __popc(mask);
        }
        n_active = write;
        __syncwarp(kFullMask);
    }

    if (lane == 0) card_out[slot] = count;
}

// Main QT clustering algorithm
//
// Each round selects the seed whose greedy cluster is largest (lowest seed index
// on ties), exactly as the sequential algorithm does. Seeds are evaluated in
// batches in decreasing order of their upper bound, and a seed is skipped once its
// bound is below the best cardinality found so far. Such a seed cannot be the
// answer, so the selected cluster does not change.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    // Scratch holds a neighbor index, a maximum distance, a state and an active-list
    // slot for each neighbor of every seed in a batch
    const size_t bytes_per_entry = sizeof(int) + sizeof(double) + sizeof(int) + sizeof(int);
    const size_t entry_cap = std::max<size_t>(
        N, std::min<size_t>(static_cast<size_t>(N) * N, kScratchBudget / bytes_per_entry));

    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_offsets = nullptr;
    int* d_bound = nullptr;
    int* d_card = nullptr;
    int* d_nbr = nullptr;
    double* d_dmax = nullptr;
    int* d_state = nullptr;
    int* d_active = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, N * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_seeds, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_offsets, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bound, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_card, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_nbr, entry_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_dmax, entry_cap * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_state, entry_cap * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_active, entry_cap * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), N * sizeof(Point), cudaMemcpyHostToDevice));

    std::vector<unsigned char> clustered_h(N, 0);
    std::vector<int> unclustered(N);
    std::iota(unclustered.begin(), unclustered.end(), 0);

    std::vector<int> bound_h(N);
    std::vector<int> visit(N);
    std::vector<int> batch_seed(N);
    std::vector<int> batch_offset(N);
    std::vector<int> batch_bound(N);
    std::vector<int> card_h(N);
    std::vector<int> best_nbr(N);
    std::vector<int> best_state(N);

    // Main clustering loop
    while (!unclustered.empty()) {
        const int U = static_cast<int>(unclustered.size());

        CUDA_CHECK(cudaMemcpy(d_clustered, clustered_h.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered.data(), U * sizeof(int), cudaMemcpyHostToDevice));
        boundKernel<<<(U + kWarpsPerBlock - 1) / kWarpsPerBlock, kWarpsPerBlock * 32>>>(
            d_points, d_clustered, N, threshold, d_seeds, U, d_bound);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(bound_h.data(), d_bound, U * sizeof(int), cudaMemcpyDeviceToHost));

        // Visit seeds by decreasing bound; ties keep ascending seed order
        visit.resize(U);
        std::iota(visit.begin(), visit.end(), 0);
        std::sort(visit.begin(), visit.end(), [&bound_h](int a, int b) {
            return bound_h[a] != bound_h[b] ? bound_h[a] > bound_h[b] : a < b;
        });

        int max_cardinality = -1;
        int best_seed = -1;
        int best_len = 0;
        size_t pos = 0;
        size_t batch_limit = std::min<size_t>(kFirstBatch, U);
        while (pos < static_cast<size_t>(U)) {
            // Sorted by bound, so once one seed cannot beat the best, none after it can
            if (bound_h[visit[pos]] < max_cardinality) break;

            // Gather a batch whose neighbor lists fit in the scratch buffers
            size_t nb = 0;
            size_t entries = 0;
            while (nb < batch_limit && pos + nb < static_cast<size_t>(U) &&
                   bound_h[visit[pos + nb]] >= max_cardinality) {
                const size_t len = static_cast<size_t>(bound_h[visit[pos + nb]]) - 1;
                if (nb > 0 && entries + len > entry_cap) break;
                batch_seed[nb] = unclustered[visit[pos + nb]];
                batch_offset[nb] = static_cast<int>(entries);
                batch_bound[nb] = bound_h[visit[pos + nb]];
                entries += len;
                ++nb;
            }

            CUDA_CHECK(cudaMemcpy(d_seeds, batch_seed.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_offsets, batch_offset.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
            growClusterKernel<<<static_cast<unsigned>((nb + kWarpsPerBlock - 1) / kWarpsPerBlock),
                                kWarpsPerBlock * 32>>>(
                d_points, d_clustered, N, threshold, d_seeds, static_cast<int>(nb), d_offsets,
                d_nbr, d_dmax, d_state, d_active, d_card);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(card_h.data(), d_card, nb * sizeof(int), cudaMemcpyDeviceToHost));

            // Best seed in this batch: largest cardinality, then lowest seed index
            size_t k_best = 0;
            for (size_t k = 1; k < nb; ++k) {
                if (card_h[k] > card_h[k_best] ||
                    (card_h[k] == card_h[k_best] && batch_seed[k] < batch_seed[k_best])) {
                    k_best = k;
                }
            }
            if (card_h[k_best] > max_cardinality ||
                (card_h[k_best] == max_cardinality && batch_seed[k_best] < best_seed)) {
                max_cardinality = card_h[k_best];
                best_seed = batch_seed[k_best];
                best_len = batch_bound[k_best] - 1;
                const size_t off = static_cast<size_t>(batch_offset[k_best]);
                CUDA_CHECK(cudaMemcpy(best_nbr.data(), d_nbr + off,
                                      best_len * sizeof(int), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(best_state.data(), d_state + off,
                                      best_len * sizeof(int), cudaMemcpyDeviceToHost));
            }

            pos += nb;
            batch_limit = std::min<size_t>(batch_limit * 2, U);
        }

        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members.assign(max_cardinality, -1);
            cluster.members[0] = best_seed;
            for (int j = 0; j < best_len; ++j) {
                if (best_state[j] > 0) cluster.members[best_state[j]] = best_nbr[j];
            }
            clusters.push_back(cluster);

            // Mark all members as clustered
            for (int m : cluster.members) {
                clustered_h[m] = 1;
            }

            // Remove clustered points from unclustered list
            unclustered.erase(
                std::remove_if(unclustered.begin(), unclustered.end(),
                              [&clustered_h](int idx) { return clustered_h[idx] != 0; }),
                unclustered.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_offsets));
    CUDA_CHECK(cudaFree(d_bound));
    CUDA_CHECK(cudaFree(d_card));
    CUDA_CHECK(cudaFree(d_nbr));
    CUDA_CHECK(cudaFree(d_dmax));
    CUDA_CHECK(cudaFree(d_state));
    CUDA_CHECK(cudaFree(d_active));

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

    // Create the CUDA context before timing so one-time driver setup is not counted
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
