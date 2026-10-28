// QT Clustering Benchmark - CUDA GPU Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// GPU parallelization:
//   - Every unclustered seed grows its candidate cluster in its own CUDA
//     block. All seeds of one round are processed concurrently.
//   - Within a block, the threads cooperatively scan the candidates and
//     reduce to the candidate with the smallest max-distance-to-cluster
//     (ties broken by lowest index, matching the sequential scan order).
//   - The max-distance of every candidate to the cluster is maintained
//     incrementally (max over members is order independent and exact), so
//     each growth step costs O(N) instead of O(N * |cluster|).
//   - The seed with the largest cardinality is selected on the host
//     (first seed wins ties), and its member list is regenerated on the GPU.

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

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err_));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

// Threads per block used by the cluster growth kernel (must be a multiple of 32, <= 1024)
static constexpr int kBlockSize = 256;
// Per-block max-distance scratch is kept in shared memory when it fits
static constexpr size_t kMaxSharedScratchBytes = 47 * 1024;
// Budget for global-memory scratch when the point count does not fit in shared memory
static constexpr size_t kGlobalScratchBudgetBytes = size_t(1) << 30;

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

// Calculate Euclidean distance between two points (host and device)
__host__ __device__ inline double distance(const Point& p1, const Point& p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Lexicographic "less than" on (value, index): smaller value wins, ties go to lower index
__device__ __forceinline__ bool lexLess(double av, int ai, double bv, int bi) {
    return (av < bv) || (av == bv && ai < bi);
}

// Grows the candidate cluster for one seed per block.
//
// Equivalent to the sequential generateCandidateCluster/findClosestPoint pair:
//   - a candidate is eligible when it is not already clustered, not already in
//     this cluster, and its max distance to the cluster members is < threshold;
//   - among eligible candidates the one with the smallest max distance is added
//     (lowest index on ties); growth stops when no candidate is eligible.
// Clustered and in-cluster candidates get max-distance = +inf, which can never
// be below a threshold, so they are excluded without a separate flag array.
//
// seeds[b]        : seed handled by block b
// cardinalities[b]: receives the size of the grown cluster for seed b
// out_members     : if non-null, receives the member list (single-seed launch only)
// global_scratch  : per-block max-distance storage (used when kSharedScratch == false)
template <bool kSharedScratch>
__global__ void growClusterKernel(const Point* __restrict__ points,
                                  const unsigned char* __restrict__ clustered,
                                  const int* __restrict__ seeds,
                                  int* __restrict__ cardinalities,
                                  int* __restrict__ out_members,
                                  double* __restrict__ global_scratch,
                                  const int N,
                                  const double threshold) {
    extern __shared__ double s_maxd[];
    __shared__ double s_red_v[kBlockSize / 32];
    __shared__ int s_red_i[kBlockSize / 32];
    __shared__ double s_best_v;
    __shared__ int s_best_i;

    double* maxd = kSharedScratch ? s_maxd
                                  : global_scratch + static_cast<size_t>(blockIdx.x) * N;

    const int tid = threadIdx.x;
    const int seed = seeds[blockIdx.x];
    const double INF = HUGE_VAL;

    // Initial max-distance: 0 for free candidates, +inf for already clustered ones
    for (int i = tid; i < N; i += kBlockSize) {
        maxd[i] = clustered[i] ? INF : 0.0;
    }
    __syncthreads();

    int count = 0;
    int best = seed;
    while (true) {
        // Add 'best' to the cluster and update every candidate's max distance
        const Point pb = points[best];
        for (int i = tid; i < N; i += kBlockSize) {
            if (i == best) {
                maxd[i] = INF;
            } else {
                const double d = distance(points[i], pb);
                maxd[i] = fmax(maxd[i], d);
            }
        }
        if (tid == 0 && out_members) {
            out_members[count] = best;
        }
        ++count;

        // Find the eligible candidate with the smallest (max distance, index)
        double lv = INF;
        int li = INT_MAX;
        for (int i = tid; i < N; i += kBlockSize) {
            const double v = maxd[i];
            if (lexLess(v, i, lv, li)) {
                lv = v;
                li = i;
            }
        }
        for (int off = 16; off > 0; off >>= 1) {
            const double ov = __shfl_down_sync(0xffffffffu, lv, off);
            const int oi = __shfl_down_sync(0xffffffffu, li, off);
            if (lexLess(ov, oi, lv, li)) {
                lv = ov;
                li = oi;
            }
        }
        const int lane = tid & 31;
        const int wid = tid >> 5;
        if (lane == 0) {
            s_red_v[wid] = lv;
            s_red_i[wid] = li;
        }
        __syncthreads();
        if (wid == 0) {
            constexpr int kNumWarps = kBlockSize / 32;
            lv = (lane < kNumWarps) ? s_red_v[lane] : INF;
            li = (lane < kNumWarps) ? s_red_i[lane] : INT_MAX;
            for (int off = 16; off > 0; off >>= 1) {
                const double ov = __shfl_down_sync(0xffffffffu, lv, off);
                const int oi = __shfl_down_sync(0xffffffffu, li, off);
                if (lexLess(ov, oi, lv, li)) {
                    lv = ov;
                    li = oi;
                }
            }
            if (lane == 0) {
                s_best_v = lv;
                s_best_i = li;
            }
        }
        __syncthreads();

        // Uniform across the block: stop when no candidate keeps diameter < threshold
        const double bv = s_best_v;
        const int bi = s_best_i;
        if (!(bv < threshold)) {
            break;
        }
        best = bi;
    }

    if (tid == 0) {
        cardinalities[blockIdx.x] = count;
    }
}

// Main QT clustering algorithm (GPU)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) {
        return clusters;
    }

    const size_t scratch_bytes_per_seed = static_cast<size_t>(N) * sizeof(double);
    const bool useShared = scratch_bytes_per_seed <= kMaxSharedScratchBytes;
    int maxChunk = N;
    if (!useShared) {
        maxChunk = static_cast<int>(
            std::max<size_t>(1, std::min<size_t>(N, kGlobalScratchBudgetBytes / scratch_bytes_per_seed)));
    }

    // Device buffers
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    int* d_members = nullptr;
    double* d_scratch = nullptr;
    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * N));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(unsigned char) * N));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_cards, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * N));
    if (!useShared) {
        CUDA_CHECK(cudaMalloc(&d_scratch, scratch_bytes_per_seed * maxChunk));
    }
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * N, cudaMemcpyHostToDevice));

    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<int> cardinalities(N);
    std::vector<int> best_cluster_members(N);

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int M = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(), sizeof(int) * M,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), sizeof(unsigned char) * N,
                              cudaMemcpyHostToDevice));

        // Grow candidate clusters from every unclustered seed
        for (int off = 0; off < M; off += maxChunk) {
            const int cnt = std::min(maxChunk, M - off);
            if (useShared) {
                growClusterKernel<true><<<cnt, kBlockSize, scratch_bytes_per_seed>>>(
                    d_points, d_clustered, d_seeds + off, d_cards + off, nullptr, nullptr,
                    N, threshold);
            } else {
                growClusterKernel<false><<<cnt, kBlockSize, 0>>>(
                    d_points, d_clustered, d_seeds + off, d_cards + off, nullptr, d_scratch,
                    N, threshold);
            }
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpy(cardinalities.data(), d_cards, sizeof(int) * M,
                              cudaMemcpyDeviceToHost));

        // Select the seed with the largest cluster (first seed wins ties)
        int max_cardinality = -1;
        int best_pos = -1;
        for (int i = 0; i < M; ++i) {
            if (cardinalities[i] > max_cardinality) {
                max_cardinality = cardinalities[i];
                best_pos = i;
            }
        }

        if (best_pos < 0 || max_cardinality <= 0) {
            // No more clusters can be formed
            break;
        }

        // Regenerate the member list of the winning seed
        const int best_seed = unclustered_indices[best_pos];
        if (useShared) {
            growClusterKernel<true><<<1, kBlockSize, scratch_bytes_per_seed>>>(
                d_points, d_clustered, d_seeds + best_pos, d_cards + best_pos, d_members,
                nullptr, N, threshold);
        } else {
            growClusterKernel<false><<<1, kBlockSize, 0>>>(
                d_points, d_clustered, d_seeds + best_pos, d_cards + best_pos, d_members,
                d_scratch, N, threshold);
        }
        CUDA_CHECK(cudaGetLastError());
        best_cluster_members.resize(max_cardinality);
        CUDA_CHECK(cudaMemcpy(best_cluster_members.data(), d_members,
                              sizeof(int) * max_cardinality, cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_cluster_members;
        clusters.push_back(cluster);

        // Mark all members as clustered
        for (size_t i = 0; i < best_cluster_members.size(); ++i) {
            clustered[best_cluster_members[i]] = 1;
        }

        // Remove clustered points from unclustered list
        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[idx] != 0; }),
            unclustered_indices.end()
        );
    }

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_cards));
    CUDA_CHECK(cudaFree(d_members));
    if (d_scratch) {
        CUDA_CHECK(cudaFree(d_scratch));
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
    
    // Initialize the CUDA context outside the timed region (one-time setup cost)
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
