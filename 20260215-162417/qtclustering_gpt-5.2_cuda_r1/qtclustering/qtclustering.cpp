// QT Clustering Benchmark - CUDA Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

static inline void cudaCheck(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s\n", file, line, cudaGetErrorString(err));
        std::fflush(stderr);
        std::exit(1);
    }
}

#define CUDA_CHECK(x) cudaCheck((x), __FILE__, __LINE__)

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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold
// Returns -1 if no such point exists
int findClosestPoint(const std::vector<int>& cluster_members,
                     const std::vector<bool>& clustered,
                     const std::vector<bool>& in_cluster,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(points[candidate], points[member]);
            max_dist = std::max(max_dist, dist);
        }
        
        // If adding this point keeps diameter below threshold and is better than current best
        if (max_dist < threshold && max_dist < min_diameter) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<bool>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              std::vector<int>* cluster_members = nullptr) {
    std::vector<bool> in_cluster(point_count, false);
    std::vector<int> members;
    
    // Add seed point
    in_cluster[seed_point] = true;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(members, clustered, in_cluster, points, 
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = true;
        members.push_back(closest);
    }
    
    // Copy members if requested
    if (cluster_members) {
        *cluster_members = members;
    }
    
    return static_cast<int>(members.size());
}

namespace {

static __device__ __forceinline__ double dist2_device(const Point* __restrict__ pts, int i, int j) {
    const double dx = pts[i].x - pts[j].x;
    const double dy = pts[i].y - pts[j].y;
    return dx * dx + dy * dy;
}

static __device__ __forceinline__ bool better_pair(double v_a, int i_a, double v_b, int i_b) {
    // Return true if (v_a, i_a) is better (smaller v, tie smaller i). i<0 means invalid.
    if (i_a < 0) return false;
    if (i_b < 0) return true;
    if (v_a < v_b) return true;
    if (v_a > v_b) return false;
    return i_a < i_b;
}

__global__ void qt_cardinalities_kernel(const Point* __restrict__ pts,
                                       const std::uint8_t* __restrict__ clustered,
                                       int N,
                                       double thresh2,
                                       const int* __restrict__ seeds,
                                       int M,
                                       int* __restrict__ out_card) {
    const int sidx = static_cast<int>(blockIdx.x);
    if (sidx >= M) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int seed = seeds[sidx];

    extern __shared__ unsigned char smem[];
    std::uint8_t* in_cluster = reinterpret_cast<std::uint8_t*>(smem);
    const size_t in_bytes = static_cast<size_t>((N + 7) & ~7);
    double* cand = reinterpret_cast<double*>(smem + in_bytes);
    double* s_bestVal = cand + N;
    int* s_bestIdx = reinterpret_cast<int*>(s_bestVal + blockDim.x);

    __shared__ int s_clusterSize;
    __shared__ int s_chosen;
    __shared__ int s_last;

    const double INF = 1.0e300;

    for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
        in_cluster[j] = 0;
    }
    __syncthreads();

    if (clustered[seed]) {
        if (tid == 0) out_card[sidx] = -1;
        return;
    }

    if (tid == 0) {
        in_cluster[seed] = 1;
        s_clusterSize = 1;
        s_last = seed;
        s_chosen = -1;
    }
    __syncthreads();

    for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
        if (j == seed || clustered[j]) {
            cand[j] = INF;
        } else {
            cand[j] = dist2_device(pts, seed, j);
        }
    }
    __syncthreads();

    while (true) {
        // Find best candidate: min cand[j] subject to cand[j] < thresh2 and not clustered/in_cluster.
        double bestV = INF;
        int bestI = -1;
        for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
            if (clustered[j] || in_cluster[j]) continue;
            const double v = cand[j];
            if (v < thresh2) {
                if (bestI < 0 || v < bestV || (v == bestV && j < bestI)) {
                    bestV = v;
                    bestI = j;
                }
            }
        }

        s_bestVal[tid] = bestV;
        s_bestIdx[tid] = bestI;
        __syncthreads();

        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (static_cast<unsigned int>(tid) < stride) {
                const double v_a = s_bestVal[tid];
                const int i_a = s_bestIdx[tid];
                const double v_b = s_bestVal[tid + stride];
                const int i_b = s_bestIdx[tid + stride];

                if (!better_pair(v_a, i_a, v_b, i_b)) {
                    s_bestVal[tid] = v_b;
                    s_bestIdx[tid] = i_b;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_chosen = s_bestIdx[0];
            if (s_chosen >= 0) {
                in_cluster[s_chosen] = 1;
                cand[s_chosen] = INF;
                s_last = s_chosen;
                s_clusterSize++;
            }
        }
        __syncthreads();

        if (s_chosen < 0 || s_clusterSize >= N) break;

        // Update cand[j] = max(cand[j], dist2(j, last_added)).
        const int last = s_last;
        for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
            if (clustered[j] || in_cluster[j]) {
                cand[j] = INF;
            } else {
                const double d2 = dist2_device(pts, last, j);
                const double prev = cand[j];
                cand[j] = (d2 > prev) ? d2 : prev;
            }
        }
        __syncthreads();
    }

    if (tid == 0) out_card[sidx] = s_clusterSize;
}

__global__ void qt_members_kernel(const Point* __restrict__ pts,
                                 const std::uint8_t* __restrict__ clustered,
                                 int N,
                                 double thresh2,
                                 int seed,
                                 int* __restrict__ out_members,
                                 int* __restrict__ out_size) {
    const int tid = static_cast<int>(threadIdx.x);

    extern __shared__ unsigned char smem[];
    std::uint8_t* in_cluster = reinterpret_cast<std::uint8_t*>(smem);
    const size_t in_bytes = static_cast<size_t>((N + 7) & ~7);
    double* cand = reinterpret_cast<double*>(smem + in_bytes);
    double* s_bestVal = cand + N;
    int* s_bestIdx = reinterpret_cast<int*>(s_bestVal + blockDim.x);

    __shared__ int s_clusterSize;
    __shared__ int s_chosen;
    __shared__ int s_last;

    const double INF = 1.0e300;

    for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
        in_cluster[j] = 0;
    }
    __syncthreads();

    if (tid == 0) {
        in_cluster[seed] = 1;
        out_members[0] = seed;
        s_clusterSize = 1;
        s_last = seed;
        s_chosen = -1;
    }
    __syncthreads();

    for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
        if (j == seed || clustered[j]) {
            cand[j] = INF;
        } else {
            cand[j] = dist2_device(pts, seed, j);
        }
    }
    __syncthreads();

    while (true) {
        double bestV = INF;
        int bestI = -1;
        for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
            if (clustered[j] || in_cluster[j]) continue;
            const double v = cand[j];
            if (v < thresh2) {
                if (bestI < 0 || v < bestV || (v == bestV && j < bestI)) {
                    bestV = v;
                    bestI = j;
                }
            }
        }

        s_bestVal[tid] = bestV;
        s_bestIdx[tid] = bestI;
        __syncthreads();

        for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (static_cast<unsigned int>(tid) < stride) {
                const double v_a = s_bestVal[tid];
                const int i_a = s_bestIdx[tid];
                const double v_b = s_bestVal[tid + stride];
                const int i_b = s_bestIdx[tid + stride];

                if (!better_pair(v_a, i_a, v_b, i_b)) {
                    s_bestVal[tid] = v_b;
                    s_bestIdx[tid] = i_b;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            s_chosen = s_bestIdx[0];
            if (s_chosen >= 0) {
                in_cluster[s_chosen] = 1;
                cand[s_chosen] = INF;
                s_last = s_chosen;
                out_members[s_clusterSize] = s_chosen;
                s_clusterSize++;
            }
        }
        __syncthreads();

        if (s_chosen < 0 || s_clusterSize >= N) break;

        const int last = s_last;
        for (int j = tid; j < N; j += static_cast<int>(blockDim.x)) {
            if (clustered[j] || in_cluster[j]) {
                cand[j] = INF;
            } else {
                const double d2 = dist2_device(pts, last, j);
                const double prev = cand[j];
                cand[j] = (d2 > prev) ? d2 : prev;
            }
        }
        __syncthreads();
    }

    if (tid == 0) *out_size = s_clusterSize;
}

static size_t qt_shared_bytes(int N, int blockSize) {
    const size_t in_bytes = static_cast<size_t>((N + 7) & ~7);
    const size_t cand_bytes = sizeof(double) * static_cast<size_t>(N);
    const size_t reduce_bytes = (sizeof(double) + sizeof(int)) * static_cast<size_t>(blockSize);
    return in_bytes + cand_bytes + reduce_bytes;
}

} // namespace

// Main QT clustering algorithm (CUDA implementation)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                 const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<Cluster> clusters;
    if (N == 0) return clusters;

    CUDA_CHECK(cudaSetDevice(0));

    Point* d_points = nullptr;
    std::uint8_t* d_clustered = nullptr;
    int* d_seeds = nullptr;
    int* d_cards = nullptr;
    int* d_members = nullptr;
    int* d_memberCount = nullptr;

    CUDA_CHECK(cudaMalloc(&d_points, sizeof(Point) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_clustered, sizeof(std::uint8_t) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_seeds, sizeof(int) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_cards, sizeof(int) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_members, sizeof(int) * static_cast<size_t>(N)));
    CUDA_CHECK(cudaMalloc(&d_memberCount, sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points, points.data(), sizeof(Point) * static_cast<size_t>(N), cudaMemcpyHostToDevice));

    std::vector<std::uint8_t> clustered(static_cast<size_t>(N), 0);
    CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), sizeof(std::uint8_t) * static_cast<size_t>(N), cudaMemcpyHostToDevice));

    std::vector<int> unclustered_indices;
    unclustered_indices.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) unclustered_indices.push_back(i);

    const double thresh2 = threshold * threshold;
    constexpr int BLOCK = 256;
    const size_t sharedBytes = qt_shared_bytes(N, BLOCK);

    while (!unclustered_indices.empty()) {
        const int M = static_cast<int>(unclustered_indices.size());

        CUDA_CHECK(cudaMemcpy(d_seeds, unclustered_indices.data(), sizeof(int) * static_cast<size_t>(M), cudaMemcpyHostToDevice));

        qt_cardinalities_kernel<<<M, BLOCK, sharedBytes>>>(d_points, d_clustered, N, thresh2, d_seeds, M, d_cards);
        CUDA_CHECK(cudaGetLastError());

        std::vector<int> cards(static_cast<size_t>(M));
        CUDA_CHECK(cudaMemcpy(cards.data(), d_cards, sizeof(int) * static_cast<size_t>(M), cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_seed = -1;
        for (int i = 0; i < M; ++i) {
            const int seed = unclustered_indices[static_cast<size_t>(i)];
            const int card = cards[static_cast<size_t>(i)];
            if (card > max_cardinality) {
                max_cardinality = card;
                best_seed = seed;
            }
        }

        if (best_seed < 0 || max_cardinality <= 0) break;

        qt_members_kernel<<<1, BLOCK, sharedBytes>>>(d_points, d_clustered, N, thresh2, best_seed, d_members, d_memberCount);
        CUDA_CHECK(cudaGetLastError());

        int memberCount = 0;
        CUDA_CHECK(cudaMemcpy(&memberCount, d_memberCount, sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> best_members(static_cast<size_t>(memberCount));
        CUDA_CHECK(cudaMemcpy(best_members.data(), d_members, sizeof(int) * static_cast<size_t>(memberCount), cudaMemcpyDeviceToHost));

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = std::move(best_members);
        clusters.push_back(cluster);

        for (int idx : clusters.back().members) clustered[static_cast<size_t>(idx)] = 1;
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), sizeof(std::uint8_t) * static_cast<size_t>(N), cudaMemcpyHostToDevice));

        unclustered_indices.erase(
            std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                           [&clustered](int idx) { return clustered[static_cast<size_t>(idx)] != 0; }),
            unclustered_indices.end());
    }

    CUDA_CHECK(cudaFree(d_memberCount));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_cards));
    CUDA_CHECK(cudaFree(d_seeds));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_points));

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
