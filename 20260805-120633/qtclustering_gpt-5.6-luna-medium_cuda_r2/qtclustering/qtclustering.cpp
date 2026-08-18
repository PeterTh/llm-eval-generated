// QT Clustering Benchmark - CUDA implementation
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
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

namespace {

constexpr int CUDA_BLOCK_SIZE = 256;

inline void cudaCheck(const cudaError_t error, const char* const operation) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                     cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

#define CUDA_CHECK(call) cudaCheck((call), #call)

__device__ __forceinline__ double deviceDistance(const Point a, const Point b) {
    const double dx = a.x - b.x;
    const double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

// Initialize one independent candidate cluster per possible seed.  The
// distance matrix stores the current maximum distance to that seed's cluster.
__global__ void initializeCandidateClusters(const Point* points,
                                             const unsigned char* clustered,
                                             const int point_count,
                                             int* members,
                                             int* member_counts,
                                             unsigned char* in_cluster,
                                             double* max_distances) {
    const int flat = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = point_count * point_count;
    if (flat >= total) return;

    const int seed = flat / point_count;
    const int candidate = flat - seed * point_count;
    const bool active = clustered[seed] == 0;
    in_cluster[flat] = static_cast<unsigned char>(active && candidate == seed);
    max_distances[flat] = deviceDistance(points[seed], points[candidate]);

    if (candidate == 0) {
        member_counts[seed] = active ? 1 : 0;
        members[seed * point_count] = seed;
    }
}

// One block handles one seed.  Threads inspect candidates in parallel, then
// cooperatively update the selected seed's distance frontier for the next
// expansion.  The index tie-break makes the reduction identical to the
// original ascending candidate loop.
__global__ void expandCandidateClusters(const Point* points,
                                         const unsigned char* clustered,
                                         const int point_count,
                                         const double threshold,
                                         int* members,
                                         int* member_counts,
                                         unsigned char* in_cluster,
                                         double* max_distances,
                                         int* selected_points,
                                         int* any_added) {
    const int seed = blockIdx.x;
    if (seed >= point_count) return;

    __shared__ double best_dist[CUDA_BLOCK_SIZE];
    __shared__ int best_index[CUDA_BLOCK_SIZE];
    __shared__ int selected;

    const int lane = threadIdx.x;
    const int count = member_counts[seed];
    const int offset = seed * point_count;
    double local_dist = 1.7976931348623157e308;
    int local_index = point_count;

    for (int candidate = lane; candidate < point_count; candidate += blockDim.x) {
        if (clustered[candidate] == 0 && in_cluster[offset + candidate] == 0) {
            const double candidate_dist = max_distances[offset + candidate];
            if (candidate_dist < threshold &&
                (candidate_dist < local_dist ||
                 (candidate_dist == local_dist && candidate < local_index))) {
                local_dist = candidate_dist;
                local_index = candidate;
            }
        }
    }

    best_dist[lane] = local_dist;
    best_index[lane] = local_index;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (lane < stride) {
            const double other_dist = best_dist[lane + stride];
            const int other_index = best_index[lane + stride];
            if (other_dist < best_dist[lane] ||
                (other_dist == best_dist[lane] && other_index < best_index[lane])) {
                best_dist[lane] = other_dist;
                best_index[lane] = other_index;
            }
        }
        __syncthreads();
    }

    if (lane == 0) {
        selected = best_index[0] < point_count ? best_index[0] : -1;
        selected_points[seed] = selected;
        if (selected >= 0) {
            members[offset + count] = selected;
            member_counts[seed] = count + 1;
            in_cluster[offset + selected] = 1;
            atomicExch(any_added, 1);
        }
    }
    __syncthreads();

    if (selected >= 0) {
        const Point selected_point = points[selected];
        for (int candidate = lane; candidate < point_count; candidate += blockDim.x) {
            const double new_dist = deviceDistance(points[candidate], selected_point);
            if (new_dist > max_distances[offset + candidate]) {
                max_distances[offset + candidate] = new_dist;
            }
        }
    }
}

} // namespace

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

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    // These buffers are reused for every outer QT round.  The O(N^2) state is
    // substantially smaller than recomputing every candidate's diameter at
    // each expansion and keeps all fine-grained work on the GPU.
    Point* d_points = nullptr;
    unsigned char* d_clustered = nullptr;
    int* d_members = nullptr;
    int* d_member_counts = nullptr;
    unsigned char* d_in_cluster = nullptr;
    double* d_max_distances = nullptr;
    int* d_selected_points = nullptr;
    int* d_any_added = nullptr;
    const size_t matrix_elements = static_cast<size_t>(N) * N;

    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_clustered, static_cast<size_t>(N) * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_members, matrix_elements * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_member_counts, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, matrix_elements * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_max_distances, matrix_elements * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_selected_points, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_any_added, sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(N) * sizeof(Point),
                          cudaMemcpyHostToDevice));

    int unclustered_count = N;
    while (unclustered_count > 0) {
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(),
                              static_cast<size_t>(N) * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));

        const int init_blocks = static_cast<int>((matrix_elements + CUDA_BLOCK_SIZE - 1) /
                                                   CUDA_BLOCK_SIZE);
        initializeCandidateClusters<<<init_blocks, CUDA_BLOCK_SIZE>>>(
            d_points, d_clustered, N, d_members, d_member_counts,
            d_in_cluster, d_max_distances);
        CUDA_CHECK(cudaGetLastError());

        // Every expansion adds at most one point to each candidate cluster.
        // The host only synchronizes once per expansion to determine whether
        // all candidate clusters have reached their quality threshold.
        int any_added = 1;
        while (any_added != 0) {
            any_added = 0;
            CUDA_CHECK(cudaMemcpy(d_any_added, &any_added, sizeof(int),
                                  cudaMemcpyHostToDevice));
            expandCandidateClusters<<<N, CUDA_BLOCK_SIZE>>>(
                d_points, d_clustered, N, threshold, d_members, d_member_counts,
                d_in_cluster, d_max_distances, d_selected_points, d_any_added);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&any_added, d_any_added, sizeof(int),
                                  cudaMemcpyDeviceToHost));
        }

        std::vector<int> member_counts(N);
        CUDA_CHECK(cudaMemcpy(member_counts.data(), d_member_counts,
                              static_cast<size_t>(N) * sizeof(int),
                              cudaMemcpyDeviceToHost));

        int max_cardinality = -1;
        int best_seed = -1;
        for (int seed = 0; seed < N; ++seed) {
            const int cardinality = member_counts[seed];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            std::vector<int> best_cluster_members(max_cardinality);
            CUDA_CHECK(cudaMemcpy(best_cluster_members.data(),
                                  d_members + static_cast<size_t>(best_seed) * N,
                                  static_cast<size_t>(max_cardinality) * sizeof(int),
                                  cudaMemcpyDeviceToHost));

            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best_cluster_members;
            clusters.push_back(cluster);
            
            for (const int member : best_cluster_members) {
                if (clustered[member] == 0) {
                    clustered[member] = 1;
                    --unclustered_count;
                }
            }
        } else {
            break;
        }
    }

    CUDA_CHECK(cudaFree(d_any_added));
    CUDA_CHECK(cudaFree(d_selected_points));
    CUDA_CHECK(cudaFree(d_max_distances));
    CUDA_CHECK(cudaFree(d_in_cluster));
    CUDA_CHECK(cudaFree(d_member_counts));
    CUDA_CHECK(cudaFree(d_members));
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
