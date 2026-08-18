// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cfloat>
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

#define CUDA_CHECK(call) do {                                                   \
    const cudaError_t error_ = (call);                                          \
    if (error_ != cudaSuccess) {                                                \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                     cudaGetErrorString(error_));                               \
        std::exit(EXIT_FAILURE);                                                \
    }                                                                           \
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

// Store squared distances transposed by member.  During candidate evaluation,
// adjacent threads then read adjacent addresses for every current member.
__global__ void buildDistanceMatrix(const Point* __restrict__ points,
                                    double* __restrict__ distances, int n) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    const int member = blockIdx.y * blockDim.y + threadIdx.y;
    if (candidate < n && member < n) {
        const double dx = points[candidate].x - points[member].x;
        const double dy = points[candidate].y - points[member].y;
        distances[static_cast<size_t>(member) * n + candidate] = dx * dx + dy * dy;
    }
}

// A block constructs one seed's candidate cluster.  Candidate scoring is
// parallel; the reduction compares (diameter, index), preserving the original
// deterministic first-index tie break.
__global__ void generateAllCandidates(const double* __restrict__ distances,
                                      const unsigned char* __restrict__ clustered,
                                      unsigned char* __restrict__ in_cluster,
                                      int* __restrict__ members,
                                      int* __restrict__ lengths,
                                      double threshold_squared, int n) {
    const int seed = blockIdx.x;
    if (clustered[seed]) return;

    unsigned char* mine = in_cluster + static_cast<size_t>(seed) * n;
    int* my_members = members + static_cast<size_t>(seed) * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) mine[i] = 0;
    __syncthreads();

    if (threadIdx.x == 0) {
        mine[seed] = 1;
        my_members[0] = seed;
    }
    __shared__ int count;
    __shared__ double best_distance[256];
    __shared__ int best_index[256];
    if (threadIdx.x == 0) count = 1;
    __syncthreads();

    while (count < n) {
        double local_distance = DBL_MAX;
        int local_index = n;
        for (int candidate = threadIdx.x; candidate < n; candidate += blockDim.x) {
            if (clustered[candidate] || mine[candidate]) continue;
            double maximum = 0.0;
            for (int j = 0; j < count; ++j)
                maximum = fmax(maximum, distances[static_cast<size_t>(my_members[j]) * n + candidate]);
            if (maximum < threshold_squared &&
                (maximum < local_distance || (maximum == local_distance && candidate < local_index))) {
                local_distance = maximum;
                local_index = candidate;
            }
        }
        best_distance[threadIdx.x] = local_distance;
        best_index[threadIdx.x] = local_index;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (threadIdx.x < stride) {
                const double other_d = best_distance[threadIdx.x + stride];
                const int other_i = best_index[threadIdx.x + stride];
                if (other_d < best_distance[threadIdx.x] ||
                    (other_d == best_distance[threadIdx.x] && other_i < best_index[threadIdx.x])) {
                    best_distance[threadIdx.x] = other_d;
                    best_index[threadIdx.x] = other_i;
                }
            }
            __syncthreads();
        }
        if (best_index[0] == n) break;
        if (threadIdx.x == 0) {
            const int chosen = best_index[0];
            mine[chosen] = 1;
            my_members[count++] = chosen;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) lengths[seed] = count;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<Cluster> clusters;

    Point* d_points = nullptr;
    double* d_distances = nullptr;
    unsigned char *d_clustered = nullptr, *d_in_cluster = nullptr;
    int *d_members = nullptr, *d_lengths = nullptr;
    const size_t cells = static_cast<size_t>(N) * N;
    CUDA_CHECK(cudaMalloc(&d_points, static_cast<size_t>(N) * sizeof(Point)));
    CUDA_CHECK(cudaMalloc(&d_distances, cells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_clustered, N));
    CUDA_CHECK(cudaMalloc(&d_in_cluster, cells));
    CUDA_CHECK(cudaMalloc(&d_members, cells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_lengths, static_cast<size_t>(N) * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_points, points.data(), static_cast<size_t>(N) * sizeof(Point), cudaMemcpyHostToDevice));
    const dim3 threads2d(16, 16), blocks2d((N + 15) / 16, (N + 15) / 16);
    buildDistanceMatrix<<<blocks2d, threads2d>>>(d_points, d_distances, N);
    CUDA_CHECK(cudaGetLastError());

    std::vector<int> lengths(N);
    int remaining = N;
    while (remaining > 0) {
        CUDA_CHECK(cudaMemcpy(d_clustered, clustered.data(), N, cudaMemcpyHostToDevice));
        generateAllCandidates<<<N, 256>>>(d_distances, d_clustered, d_in_cluster,
                                          d_members, d_lengths, threshold * threshold, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(lengths.data(), d_lengths, static_cast<size_t>(N) * sizeof(int), cudaMemcpyDeviceToHost));
        int best_seed = -1;
        for (int seed = 0; seed < N; ++seed)
            if (!clustered[seed] && (best_seed < 0 || lengths[seed] > lengths[best_seed])) best_seed = seed;
        if (best_seed < 0) break;
        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members.resize(lengths[best_seed]);
        CUDA_CHECK(cudaMemcpy(cluster.members.data(), d_members + static_cast<size_t>(best_seed) * N,
                              cluster.members.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (int member : cluster.members) {
            clustered[member] = 1;
            --remaining;
        }
        clusters.push_back(std::move(cluster));
    }
    CUDA_CHECK(cudaFree(d_lengths));
    CUDA_CHECK(cudaFree(d_members));
    CUDA_CHECK(cudaFree(d_in_cluster));
    CUDA_CHECK(cudaFree(d_clustered));
    CUDA_CHECK(cudaFree(d_distances));
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
