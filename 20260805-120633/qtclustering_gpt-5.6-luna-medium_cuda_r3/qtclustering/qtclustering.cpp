// QT Clustering Benchmark - Simplified Sequential Version
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

[[noreturn]] void cudaFailure(const cudaError_t error, const char* operation) {
    std::fprintf(stderr, "CUDA error during %s: %s\n", operation,
                 cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void cudaCheck(const cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(error, operation);
}

// Each thread evaluates one candidate.  The reduction/selection is performed
// in index order on the host so that ties have precisely the original meaning.
struct CandidateBest {
    double score;
    int index;
};

__global__ void evaluateCandidates(const Point* __restrict__ points,
                                    const unsigned char* __restrict__ clustered,
                                    const int* __restrict__ members,
                                    const int member_count,
                                    const double threshold,
                                    const int point_count,
                                    CandidateBest* __restrict__ block_results) {
    const int candidate = blockIdx.x * blockDim.x + threadIdx.x;
    double score = INFINITY;
    int index = -1;
    if (candidate < point_count && !clustered[candidate]) {
        bool is_member = false;
        for (int i = 0; i < member_count; ++i) {
            if (members[i] == candidate) {
                is_member = true;
                break;
            }
        }
        if (!is_member) {
            double max_dist = 0.0;
            const Point candidate_point = points[candidate];
            for (int i = 0; i < member_count; ++i) {
                const Point member_point = points[members[i]];
                const double dx = candidate_point.x - member_point.x;
                const double dy = candidate_point.y - member_point.y;
                const double dist = sqrt(dx * dx + dy * dy);
                if (dist > max_dist) max_dist = dist;
            }
            if (max_dist < threshold) {
                score = max_dist;
                index = candidate;
            }
        }
    }

    __shared__ double shared_scores[256];
    __shared__ int shared_indices[256];
    shared_scores[threadIdx.x] = score;
    shared_indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const double other_score = shared_scores[threadIdx.x + stride];
            const int other_index = shared_indices[threadIdx.x + stride];
            if (other_score < shared_scores[threadIdx.x] ||
                (other_score == shared_scores[threadIdx.x] &&
                 other_index >= 0 &&
                 (shared_indices[threadIdx.x] < 0 ||
                  other_index < shared_indices[threadIdx.x]))) {
                shared_scores[threadIdx.x] = other_score;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_results[blockIdx.x] = {shared_scores[0], shared_indices[0]};
    }
}

class CudaSearch {
  public:
    explicit CudaSearch(const std::vector<Point>& points)
        : point_count_(static_cast<int>(points.size())) {
        cudaCheck(cudaSetDevice(0), "selecting GPU 0");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device_points_),
                             points.size() * sizeof(Point)), "allocating points");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device_clustered_),
                             points.size() * sizeof(unsigned char)),
                  "allocating clustered bitmap");
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device_members_),
                             points.size() * sizeof(int)), "allocating members");
        block_count_ = (point_count_ + threadsPerBlock - 1) / threadsPerBlock;
        cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device_results_),
                             block_count_ * sizeof(CandidateBest)),
                  "allocating reduction results");
        cudaCheck(cudaMemcpy(device_points_, points.data(),
                             points.size() * sizeof(Point), cudaMemcpyHostToDevice),
                  "copying points");
        host_results_.resize(block_count_);
    }

    void updateClustered(const std::vector<unsigned char>& clustered) {
        cudaCheck(cudaMemcpy(device_clustered_, clustered.data(),
                             clustered.size() * sizeof(unsigned char),
                             cudaMemcpyHostToDevice), "copying clustered bitmap");
    }

    ~CudaSearch() {
        cudaFree(device_results_);
        cudaFree(device_members_);
        cudaFree(device_clustered_);
        cudaFree(device_points_);
    }

    int closestPoint(const std::vector<int>& members,
                     const std::vector<unsigned char>& /*clustered*/,
                     const double threshold) {
        cudaCheck(cudaMemcpy(device_members_, members.data(),
                             members.size() * sizeof(int), cudaMemcpyHostToDevice),
                  "copying cluster members");

        const int blocks = block_count_;
        evaluateCandidates<<<blocks, threadsPerBlock>>>(
            device_points_, device_clustered_, device_members_,
            static_cast<int>(members.size()), threshold, point_count_, device_results_);
        cudaCheck(cudaGetLastError(), "launching candidate evaluation");
        cudaCheck(cudaMemcpy(host_results_.data(), device_results_,
                             host_results_.size() * sizeof(CandidateBest), cudaMemcpyDeviceToHost),
                  "copying candidate scores");

        int closest = -1;
        double minimum = std::numeric_limits<double>::max();
        for (const CandidateBest result : host_results_) {
            // Strict comparison deliberately retains the lowest index on ties.
            if (result.score < minimum ||
                (result.score == minimum && result.index >= 0 &&
                 (closest < 0 || result.index < closest))) {
                minimum = result.score;
                closest = result.index;
            }
        }
        return closest;
    }

  private:
    static constexpr int threadsPerBlock = 256;
    int point_count_;
    int block_count_;
    Point* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    int* device_members_ = nullptr;
    CandidateBest* device_results_ = nullptr;
    std::vector<CandidateBest> host_results_;
};

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

int generateCandidateClusterCuda(const int seed_point,
                                  const std::vector<unsigned char>& clustered,
                                  const double threshold,
                                  CudaSearch& search,
                                  std::vector<int>* cluster_members) {
    std::vector<int> members;
    members.reserve(clustered.size());
    members.push_back(seed_point);

    while (static_cast<int>(members.size()) < static_cast<int>(clustered.size())) {
        const int closest = search.closestPoint(members, clustered, threshold);
        if (closest < 0) break;
        members.push_back(closest);
    }

    if (cluster_members) *cluster_members = members;
    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    CudaSearch search(points);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        search.updateClustered(clustered);
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Try each unclustered point as a seed
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            if (clustered[seed]) continue;
            
            std::vector<int> candidate_members;
            const int cardinality = generateCandidateClusterCuda(
                seed, clustered, threshold, search, &candidate_members);
            
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
                best_cluster_members = candidate_members;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
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
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );
        } else {
            // No more clusters can be formed
            break;
        }
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
