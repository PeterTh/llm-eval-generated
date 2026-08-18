// QT Clustering Benchmark - Simplified Sequential Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation,
                cudaGetErrorString(status));
        std::exit(EXIT_FAILURE);
    }
}

// A block builds one candidate cluster.  Candidate additions are necessarily
// sequential, but evaluating the diameter of every possible next point is
// independent and is distributed across the block's threads.
__global__ void buildCandidateClusters(const Point* points,
                                       const unsigned char* clustered,
                                       const int* seeds,
                                       int seed_count,
                                       double threshold,
                                       int point_count,
                                       int* member_lists,
                                       int* cardinalities) {
    const int slot = blockIdx.x;
    if (slot >= seed_count) return;

    const int tid = threadIdx.x;
    int* const members = member_lists + static_cast<size_t>(slot) * point_count;
    __shared__ double reduction_distance[CUDA_BLOCK_SIZE];
    __shared__ int reduction_index[CUDA_BLOCK_SIZE];
    __shared__ int selected_point;
    __shared__ int has_selection;
    __shared__ int member_count;

    if (tid == 0) {
        members[0] = seeds[slot];
        member_count = 1;
    }
    __syncthreads();

    while (member_count < point_count) {
        double thread_best_distance = DBL_MAX;
        int thread_best_index = INT_MAX;

        for (int candidate = tid; candidate < point_count; candidate += blockDim.x) {
            if (clustered[candidate]) continue;

            double max_distance = 0.0;
            bool already_member = false;
            for (int i = 0; i < member_count; ++i) {
                const int member = members[i];
                if (candidate == member) {
                    already_member = true;
                    break;
                }
                const double dx = points[candidate].x - points[member].x;
                const double dy = points[candidate].y - points[member].y;
                const double point_distance = sqrt(dx * dx + dy * dy);
                if (point_distance > max_distance) max_distance = point_distance;
            }

            if (!already_member && max_distance < threshold &&
                (max_distance < thread_best_distance ||
                 (max_distance == thread_best_distance && candidate < thread_best_index))) {
                thread_best_distance = max_distance;
                thread_best_index = candidate;
            }
        }

        reduction_distance[tid] = thread_best_distance;
        reduction_index[tid] = thread_best_index;
        __syncthreads();

        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (tid < offset) {
                const double other_distance = reduction_distance[tid + offset];
                const int other_index = reduction_index[tid + offset];
                if (other_distance < reduction_distance[tid] ||
                    (other_distance == reduction_distance[tid] &&
                     other_index < reduction_index[tid])) {
                    reduction_distance[tid] = other_distance;
                    reduction_index[tid] = other_index;
                }
            }
            __syncthreads();
        }

        if (tid == 0) {
            selected_point = reduction_index[0];
            has_selection = selected_point != INT_MAX;
            if (has_selection) members[member_count++] = selected_point;
        }
        __syncthreads();
        if (!has_selection) break;
    }

    if (tid == 0) cardinalities[slot] = member_count;
}

class CandidateClusterWorkspace {
public:
    explicit CandidateClusterWorkspace(int point_count)
        : point_count_(point_count) {
        const size_t member_bytes = static_cast<size_t>(point_count_) * point_count_ * sizeof(int);
        checkCuda(cudaMalloc(&device_points_, static_cast<size_t>(point_count_) * sizeof(Point)),
                  "allocating point data");
        checkCuda(cudaMalloc(&device_clustered_, static_cast<size_t>(point_count_) * sizeof(unsigned char)),
                  "allocating clustered flags");
        checkCuda(cudaMalloc(&device_seeds_, static_cast<size_t>(point_count_) * sizeof(int)),
                  "allocating seed list");
        checkCuda(cudaMalloc(&device_members_, member_bytes), "allocating candidate members");
        checkCuda(cudaMalloc(&device_cardinalities_, static_cast<size_t>(point_count_) * sizeof(int)),
                  "allocating candidate cardinalities");
    }

    ~CandidateClusterWorkspace() {
        cudaFree(device_cardinalities_);
        cudaFree(device_members_);
        cudaFree(device_seeds_);
        cudaFree(device_clustered_);
        cudaFree(device_points_);
    }

    CandidateClusterWorkspace(const CandidateClusterWorkspace&) = delete;
    CandidateClusterWorkspace& operator=(const CandidateClusterWorkspace&) = delete;

    void uploadPoints(const std::vector<Point>& points) {
        checkCuda(cudaMemcpy(device_points_, points.data(), static_cast<size_t>(point_count_) * sizeof(Point),
                             cudaMemcpyHostToDevice), "uploading point data");
    }

    void build(const std::vector<unsigned char>& clustered,
               const std::vector<int>& seeds,
               double threshold,
               std::vector<int>& cardinalities) {
        const int seed_count = static_cast<int>(seeds.size());
        checkCuda(cudaMemcpy(device_clustered_, clustered.data(), static_cast<size_t>(point_count_),
                             cudaMemcpyHostToDevice), "uploading clustered flags");
        checkCuda(cudaMemcpy(device_seeds_, seeds.data(), static_cast<size_t>(seed_count) * sizeof(int),
                             cudaMemcpyHostToDevice), "uploading seed list");
        buildCandidateClusters<<<seed_count, CUDA_BLOCK_SIZE>>>(
            device_points_, device_clustered_, device_seeds_, seed_count, threshold,
            point_count_, device_members_, device_cardinalities_);
        checkCuda(cudaGetLastError(), "launching candidate-cluster kernel");
        checkCuda(cudaMemcpy(cardinalities.data(), device_cardinalities_,
                             static_cast<size_t>(seed_count) * sizeof(int), cudaMemcpyDeviceToHost),
                  "downloading candidate cardinalities");
    }

    void downloadMembers(int seed_slot, int cardinality, std::vector<int>& members) const {
        members.resize(cardinality);
        checkCuda(cudaMemcpy(members.data(),
                             device_members_ + static_cast<size_t>(seed_slot) * point_count_,
                             static_cast<size_t>(cardinality) * sizeof(int), cudaMemcpyDeviceToHost),
                  "downloading selected cluster");
    }

private:
    int point_count_;
    Point* device_points_ = nullptr;
    unsigned char* device_clustered_ = nullptr;
    int* device_seeds_ = nullptr;
    int* device_members_ = nullptr;
    int* device_cardinalities_ = nullptr;
};

}  // namespace

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

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    // Bytes are used instead of vector<bool> so the current clustering state
    // can be copied directly to the device before each QT selection round.
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    CandidateClusterWorkspace workspace(N);
    workspace.uploadPoints(points);
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        int best_seed_slot = -1;
        std::vector<int> best_cluster_members;

        // Every remaining seed is constructed independently on the GPU.  The
        // host performs the final ordered reduction, matching the original
        // first-seed-wins rule for equal cardinalities.
        std::vector<int> cardinalities(unclustered_indices.size());
        workspace.build(clustered, unclustered_indices, threshold, cardinalities);
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int cardinality = cardinalities[i];
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = unclustered_indices[i];
                best_seed_slot = static_cast<int>(i);
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            workspace.downloadMembers(best_seed_slot, max_cardinality, best_cluster_members);
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
