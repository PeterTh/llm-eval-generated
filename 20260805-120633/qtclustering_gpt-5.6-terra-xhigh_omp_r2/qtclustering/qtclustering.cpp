// QT Clustering Benchmark - OpenMP Parallel Version
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
#include <omp.h>
#include <utility>
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

// Each seed evaluation needs private membership state.  Keeping one of these
// workspaces per OpenMP worker avoids repeatedly allocating and clearing an
// O(N) bitmap for every candidate cluster.  A generation stamp makes resetting
// the membership state O(number of members) instead of O(N).
class CandidateWorkspace {
public:
    explicit CandidateWorkspace(const int point_count)
        : membership_generation_(point_count, 0) {}

    void beginCluster(const int seed_point) {
        ++generation_;
        if (generation_ == 0) {
            std::fill(membership_generation_.begin(), membership_generation_.end(), 0);
            generation_ = 1;
        }

        members_.clear();
        membership_generation_[seed_point] = generation_;
        members_.push_back(seed_point);
    }

    bool contains(const int point) const {
        return membership_generation_[point] == generation_;
    }

    void add(const int point) {
        membership_generation_[point] = generation_;
        members_.push_back(point);
    }

    const std::vector<int>& members() const {
        return members_;
    }

private:
    std::vector<unsigned int> membership_generation_;
    std::vector<int> members_;
    unsigned int generation_ = 0;
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
                     const std::vector<unsigned char>& clustered,
                     const CandidateWorkspace& workspace,
                     const std::vector<Point>& points,
                     const double threshold,
                     const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || workspace.contains(candidate)) continue;
        
        // Calculate the maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        const double cutoff = std::min(threshold, min_diameter);
        const Point& candidate_point = points[candidate];
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double dist = distance(candidate_point, points[member]);
            max_dist = std::max(max_dist, dist);

            // The maximum distance can only increase.  A candidate at or
            // above either limit cannot satisfy the strict comparisons below,
            // so the remaining distance calculations cannot affect the result.
            if (max_dist >= cutoff) break;
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
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const double threshold,
                              const int point_count,
                              CandidateWorkspace& workspace) {
    workspace.beginCluster(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(workspace.members().size()) < point_count) {
        // Find closest point to current cluster that maintains diameter < threshold
        const int closest = findClosestPoint(workspace.members(), clustered, workspace, points,
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        workspace.add(closest);
    }

    return static_cast<int>(workspace.members().size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    unclustered_indices.reserve(N);
    clusters.reserve(N);

    // Workspaces are indexed by OpenMP thread ID and therefore never shared
    // between simultaneous candidate-cluster evaluations.
    std::vector<CandidateWorkspace> workspaces;
    const int worker_count = std::min(N, omp_get_max_threads());
    workspaces.reserve(worker_count);
    for (int worker = 0; worker < worker_count; ++worker) {
        workspaces.emplace_back(N);
    }
    std::vector<std::vector<int>> worker_best_members(worker_count);
    std::vector<int> worker_best_cardinalities(worker_count);
    std::vector<int> worker_best_positions(worker_count);
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int active_worker_count = std::min(
            worker_count, static_cast<int>(unclustered_indices.size()));
        int max_cardinality = -1;
        int best_seed = -1;
        int best_worker = -1;
        std::fill(worker_best_cardinalities.begin(), worker_best_cardinalities.end(), -1);
        std::fill(worker_best_positions.begin(), worker_best_positions.end(),
                  std::numeric_limits<int>::max());
        
        // Candidate clusters are independent while the current set of
        // clustered points is read-only.  Dynamic scheduling balances seeds
        // whose candidate clusters have very different construction costs.
#pragma omp parallel default(none) shared(clustered, points, threshold, N, unclustered_indices, workspaces, worker_best_members, worker_best_cardinalities, worker_best_positions) num_threads(active_worker_count)
        {
            const int worker = omp_get_thread_num();
            CandidateWorkspace& workspace = workspaces[worker];
            std::vector<int>& local_best_members = worker_best_members[worker];
            int local_best_cardinality = -1;
            int local_best_position = std::numeric_limits<int>::max();

#pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < static_cast<int>(unclustered_indices.size()); ++i) {
                const int seed = unclustered_indices[i];
                const int cardinality = generateCandidateCluster(
                    seed, clustered, points, threshold, N, workspace);

                // Dynamic scheduling does not visit a worker's seeds in
                // index order, so apply the same first-seed tie break locally
                // before the deterministic global selection below.
                if (cardinality > local_best_cardinality ||
                    (cardinality == local_best_cardinality && i < local_best_position)) {
                    local_best_cardinality = cardinality;
                    local_best_position = i;
                    local_best_members = workspace.members();
                }
            }

            worker_best_cardinalities[worker] = local_best_cardinality;
            worker_best_positions[worker] = local_best_position;
        }

        // Select serially in the original order.  Besides being inexpensive,
        // this retains the sequential algorithm's first-seed tie break.
        int best_position = std::numeric_limits<int>::max();
        for (int worker = 0; worker < active_worker_count; ++worker) {
            const int cardinality = worker_best_cardinalities[worker];
            const int position = worker_best_positions[worker];
            if (cardinality > max_cardinality ||
                (cardinality == max_cardinality && position < best_position)) {
                max_cardinality = cardinality;
                best_position = position;
                best_seed = unclustered_indices[position];
                best_worker = worker;
            }
        }
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = std::move(worker_best_members[best_worker]);
            clusters.push_back(std::move(cluster));
            
            // Mark all members as clustered
            for (const int member : clusters.back().members) {
                clustered[member] = 1;
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
