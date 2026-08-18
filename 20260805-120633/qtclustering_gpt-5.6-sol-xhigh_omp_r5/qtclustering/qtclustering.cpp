// QT Clustering Benchmark - OpenMP shared-memory version
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
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <omp.h>

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

// All seed clusters in one QT round read the same point set.  Precomputing the
// distances once both removes repeated square roots and makes the innermost
// updates read a contiguous matrix row.  Each neighbor row remains sorted by
// point index, which preserves the sequential implementation's tie breaking.
class DistanceData {
  public:
    DistanceData(const std::vector<Point>& points, const double threshold)
        : point_count_(static_cast<int>(points.size())),
          values_(new double[static_cast<size_t>(point_count_) * point_count_]),
          neighbors_(point_count_) {
        constexpr double PI = 3.14159265358979323846;
        const double area_fraction = std::min(
            1.0, PI * threshold * threshold / (MAX_WIDTH * MAX_HEIGHT));
        const size_t estimated_neighbors = std::min(
            static_cast<size_t>(point_count_ > 0 ? point_count_ - 1 : 0),
            static_cast<size_t>(point_count_ * area_fraction * 2.0) + 8);

        // Row ownership gives deterministic results and parallel first-touch
        // placement of the potentially large matrix on NUMA systems.
        #pragma omp parallel for schedule(static)
        for (int member = 0; member < point_count_; ++member) {
            double* const row = values_.get() + static_cast<size_t>(member) * point_count_;
            std::vector<int>& row_neighbors = neighbors_[member];
            row_neighbors.reserve(estimated_neighbors);

            for (int candidate = 0; candidate < point_count_; ++candidate) {
                // Keep the argument order used by the original candidate scan.
                const double dist = distance(points[candidate], points[member]);
                row[candidate] = dist;
                if (candidate != member && dist < threshold) {
                    row_neighbors.push_back(candidate);
                }
            }
        }
    }

    const double* row(const int member) const {
        return values_.get() + static_cast<size_t>(member) * point_count_;
    }

    const std::vector<int>& neighbors(const int point) const {
        return neighbors_[point];
    }

  private:
    int point_count_;
    std::unique_ptr<double[]> values_;
    std::vector<std::vector<int>> neighbors_;
};

struct CandidateWorkspace {
    std::vector<int> members;
    std::vector<int> eligible_points;
    std::vector<double> max_distances;

    void reserve(const size_t point_count) {
        members.reserve(point_count);
        eligible_points.reserve(point_count);
        max_distances.reserve(point_count);
    }
};

// Generate one greedy candidate cluster.  A candidate's diameter can only
// increase as members are added, so points that reach the threshold are
// permanently removed.  The running maximum is updated using only the newest
// member instead of rescanning all previous members.
int generateCandidateCluster(const int seed_point,
                             const std::vector<uint8_t>& clustered,
                             const DistanceData& distances,
                             const double threshold,
                             CandidateWorkspace& workspace) {
    std::vector<int>& members = workspace.members;
    std::vector<int>& eligible = workspace.eligible_points;
    std::vector<double>& max_distances = workspace.max_distances;
    members.clear();
    eligible.clear();
    max_distances.clear();
    members.push_back(seed_point);

    // Every member of a valid cluster must be within the threshold of its
    // seed, so the precomputed seed neighborhood is the complete initial set.
    const double* seed_distances = distances.row(seed_point);
    int closest_slot = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (const int candidate : distances.neighbors(seed_point)) {
        if (clustered[candidate]) continue;

        const double dist = seed_distances[candidate];
        eligible.push_back(candidate);
        max_distances.push_back(dist);
        if (dist < min_diameter) {
            min_diameter = dist;
            closest_slot = static_cast<int>(eligible.size()) - 1;
        }
    }

    while (closest_slot >= 0) {
        const int selected_slot = closest_slot;
        const int newest_member = eligible[closest_slot];
        members.push_back(newest_member);

        const double* newest_distances = distances.row(newest_member);
        size_t write = 0;
        closest_slot = -1;
        min_diameter = std::numeric_limits<double>::max();

        // Filtering in place retains ascending point order and therefore the
        // original winner when two candidates have the same diameter.
        for (size_t read = 0; read < eligible.size(); ++read) {
            if (static_cast<int>(read) == selected_slot) continue;
            const int candidate = eligible[read];
            const double max_dist = std::max(max_distances[read],
                                             newest_distances[candidate]);
            if (max_dist < threshold) {
                eligible[write] = candidate;
                max_distances[write] = max_dist;
                if (max_dist < min_diameter) {
                    min_diameter = max_dist;
                    closest_slot = static_cast<int>(write);
                }
                ++write;
            }
        }
        eligible.resize(write);
        max_distances.resize(write);
    }

    return static_cast<int>(members.size());
}

struct alignas(64) ThreadBest {
    int cardinality = -1;
    int seed = -1;
    std::vector<int> members;
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    const DistanceData distances(points, threshold);
    std::vector<uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    clusters.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices[i] = i;
    }

    // Keep one OpenMP team alive across all QT rounds.  This avoids repeatedly
    // creating a team as the active point set shrinks.
    std::vector<ThreadBest> thread_best(omp_get_max_threads());
    bool finished = false;

    #pragma omp parallel shared(finished, thread_best, unclustered_indices, clustered, clusters)
    {
        const int thread_id = omp_get_thread_num();
        CandidateWorkspace workspace;
        workspace.reserve(N);

        while (true) {
            #pragma omp single
            {
                finished = unclustered_indices.empty();
            }

            if (finished) break;

            ThreadBest& local_best = thread_best[thread_id];
            local_best.cardinality = -1;
            local_best.seed = -1;
            local_best.members.clear();
            const int active_count = static_cast<int>(unclustered_indices.size());

            // Candidate costs vary sharply with local point density. Unit-sized
            // dynamic work distribution prevents dense seeds from stranding a
            // thread while the rest of the team waits at the round barrier.
            #pragma omp for schedule(dynamic, 1)
            for (int i = 0; i < active_count; ++i) {
                const int seed = unclustered_indices[i];
                const int cardinality = generateCandidateCluster(
                    seed, clustered, distances, threshold, workspace);

                if (cardinality > local_best.cardinality ||
                    (cardinality == local_best.cardinality && seed < local_best.seed)) {
                    local_best.cardinality = cardinality;
                    local_best.seed = seed;
                    local_best.members = workspace.members;
                }
            }

            // The ordered tie break makes the reduction independent of thread
            // count and scheduling, matching the original ascending seed scan.
            #pragma omp single
            {
                int winner = -1;
                const int team_size = omp_get_num_threads();
                for (int t = 0; t < team_size; ++t) {
                    if (thread_best[t].seed < 0) continue;
                    if (winner < 0 ||
                        thread_best[t].cardinality > thread_best[winner].cardinality ||
                        (thread_best[t].cardinality == thread_best[winner].cardinality &&
                         thread_best[t].seed < thread_best[winner].seed)) {
                        winner = t;
                    }
                }

                if (winner >= 0 && thread_best[winner].cardinality > 0) {
                    Cluster cluster;
                    cluster.seed_point = thread_best[winner].seed;
                    cluster.members = thread_best[winner].members;
                    for (const int member : cluster.members) {
                        clustered[member] = 1;
                    }
                    clusters.push_back(std::move(cluster));

                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                       [&clustered](const int idx) {
                                           return clustered[idx] != 0;
                                       }),
                        unclustered_indices.end());
                } else {
                    unclustered_indices.clear();
                }
            }
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
