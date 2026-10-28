// QT Clustering Benchmark - OpenMP Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization: candidate clusters for all seed points are grown
// concurrently with OpenMP. Each thread keeps its own best candidate and the
// results are merged with the same tie-breaking rule as the sequential
// algorithm (largest cardinality, then lowest seed index), so the output is
// identical to the sequential version.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// Per-thread scratch space reused across seed points
struct Workspace {
    std::vector<double> max_dist;  // max distance from each point to the cluster so far
    std::vector<int> active;       // candidates still eligible, in ascending index order

    explicit Workspace(const int point_count)
        : max_dist(point_count, 0.0) {
        active.reserve(point_count);
    }
};

// Generate a candidate cluster starting from a seed point.
//
// Equivalent to repeatedly adding the unclustered point whose maximum distance
// to the current members is smallest (ties -> lowest index) while that maximum
// distance stays below the threshold.
//
// The maximum distance of a candidate to the cluster is non-decreasing as
// members are added, so it is maintained incrementally, and candidates that
// reach the threshold are dropped permanently.
//
// Pruning: the final cardinality is at most members + active candidates. If
// that bound is strictly below a cardinality already achieved by some other
// seed (best_known), this candidate can never win, so growth stops early and
// -1 is returned.
//
// On return, `members` holds the cluster in insertion order.
int generateCandidateCluster(const int seed_point,
                             const std::vector<bool>& clustered,
                             const std::vector<Point>& points,
                             const double threshold,
                             const int point_count,
                             const std::atomic<int>& best_known,
                             Workspace& ws,
                             std::vector<int>& members) {
    std::vector<double>& max_dist = ws.max_dist;
    std::vector<int>& active = ws.active;

    members.clear();
    members.push_back(seed_point);
    active.clear();

    // Initial candidates: unclustered points within threshold of the seed
    const Point& seed = points[seed_point];
    for (int c = 0; c < point_count; ++c) {
        if (clustered[c] || c == seed_point) continue;
        const double d = distance(points[c], seed);
        if (d < threshold) {
            max_dist[c] = d;
            active.push_back(c);
        }
    }

    // Select the candidate with smallest max distance (first index on ties)
    int next = -1;
    double next_d = std::numeric_limits<double>::max();
    for (const int c : active) {
        if (max_dist[c] < next_d) {
            next_d = max_dist[c];
            next = c;
        }
    }

    while (next >= 0) {
        if (static_cast<int>(members.size() + active.size()) <
            best_known.load(std::memory_order_relaxed)) {
            return -1;  // Cannot reach the cardinality already achieved elsewhere
        }

        const int added = next;
        members.push_back(added);
        const Point& added_pt = points[added];

        // Update max distances of remaining candidates, drop those that reach
        // the threshold, and select the next closest candidate in the same pass
        next = -1;
        next_d = std::numeric_limits<double>::max();
        size_t w = 0;
        for (size_t r = 0; r < active.size(); ++r) {
            const int c = active[r];
            if (c == added) continue;

            const double d = distance(points[c], added_pt);
            const double m = std::max(max_dist[c], d);
            if (m < threshold) {
                max_dist[c] = m;
                active[w++] = c;
                if (m < next_d) {
                    next_d = m;
                    next = c;
                }
            }
        }
        active.resize(w);
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        int max_cardinality = -1;
        int best_seed = -1;
        std::vector<int> best_cluster_members;
        
        // Largest cardinality achieved so far by any seed (lower bound used for pruning)
        std::atomic<int> best_known{0};
        const int seed_count = static_cast<int>(unclustered_indices.size());
        
        // Try each unclustered point as a seed, in parallel. Every thread tracks the
        // best (largest cardinality, then lowest seed index) candidate it has seen;
        // these are merged with the same rule, which yields the same winner as the
        // sequential scan in ascending seed order.
        #pragma omp parallel
        {
            Workspace ws(N);
            std::vector<int> candidate_members;
            int t_cardinality = -1;
            int t_seed = -1;
            std::vector<int> t_members;
            
            #pragma omp for schedule(dynamic, 4) nowait
            for (int i = 0; i < seed_count; ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;
                
                const int cardinality = generateCandidateCluster(seed, clustered, points,
                                                                 threshold, N, best_known,
                                                                 ws, candidate_members);
                if (cardinality < 0) continue;  // Pruned: cannot win
                
                if (cardinality > t_cardinality ||
                    (cardinality == t_cardinality && seed < t_seed)) {
                    t_cardinality = cardinality;
                    t_seed = seed;
                    t_members.swap(candidate_members);
                    
                    // Publish the achieved cardinality for pruning in other threads
                    int cur = best_known.load(std::memory_order_relaxed);
                    while (cur < cardinality &&
                           !best_known.compare_exchange_weak(cur, cardinality,
                                                             std::memory_order_relaxed)) {
                    }
                }
            }
            
            #pragma omp critical(qt_merge_best)
            {
                if (t_cardinality > max_cardinality ||
                    (t_cardinality == max_cardinality && t_seed >= 0 && t_seed < best_seed)) {
                    max_cardinality = t_cardinality;
                    best_seed = t_seed;
                    best_cluster_members = t_members;
                }
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
                clustered[best_cluster_members[i]] = true;
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
    
    // Diameter of each cluster (max distance between any two points), computed in parallel
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& members = clusters[c].members;
        double max_diameter = 0.0;
        for (size_t i = 0; i < members.size(); ++i) {
            for (size_t j = i + 1; j < members.size(); ++j) {
                const double dist = distance(points[members[i]], points[members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        const double max_diameter = diameters[c];
        
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
