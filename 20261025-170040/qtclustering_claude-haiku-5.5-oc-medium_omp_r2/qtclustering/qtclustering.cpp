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
#include <numeric>
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

// Candidate point that may still join the cluster being grown.
// max_dist is the maximum distance from idx to every current cluster member.
struct Candidate {
    int idx;
    double max_dist;
};

// Per-thread scratch space, reused across seeds to avoid repeated allocation.
struct ThreadScratch {
    std::vector<Candidate> active;   // candidates still eligible (max_dist < threshold)
    std::vector<int> members;        // members of the cluster currently being grown
    int best_card = -1;              // best cardinality seen by this thread
    int best_pos = -1;               // position in unclustered list of that best seed
    std::vector<int> best_members;   // members of that best candidate cluster
};

// Grow a candidate cluster from seed_point using greedy QT expansion.
// Each step adds the eligible candidate with the smallest max-distance to the
// current members (ties broken by lowest index), exactly as the original
// findClosestPoint() did. Distances to the cluster are maintained incrementally:
// a candidate's max-distance only grows as members are added, so once it reaches
// the threshold it can never become eligible again and is dropped.
// Result is left in s.members. Returns the cardinality of the cluster.
int growCandidateCluster(const int seed_point,
                         const std::vector<int>& unclustered,
                         const std::vector<unsigned char>& clustered,
                         const std::vector<Point>& points,
                         const double threshold,
                         ThreadScratch& s) {
    s.members.clear();
    s.members.push_back(seed_point);

    s.active.clear();
    const Point& seed = points[seed_point];
    for (const int c : unclustered) {
        if (c == seed_point || clustered[c]) continue;
        const double d = distance(points[c], seed);
        if (d < threshold) {
            s.active.push_back({c, d});
        }
    }

    while (!s.active.empty()) {
        // Select the eligible candidate with minimum max-distance (lowest index on ties)
        size_t best_pos = 0;
        for (size_t p = 1; p < s.active.size(); ++p) {
            const Candidate& a = s.active[p];
            const Candidate& b = s.active[best_pos];
            if (a.max_dist < b.max_dist || (a.max_dist == b.max_dist && a.idx < b.idx)) {
                best_pos = p;
            }
        }
        const int best = s.active[best_pos].idx;
        s.active[best_pos] = s.active.back();
        s.active.pop_back();
        s.members.push_back(best);

        // Update max-distances with the newly added member and drop candidates
        // that can no longer satisfy the diameter threshold.
        const Point& added = points[best];
        size_t w = 0;
        for (size_t r = 0; r < s.active.size(); ++r) {
            Candidate c = s.active[r];
            c.max_dist = std::max(c.max_dist, distance(points[c.idx], added));
            if (c.max_dist < threshold) {
                s.active[w++] = c;
            }
        }
        s.active.resize(w);
    }

    return static_cast<int>(s.members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<Cluster> clusters;

    // One scratch buffer per OpenMP thread, reused across all iterations
    std::vector<ThreadScratch> scratch(omp_get_max_threads());

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int U = static_cast<int>(unclustered_indices.size());

        for (auto& s : scratch) {
            s.best_card = -1;
            s.best_pos = -1;
        }

        // Evaluate every unclustered point as a seed in parallel. Each thread keeps
        // its own best (largest cardinality, then earliest position), so the global
        // reduction below reproduces the sequential "first seed with max size wins" rule.
        #pragma omp parallel
        {
            ThreadScratch& s = scratch[omp_get_thread_num()];

            #pragma omp for schedule(dynamic, 1) nowait
            for (int i = 0; i < U; ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                const int cardinality = growCandidateCluster(seed, unclustered_indices,
                                                             clustered, points,
                                                             threshold, s);

                if (cardinality > s.best_card ||
                    (cardinality == s.best_card && i < s.best_pos)) {
                    s.best_card = cardinality;
                    s.best_pos = i;
                    s.best_members = s.members;
                }
            }
        }

        // Reduce across threads: max cardinality, ties -> earliest position
        int max_cardinality = -1;
        int best_pos = -1;
        const ThreadScratch* winner = nullptr;
        for (const auto& s : scratch) {
            if (s.best_card > max_cardinality ||
                (s.best_card == max_cardinality && s.best_card > 0 && s.best_pos < best_pos)) {
                max_cardinality = s.best_card;
                best_pos = s.best_pos;
                winner = &s;
            }
        }

        // If we found a cluster, add it
        if (winner != nullptr && best_pos >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = unclustered_indices[best_pos];
            cluster.members = winner->best_members;

            // Mark all members as clustered
            for (const int m : cluster.members) {
                clustered[m] = 1;
            }
            clusters.push_back(std::move(cluster));

            // Remove clustered points from unclustered list (order preserved)
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx] != 0; }),
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
    
    // Compute diameters (max distance between any two members) in parallel
    const long long C = static_cast<long long>(clusters.size());
    std::vector<double> diameters(clusters.size(), 0.0);
    #pragma omp parallel for schedule(dynamic, 1)
    for (long long c = 0; c < C; ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        for (size_t i = 0; i < cluster.members.size(); ++i) {
            for (size_t j = i + 1; j < cluster.members.size(); ++j) {
                const double dist = distance(points[cluster.members[i]],
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }

    // Check each cluster (sequential, so output order is deterministic)
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
