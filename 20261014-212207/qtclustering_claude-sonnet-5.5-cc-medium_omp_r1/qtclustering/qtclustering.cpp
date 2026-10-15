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

// Grow the candidate cluster for one seed. The unclustered points within
// `threshold` of the seed are the only possible members (a member's maximum
// distance to the cluster can only grow), so we keep just those, along with
// their running maximum distance to the current members. Candidates are kept
// in ascending index order so tie-breaking matches a full scan.
static void growCluster(const int pos,
                        const std::vector<int>& ids,
                        const std::vector<double>& ux,
                        const std::vector<double>& uy,
                        const double threshold,
                        std::vector<int>& cand,
                        std::vector<double>& md,
                        std::vector<int>& members) {
    const int U = static_cast<int>(ids.size());
    const double sx = ux[pos], sy = uy[pos];
    cand.clear();
    md.clear();
    members.clear();
    members.push_back(ids[pos]);

    for (int j = 0; j < U; ++j) {
        if (j == pos) continue;
        const double dx = sx - ux[j], dy = sy - uy[j];
        const double d = std::sqrt(dx * dx + dy * dy);
        if (d < threshold) {
            cand.push_back(j);
            md.push_back(d);
        }
    }

    int n = static_cast<int>(cand.size());
    while (n > 0) {
        // Closest candidate (lowest index on ties)
        int best = 0;
        double bd = md[0];
        for (int j = 1; j < n; ++j) {
            if (md[j] < bd) { bd = md[j]; best = j; }
        }
        const int bp = cand[best];
        members.push_back(ids[bp]);
        const double bx = ux[bp], by = uy[bp];

        // Update running max distances and drop candidates exceeding threshold
        int w = 0;
        for (int j = 0; j < n; ++j) {
            if (j == best) continue;
            const int c = cand[j];
            const double dx = ux[c] - bx, dy = uy[c] - by;
            const double d = std::sqrt(dx * dx + dy * dy);
            const double nd = std::max(md[j], d);
            if (nd < threshold) {
                cand[w] = c;
                md[w] = nd;
                ++w;
            }
        }
        n = w;
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> ids(N);
    std::vector<double> ux(N), uy(N);
    std::vector<Cluster> clusters;
    std::vector<char> clustered(N, 0);

    for (int i = 0; i < N; ++i) {
        ids[i] = i;
        ux[i] = points[i].x;
        uy[i] = points[i].y;
    }

    while (!ids.empty()) {
        const int U = static_cast<int>(ids.size());
        int max_cardinality = -1;
        int best_pos = -1;
        std::vector<int> best_members;

        #pragma omp parallel
        {
            std::vector<int> cand, members;
            std::vector<double> md;
            int local_card = -1, local_pos = -1;
            std::vector<int> local_members;

            #pragma omp for schedule(dynamic, 4) nowait
            for (int i = 0; i < U; ++i) {
                growCluster(i, ids, ux, uy, threshold, cand, md, members);
                const int card = static_cast<int>(members.size());
                // Within a thread positions are not necessarily increasing
                // under dynamic scheduling, so break ties on position.
                if (card > local_card || (card == local_card && i < local_pos)) {
                    local_card = card;
                    local_pos = i;
                    local_members.swap(members);
                }
            }

            #pragma omp critical
            {
                if (local_card > max_cardinality ||
                    (local_card == max_cardinality && local_pos >= 0 && local_pos < best_pos)) {
                    max_cardinality = local_card;
                    best_pos = local_pos;
                    best_members.swap(local_members);
                }
            }
        }

        if (best_pos < 0 || max_cardinality <= 0) break;

        Cluster cluster;
        cluster.seed_point = ids[best_pos];
        cluster.members = std::move(best_members);
        for (int m : cluster.members) clustered[m] = 1;
        clusters.push_back(std::move(cluster));

        // Compact the unclustered arrays
        int w = 0;
        for (int i = 0; i < U; ++i) {
            if (!clustered[ids[i]]) {
                ids[w] = ids[i];
                ux[w] = ux[i];
                uy[w] = uy[i];
                ++w;
            }
        }
        ids.resize(w);
        ux.resize(w);
        uy.resize(w);
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
