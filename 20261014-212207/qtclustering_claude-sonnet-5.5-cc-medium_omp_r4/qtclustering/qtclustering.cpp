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

// Main QT clustering algorithm.
// Seeds are evaluated in parallel. For each seed, the running maximum distance
// from every candidate to the cluster is maintained incrementally (identical
// values to recomputing the max over all members), and candidates whose maximum
// reaches the threshold are dropped permanently since it can only grow.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> unclustered_indices(N);
    std::vector<char> clustered(N, 0);
    std::vector<Cluster> clusters;

    for (int i = 0; i < N; ++i) unclustered_indices[i] = i;

    // Contiguous coordinates of the currently unclustered points
    std::vector<double> ux(N), uy(N);
    for (int i = 0; i < N; ++i) {
        ux[i] = points[i].x;
        uy[i] = points[i].y;
    }

    int M = N;                 // number of unclustered points
    int g_card = -1;           // best cardinality this round
    int g_pos = -1;            // position of best seed in unclustered list
    std::vector<int> g_members;

    #pragma omp parallel
    {
        // Thread-local scratch buffers
        std::vector<double> cx(N), cy(N), cd(N);
        std::vector<int> cid(N);
        std::vector<int> members, best_local;
        members.reserve(N);

        while (M > 0) {
            int local_card = -1;
            int local_pos = -1;

            #pragma omp for schedule(dynamic, 1) nowait
            for (int s = 0; s < M; ++s) {
                const double sx = ux[s], sy = uy[s];
                int n = 0;
                for (int j = 0; j < M; ++j) {
                    if (j == s) continue;
                    const double dx = ux[j] - sx;
                    const double dy = uy[j] - sy;
                    const double d = std::sqrt(dx * dx + dy * dy);
                    if (d < threshold) {
                        cx[n] = ux[j]; cy[n] = uy[j]; cd[n] = d; cid[n] = j;
                        ++n;
                    }
                }
                members.clear();
                members.push_back(unclustered_indices[s]);

                while (n > 0) {
                    // Closest candidate (first one on ties)
                    int bi = 0;
                    double bd = cd[0];
                    for (int j = 1; j < n; ++j) {
                        if (cd[j] < bd) { bd = cd[j]; bi = j; }
                    }
                    const double mx = cx[bi], my = cy[bi];
                    members.push_back(unclustered_indices[cid[bi]]);

                    // Update running max distances and compact survivors
                    int m = 0;
                    for (int j = 0; j < n; ++j) {
                        if (j == bi) continue;
                        const double dx = cx[j] - mx;
                        const double dy = cy[j] - my;
                        const double d = std::max(cd[j], std::sqrt(dx * dx + dy * dy));
                        if (d < threshold) {
                            cx[m] = cx[j]; cy[m] = cy[j]; cd[m] = d; cid[m] = cid[j];
                            ++m;
                        }
                    }
                    n = m;
                }

                const int card = static_cast<int>(members.size());
                if (card > local_card || (card == local_card && s < local_pos)) {
                    local_card = card;
                    local_pos = s;
                    best_local = members;
                }
            }

            // Reduce: largest cardinality, earliest seed position on ties
            #pragma omp critical(qt_best)
            {
                if (local_card > g_card || (local_card == g_card && local_card >= 0 && local_pos < g_pos)) {
                    g_card = local_card;
                    g_pos = local_pos;
                    g_members = best_local;
                }
            }
            #pragma omp barrier

            #pragma omp single
            {
                Cluster cluster;
                cluster.seed_point = unclustered_indices[g_pos];
                cluster.members = g_members;
                for (int m : g_members) clustered[m] = 1;
                clusters.push_back(std::move(cluster));

                int k = 0;
                for (int i = 0; i < M; ++i) {
                    const int idx = unclustered_indices[i];
                    if (!clustered[idx]) {
                        unclustered_indices[k] = idx;
                        ux[k] = ux[i];
                        uy[k] = uy[i];
                        ++k;
                    }
                }
                unclustered_indices.resize(k);
                M = k;
                g_card = -1;
                g_pos = -1;
            }   // implicit barrier
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
