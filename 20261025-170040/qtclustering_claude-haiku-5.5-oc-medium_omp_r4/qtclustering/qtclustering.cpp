// QT Clustering Benchmark - OpenMP Parallel Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization (OpenMP, shared memory):
//   - In every clustering round, the candidate cluster for each unclustered
//     seed is built concurrently. Each thread keeps its own best candidate and
//     the per-thread bests are merged with a deterministic total order
//     (largest cardinality, ties broken by smallest seed index), so the result
//     is identical to the sequential algorithm regardless of scheduling.
//   - Cluster diameter validation is computed in parallel; output is printed
//     sequentially in the original order.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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
// (inherently sequential: the random stream defines the dataset)
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

// Find the closest unclustered point to the current cluster that maintains diameter < threshold.
// Returns -1 if no such point exists.
// Candidates are examined in index order and ties keep the lowest index.
// The distance scan for a candidate stops as soon as the candidate can no longer
// be accepted; this makes the same decisions as computing the full maximum.
static int findClosestPoint(const int* members,
                            const int member_count,
                            const unsigned char* clustered,
                            const unsigned char* in_cluster,
                            const Point* points,
                            const double threshold,
                            const int point_count) {
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    
    for (int candidate = 0; candidate < point_count; ++candidate) {
        // Skip if already clustered or already in this cluster
        if (clustered[candidate] || in_cluster[candidate]) continue;
        
        // Candidate is accepted only if max_dist < min(threshold, min_diameter)
        const double limit = std::min(threshold, min_diameter);
        const Point& p = points[candidate];
        
        // Maximum distance from candidate to all cluster members
        double max_dist = 0.0;
        bool rejected = false;
        for (int i = 0; i < member_count; ++i) {
            const double dist = distance(p, points[members[i]]);
            if (dist > max_dist) {
                max_dist = dist;
                if (max_dist >= limit) {
                    rejected = true;
                    break;
                }
            }
        }
        
        if (!rejected && max_dist < limit) {
            min_diameter = max_dist;
            closest_point = candidate;
        }
    }
    
    return closest_point;
}

// Grow a candidate cluster from seed_point.
// The members are written to `members` (cleared first). `in_cluster` is per-thread
// scratch of size point_count that must be all zero on entry; it is restored to
// all zero before returning.
// Returns the cardinality (size) of the cluster.
static int growCandidateCluster(const int seed_point,
                                const unsigned char* clustered,
                                const Point* points,
                                const double threshold,
                                const int point_count,
                                std::vector<int>& members,
                                unsigned char* in_cluster) {
    members.clear();
    
    // Add seed point
    in_cluster[seed_point] = 1;
    members.push_back(seed_point);
    
    // Iteratively add closest points
    while (static_cast<int>(members.size()) < point_count) {
        const int closest = findClosestPoint(members.data(), static_cast<int>(members.size()),
                                             clustered, in_cluster, points,
                                             threshold, point_count);
        
        if (closest < 0) break; // No more points can be added
        
        in_cluster[closest] = 1;
        members.push_back(closest);
    }
    
    // Reset scratch for the next call
    for (int m : members) {
        in_cluster[m] = 0;
    }
    
    return static_cast<int>(members.size());
}

// Best candidate seed found so far within a round
struct SeedCandidate {
    int cardinality = -1;
    int seed = -1;
    std::vector<int> members;
};

// Sequential selection rule: largest cardinality wins; among equal cardinalities
// the smallest seed index wins (seeds are visited in ascending order sequentially).
static inline bool isBetterCandidate(const int cardinality, const int seed,
                                     const int best_cardinality, const int best_seed) {
    return cardinality > best_cardinality ||
           (cardinality == best_cardinality && seed < best_seed);
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    
    const Point* pts = points.data();
    
    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        const int num_unclustered = static_cast<int>(unclustered_indices.size());
        SeedCandidate best;
        
        // Try each unclustered point as a seed, in parallel
        #pragma omp parallel
        {
            SeedCandidate local;
            std::vector<int> scratch;
            std::vector<unsigned char> in_cluster(N, 0);
            
            #pragma omp for schedule(dynamic, 1) nowait
            for (int i = 0; i < num_unclustered; ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;
                
                const int cardinality = growCandidateCluster(seed, clustered.data(), pts,
                                                             threshold, N,
                                                             scratch, in_cluster.data());
                
                if (isBetterCandidate(cardinality, seed, local.cardinality, local.seed)) {
                    local.cardinality = cardinality;
                    local.seed = seed;
                    local.members.swap(scratch);
                }
            }
            
            #pragma omp critical(qt_best_seed)
            {
                if (local.seed >= 0 &&
                    isBetterCandidate(local.cardinality, local.seed,
                                      best.cardinality, best.seed)) {
                    best = std::move(local);
                }
            }
        }
        
        // If we found a cluster, add it
        if (best.seed >= 0 && best.cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best.seed;
            cluster.members = std::move(best.members);
            
            // Mark all members as clustered
            for (int m : cluster.members) {
                clustered[m] = 1;
            }
            clusters.push_back(std::move(cluster));
            
            // Remove clustered points from unclustered list
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
    
    // Compute the diameter of each cluster (max distance between any two members)
    const int num_clusters = static_cast<int>(clusters.size());
    std::vector<double> diameters(num_clusters, 0.0);
    
    #pragma omp parallel for schedule(dynamic, 1)
    for (int c = 0; c < num_clusters; ++c) {
        const auto& members = clusters[c].members;
        const int m = static_cast<int>(members.size());
        double max_diameter = 0.0;
        for (int i = 0; i < m; ++i) {
            const Point& pi = points[members[i]];
            for (int j = i + 1; j < m; ++j) {
                const double dist = distance(pi, points[members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        diameters[c] = max_diameter;
    }
    
    // Report in cluster order
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
