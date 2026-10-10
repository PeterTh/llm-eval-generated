// QT Clustering Benchmark - OpenMP shared-memory version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
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

struct CandidatePoint {
    double max_distance;
    int point;
};

// Each distance to an added member is evaluated once. Candidates that fail the
// threshold can never become eligible again, since their maximum only grows.
// Stable compaction retains point-index order and the original distance ties.
void generateCandidateCluster(const int seed_point,
                              const std::vector<int>& unclustered_indices,
                              const std::vector<Point>& points,
                              const double threshold,
                              std::vector<CandidatePoint>& candidates,
                              std::vector<int>& members) {
    candidates.clear();
    members.clear();
    members.push_back(seed_point);

    int closest = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (const int candidate : unclustered_indices) {
        if (candidate == seed_point) continue;
        const double max_dist = std::max(0.0, distance(points[candidate], points[seed_point]));
        if (max_dist < threshold) {
            candidates.push_back({max_dist, candidate});
            if (max_dist < min_diameter) {
                min_diameter = max_dist;
                closest = candidate;
            }
        }
    }

    while (closest >= 0) {
        members.push_back(closest);
        const int added = closest;
        closest = -1;
        min_diameter = std::numeric_limits<double>::max();
        size_t remaining = 0;
        for (const CandidatePoint& candidate : candidates) {
            if (candidate.point == added) continue;
            const double max_dist = std::max(candidate.max_distance,
                                            distance(points[candidate.point], points[added]));
            if (max_dist < threshold) {
                candidates[remaining++] = {max_dist, candidate.point};
                if (max_dist < min_diameter) {
                    min_diameter = max_dist;
                    closest = candidate.point;
                }
            }
        }
        candidates.resize(remaining);
    }
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::vector<Cluster> clusters;
    clusters.reserve(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);

    // Removing points outside a seed's greedy cluster cannot change any of its
    // choices. Reuse such clusters, and only rebuild ones that lost a member.
    std::vector<std::vector<int>> candidate_clusters(N);
    std::uint64_t best_candidate = 0;

    // Keep one team and one scratch buffer per worker for all clustering rounds.
    #pragma omp parallel default(none) shared(points, threshold, N, clustered, unclustered_indices, clusters, candidate_clusters, best_candidate)
    {
        std::vector<CandidatePoint> candidates;
        candidates.reserve(N);

        while (!unclustered_indices.empty()) {
            #pragma omp for schedule(dynamic, 8) reduction(max:best_candidate)
            for (int i = 0; i < static_cast<int>(unclustered_indices.size()); ++i) {
                const int seed = unclustered_indices[i];
                auto& members = candidate_clusters[seed];
                bool rebuild = members.empty();
                if (!rebuild) {
                    for (const int member : members) {
                        if (clustered[member]) {
                            rebuild = true;
                            break;
                        }
                    }
                }
                if (rebuild) {
                    generateCandidateCluster(seed, unclustered_indices, points,
                                             threshold, candidates, members);
                }

                // Integer max selects the largest cluster, then the lowest
                // seed index, independently of thread execution order.
                const std::uint64_t rank = (static_cast<std::uint64_t>(members.size()) << 32)
                                        | static_cast<std::uint32_t>(N - seed);
                best_candidate = std::max(best_candidate, rank);
            }

            #pragma omp single
            {
                if ((best_candidate >> 32) == 1) {
                    // No remaining pair can form a cluster. The sequential
                    // algorithm would emit these singletons in seed order.
                    for (const int seed : unclustered_indices) {
                        clusters.push_back({std::move(candidate_clusters[seed]), seed});
                    }
                    unclustered_indices.clear();
                } else {
                    const int best_seed = N - static_cast<int>(static_cast<std::uint32_t>(best_candidate));
                    clusters.push_back({std::move(candidate_clusters[best_seed]), best_seed});
                    for (const int member : clusters.back().members) {
                        clustered[member] = 1;
                    }
                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                       [&clustered](int idx) { return clustered[idx] != 0; }),
                        unclustered_indices.end());
                }
                best_candidate = 0;
            } // Barrier publishes membership changes before the next round.
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
