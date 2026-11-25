// QT Clustering Benchmark - OpenMP Version
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
#include <numeric>
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

struct Neighbor {
    int index;
    double diameter;

    bool precedes(const Neighbor& other) const {
        return diameter < other.diameter || (diameter == other.diameter && index < other.index);
    }
};

static void generateCandidateCluster(const int seed,
                                     const std::vector<unsigned char>& clustered,
                                     const std::vector<Point>& points,
                                     const std::vector<int>& by_x,
                                     const std::vector<int>& x_position,
                                     const double threshold,
                                     std::vector<Neighbor>& neighbors,
                                     std::vector<int>& members) {
    neighbors.clear();
    members.clear();
    members.push_back(seed);

    auto first = by_x.begin();
    auto last = by_x.end();
    if (!x_position.empty()) {
        const double seed_x = points[seed].x;
        const auto within_x_range = [&](int index) {
            const double dx = points[index].x - seed_x;
            return std::sqrt(dx * dx) < threshold;
        };
        const auto middle = first + x_position[seed];
        first = std::partition_point(first, middle, [&](int index) {
            return !within_x_range(index);
        });
        last = std::partition_point(middle, last, within_x_range);
    }

    size_t closest = 0;
    for (auto it = first; it != last; ++it) {
        const int index = *it;
        if (index == seed || clustered[index]) continue;
        const double diameter = std::max(0.0, distance(points[index], points[seed]));
        if (diameter < threshold) {
            const Neighbor neighbor{index, diameter};
            if (neighbors.empty() || neighbor.precedes(neighbors[closest])) {
                closest = neighbors.size();
            }
            neighbors.push_back(neighbor);
        }
    }

    while (!neighbors.empty()) {
        const int member = neighbors[closest].index;
        members.push_back(member);
        neighbors[closest] = neighbors.back();
        neighbors.pop_back();

        // Update each maximum using just the new member. Once a point fails
        // the threshold, its maximum can only stay the same or increase.
        const Point& added = points[member];
        closest = 0;
        size_t kept = 0;
        for (Neighbor neighbor : neighbors) {
            neighbor.diameter = std::max(neighbor.diameter, distance(points[neighbor.index], added));
            if (neighbor.diameter < threshold) {
                if (kept == 0 || neighbor.precedes(neighbors[closest])) {
                    closest = kept;
                }
                neighbors[kept++] = neighbor;
            }
        }
        neighbors.resize(kept);
    }
}

struct Bounds {
    Point lower, upper;

    bool overlaps(const Bounds& other) const {
        return lower.x <= other.upper.x && other.lower.x <= upper.x &&
               lower.y <= other.upper.y && other.lower.y <= upper.y;
    }
};

static Bounds memberBounds(const std::vector<int>& members, const std::vector<Point>& points) {
    const Point& first = points[members.front()];
    Bounds bounds{first, first};
    for (int index : members) {
        const Point& point = points[index];
        bounds.lower.x = std::min(bounds.lower.x, point.x);
        bounds.lower.y = std::min(bounds.lower.y, point.y);
        bounds.upper.x = std::max(bounds.upper.x, point.x);
        bounds.upper.y = std::max(bounds.upper.y, point.y);
    }
    return bounds;
}

struct CachedCandidate {
    size_t size = 0;
    Bounds bounds{};
    std::vector<int> members;
};

struct alignas(64) CandidateWinner {
    int seed = -1;
    size_t size = 0;

    bool accepts(int candidate_seed, size_t candidate_size) const {
        return candidate_size > size || (candidate_size == size && candidate_seed < seed);
    }
};

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int point_count = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<int> unclustered_indices(point_count);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<Cluster> clusters;

    // Only points within range along the x axis can pass the distance test.
    // Use a full scan for nonfinite coordinates to retain the distance
    // comparisons of the original algorithm, including its handling of NaN.
    std::vector<int> by_x = unclustered_indices;
    std::vector<int> x_position;
    if (std::all_of(points.begin(), points.end(), [](const Point& point) {
            return std::isfinite(point.x) && std::isfinite(point.y);
        })) {
        std::sort(by_x.begin(), by_x.end(), [&](int a, int b) {
            return points[a].x < points[b].x;
        });
        x_position.resize(point_count);
        for (int i = 0; i < point_count; ++i) {
            x_position[by_x[i]] = i;
        }
    }

    // Bound cached member storage even when candidate clusters are dense.
    constexpr size_t cache_budget = 64 * 1024 * 1024;
    const size_t cache_limit = cache_budget / sizeof(int) / std::max(1, point_count);
    std::vector<CachedCandidate> candidates(point_count);
    Bounds last_selected_bounds{};

    // Avoid launching more workers than the initial seed workload can use.
    const int thread_count = std::min(omp_get_max_threads(), 1 + std::max(0, point_count - 1) / 64);
    std::vector<CandidateWinner> winners(thread_count);

    #pragma omp parallel num_threads(thread_count)
    {
        std::vector<Neighbor> neighbors;
        std::vector<int> members;
        CandidateWinner& local = winners[omp_get_thread_num()];

        while (!unclustered_indices.empty()) {
            local.seed = -1;
            local.size = 0;

            #pragma omp for schedule(guided, 4)
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                const int seed = unclustered_indices[i];
                auto& cached = candidates[seed];

                // Removing points that were never selected cannot change any
                // greedy choice in the cached sequence. A large candidate
                // keeps only its size and bounds to limit cache memory.
                const bool changed = cached.members.empty()
                    ? x_position.empty() || cached.bounds.overlaps(last_selected_bounds)
                    : std::any_of(cached.members.begin(), cached.members.end(),
                                  [&](int index) { return clustered[index] != 0; });
                if (cached.size == 0 || changed) {
                    generateCandidateCluster(seed, clustered, points, by_x, x_position,
                                             threshold, neighbors, members);
                    cached.size = members.size();
                    if (members.size() <= cache_limit) {
                        cached.members.assign(members.begin(), members.end());
                    } else {
                        cached.members.clear();
                        cached.bounds = memberBounds(members, points);
                    }
                }

                if (local.accepts(seed, cached.size)) {
                    local.seed = seed;
                    local.size = cached.size;
                }
            }

            #pragma omp single
            {
                CandidateWinner* best = &winners.front();
                for (auto& winner : winners) {
                    if (best->accepts(winner.seed, winner.size)) {
                        best = &winner;
                    }
                }

                if (best->size == 1) {
                    // No eligible pair remains. All subsequent winners are
                    // singletons in the existing order of the active seeds.
                    for (int seed : unclustered_indices) {
                        clusters.push_back({{seed}, seed});
                    }
                    unclustered_indices.clear();
                } else {
                    auto& cached = candidates[best->seed];
                    if (cached.members.empty()) {
                        generateCandidateCluster(best->seed, clustered, points, by_x, x_position,
                                                 threshold, neighbors, members);
                    }
                    auto& selected = cached.members.empty() ? members : cached.members;
                    last_selected_bounds = memberBounds(selected, points);
                    for (int member : selected) {
                        clustered[member] = 1;
                    }
                    clusters.push_back({std::move(selected), best->seed});
                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                       [&](int index) { return clustered[index] != 0; }),
                        unclustered_indices.end());
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
