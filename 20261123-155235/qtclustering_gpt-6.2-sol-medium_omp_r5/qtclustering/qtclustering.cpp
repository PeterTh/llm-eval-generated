// QT Clustering Benchmark
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <set>
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

namespace {

struct PointRange {
    int begin, end;
};

struct CandidatePoint {
    int index;
    double diameter;
};

struct alignas(64) CandidateScratch {
    std::vector<CandidatePoint> available;
};

// A candidate's running maximum only changes when a new member is added.
void generateCandidateCluster(
    int seed, const std::vector<Point>& points, double threshold,
    const std::vector<int>& sorted_indices, PointRange range,
    const std::vector<unsigned char>& clustered,
    std::vector<CandidatePoint>& available, std::vector<int>& members) {
    members.clear();
    members.push_back(seed);
    available.clear();
    for (int i = range.begin; i < range.end; ++i) {
        const int index = sorted_indices[i];
        if (index != seed && !clustered[index]) {
            available.push_back({index, 0.0});
        }
    }

    size_t count = available.size();
    while (count > 0) {
        const Point latest = points[members.back()];
        double best_diameter = threshold;
        int best_index = -1;
        size_t best_position = 0;
        size_t kept = 0;
        for (size_t i = 0; i < count; ++i) {
            const int index = available[i].index;
            const double diameter = std::max(available[i].diameter,
                                             distance(points[index], latest));
            if (diameter < threshold) {
                if (diameter < best_diameter ||
                    (diameter == best_diameter && index < best_index)) {
                    best_diameter = diameter;
                    best_index = index;
                    best_position = kept;
                }
                available[kept++] = {index, diameter};
            }
        }
        if (best_index < 0) {
            break;
        }
        members.push_back(best_index);
        // A point rejected by the diameter test can never become eligible again.
        count = kept - 1;
        available[best_position] = available[count];
    }
}

} // namespace

std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int point_count = static_cast<int>(points.size());
    if (point_count == 0) {
        return {};
    }

    // The x separation gives a lower bound on the Euclidean distance.
    std::vector<int> sorted_indices(point_count);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    const bool finite_points = std::all_of(points.begin(), points.end(), [](const Point& point) {
        return std::isfinite(point.x) && std::isfinite(point.y);
    });
    if (finite_points) {
        std::sort(sorted_indices.begin(), sorted_indices.end(), [&](int a, int b) {
            return points[a].x < points[b].x;
        });
    }

    // Keep enough seeds per worker to amortize scheduling and team setup.
    const int max_threads = std::min({omp_get_max_threads(), point_count,
                                      std::max(2, 1 + (point_count - 1) / 128)});
    std::vector<PointRange> ranges(point_count);
    std::vector<unsigned char> clustered(point_count, 0);
    std::vector<std::vector<int>> candidates(point_count);
    std::vector<CandidateScratch> scratch(max_threads);
    const double limit = std::min(threshold, std::numeric_limits<double>::max());
    const int first_seed_position = static_cast<int>(
        std::find(sorted_indices.begin(), sorted_indices.end(), 0) - sorted_indices.begin());
    std::atomic<bool> full_cluster{false};

    #pragma omp parallel for num_threads(max_threads) schedule(dynamic, 4)
    for (int work = 0; work < point_count; ++work) {
        if (full_cluster.load(std::memory_order_relaxed)) {
            continue;
        }
        // Give seed zero a head start. If it takes every point, no other seed can beat it.
        int pos = work;
        if (work == 0) {
            pos = first_seed_position;
        } else if (work == first_seed_position) {
            pos = 0;
        }
        const int seed = sorted_indices[pos];
        PointRange range{0, point_count};
        if (finite_points) {
            const auto is_close = [&](int index) {
                const double dx = points[index].x - points[seed].x;
                return std::sqrt(dx * dx) < limit;
            };
            const auto center = sorted_indices.begin() + pos;
            const auto first = std::partition_point(sorted_indices.begin(), center,
                                                    [&](int index) { return !is_close(index); });
            const auto last = std::partition_point(center + 1, sorted_indices.end(), is_close);
            range = {static_cast<int>(first - sorted_indices.begin()),
                     static_cast<int>(last - sorted_indices.begin())};
        }
        ranges[seed] = range;
        generateCandidateCluster(seed, points, limit, sorted_indices, range, clustered,
                                 scratch[omp_get_thread_num()].available, candidates[seed]);
        if (seed == 0 && candidates[seed].size() == points.size()) {
            full_cluster.store(true, std::memory_order_relaxed);
        }
    }

    std::vector<Cluster> clusters;
    if (full_cluster.load(std::memory_order_relaxed)) {
        clusters.push_back({std::move(candidates[0]), 0});
        return clusters;
    }

    // Rank by largest candidate first, breaking ties by lowest seed index.
    using Rank = std::pair<size_t, int>;
    std::set<Rank> ranking;
    for (int seed = 0; seed < point_count; ++seed) {
        ranking.emplace(candidates[seed].size(), -seed);
    }

    std::vector<int> affected;
    std::vector<size_t> old_sizes(point_count);
    while (!ranking.empty()) {
        if (ranking.rbegin()->first == 1) {
            // All remaining candidates are singletons, in seed order.
            for (auto it = ranking.rbegin(); it != ranking.rend(); ++it) {
                const int seed = -it->second;
                clusters.push_back({std::move(candidates[seed]), seed});
            }
            break;
        }
        const int best_seed = -ranking.rbegin()->second;
        auto& cluster = clusters.emplace_back(Cluster{std::move(candidates[best_seed]), best_seed});
        ranking.erase(std::prev(ranking.end()));

        // Any surviving seed whose candidate lost a member is within the
        // combined x ranges of the removed points.
        int begin = point_count;
        int end = 0;
        for (const int member : cluster.members) {
            clustered[member] = 1;
            if (member != best_seed) {
                ranking.erase({candidates[member].size(), -member});
                std::vector<int>().swap(candidates[member]);
            }
            begin = std::min(begin, ranges[member].begin);
            end = std::max(end, ranges[member].end);
        }

        affected.clear();
        for (int pos = begin; pos < end; ++pos) {
            const int seed = sorted_indices[pos];
            if (!clustered[seed] && candidates[seed].size() > 1) {
                affected.push_back(seed);
            }
        }
        if (affected.empty()) {
            continue;
        }

        const int affected_count = static_cast<int>(affected.size());
        const int threads = std::min(max_threads, 1 + (affected_count - 1) / 32);
        #pragma omp parallel for num_threads(threads) schedule(dynamic, 4)
        for (int i = 0; i < affected_count; ++i) {
            const int seed = affected[i];
            auto& members = candidates[seed];
            old_sizes[i] = members.size();
            // Removing points leaves the same greedy choices if none of
            // the chosen members were removed.
            if (std::any_of(members.begin(), members.end(),
                            [&](int member) { return clustered[member] != 0; })) {
                generateCandidateCluster(seed, points, limit, sorted_indices, ranges[seed],
                                         clustered, scratch[omp_get_thread_num()].available,
                                         members);
            }
        }

        for (int i = 0; i < affected_count; ++i) {
            const int seed = affected[i];
            const size_t new_size = candidates[seed].size();
            if (new_size != old_sizes[i]) {
                ranking.erase({old_sizes[i], -seed});
                ranking.emplace(new_size, -seed);
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
    const double time_sec = std::chrono::duration<double>(cluster_end - cluster_start).count();
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
