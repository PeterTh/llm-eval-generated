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
#include <memory>
#include <new>
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

// Cache pairwise distances when the data set is small enough for the cache to
// remain practical.  This removes repeated square roots from the cubic QT
// search without imposing quadratic memory use on large inputs.
class DistanceCache {
public:
    explicit DistanceCache(const std::vector<Point>& points) {
        const size_t point_count = points.size();
        constexpr size_t max_cache_bytes = 256ULL * 1024ULL * 1024ULL;
        constexpr size_t max_cache_entries = max_cache_bytes / sizeof(double);

        if (point_count < 2 ||
            point_count - 1 > (2 * max_cache_entries) / point_count) {
            return;
        }

        const size_t entry_count = point_count * (point_count - 1) / 2;
        try {
            distances_.reset(new double[entry_count]);
        } catch (const std::bad_alloc&) {
            return;
        }

        // Rows have increasing lengths, so guided scheduling balances the
        // triangular matrix construction across the OpenMP team.
        const int worker_count = std::min(
            omp_get_max_threads(),
            std::max(1, static_cast<int>(point_count) / 16));
        #pragma omp parallel for num_threads(worker_count) schedule(guided)
        for (int first = 1; first < static_cast<int>(point_count); ++first) {
            const size_t row_offset = static_cast<size_t>(first) *
                                      static_cast<size_t>(first - 1) / 2;
            for (int second = 0; second < first; ++second) {
                distances_[row_offset + static_cast<size_t>(second)] =
                    distance(points[first], points[second]);
            }
        }
    }

    inline double between(const int first, const int second,
                          const std::vector<Point>& points) const {
        if (!distances_) {
            return distance(points[first], points[second]);
        }

        const int row = std::max(first, second);
        const int column = std::min(first, second);
        return distances_[static_cast<size_t>(row) *
                          static_cast<size_t>(row - 1) / 2 +
                          static_cast<size_t>(column)];
    }

private:
    std::unique_ptr<double[]> distances_;
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Point>& points,
                              const DistanceCache& distance_cache,
                              const double threshold,
                              const int point_count,
                              std::vector<int>& in_cluster_marks,
                              int& next_cluster_mark,
                              std::vector<double>& candidate_diameters,
                              std::vector<int>* cluster_members = nullptr) {
    // A thread-local generation marker makes each candidate construction
    // independent without repeatedly clearing an O(N) boolean vector.
    if (next_cluster_mark == std::numeric_limits<int>::max()) {
        std::fill(in_cluster_marks.begin(), in_cluster_marks.end(), 0);
        next_cluster_mark = 0;
    }
    const int cluster_mark = ++next_cluster_mark;
    in_cluster_marks[seed_point] = cluster_mark;

    int member_count = 1;
    int newest_member = seed_point;
    if (cluster_members) {
        cluster_members->clear();
        cluster_members->push_back(seed_point);
    }

    // The diameter of a candidate after adding a new member is the previous
    // diameter maxed with only that new distance.  This is exactly the same
    // value as recomputing over all members, but avoids repeated work.
    while (member_count < point_count) {
        int closest_point = -1;
        double min_diameter = std::numeric_limits<double>::max();

        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || in_cluster_marks[candidate] == cluster_mark) {
                continue;
            }

            const double newest_distance =
                distance_cache.between(candidate, newest_member, points);
            const double diameter = member_count == 1
                ? newest_distance
                : std::max(candidate_diameters[candidate], newest_distance);
            candidate_diameters[candidate] = diameter;

            // Strict comparisons retain the original ascending-index tie break.
            if (diameter < threshold && diameter < min_diameter) {
                min_diameter = diameter;
                closest_point = candidate;
            }
        }

        if (closest_point < 0) {
            break;
        }

        in_cluster_marks[closest_point] = cluster_mark;
        newest_member = closest_point;
        ++member_count;
        if (cluster_members) {
            cluster_members->push_back(closest_point);
        }
    }

    return member_count;
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    std::vector<int> candidate_cardinalities(N);
    const DistanceCache distance_cache(points);
    
    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    clusters.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    int unclustered_count = N;
    bool finished = false;

    // Candidate clusters are independent while the clustered set is fixed.
    // Keep one OpenMP team alive for all rounds so its per-thread workspaces
    // are reused instead of allocating them once per seed or round.
    // Starting with at least 16 seeds per worker avoids a large idle team on
    // small inputs while respecting a lower OMP_NUM_THREADS request.
    const int worker_count = std::min(omp_get_max_threads(), std::max(1, N / 16));
    #pragma omp parallel num_threads(worker_count) default(none) \
        shared(candidate_cardinalities, clustered, clusters, distance_cache, finished, \
               N, points, threshold, unclustered_count, unclustered_indices)
    {
        std::vector<int> in_cluster_marks(N, 0);
        std::vector<double> candidate_diameters(N);
        int next_cluster_mark = 0;

        while (true) {
            #pragma omp for schedule(guided)
            for (int i = 0; i < unclustered_count; ++i) {
                const int seed = unclustered_indices[i];
                candidate_cardinalities[i] = generateCandidateCluster(
                    seed, clustered, points, distance_cache, threshold, N,
                    in_cluster_marks, next_cluster_mark, candidate_diameters);
            }

            // Selection and state changes are deliberately serial: scanning
            // cardinalities in seed order preserves the original tie policy.
            #pragma omp single
            {
                int max_cardinality = -1;
                int best_seed = -1;
                for (int i = 0; i < unclustered_count; ++i) {
                    if (candidate_cardinalities[i] > max_cardinality) {
                        max_cardinality = candidate_cardinalities[i];
                        best_seed = unclustered_indices[i];
                    }
                }

                if (best_seed >= 0 && max_cardinality > 0) {
                    std::vector<int> best_cluster_members;
                    best_cluster_members.reserve(static_cast<size_t>(max_cardinality));
                    generateCandidateCluster(
                        best_seed, clustered, points, distance_cache, threshold, N,
                        in_cluster_marks, next_cluster_mark, candidate_diameters,
                        &best_cluster_members);

                    clusters.push_back({std::move(best_cluster_members), best_seed});
                    for (const int member : clusters.back().members) {
                        clustered[member] = 1;
                    }

                    const auto new_end = std::remove_if(
                        unclustered_indices.begin(), unclustered_indices.end(),
                        [&clustered](const int idx) { return clustered[idx] != 0; });
                    unclustered_count = static_cast<int>(
                        new_end - unclustered_indices.begin());
                    unclustered_indices.erase(new_end, unclustered_indices.end());
                } else {
                    finished = true;
                }
            }

            // The implicit barrier at the end of single publishes the updated
            // clustered set and work range before the next parallel round.
            if (finished) {
                break;
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
