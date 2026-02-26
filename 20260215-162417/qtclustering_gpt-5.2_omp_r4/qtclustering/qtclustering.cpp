// QT Clustering Benchmark - OpenMP Parallel Version
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

// Calculate Euclidean distance between two points (for validation/printing)
inline double distance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

inline double distanceSquared(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Precompute squared distances (avoids repeated sqrt in the hot path)
static std::vector<double> computeDistanceSquaredMatrix(const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    std::vector<double> dist2(static_cast<size_t>(N) * static_cast<size_t>(N));

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        const Point pi = points[i];
        double* row = dist2.data() + static_cast<size_t>(i) * static_cast<size_t>(N);
        for (int j = 0; j < N; ++j) {
            const double dx = pi.x - points[j].x;
            const double dy = pi.y - points[j].y;
            row[j] = dx * dx + dy * dy;
        }
    }

    return dist2;
}

// Find the closest unclustered point to the current cluster that maintains diameter^2 < threshold2
// Returns -1 if no such point exists
static int findClosestPoint(const std::vector<int>& cluster_members,
                            const std::vector<uint8_t>& clustered,
                            const std::vector<uint8_t>& in_cluster,
                            const Point* points,
                            const double* dist2, // optional; nullptr => compute on the fly
                            const double threshold2,
                            const int point_count) {
    int closest_point = -1;
    double min_diameter2 = std::numeric_limits<double>::max();

    // Try each unclustered point as a candidate
    for (int candidate = 0; candidate < point_count; ++candidate) {
        if (clustered[candidate] || in_cluster[candidate]) continue;

        const Point pc = points[candidate];
        const double* row = dist2 ? (dist2 + static_cast<size_t>(candidate) * static_cast<size_t>(point_count)) : nullptr;

        // Calculate the maximum distance^2 from candidate to all cluster members
        double max_dist2 = 0.0;
        for (size_t i = 0; i < cluster_members.size(); ++i) {
            const int member = cluster_members[i];
            const double d2 = row ? row[member] : distanceSquared(pc, points[member]);
            if (d2 > max_dist2) max_dist2 = d2;

            // Prune if this candidate is already worse than current best or violates threshold
            if (max_dist2 >= threshold2 || max_dist2 >= min_diameter2) {
                break;
            }
        }

        if (max_dist2 < threshold2 && max_dist2 < min_diameter2) {
            min_diameter2 = max_dist2;
            closest_point = candidate;
        }
    }

    return closest_point;
}

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
static int generateCandidateCluster(const int seed_point,
                                    const std::vector<uint8_t>& clustered,
                                    const Point* points,
                                    const double* dist2, // optional; nullptr => compute on the fly
                                    const double threshold2,
                                    const int point_count,
                                    std::vector<int>& out_members) {
    std::vector<uint8_t> in_cluster(static_cast<size_t>(point_count), 0);
    out_members.clear();
    out_members.reserve(static_cast<size_t>(point_count));

    // Add seed point
    in_cluster[seed_point] = 1;
    out_members.push_back(seed_point);

    // Iteratively add closest points
    while (static_cast<int>(out_members.size()) < point_count) {
        const int closest =
            findClosestPoint(out_members, clustered, in_cluster, points, dist2, threshold2, point_count);
        if (closest < 0) break;

        in_cluster[closest] = 1;
        out_members.push_back(closest);
    }

    return static_cast<int>(out_members.size());
}

// Main QT clustering algorithm
static std::vector<Cluster> qtClustering(const std::vector<Point>& points, const double threshold) {
    const int N = static_cast<int>(points.size());
    const double threshold2 = threshold * threshold;

    // Keep the algorithm robust for larger N by bounding O(N^2) memory.
    constexpr int64_t kMaxDist2Elements = 32LL * 1024LL * 1024LL; // 256 MiB of doubles
    const int64_t n64 = static_cast<int64_t>(N);
    const bool use_dist2_matrix = (n64 > 0) && (n64 <= (std::numeric_limits<int64_t>::max() / n64)) &&
                                  (n64 * n64 <= kMaxDist2Elements);

    std::vector<double> dist2_storage;
    const double* dist2 = nullptr;
    if (use_dist2_matrix) {
        dist2_storage = computeDistanceSquaredMatrix(points);
        dist2 = dist2_storage.data();
    }

    std::vector<uint8_t> clustered(static_cast<size_t>(N), 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    unclustered_indices.reserve(static_cast<size_t>(N));
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    int max_cardinality = -1;
    int best_seed = -1;
    size_t best_order = std::numeric_limits<size_t>::max();
    std::vector<int> best_cluster_members;
    bool stop = false;

    #pragma omp parallel
    {
        std::vector<int> local_best_members;
        std::vector<int> candidate_members;

        while (true) {
            #pragma omp single
            {
                if (unclustered_indices.empty()) {
                    stop = true;
                }
                max_cardinality = -1;
                best_seed = -1;
                best_order = std::numeric_limits<size_t>::max();
                best_cluster_members.clear();
            }

            #pragma omp barrier
            if (stop) break;

            int local_max_cardinality = -1;
            int local_best_seed = -1;
            size_t local_best_order = std::numeric_limits<size_t>::max();
            local_best_members.clear();

            #pragma omp for schedule(dynamic, 1) nowait
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                const int seed = unclustered_indices[i];
                if (clustered[seed]) continue;

                const int cardinality =
                    generateCandidateCluster(seed, clustered, points.data(), dist2, threshold2, N, candidate_members);

                if (cardinality > local_max_cardinality ||
                    (cardinality == local_max_cardinality && cardinality >= 0 && i < local_best_order)) {
                    local_max_cardinality = cardinality;
                    local_best_seed = seed;
                    local_best_order = i;
                    local_best_members.swap(candidate_members);
                    candidate_members.clear();
                }
            }

            #pragma omp critical
            {
                if (local_max_cardinality > max_cardinality ||
                    (local_max_cardinality == max_cardinality && local_max_cardinality >= 0 && local_best_order < best_order)) {
                    max_cardinality = local_max_cardinality;
                    best_seed = local_best_seed;
                    best_order = local_best_order;
                    best_cluster_members = std::move(local_best_members);
                }
            }

            #pragma omp barrier

            #pragma omp single
            {
                if (best_seed >= 0 && max_cardinality > 0) {
                    Cluster cluster;
                    cluster.seed_point = best_seed;
                    cluster.members = std::move(best_cluster_members);
                    clusters.push_back(std::move(cluster));

                    for (size_t i = 0; i < clusters.back().members.size(); ++i) {
                        clustered[clusters.back().members[i]] = 1;
                    }

                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                       [&clustered](int idx) { return clustered[idx] != 0; }),
                        unclustered_indices.end());
                } else {
                    stop = true;
                }
            }

            #pragma omp barrier
            if (stop) break;
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
                const double dist = distance(points[cluster.members[i]], points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }

        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", c, cluster.members.size(), cluster.seed_point,
                   max_diameter);
        }

        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", c, max_diameter, threshold);
            valid = false;
        }
    }

    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n", member, membership[member], c);
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

    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n", points.size(), clustered_count,
           points.size() - clustered_count);

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
        printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", num_points, threshold);
        return 1;
    }

    printf("QT Clustering Benchmark\n");
    printf("Number of points: %d\n", num_points);
    printf("Distance threshold: %.2f\n", threshold);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Generate synthetic data
    std::vector<Point> points(static_cast<size_t>(num_points));
    generateSyntheticData(points, num_points);

    // Perform QT clustering
    auto cluster_start = std::chrono::high_resolution_clock::now();

    const std::vector<Cluster> clusters = qtClustering(points, threshold);

    auto cluster_end = std::chrono::high_resolution_clock::now();
    auto cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(cluster_end - cluster_start);

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

    const double avg_cluster_size = clusters.empty() ? 0.0 : static_cast<double>(total_clustered) / clusters.size();

    printf("Points clustered: %d / %d (%.1f%%)\n", total_clustered, num_points, 100.0 * total_clustered / num_points);
    printf("Average cluster size: %.2f\n", avg_cluster_size);
    printf("Maximum cluster size: %d\n", max_cluster_size);

    // Performance metrics
    const double time_sec = cluster_time.count() / 1000.0;
    const double clusters_per_sec = clusters.size() / time_sec;
    const double points_per_sec = num_points / time_sec;
    printf("Performance: %.1f clusters/s, %.1f points/s\n", clusters_per_sec, points_per_sec);

    // Print results for external validation
    if (printResults) {
        // Serialize cluster membership for hashing
        std::vector<double> membershipData;
        membershipData.reserve(static_cast<size_t>(num_points));
        std::vector<int> membership(static_cast<size_t>(num_points), -1);
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
        }
        printf("Validation: FAILED\n");
        return 1;
    }

    return 0;
}
