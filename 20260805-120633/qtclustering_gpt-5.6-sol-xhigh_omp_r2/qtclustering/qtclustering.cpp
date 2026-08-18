// QT Clustering Benchmark - OpenMP Version
// 
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
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

inline double squaredDistance(const Point& p1, const Point& p2) {
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}

// Scratch storage is private to an OpenMP worker and reused for every seed it
// processes.  Epoch tags avoid clearing the membership array for every seed.
struct CandidateWorkspace {
    explicit CandidateWorkspace(const int point_count)
        : membership_epoch(static_cast<size_t>(point_count), 0),
          max_squared_distances(static_cast<size_t>(point_count)) {
        members.reserve(static_cast<size_t>(point_count));
    }

    void startCandidate(const int seed_point) {
        ++epoch;
        if (epoch == 0) {
            std::fill(membership_epoch.begin(), membership_epoch.end(), 0);
            ++epoch;
        }
        members.clear();
        members.push_back(seed_point);
        membership_epoch[seed_point] = epoch;
    }

    std::vector<int> members;
    std::vector<std::uint32_t> membership_epoch;
    std::vector<double> max_squared_distances;
    std::uint32_t epoch = 0;
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<int>& unclustered_indices,
                              const std::vector<Point>& points,
                              const double threshold,
                              CandidateWorkspace& workspace) {
    workspace.startCandidate(seed_point);
    int newest_member = seed_point;
    
    // Iteratively add closest points
    while (workspace.members.size() < unclustered_indices.size()) {
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();
        double smallest_squared = std::numeric_limits<double>::max();
        const bool first_member = workspace.members.size() == 1;

        // A candidate's diameter can only change when the newest member is
        // added. Updating the saved maximum is equivalent to rescanning every
        // prior member.
        for (const int candidate : unclustered_indices) {
            if (workspace.membership_epoch[candidate] == workspace.epoch) {
                continue;
            }

            const double squared_dist = squaredDistance(
                points[candidate], points[newest_member]);
            const double max_squared = first_member
                ? squared_dist
                : std::max(workspace.max_squared_distances[candidate], squared_dist);
            workspace.max_squared_distances[candidate] = max_squared;

            // Strict comparisons and ascending candidate traversal preserve
            // the sequential implementation's deterministic tie breaking.
            // sqrt is monotonic, so only a new squared-distance record can
            // improve the winner.  Computing sqrt for those records retains
            // the original floating-point threshold and tie semantics while
            // removing it from nearly all inner-loop iterations.
            if (max_squared < smallest_squared) {
                smallest_squared = max_squared;
                const double max_dist = std::sqrt(max_squared);
                if (max_dist < threshold && max_dist < min_diameter) {
                    min_diameter = max_dist;
                    closest = candidate;
                }
            }
        }
        
        if (closest < 0) break; // No more points can be added
        
        workspace.membership_epoch[closest] = workspace.epoch;
        workspace.members.push_back(closest);
        newest_member = closest;
    }
    
    return static_cast<int>(workspace.members.size());
}

// Cache-line separation keeps independently written per-worker results from
// contending during a clustering round.
struct alignas(64) ThreadBest {
    int cardinality = -1;
    int seed = std::numeric_limits<int>::max();
    std::vector<int> members;
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(static_cast<size_t>(N), 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    unclustered_indices.reserve(static_cast<size_t>(N));
    clusters.reserve(static_cast<size_t>(N));
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }
    
    // Keep one OpenMP team alive across all sequential QT rounds.  Candidate
    // clusters for different seeds share only read-only state and therefore
    // scale independently; a deterministic per-thread reduction selects the
    // same lowest-index seed on cardinality ties as the original loop.
    std::vector<ThreadBest> thread_best(static_cast<size_t>(omp_get_max_threads()));
    size_t round_size = 0;
    bool finished = false;

    #pragma omp parallel default(none) \
        shared(points, clustered, unclustered_indices, clusters, thread_best, \
               round_size, finished) firstprivate(N, threshold)
    {
        const int thread_id = omp_get_thread_num();
        CandidateWorkspace workspace(N);

        while (true) {
            #pragma omp single
            {
                finished = finished || unclustered_indices.empty();
                round_size = unclustered_indices.size();
            }

            if (finished) {
                break;
            }

            ThreadBest& local_best = thread_best[static_cast<size_t>(thread_id)];
            local_best.cardinality = -1;
            local_best.seed = std::numeric_limits<int>::max();
            local_best.members.clear();

            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < round_size; ++i) {
                const int seed = unclustered_indices[i];
                const int cardinality = generateCandidateCluster(
                    seed, unclustered_indices, points, threshold, workspace);

                if (cardinality > local_best.cardinality ||
                    (cardinality == local_best.cardinality && seed < local_best.seed)) {
                    local_best.cardinality = cardinality;
                    local_best.seed = seed;
                    local_best.members = workspace.members;
                }
            }

            #pragma omp single
            {
                int winning_thread = -1;
                int max_cardinality = -1;
                int best_seed = std::numeric_limits<int>::max();
                const int team_size = omp_get_num_threads();

                for (int thread = 0; thread < team_size; ++thread) {
                    const ThreadBest& candidate = thread_best[static_cast<size_t>(thread)];
                    if (candidate.cardinality > max_cardinality ||
                        (candidate.cardinality == max_cardinality &&
                         candidate.seed < best_seed)) {
                        winning_thread = thread;
                        max_cardinality = candidate.cardinality;
                        best_seed = candidate.seed;
                    }
                }

                if (winning_thread >= 0 && max_cardinality > 0) {
                    Cluster cluster;
                    cluster.seed_point = best_seed;
                    cluster.members = std::move(
                        thread_best[static_cast<size_t>(winning_thread)].members);

                    for (const int member : cluster.members) {
                        clustered[member] = 1;
                    }
                    clusters.push_back(std::move(cluster));

                    unclustered_indices.erase(
                        std::remove_if(
                            unclustered_indices.begin(), unclustered_indices.end(),
                            [&clustered](const int idx) { return clustered[idx] != 0; }),
                        unclustered_indices.end());
                } else {
                    finished = true;
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
