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

// Per-thread storage is retained for the whole clustering operation.  This
// avoids allocating O(N) temporary arrays for every seed point.
struct alignas(64) CandidateWorkspace {
    std::vector<double> max_distances;
    std::vector<int> members;
};

struct alignas(64) ThreadBest {
    int cardinality = -1;
    int seed_point = -1;
    std::vector<int> members;

    void reset() {
        cardinality = -1;
        seed_point = -1;
        members.clear();
    }
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

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_position,
                              const std::vector<int>& unclustered_indices,
                              const std::vector<double>& distances,
                              const double threshold,
                              const int point_count,
                              CandidateWorkspace& workspace) {
    const int active_count = static_cast<int>(unclustered_indices.size());
    workspace.max_distances.resize(active_count);
    workspace.members.clear();
    workspace.members.reserve(active_count);

    const int seed_point = unclustered_indices[seed_position];
    workspace.members.push_back(seed_point);

    // Initialize directly from the seed row and choose the first addition in
    // the same pass.  This saves a full scratch-array clear for every seed.
    int closest_position = -1;
    double min_diameter = threshold;
    const double* seed_distances =
        distances.data() + static_cast<size_t>(seed_point) * point_count;
    for (int position = 0; position < active_count; ++position) {
        if (position == seed_position) {
            workspace.max_distances[position] =
                std::numeric_limits<double>::infinity();
            continue;
        }

        const double candidate_diameter =
            seed_distances[unclustered_indices[position]];
        workspace.max_distances[position] = candidate_diameter;
        if (candidate_diameter < min_diameter) {
            min_diameter = candidate_diameter;
            closest_position = position;
        }
    }

    if (closest_position < 0) {
        return 1;
    }

    workspace.max_distances[closest_position] =
        std::numeric_limits<double>::infinity();
    int newest_member = unclustered_indices[closest_position];
    workspace.members.push_back(newest_member);

    // max_distances[p] is maintained incrementally.  After a point is added,
    // only its distance to every remaining point is new; recomputing distances
    // to all earlier members would produce the same maximum at much higher cost.
    while (static_cast<int>(workspace.members.size()) < active_count) {
        closest_position = -1;
        min_diameter = threshold;
        const double* distance_row =
            distances.data() + static_cast<size_t>(newest_member) * point_count;

        // unclustered_indices remains sorted, so strict comparison retains the
        // original lowest-index tie breaking.
        for (int position = 0; position < active_count; ++position) {
            const int candidate = unclustered_indices[position];
            const double candidate_diameter = std::max(
                workspace.max_distances[position], distance_row[candidate]);
            workspace.max_distances[position] = candidate_diameter;

            if (candidate_diameter < min_diameter) {
                min_diameter = candidate_diameter;
                closest_position = position;
            }
        }

        if (closest_position < 0) {
            break;
        }

        // Infinity makes selected positions ineligible in subsequent scans
        // without a separate membership bitmap or an unpredictable branch.
        workspace.max_distances[closest_position] =
            std::numeric_limits<double>::infinity();
        newest_member = unclustered_indices[closest_position];
        workspace.members.push_back(newest_member);
    }

    return static_cast<int>(workspace.members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Initialize unclustered indices
    unclustered_indices.reserve(N);
    clusters.reserve(N);
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Distances are reused by many seed candidates.  A full row-major matrix
    // makes the update for a newly-added member a cache-friendly row traversal
    // and computes each directed distance only once.
    const size_t matrix_size = static_cast<size_t>(N) * static_cast<size_t>(N);
    std::vector<double> distances(matrix_size);

    const int max_threads = omp_get_max_threads();
    std::vector<CandidateWorkspace> workspaces(max_threads);
    std::vector<ThreadBest> thread_bests(max_threads);
    std::vector<unsigned char> selected(N, 0);
    bool done = unclustered_indices.empty();

    // Keep one OpenMP team alive across all QT iterations.  This is important
    // as clusters get smaller: repeatedly creating a team would eventually
    // cost more than the remaining candidate work.
#pragma omp parallel shared(done, distances, unclustered_indices, clusters, workspaces, thread_bests, selected)
    {
        const int thread_id = omp_get_thread_num();
        CandidateWorkspace& workspace = workspaces[thread_id];
        ThreadBest& local_best = thread_bests[thread_id];

#pragma omp for schedule(static)
        for (int member = 0; member < N; ++member) {
            double* distance_row =
                distances.data() + static_cast<size_t>(member) * N;
#pragma omp simd
            for (int candidate = 0; candidate < N; ++candidate) {
                // Keep the argument order identical to the sequential search.
                distance_row[candidate] =
                    distance(points[candidate], points[member]);
            }
        }

        while (!done) {
            local_best.reset();

            // Every seed candidate reads the same immutable active-point set.
            // Dynamic scheduling balances seeds whose clusters grow to
            // different sizes.
#pragma omp for schedule(dynamic, 1)
            for (int position = 0;
                 position < static_cast<int>(unclustered_indices.size());
                 ++position) {
                const int seed = unclustered_indices[position];
                const int cardinality = generateCandidateCluster(
                    position, unclustered_indices, distances, threshold, N,
                    workspace);

                if (cardinality > local_best.cardinality ||
                    (cardinality == local_best.cardinality &&
                     seed < local_best.seed_point)) {
                    local_best.cardinality = cardinality;
                    local_best.seed_point = seed;
                    local_best.members = workspace.members;
                }
            }

            // The serial choice preserves the original deterministic rule:
            // largest cardinality, then the first (lowest-index) seed.
#pragma omp single
            {
                const int team_size = omp_get_num_threads();
                int winner = -1;
                for (int thread = 0; thread < team_size; ++thread) {
                    if (thread_bests[thread].cardinality < 0) {
                        continue;
                    }
                    if (winner < 0 ||
                        thread_bests[thread].cardinality >
                            thread_bests[winner].cardinality ||
                        (thread_bests[thread].cardinality ==
                             thread_bests[winner].cardinality &&
                         thread_bests[thread].seed_point <
                             thread_bests[winner].seed_point)) {
                        winner = thread;
                    }
                }

                if (winner >= 0 && thread_bests[winner].cardinality > 0) {
                    Cluster cluster;
                    cluster.seed_point = thread_bests[winner].seed_point;
                    cluster.members = thread_bests[winner].members;

                    for (const int member : cluster.members) {
                        selected[member] = 1;
                    }

                    unclustered_indices.erase(
                        std::remove_if(
                            unclustered_indices.begin(),
                            unclustered_indices.end(),
                            [&selected](const int point) {
                                return selected[point] != 0;
                            }),
                        unclustered_indices.end());
                    clusters.push_back(std::move(cluster));
                } else {
                    done = true;
                }

                if (unclustered_indices.empty()) {
                    done = true;
                }
            }
            // The implicit single barrier publishes the updated active set and
            // done flag before every thread evaluates the next iteration.
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
