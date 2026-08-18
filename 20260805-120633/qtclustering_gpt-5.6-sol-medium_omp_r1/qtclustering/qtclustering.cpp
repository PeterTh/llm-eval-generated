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
#include <cstdint>
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

struct CandidateWorkspace {
    explicit CandidateWorkspace(const int point_count)
        : in_cluster(point_count), max_distance(point_count) {
        members.reserve(point_count);
    }

    std::vector<std::uint8_t> in_cluster;
    std::vector<double> max_distance;
    std::vector<int> members;
};

// Build one seed's cluster.  max_distance is updated only for the newly added
// member, instead of recalculating all earlier member distances on every pass.
int generateCandidateCluster(const int seed_point,
                             const std::vector<std::uint8_t>& clustered,
                             const std::vector<double>& distances,
                             const double threshold,
                             const int point_count,
                             CandidateWorkspace& workspace) {
    std::fill(workspace.in_cluster.begin(), workspace.in_cluster.end(), 0);
    workspace.members.clear();
    workspace.in_cluster[seed_point] = 1;
    workspace.members.push_back(seed_point);

    const double* seed_distances = distances.data() +
        static_cast<size_t>(seed_point) * point_count;
    std::copy_n(seed_distances, point_count, workspace.max_distance.begin());

    while (static_cast<int>(workspace.members.size()) < point_count) {
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();

        // The ascending scan and strict comparison preserve the sequential
        // implementation's deterministic tie breaking.
        for (int candidate = 0; candidate < point_count; ++candidate) {
            if (clustered[candidate] || workspace.in_cluster[candidate]) continue;
            const double candidate_diameter = workspace.max_distance[candidate];
            if (candidate_diameter < threshold && candidate_diameter < min_diameter) {
                min_diameter = candidate_diameter;
                closest = candidate;
            }
        }

        if (closest < 0) break;
        workspace.in_cluster[closest] = 1;
        workspace.members.push_back(closest);

        const double* new_distances = distances.data() +
            static_cast<size_t>(closest) * point_count;
        #pragma omp simd
        for (int candidate = 0; candidate < point_count; ++candidate) {
            workspace.max_distance[candidate] =
                std::max(workspace.max_distance[candidate], new_distances[candidate]);
        }
    }

    return static_cast<int>(workspace.members.size());
}

struct alignas(64) ThreadBest {
    int cardinality = -1;
    int seed_position = -1;
    int seed = -1;
    std::vector<int> members;
};

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<std::uint8_t> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;
    unclustered_indices.reserve(N);

    // Candidate clusters reuse point-to-point distances extensively.  A dense
    // row-major table trades O(N^2) storage for substantially less arithmetic
    // and regular, cache-friendly reads in the timed clustering phase.
    std::vector<double> distances(static_cast<size_t>(N) * N);
    const int max_threads = std::min(omp_get_max_threads(), N);
    #pragma omp parallel for schedule(static) num_threads(max_threads)
    for (int i = 0; i < N; ++i) {
        double* row = distances.data() + static_cast<size_t>(i) * N;
        #pragma omp simd
        for (int j = 0; j < N; ++j) {
            row[j] = distance(points[i], points[j]);
        }
    }
    
    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    std::vector<ThreadBest> thread_bests(max_threads);
    std::vector<CandidateWorkspace> workspaces;
    workspaces.reserve(max_threads);
    for (int thread = 0; thread < max_threads; ++thread) {
        workspaces.emplace_back(N);
    }
    
    // Main clustering loop
    while (!unclustered_indices.empty()) {
        for (ThreadBest& best : thread_bests) {
            best.cardinality = -1;
            best.seed_position = -1;
            best.seed = -1;
        }

        // Seed evaluations read the same immutable clustered state and are
        // independent. Dynamic scheduling balances seeds whose clusters grow
        // to different sizes.
        const int active_threads = std::min(
            max_threads, static_cast<int>(unclustered_indices.size()));
        #pragma omp parallel num_threads(active_threads)
        {
            const int thread = omp_get_thread_num();
            ThreadBest& local_best = thread_bests[thread];
            CandidateWorkspace& workspace = workspaces[thread];

            #pragma omp for schedule(dynamic, 1)
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                const int seed = unclustered_indices[i];
                const int cardinality = generateCandidateCluster(
                    seed, clustered, distances, threshold, N, workspace);

                if (cardinality > local_best.cardinality ||
                    (cardinality == local_best.cardinality &&
                     (local_best.seed_position < 0 ||
                      static_cast<int>(i) < local_best.seed_position))) {
                    local_best.cardinality = cardinality;
                    local_best.seed_position = static_cast<int>(i);
                    local_best.seed = seed;
                    local_best.members = workspace.members;
                }
            }
        }

        const ThreadBest* best = nullptr;
        for (const ThreadBest& candidate : thread_bests) {
            if (candidate.seed < 0) continue;
            if (!best || candidate.cardinality > best->cardinality ||
                (candidate.cardinality == best->cardinality &&
                 candidate.seed_position < best->seed_position)) {
                best = &candidate;
            }
        }

        const int max_cardinality = best ? best->cardinality : -1;
        const int best_seed = best ? best->seed : -1;
        
        // If we found a cluster, add it
        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = best->members;
            clusters.push_back(std::move(cluster));
            
            // Mark all members as clustered
            for (const int member : best->members) {
                clustered[member] = 1;
            }
            
            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
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
