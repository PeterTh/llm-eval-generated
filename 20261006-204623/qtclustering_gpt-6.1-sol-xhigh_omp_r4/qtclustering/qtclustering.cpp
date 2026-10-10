// QT Clustering Benchmark - OpenMP shared-memory version
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
#include <numeric>
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

// Only pairs strictly below the threshold can occur in the same cluster.
// Keep neighbors sorted by point index, with the original double distances:
// comparing squared distances could change ties after sqrt rounding.
struct Neighborhood {
    std::vector<int> indices;
    std::vector<double> distances;
};

// Reused by one worker for all of its seeds, without shared scratch storage.
struct CandidateWorkspace {
    std::vector<int> indices;
    std::vector<double> diameters;
};

void generateCandidateCluster(const int seed,
                              const std::vector<unsigned char>& clustered,
                              const std::vector<Neighborhood>& neighborhoods,
                              CandidateWorkspace& workspace,
                              std::vector<int>& members) {
    workspace.indices.clear();
    workspace.diameters.clear();
    const auto& initial = neighborhoods[seed];
    workspace.indices.reserve(initial.indices.size());
    workspace.diameters.reserve(initial.indices.size());
    for (size_t i = 0; i < initial.indices.size(); ++i) {
        const int candidate = initial.indices[i];
        if (!clustered[candidate]) {
            workspace.indices.push_back(candidate);
            workspace.diameters.push_back(initial.distances[i]);
        }
    }

    members.clear();
    members.push_back(seed);
    while (!workspace.indices.empty()) {
        double min_diameter = std::numeric_limits<double>::max();
        int closest = -1;
        for (size_t i = 0; i < workspace.indices.size(); ++i) {
            if (workspace.diameters[i] < min_diameter) {
                min_diameter = workspace.diameters[i];
                closest = workspace.indices[i];
            }
        }
        if (closest < 0) break;
        members.push_back(closest);

        // Intersect with the new member's neighbors. A rejected point can
        // never become eligible again as the cluster grows. The selected
        // point is also removed, since a neighborhood excludes its own point.
        const auto& neighbors = neighborhoods[closest];
        size_t next_neighbor = 0;
        size_t kept = 0;
        for (size_t i = 0; i < workspace.indices.size(); ++i) {
            const int candidate = workspace.indices[i];
            while (next_neighbor < neighbors.indices.size() &&
                   neighbors.indices[next_neighbor] < candidate) {
                ++next_neighbor;
            }
            if (next_neighbor == neighbors.indices.size()) break;
            if (neighbors.indices[next_neighbor] == candidate) {
                workspace.indices[kept] = candidate;
                workspace.diameters[kept] = std::max(workspace.diameters[i],
                                                   neighbors.distances[next_neighbor]);
                ++kept;
                ++next_neighbor;
            }
        }
        workspace.indices.resize(kept);
        workspace.diameters.resize(kept);
    }
}

// Separate cache lines keep the per-worker reduction results independent.
struct alignas(64) SeedResult {
    int cardinality = -1;
    int seed = -1;
};

inline bool betterSeed(const int cardinality, const int seed, const SeedResult& best) {
    return cardinality > best.cardinality ||
           (cardinality == best.cardinality && seed < best.seed);
}

// The winning clusters are committed sequentially, but all seeds in a round
// see the same immutable clustered flags and are evaluated in parallel.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices(N);
    std::iota(unclustered_indices.begin(), unclustered_indices.end(), 0);
    std::vector<Cluster> clusters;

    std::vector<Neighborhood> neighborhoods(N);
    std::vector<std::vector<int>> candidate_members(N);
    std::vector<SeedResult> worker_results(omp_get_max_threads());
    bool complete_neighborhoods = true;

    // One persistent team avoids creating threads for each clustering round.
    #pragma omp parallel
    {
        const int worker = omp_get_thread_num();
        CandidateWorkspace workspace;
        std::vector<double> distances(N);

        #pragma omp for schedule(static) reduction(&&:complete_neighborhoods)
        for (int seed = 0; seed < N; ++seed) {
            #pragma omp simd
            for (int candidate = 0; candidate < N; ++candidate) {
                distances[candidate] = std::max(0.0, distance(points[candidate], points[seed]));
            }
            size_t count = 0;
            for (int candidate = 0; candidate < N; ++candidate) {
                if (candidate != seed && distances[candidate] < threshold) ++count;
            }
            auto& neighbors = neighborhoods[seed];
            neighbors.indices.reserve(count);
            neighbors.distances.reserve(count);
            for (int candidate = 0; candidate < N; ++candidate) {
                if (candidate != seed && distances[candidate] < threshold) {
                    neighbors.indices.push_back(candidate);
                    neighbors.distances.push_back(distances[candidate]);
                }
            }
            complete_neighborhoods = complete_neighborhoods && count == static_cast<size_t>(N - 1);
        }

        #pragma omp single
        {
            // When every pair is eligible, the first seed wins with all N
            // points. Preserve its greedy member order without trying N seeds.
            if (N > 0 && complete_neighborhoods) {
                generateCandidateCluster(0, clustered, neighborhoods, workspace, candidate_members[0]);
                clusters.push_back({std::move(candidate_members[0]), 0});
                unclustered_indices.clear();
            }
        }
        distances.clear();
        distances.shrink_to_fit();

        while (!unclustered_indices.empty()) {
            SeedResult local_best;

            // Cluster sizes vary widely, so distribute seeds dynamically.
            #pragma omp for schedule(dynamic, 8) nowait
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                const int seed = unclustered_indices[i];
                auto& members = candidate_members[seed];

                // Removing points outside a cached candidate cannot change
                // any of its greedy choices. Rebuild only if a member was lost.
                if (members.empty() ||
                    std::any_of(members.begin(), members.end(),
                                [&clustered](int member) { return clustered[member] != 0; })) {
                    generateCandidateCluster(seed, clustered, neighborhoods, workspace, members);
                }
                const int cardinality = static_cast<int>(members.size());
                if (betterSeed(cardinality, seed, local_best)) {
                    local_best = {cardinality, seed};
                }
            }
            worker_results[worker] = local_best;

            #pragma omp barrier
            #pragma omp single
            {
                SeedResult best;
                for (int i = 0; i < omp_get_num_threads(); ++i) {
                    const auto& result = worker_results[i];
                    if (betterSeed(result.cardinality, result.seed, best)) best = result;
                }

                // A tie selects the lowest seed, just as the original ascending
                // scan did, regardless of the order in which workers finish.
                if (best.cardinality == 1) {
                    // All remaining clusters are singletons in ascending order.
                    for (int seed : unclustered_indices) clusters.push_back({{seed}, seed});
                    unclustered_indices.clear();
                } else {
                    clusters.push_back({std::move(candidate_members[best.seed]), best.seed});
                    for (int member : clusters.back().members) clustered[member] = 1;
                    unclustered_indices.erase(
                        std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                                       [&clustered](int idx) { return clustered[idx] != 0; }),
                        unclustered_indices.end());
                }
            } // The single barrier publishes flags and indices before the next round.
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
