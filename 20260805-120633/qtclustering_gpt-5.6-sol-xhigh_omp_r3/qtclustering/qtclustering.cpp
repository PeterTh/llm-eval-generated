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
#include <omp.h>
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
    const double dx = p1.x - p2.x;
    const double dy = p1.y - p2.y;
    return std::sqrt(dx * dx + dy * dy);
}

struct CandidateState {
    int point;
    double diameter;
};

// Reused by one OpenMP worker. Keeping these allocations thread-local and alive
// across seeds is important because candidate generation is the inner hot path.
struct CandidateWorkspace {
    std::vector<int> members;
    std::vector<CandidateState> active;
};

// Each worker publishes only its best seed. Cache-line alignment prevents
// workers from contending on adjacent result metadata.
struct alignas(64) ThreadBest {
    std::vector<int> members;
    int cardinality = -1;
    int seed = -1;
    int seed_position = std::numeric_limits<int>::max();
};

// Generate a candidate cluster starting from a seed point
// Returns the cardinality (size) of the cluster
int generateCandidateCluster(const int seed_point,
                              const std::vector<int>& unclustered_indices,
                              const std::vector<Point>& points,
                              const double threshold,
                              CandidateWorkspace& workspace) {
    auto& members = workspace.members;
    auto& active = workspace.active;
    members.clear();
    active.clear();
    members.reserve(unclustered_indices.size());
    active.reserve(unclustered_indices.size());

    members.push_back(seed_point);

    // For a one-member cluster, each point's diameter is its distance to the
    // seed. The vectors remain in point-index order, preserving the original
    // first-point tie break.
    int closest_point = -1;
    double min_diameter = std::numeric_limits<double>::max();
    for (const int point : unclustered_indices) {
        if (point == seed_point) {
            continue;
        }

        const double candidate_diameter = distance(points[point], points[seed_point]);
        if (candidate_diameter < threshold) {
            active.push_back({point, candidate_diameter});
            if (candidate_diameter < min_diameter) {
                min_diameter = candidate_diameter;
                closest_point = point;
            }
        }
    }

    // A cluster diameter can only increase. Cache each surviving point's
    // diameter and update it for the newly added member, instead of rescanning
    // every prior member. Points that reach the threshold can be discarded
    // permanently. This is equivalent to the original max-distance scan.
    while (closest_point >= 0) {
        members.push_back(closest_point);

        int next_closest = -1;
        min_diameter = std::numeric_limits<double>::max();
        size_t write = 0;

        for (size_t read = 0; read < active.size(); ++read) {
            CandidateState candidate = active[read];
            if (candidate.point == closest_point) {
                continue;
            }

            candidate.diameter = std::max(
                candidate.diameter,
                distance(points[candidate.point], points[closest_point]));

            if (candidate.diameter < threshold) {
                active[write++] = candidate;
                if (candidate.diameter < min_diameter) {
                    min_diameter = candidate.diameter;
                    next_closest = candidate.point;
                }
            }
        }

        active.resize(write);
        closest_point = next_closest;
    }

    return static_cast<int>(members.size());
}

// Main QT clustering algorithm
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<unsigned char> clustered(N, 0);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    unclustered_indices.reserve(N);
    clusters.reserve(N);

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Disable runtime team shrinking so every QT iteration uses the OpenMP
    // team requested by the environment. The team itself persists across all
    // clusters, avoiding repeated parallel-region startup costs.
    omp_set_dynamic(0);
    std::vector<ThreadBest> thread_bests(static_cast<size_t>(omp_get_max_threads()));
    int remaining_count = N;

#pragma omp parallel shared(thread_bests, unclustered_indices, clustered, clusters, remaining_count)
    {
        const int thread_id = omp_get_thread_num();
        CandidateWorkspace workspace;
        ThreadBest local_best;

        while (true) {
            if (remaining_count == 0) {
                break;
            }

            local_best.cardinality = -1;
            local_best.seed = -1;
            local_best.seed_position = std::numeric_limits<int>::max();
            local_best.members.clear();

            // Candidate clusters are independent while the set of already
            // clustered points is fixed. Guided scheduling balances seeds
            // that produce different cluster sizes with low queue overhead.
#pragma omp for schedule(guided, 1) nowait
            for (int i = 0; i < remaining_count; ++i) {
                const int seed = unclustered_indices[static_cast<size_t>(i)];
                const int cardinality = generateCandidateCluster(
                    seed, unclustered_indices, points, threshold, workspace);

                if (cardinality > local_best.cardinality ||
                    (cardinality == local_best.cardinality &&
                     i < local_best.seed_position)) {
                    local_best.cardinality = cardinality;
                    local_best.seed = seed;
                    local_best.seed_position = i;
                    local_best.members.assign(workspace.members.begin(),
                                              workspace.members.end());
                }
            }

            ThreadBest& published = thread_bests[static_cast<size_t>(thread_id)];
            published.cardinality = local_best.cardinality;
            published.seed = local_best.seed;
            published.seed_position = local_best.seed_position;
            published.members.swap(local_best.members);

#pragma omp barrier
#pragma omp single
            {
                int best_thread = -1;
                int max_cardinality = -1;
                int best_position = std::numeric_limits<int>::max();
                const int team_size = omp_get_num_threads();

                // Explicitly reduce by cardinality and then original seed
                // position. This makes results independent of scheduling and
                // exactly preserves the sequential tie rule.
                for (int t = 0; t < team_size; ++t) {
                    const ThreadBest& candidate = thread_bests[static_cast<size_t>(t)];
                    if (candidate.cardinality > max_cardinality ||
                        (candidate.cardinality == max_cardinality &&
                         candidate.seed_position < best_position)) {
                        max_cardinality = candidate.cardinality;
                        best_position = candidate.seed_position;
                        best_thread = t;
                    }
                }

                ThreadBest& best = thread_bests[static_cast<size_t>(best_thread)];
                Cluster cluster;
                cluster.seed_point = best.seed;
                cluster.members.swap(best.members);

                for (const int member : cluster.members) {
                    clustered[static_cast<size_t>(member)] = 1;
                }
                clusters.push_back(std::move(cluster));

                unclustered_indices.erase(
                    std::remove_if(
                        unclustered_indices.begin(), unclustered_indices.end(),
                        [&clustered](const int idx) {
                            return clustered[static_cast<size_t>(idx)] != 0;
                        }),
                    unclustered_indices.end());
                remaining_count = static_cast<int>(unclustered_indices.size());
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
