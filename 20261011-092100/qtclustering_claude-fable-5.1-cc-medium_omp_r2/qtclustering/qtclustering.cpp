// QT Clustering Benchmark - OpenMP Parallel Version
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

// Per-thread scratch space reused across candidate cluster generations
struct Workspace {
    std::vector<double> max_dist;   // max distance from candidate to current cluster members
    std::vector<int> alive;         // eligible candidates (ascending index order)
    std::vector<int> next_alive;
};

// Generate a candidate cluster starting from a seed point.
//
// Semantics are identical to the original greedy procedure: at each step the
// unclustered point not yet in the cluster with the smallest maximum distance
// to all members (ties -> lowest index) is added, provided that distance is
// below the threshold. The maximum distance of every candidate to the cluster
// is maintained incrementally, and candidates whose maximum distance already
// reached the threshold are dropped permanently (it can only grow).
void generateCandidateCluster(const int seed_point,
                              const std::vector<int>& unclustered_indices,
                              const std::vector<double>& px,
                              const std::vector<double>& py,
                              const double threshold,
                              Workspace& ws,
                              std::vector<int>& members) {
    std::vector<double>& max_dist = ws.max_dist;
    std::vector<int>& alive = ws.alive;
    std::vector<int>& next_alive = ws.next_alive;

    members.clear();
    members.push_back(seed_point);

    // Initial alive set: all unclustered points except the seed, with their
    // distance to the seed, keeping only those below the threshold.
    alive.clear();
    {
        const double sx = px[seed_point];
        const double sy = py[seed_point];
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int c = unclustered_indices[i];
            if (c == seed_point) continue;
            const double dx = px[c] - sx;
            const double dy = py[c] - sy;
            const double d = std::sqrt(dx * dx + dy * dy);
            if (d < threshold) {
                max_dist[c] = d;
                alive.push_back(c);
            }
        }
    }

    while (!alive.empty()) {
        // Pick the alive candidate with the minimum max-distance (lowest index on ties)
        int closest = -1;
        double min_diameter = std::numeric_limits<double>::max();
        for (size_t i = 0; i < alive.size(); ++i) {
            const int c = alive[i];
            const double md = max_dist[c];
            if (md < min_diameter) {
                min_diameter = md;
                closest = c;
            }
        }
        if (closest < 0) break;  // cannot happen (all alive are < threshold), kept for safety

        members.push_back(closest);

        // Update max distances w.r.t. the new member and compact the alive list
        const double nx = px[closest];
        const double ny = py[closest];
        next_alive.clear();
        for (size_t i = 0; i < alive.size(); ++i) {
            const int c = alive[i];
            if (c == closest) continue;
            const double dx = px[c] - nx;
            const double dy = py[c] - ny;
            const double d = std::sqrt(dx * dx + dy * dy);
            const double md = std::max(max_dist[c], d);
            if (md < threshold) {
                max_dist[c] = md;
                next_alive.push_back(c);
            }
        }
        alive.swap(next_alive);
    }
}

// Main QT clustering algorithm
//
// Parallel over seeds with OpenMP. Candidate clusters are cached per seed:
// a seed's greedy candidate cluster only depends on the set of unclustered
// points it draws from, and removing points that are *not* in the candidate
// cluster never changes the greedy choices. Hence a cached candidate cluster
// stays valid until one of its members gets clustered, and only those seeds
// need recomputation after a cluster is committed.
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold) {
    const int N = static_cast<int>(points.size());
    std::vector<bool> clustered(N, false);
    std::vector<int> unclustered_indices;
    std::vector<Cluster> clusters;

    // Structure-of-arrays coordinates for cache-friendly access
    std::vector<double> px(N), py(N);
    for (int i = 0; i < N; ++i) {
        px[i] = points[i].x;
        py[i] = points[i].y;
    }

    // Initialize unclustered indices
    for (int i = 0; i < N; ++i) {
        unclustered_indices.push_back(i);
    }

    // Cached candidate clusters per seed and list of seeds needing recomputation
    std::vector<std::vector<int>> candidate(N);
    std::vector<int> dirty = unclustered_indices;

    // Per-thread workspaces
    const int max_threads = omp_get_max_threads();
    std::vector<Workspace> workspaces(max_threads);
    for (auto& ws : workspaces) {
        ws.max_dist.assign(N, 0.0);
        ws.alive.reserve(N);
        ws.next_alive.reserve(N);
    }

    // Main clustering loop
    while (!unclustered_indices.empty()) {
        // Recompute candidate clusters for seeds invalidated by the last commit
        const int num_dirty = static_cast<int>(dirty.size());
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_dirty; ++i) {
            Workspace& ws = workspaces[omp_get_thread_num()];
            const int seed = dirty[i];
            generateCandidateCluster(seed, unclustered_indices, px, py, threshold,
                                     ws, candidate[seed]);
        }

        // Select the largest candidate cluster (first in ascending seed order on ties)
        int max_cardinality = -1;
        int best_seed = -1;
        for (size_t i = 0; i < unclustered_indices.size(); ++i) {
            const int seed = unclustered_indices[i];
            const int cardinality = static_cast<int>(candidate[seed].size());
            if (cardinality > max_cardinality) {
                max_cardinality = cardinality;
                best_seed = seed;
            }
        }

        if (best_seed >= 0 && max_cardinality > 0) {
            Cluster cluster;
            cluster.seed_point = best_seed;
            cluster.members = candidate[best_seed];

            // Mark all members as clustered
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                clustered[cluster.members[i]] = true;
            }

            // Remove clustered points from unclustered list
            unclustered_indices.erase(
                std::remove_if(unclustered_indices.begin(), unclustered_indices.end(),
                              [&clustered](int idx) { return clustered[idx]; }),
                unclustered_indices.end()
            );

            // Seeds whose cached candidate cluster lost a member must be recomputed;
            // release the cache of committed points.
            dirty.clear();
            for (size_t i = 0; i < unclustered_indices.size(); ++i) {
                const int seed = unclustered_indices[i];
                const std::vector<int>& cm = candidate[seed];
                bool invalid = false;
                for (size_t j = 0; j < cm.size(); ++j) {
                    if (clustered[cm[j]]) { invalid = true; break; }
                }
                if (invalid) dirty.push_back(seed);
            }
            for (size_t i = 0; i < cluster.members.size(); ++i) {
                std::vector<int>().swap(candidate[cluster.members[i]]);
            }

            clusters.push_back(std::move(cluster));
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
